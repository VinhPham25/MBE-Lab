"""
Stage E — Phase 2: Effective TE (ETE) Wiring (Section 10.4b).

**STATUS: batch executed (2026-08-31, 13/13 subjects) — see
``PHASE2_ETE_AI_HANDOFF.md`` and ``config.yaml``'s
``phase_2.executed_ete_or_significance: true``.**  Phase 2's phase-based
dSPTE is still retained for documentation/comparison only (see
``CLAUDE.md`` Section 3 — "the project has moved off phase-based TE
entirely"); this ETE track exists purely as a comparison point against
Phase 4's primary amplitude-ETE track, not as a methodological pivot back
to phase. This file proves the ``11_symbolic_ete_core.py`` contract is
truly method-agnostic (Section 10.6's completion check) by wiring it to a
second, structurally different raw pipeline. Re-run manually via the CLI
below if the raw Phase 2 dSPTE pipeline or its config changes.

Key structural difference from Phase 4's wiring (``19_amplitude_ete.py``):
``compute_multivariate_dspte_matrix(Phi_win, taus, config, roi_names)`` takes an
extra required ``taus`` argument (per-ROI embedding delays computed once
per *trial*, not per window) that does not fit the
``raw_compute_fn(window_data, config, roi_names) -> (EC, delay, meta)``
contract expected by ``compute_symbolic_ete_matrix()``. This file resolves
that mismatch with a closure that binds ``taus`` per-trial before the
window loop, and also runs the 0.04-0.15 Hz narrowband filter + Hilbert
phase extraction (Phase 2-specific preprocessing that Phase 4 does not
need) before calling into the shared ETE core -- proving the core needs
zero knowledge of either detail.

Output schema (parallel to Phase 4's, when run)
------------------------------------------------
``phase_2_ete_connectivity/sub-{id}_multivariate_ete_connectivity.npz``:
  same keys as ``phase_4_ete_connectivity``'s file (Section 10.5):
  ``Tr_N_connectivity``, ``Tr_N_ete_connectivity``, ``Tr_N_null_mean``,
  ``Tr_N_null_std``, ``Tr_N_p_value``, ``Tr_N_z_score``, ``Tr_N_delays``,
  plus ``subject_id, group, trial_ids, window_counts, roi_names_used,
  connection_labels, n_trials, params_json``.

CLI
---
python3 12_multivariate_ete.py --smoke-test
python3 12_multivariate_ete.py --benchmark SUBJECT_ID [--n-windows N]
python3 12_multivariate_ete.py --subject SUBJECT_ID
python3 12_multivariate_ete.py --batch
"""

from __future__ import annotations

import functools
import importlib.util as _ilu
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping (mirrors 06_multivariate_dspte.py / 19_amplitude_ete.py)
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).parent.parent.parent    # code/
_STAGE_E_ROOT = Path(__file__).parent.parent         # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy imports
# ---------------------------------------------------------------------------
_UTILS_TE_MOD = None
_PIPELINE_MOD = None    # phase_1/02_bivariate_dspte_pipeline.py (extract_connectivity_vector)
_SEG_MOD = None         # stage_d_granger/02_trial_segmentation.py
_ETE_CORE_MOD = None    # ../11_symbolic_ete_core.py
_MULTIVARIATE_DSPTE_MOD = None  # 06_multivariate_dspte.py (raw_compute_fn source + phase-extraction helper)


def _get_utils_te() -> object:
    global _UTILS_TE_MOD
    if _UTILS_TE_MOD is None:
        spec = _ilu.spec_from_file_location("utils_te_p2ete", _STAGE_E_ROOT / "utils_te.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UTILS_TE_MOD = mod
    return _UTILS_TE_MOD


def _get_pipeline_mod() -> object:
    global _PIPELINE_MOD
    if _PIPELINE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "bivariate_dspte_pipeline_p2ete", _STAGE_E_ROOT / "phase_1" / "02_bivariate_dspte_pipeline.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PIPELINE_MOD = mod
    return _PIPELINE_MOD


def _get_seg_mod() -> object:
    global _SEG_MOD
    if _SEG_MOD is None:
        seg_path = _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py"
        spec = _ilu.spec_from_file_location("trial_seg_p2ete", seg_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEG_MOD = mod
    return _SEG_MOD


def _get_ete_core() -> object:
    global _ETE_CORE_MOD
    if _ETE_CORE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "symbolic_ete_core_p2", _STAGE_E_ROOT / "11_symbolic_ete_core.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ETE_CORE_MOD = mod
    return _ETE_CORE_MOD


def _get_multivariate_dspte_mod() -> object:
    """Return cached phase_2/06_multivariate_dspte.py module."""
    global _MULTIVARIATE_DSPTE_MOD
    if _MULTIVARIATE_DSPTE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "multivariate_dspte_p2ete", Path(__file__).parent / "06_multivariate_dspte.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MULTIVARIATE_DSPTE_MOD = mod
    return _MULTIVARIATE_DSPTE_MOD


def _load_config_if_none(config: "dict | None") -> dict:
    if config is not None:
        return config
    try:
        from utils import load_config  # noqa
        return load_config()
    except ImportError:
        return {}


def _make_multivariate_raw_compute_fn(taus_trial: list):
    """
    Bind a fixed per-trial ``taus`` list into a closure matching the
    ``raw_compute_fn(window_data, config, roi_names) -> (EC, delay, meta)``
    contract expected by ``compute_symbolic_ete_matrix()``.

    This is the resolution to the signature mismatch called out in the
    module docstring: ``compute_multivariate_dspte_matrix`` needs ``taus``
    (computed once per trial from the full-trial phase signal, per Phase
    2's design — see ``06_multivariate_dspte.py``'s
    ``_filter_and_extract_phase_trial_p2``), which the shared ETE core has
    no notion of. Phase 4's ``compute_amplitude_conditional_dste_matrix``
    computes its equivalent (per-ROI embedding delay) internally per
    window, which is why it did not need this wrapper.
    """
    multivariate_mod = _get_multivariate_dspte_mod()

    def _bound_fn(window_data: np.ndarray, config: dict, roi_names: "list[str] | None"):
        return multivariate_mod.compute_multivariate_dspte_matrix(
            Phi_win=window_data, taus=taus_trial, config=config, roi_names=roi_names,
        )

    return _bound_fn


_ROI_NAMES_CANONICAL_P2ETE = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]


# ===========================================================================
# Per-subject ETE pipeline (parallel structure to 19_amplitude_ete.py)
# ===========================================================================

def _save_subject_multivariate_ete_npz(
    subject_id: str,
    per_trial: dict,
    valid_roi_names: list[str],
    out_path: Path,
    params: dict,
    config: dict,
) -> None:
    """Save per-subject Phase 2 ETE connectivity to .npz (parallel to Phase 4's schema)."""
    utils_te = _get_utils_te()
    conn_labels = utils_te.get_connection_labels()
    try:
        group = utils_te.get_subject_group(subject_id, config=config)
    except ValueError:
        group = "Unknown"

    trial_ids = sorted(per_trial.keys(), key=lambda k: int(k.split("_")[1]))
    save_dict: dict = {}
    window_counts: list[int] = []

    array_keys = [
        "connectivity", "ete_connectivity", "null_mean", "null_std",
        "p_value", "z_score", "delays",
    ]

    for trial_id in trial_ids:
        trial_wins = per_trial.get(trial_id, {})
        win_ids = sorted(trial_wins.keys(), key=lambda k: int(k.split("_")[1]))
        n_wins = len(win_ids)
        window_counts.append(n_wins)

        for key in array_keys:
            if n_wins > 0:
                cols = [trial_wins[w][key] for w in win_ids]
                arr = np.column_stack(cols).astype(np.float64)
            else:
                arr = np.empty((20, 0), dtype=np.float64)
            save_dict[f"{trial_id}_{key}"] = arr

    save_dict["subject_id"] = np.array(subject_id)
    save_dict["group"] = np.array(group)
    save_dict["trial_ids"] = np.array(trial_ids)
    save_dict["window_counts"] = np.array(window_counts, dtype=int)
    save_dict["n_trials"] = np.array(len(trial_ids), dtype=int)
    save_dict["roi_names_used"] = np.array(valid_roi_names)
    save_dict["connection_labels"] = np.array(conn_labels)
    save_dict["params_json"] = np.array(json.dumps(params))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **save_dict)
    logger.info(
        "_save_subject_multivariate_ete_npz [%s]: saved %d trials -> %s",
        subject_id, len(trial_ids), out_path,
    )


def process_subject_multivariate_ete(
    subject_id: str,
    study: str = "expert_novice",
    config: "dict | None" = None,
    overwrite: bool = False,
    n_windows_limit: "int | None" = None,
) -> dict:
    """
    Run the Phase 2 Effective TE pipeline for one subject.

    Parallel structure to ``process_subject_amplitude_ete()``
    (``phase_4/19_amplitude_ete.py``), but includes the Phase 2-specific
    narrowband-filter + full-trial Hilbert-phase preprocessing step (via
    ``06_multivariate_dspte.py``'s private helper) before the window loop, and
    binds ``taus_trial`` into ``raw_compute_fn`` via
    ``_make_multivariate_raw_compute_fn()``.

    Batch-executed 2026-08-31 (see module docstring) — safe to re-run with
    ``overwrite=True`` if the raw Phase 2 dSPTE pipeline or its config changes.
    """
    config = _load_config_if_none(config)
    utils_te = _get_utils_te()
    pipeline_mod = _get_pipeline_mod()
    seg_mod = _get_seg_mod()
    ete_core = _get_ete_core()
    multivariate_mod = _get_multivariate_dspte_mod()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    win_cfg = te_cfg.get("window", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})
    window_size = int(win_cfg.get("size_samples", 1350))
    stride = int(win_cfg.get("stride_samples", 1350))

    out_dir = utils_te.get_te_output_dir("phase_2_ete_connectivity", config=config)
    out_path = out_dir / f"sub-{subject_id}_multivariate_ete_connectivity.npz"

    if not overwrite and out_path.exists() and n_windows_limit is None:
        logger.info(
            "process_subject_multivariate_ete [%s]: cache hit -> %s", subject_id, out_path,
        )
        return {}

    t0_subj = time.perf_counter()

    Data_Reg, valid_roi_names, sfreq = utils_te.load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    raw = utils_te.load_preprocessed_fif(subject_id, study=study, config=config)
    segments = seg_mod.get_trial_segments(raw, config=config)

    filter_enabled = bool(filt_cfg.get("enabled", True))
    filt_f_lo = float(filt_cfg.get("f_lo", 0.04))
    filt_f_hi = float(filt_cfg.get("f_hi", 0.15))
    filt_order = int(filt_cfg.get("order", 3))

    deps = multivariate_mod._get_multivariate_module_dependencies()
    core_mod = deps["core"]

    per_trial: dict = {}
    n_windows_processed_total = 0

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset: offset + 1]

        if Y_trial.shape[1] < (window_size + 1):
            logger.warning(
                "process_subject_multivariate_ete [%s %s]: trial too short — skipping.",
                subject_id, trial_id,
            )
            continue

        # Phase 2-specific preprocessing: narrowband filter + full-trial
        # Hilbert phase (once per trial), matching 06_multivariate_dspte.py exactly.
        if filter_enabled:
            Phi_trial, taus_trial = multivariate_mod._filter_and_extract_phase_trial_p2(
                Y_trial, sfreq=sfreq, f_lo=filt_f_lo, f_hi=filt_f_hi, order=filt_order,
                core_mod=core_mod, subject_id=subject_id, trial_id=trial_id,
            )
        else:
            n_rois_t, T_t = Y_trial.shape
            Phi_trial = np.zeros_like(Y_trial)
            taus_trial = []
            for roi_i in range(n_rois_t):
                ph = core_mod.extract_phase_signal(Y_trial[roi_i])
                Phi_trial[roi_i] = ph
                taus_trial.append(core_mod.compute_embedding_delay(ph))

        raw_compute_fn = _make_multivariate_raw_compute_fn(taus_trial)

        per_trial[trial_id] = {}
        win_num = 1

        for w in range(window_size, Y_trial.shape[1], stride):
            if n_windows_limit is not None and win_num > n_windows_limit:
                break

            ws = w - window_size
            we = w
            win_id = f"Win_{win_num}"
            win_num += 1

            Phi_win = Phi_trial[:, ws: we + 1]

            matrices, meta = ete_core.compute_symbolic_ete_matrix(
                raw_compute_fn=raw_compute_fn,
                window_data=Phi_win,
                config=config,
                roi_names=valid_roi_names,
                method_key="phase_2",
            )
            n_windows_processed_total += 1

            logger.info(
                "process_subject_multivariate_ete [%s %s %s]: n_valid=%d  "
                "elapsed_raw=%.2fs  elapsed_ete=%.2fs",
                subject_id, trial_id, win_id, meta.get("n_valid", 0),
                meta.get("elapsed_raw_s", 0.0), meta.get("elapsed_ete_s", 0.0),
            )

            per_trial[trial_id][win_id] = {
                "connectivity": pipeline_mod.extract_connectivity_vector(
                    matrices["raw"], valid_roi_names
                ),
                "ete_connectivity": pipeline_mod.extract_connectivity_vector(
                    matrices["ete"], valid_roi_names
                ),
                "null_mean": pipeline_mod.extract_connectivity_vector(
                    matrices["null_mean"], valid_roi_names
                ),
                "null_std": pipeline_mod.extract_connectivity_vector(
                    matrices["null_std"], valid_roi_names
                ),
                "p_value": pipeline_mod.extract_connectivity_vector(
                    matrices["p_value"], valid_roi_names
                ),
                "z_score": pipeline_mod.extract_connectivity_vector(
                    matrices["z_score"], valid_roi_names
                ),
                "delays": pipeline_mod.extract_connectivity_vector(
                    matrices["delay"].astype(float), valid_roi_names
                ),
            }

    elapsed_subj = time.perf_counter() - t0_subj

    if n_windows_limit is None:
        try:
            group = utils_te.get_subject_group(subject_id, config=config)
        except ValueError:
            group = "Unknown"

        ete_cfg = te_cfg.get("phase_2", {}).get("ete", {})
        params = {
            "phase": "phase_2_multivariate_ete",
            "study": study,
            "group": group,
            "n_surrogates": int(ete_cfg.get("n_surrogates", 500)),
            "min_shift_fraction": float(ete_cfg.get("min_shift_fraction", 0.05)),
            "floor_estimator": str(ete_cfg.get("floor_estimator", "mean")),
            "random_seed_base": int(ete_cfg.get("random_seed_base", 42)),
            "n_windows_processed": n_windows_processed_total,
            "elapsed_s": round(elapsed_subj, 2),
        }
        _save_subject_multivariate_ete_npz(
            subject_id=subject_id, per_trial=per_trial, valid_roi_names=valid_roi_names,
            out_path=out_path, params=params, config=config,
        )

    logger.info(
        "process_subject_multivariate_ete [%s]: DONE  windows=%d  elapsed=%.1f s",
        subject_id, n_windows_processed_total, elapsed_subj,
    )
    return per_trial


# ===========================================================================
# Batch runner
# ===========================================================================

def run_batch_multivariate_ete(
    study: str = "expert_novice",
    subjects: "list[str] | None" = None,
    config: "dict | None" = None,
    overwrite: bool = False,
) -> dict:
    """Batch-process all subjects for Phase 2 Effective TE."""
    config = _load_config_if_none(config)
    utils_te = _get_utils_te()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    if subjects is None:
        subjects = list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))

    t0 = time.perf_counter()
    from datetime import datetime as _dt
    batch_log: dict = {
        "run_timestamp": _dt.now().isoformat(),
        "phase": "phase_2_multivariate_ete",
        "study": study,
        "n_subjects": len(subjects),
        "subjects": {},
    }

    for sid in subjects:
        t_sub = time.perf_counter()
        try:
            res = process_subject_multivariate_ete(
                subject_id=sid, study=study, config=config, overwrite=overwrite,
            )
            n_wins = sum(len(v) for v in res.values())
            batch_log["subjects"][sid] = {
                "status": "ok", "n_trials": len(res), "n_windows_total": n_wins,
                "elapsed_s": round(time.perf_counter() - t_sub, 2),
            }
        except Exception as exc:
            batch_log["subjects"][sid] = {
                "status": "failed", "error": str(exc),
                "elapsed_s": round(time.perf_counter() - t_sub, 2),
            }
            logger.error("run_batch_multivariate_ete [%s]: FAILED -- %s", sid, exc, exc_info=True)

    n_ok = sum(1 for v in batch_log["subjects"].values() if v.get("status") == "ok")
    batch_log["summary"] = {
        "n_success": n_ok, "n_failed": len(subjects) - n_ok,
        "total_elapsed_s": round(time.perf_counter() - t0, 2),
    }

    group_dir = utils_te.get_te_output_dir("phase_2_ete_group", config=config)
    group_dir.mkdir(parents=True, exist_ok=True)
    log_path = group_dir / "multivariate_ete_batch_log.json"
    with open(log_path, "w") as fh:
        json.dump(batch_log, fh, indent=2)

    logger.info(
        "run_batch_multivariate_ete: DONE  %d/%d subjects succeeded  total=%.1f s  log -> %s",
        n_ok, len(subjects), batch_log["summary"]["total_elapsed_s"], log_path,
    )
    return batch_log


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stage E Phase 2 -- Effective TE"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--benchmark", type=str, metavar="SUBJECT_ID")
    grp.add_argument("--subject", type=str, metavar="SUBJECT_ID")
    grp.add_argument("--batch", action="store_true")
    parser.add_argument("--n-windows", type=int, default=1)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    _cfg = _load_config_if_none(None)

    if args.benchmark:
        _t0 = time.perf_counter()
        _res = process_subject_multivariate_ete(
            subject_id=args.benchmark.strip(), config=_cfg, overwrite=True,
            n_windows_limit=args.n_windows,
        )
        _elapsed = time.perf_counter() - _t0
        _n_wins = sum(len(v) for v in _res.values())
        logger.info(
            "BENCHMARK: %d windows in %.1f s -> %.2f s/window",
            _n_wins, _elapsed, _elapsed / max(1, _n_wins),
        )
    elif args.subject:
        _res = process_subject_multivariate_ete(
            subject_id=args.subject.strip(), config=_cfg, overwrite=args.overwrite,
        )
        logger.info("subject=%s  trials=%d", args.subject, len(_res))
    elif args.batch:
        _log = run_batch_multivariate_ete(config=_cfg, overwrite=args.overwrite)
        logger.info("batch done: %s", _log.get("summary", {}))
