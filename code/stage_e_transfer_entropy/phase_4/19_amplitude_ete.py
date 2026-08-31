"""
Stage E — Phase 4: Effective TE (ETE) Wiring (Section 10.4a / 10.5).

**STATUS: RUN NOW.**  Thin wiring around the method-agnostic
``11_symbolic_ete_core.py`` — subject/batch orchestration only.  All the
actual surrogate-testing logic (circular shift, per-pair ETE, full-matrix
loop) lives in the core module; this file is deliberately small.

Mirrors ``process_subject_amplitude_dste()`` /
``run_batch_amplitude_dste()`` (Section 9.5/9.6, ``14_amplitude_dste.py``)
almost exactly, but calls ``compute_symbolic_ete_matrix()`` per window
instead of calling ``compute_amplitude_conditional_dste_matrix()`` directly
— the raw connectivity is still computed (once per window, inside the ETE
core), just with the additional surrogate-testing layer wrapped around it.

Output schema
--------------
``phase_4_ete_connectivity/sub-{id}_amplitude_ete_connectivity.npz``:
  - ``Tr_N_connectivity``      — raw TE (duplicated for convenience; same
                                  values as ``phase_4_connectivity``'s file
                                  for the same subject, assuming config is
                                  unchanged between runs)
  - ``Tr_N_ete_connectivity``  — floor-corrected ETE
  - ``Tr_N_null_mean``         — surrogate null distribution mean
  - ``Tr_N_null_std``          — surrogate null distribution std
  - ``Tr_N_p_value``           — one-sided surrogate p-value
  - ``Tr_N_z_score``           — (raw_te - null_mean) / null_std
  - ``Tr_N_delays``            — optimal lag (samples), same as raw pipeline
  - ``subject_id, group, trial_ids, window_counts, roi_names_used,
    connection_labels, n_trials, params_json``

CLI
---
python3 19_amplitude_ete.py --smoke-test
python3 19_amplitude_ete.py --benchmark SUBJECT_ID [--n-windows N]
python3 19_amplitude_ete.py --subject SUBJECT_ID
python3 19_amplitude_ete.py --batch
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping (mirrors 14_amplitude_dste.py)
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
_AMPLITUDE_DSTE_MOD = None  # 14_amplitude_dste.py (raw_compute_fn source)


def _get_utils_te() -> object:
    global _UTILS_TE_MOD
    if _UTILS_TE_MOD is None:
        spec = _ilu.spec_from_file_location("utils_te_p4ete", _STAGE_E_ROOT / "utils_te.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UTILS_TE_MOD = mod
    return _UTILS_TE_MOD


def _get_pipeline_mod() -> object:
    global _PIPELINE_MOD
    if _PIPELINE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "bivariate_dspte_pipeline_p4ete", _STAGE_E_ROOT / "phase_1" / "02_bivariate_dspte_pipeline.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PIPELINE_MOD = mod
    return _PIPELINE_MOD


def _get_seg_mod() -> object:
    global _SEG_MOD
    if _SEG_MOD is None:
        seg_path = _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py"
        spec = _ilu.spec_from_file_location("trial_seg_p4ete", seg_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEG_MOD = mod
    return _SEG_MOD


def _get_ete_core() -> object:
    """Return cached ../11_symbolic_ete_core.py module."""
    global _ETE_CORE_MOD
    if _ETE_CORE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "symbolic_ete_core_p4", _STAGE_E_ROOT / "11_symbolic_ete_core.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ETE_CORE_MOD = mod
    return _ETE_CORE_MOD


def _get_amplitude_dste_mod() -> object:
    """Return cached 14_amplitude_dste.py module (source of raw_compute_fn)."""
    global _AMPLITUDE_DSTE_MOD
    if _AMPLITUDE_DSTE_MOD is None:
        spec = _ilu.spec_from_file_location(
            "amplitude_dste_p4ete", Path(__file__).parent / "14_amplitude_dste.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _AMPLITUDE_DSTE_MOD = mod
    return _AMPLITUDE_DSTE_MOD


def _load_config_if_none(config: "dict | None") -> dict:
    if config is not None:
        return config
    try:
        from utils import load_config  # noqa
        return load_config()
    except ImportError:
        return {}


_ROI_NAMES_CANONICAL_P4 = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_N_CANONICAL_P4 = 5


# ===========================================================================
# Section 10.5 — Per-subject ETE pipeline
# ===========================================================================

def _save_subject_amplitude_ete_npz(
    subject_id: str,
    per_trial: dict,
    valid_roi_names: list[str],
    out_path: Path,
    params: dict,
    config: dict,
) -> None:
    """Save per-subject Phase 4 ETE connectivity to .npz (Section 10.5 schema)."""
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
        "_save_subject_amplitude_ete_npz [%s]: saved %d trials -> %s",
        subject_id, len(trial_ids), out_path,
    )


def _load_subject_amplitude_ete_npz(path: Path) -> "dict | None":
    """Load per-subject Phase 4 ETE .npz into {trial_id: {win_id: {key: vec20}}}."""
    if not path.exists():
        return None
    data = dict(np.load(path, allow_pickle=True))
    trial_ids = list(data.get("trial_ids", np.array([])))
    array_keys = [
        "connectivity", "ete_connectivity", "null_mean", "null_std",
        "p_value", "z_score", "delays",
    ]
    result: dict = {}
    for trial_id in trial_ids:
        trial_result: dict = {}
        conn = data.get(f"{trial_id}_connectivity")
        if conn is not None and conn.ndim == 2 and conn.shape[1] > 0:
            n_wins = conn.shape[1]
            for i in range(n_wins):
                win_id = f"Win_{i + 1}"
                trial_result[win_id] = {
                    key: data[f"{trial_id}_{key}"][:, i] for key in array_keys
                }
        result[trial_id] = trial_result
    return result


def process_subject_amplitude_ete(
    subject_id: str,
    study: str = "expert_novice",
    config: "dict | None" = None,
    overwrite: bool = False,
    n_windows_limit: "int | None" = None,
) -> dict:
    """
    Run the Phase 4 Effective TE pipeline for one subject (Section 10.5).

    Mirrors ``process_subject_amplitude_dste()`` (Section 9.5) but calls
    ``compute_symbolic_ete_matrix()`` (Section 10.4) per window instead of
    ``compute_amplitude_conditional_dste_matrix()`` directly.

    Parameters
    ----------
    subject_id : str
        ASTaUND subject identifier.
    study : str
        Study key.
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        If False, load from cache when the output .npz already exists.
    n_windows_limit : int, optional
        If set, process at most this many windows per trial (used by the
        10.5 benchmark CLI mode — NOT used in the real batch run).

    Returns
    -------
    dict
        ``{trial_id: {win_id: {'connectivity', 'ete_connectivity',
        'null_mean', 'null_std', 'p_value', 'z_score', 'delays'}}}``.
    """
    config = _load_config_if_none(config)
    utils_te = _get_utils_te()
    pipeline_mod = _get_pipeline_mod()
    seg_mod = _get_seg_mod()
    ete_core = _get_ete_core()
    amp_mod = _get_amplitude_dste_mod()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    win_cfg = te_cfg.get("window", {})
    window_size = int(win_cfg.get("size_samples", 1350))
    stride = int(win_cfg.get("stride_samples", 1350))

    out_dir = utils_te.get_te_output_dir("phase_4_ete_connectivity", config=config)
    out_path = out_dir / f"sub-{subject_id}_amplitude_ete_connectivity.npz"

    if not overwrite and out_path.exists() and n_windows_limit is None:
        logger.info(
            "process_subject_amplitude_ete [%s]: cache hit -> %s", subject_id, out_path,
        )
        cached = _load_subject_amplitude_ete_npz(out_path)
        return cached if cached is not None else {}

    t0_subj = time.perf_counter()
    conn_labels = utils_te.get_connection_labels()

    Data_Reg, valid_roi_names, sfreq = utils_te.load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    raw = utils_te.load_preprocessed_fif(subject_id, study=study, config=config)
    segments = seg_mod.get_trial_segments(raw, config=config)

    per_trial: dict = {}
    n_windows_processed_total = 0

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset: offset + 1]

        if Y_trial.shape[1] < (window_size + 1):
            logger.warning(
                "process_subject_amplitude_ete [%s %s]: trial too short — skipping.",
                subject_id, trial_id,
            )
            continue

        per_trial[trial_id] = {}
        win_num = 1

        for w in range(window_size, Y_trial.shape[1], stride):
            if n_windows_limit is not None and win_num > n_windows_limit:
                break

            ws = w - window_size
            we = w
            win_id = f"Win_{win_num}"
            win_num += 1

            Y_win = Y_trial[:, ws: we + 1]

            t_win = time.perf_counter()
            matrices, meta = ete_core.compute_symbolic_ete_matrix(
                raw_compute_fn=amp_mod.compute_amplitude_conditional_dste_matrix,
                window_data=Y_win,
                config=config,
                roi_names=valid_roi_names,
                method_key="phase_4",
            )
            elapsed_win = time.perf_counter() - t_win
            n_windows_processed_total += 1

            logger.info(
                "process_subject_amplitude_ete [%s %s %s]: n_valid=%d  "
                "elapsed_raw=%.2fs  elapsed_ete=%.2fs  total=%.2fs",
                subject_id, trial_id, win_id, meta.get("n_valid", 0),
                meta.get("elapsed_raw_s", 0.0), meta.get("elapsed_ete_s", 0.0),
                elapsed_win,
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

        logger.info(
            "process_subject_amplitude_ete [%s %s]: %d windows processed",
            subject_id, trial_id, win_num - 1,
        )

    elapsed_subj = time.perf_counter() - t0_subj

    if n_windows_limit is None:
        try:
            group = utils_te.get_subject_group(subject_id, config=config)
        except ValueError:
            group = "Unknown"

        ete_cfg = te_cfg.get("phase_4", {}).get("ete", {})
        params = {
            "phase": "phase_4_amplitude_ete",
            "study": study,
            "group": group,
            "n_surrogates": int(ete_cfg.get("n_surrogates", 500)),
            "min_shift_fraction": float(ete_cfg.get("min_shift_fraction", 0.05)),
            "floor_estimator": str(ete_cfg.get("floor_estimator", "mean")),
            "random_seed_base": int(ete_cfg.get("random_seed_base", 42)),
            "n_windows_processed": n_windows_processed_total,
            "elapsed_s": round(elapsed_subj, 2),
        }
        _save_subject_amplitude_ete_npz(
            subject_id=subject_id,
            per_trial=per_trial,
            valid_roi_names=valid_roi_names,
            out_path=out_path,
            params=params,
            config=config,
        )

    logger.info(
        "process_subject_amplitude_ete [%s]: DONE  windows=%d  elapsed=%.1f s",
        subject_id, n_windows_processed_total, elapsed_subj,
    )
    return per_trial


# ===========================================================================
# Section 10.5 — Batch runner
# ===========================================================================

def run_batch_amplitude_ete(
    study: str = "expert_novice",
    subjects: "list[str] | None" = None,
    config: "dict | None" = None,
    overwrite: bool = False,
) -> dict:
    """
    Batch-process all ASTaUND subjects for Phase 4 Effective TE (Section 10.5).

    Mirrors ``run_batch_amplitude_dste()`` (Section 9.6).  Per-subject error
    isolation — a failure on one subject does not abort the batch.
    """
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
        "phase": "phase_4_amplitude_ete",
        "study": study,
        "n_subjects": len(subjects),
        "subjects": {},
    }

    for sid in subjects:
        t_sub = time.perf_counter()
        logger.info("run_batch_amplitude_ete: processing subject %s ...", sid)
        try:
            res = process_subject_amplitude_ete(
                subject_id=sid, study=study, config=config, overwrite=overwrite,
            )
            n_wins = sum(len(v) for v in res.values())
            all_ete_vecs = [
                w["ete_connectivity"] for tr in res.values() for w in tr.values()
            ]
            if all_ete_vecs:
                arr = np.stack(all_ete_vecs, axis=0)
                mean_ete = float(np.nanmean(arr))
                max_ete = float(np.nanmax(arr))
            else:
                mean_ete = 0.0
                max_ete = 0.0

            batch_log["subjects"][sid] = {
                "status": "ok",
                "n_trials": len(res),
                "n_windows_total": n_wins,
                "mean_ete": mean_ete,
                "max_ete": max_ete,
                "elapsed_s": round(time.perf_counter() - t_sub, 2),
            }
        except Exception as exc:
            batch_log["subjects"][sid] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - t_sub, 2),
            }
            logger.error(
                "run_batch_amplitude_ete [%s]: FAILED -- %s", sid, exc, exc_info=True,
            )

    n_ok = sum(1 for v in batch_log["subjects"].values() if v.get("status") == "ok")
    batch_log["summary"] = {
        "n_success": n_ok,
        "n_failed": len(subjects) - n_ok,
        "total_elapsed_s": round(time.perf_counter() - t0, 2),
    }

    group_dir = utils_te.get_te_output_dir("phase_4_ete_group", config=config)
    group_dir.mkdir(parents=True, exist_ok=True)
    log_path = group_dir / "amplitude_ete_batch_log.json"
    with open(log_path, "w") as fh:
        json.dump(batch_log, fh, indent=2)

    logger.info(
        "run_batch_amplitude_ete: DONE  %d/%d subjects succeeded  total=%.1f s  log -> %s",
        n_ok, len(subjects), batch_log["summary"]["total_elapsed_s"], log_path,
    )
    return batch_log


# ===========================================================================
# CLI
# ===========================================================================

def _run_benchmark(subject_id: str, n_windows: int, config: dict) -> None:
    """Section 10.5 benchmark mode: time N windows for one subject, extrapolate."""
    logger.info(
        "Benchmark: subject=%s  n_windows=%d  (n_surrogates=%d)",
        subject_id, n_windows,
        config["stage_e_transfer_entropy"]["phase_4"]["ete"]["n_surrogates"],
    )
    t0 = time.perf_counter()
    result = process_subject_amplitude_ete(
        subject_id=subject_id, config=config, overwrite=True,
        n_windows_limit=n_windows,
    )
    elapsed = time.perf_counter() - t0
    n_wins_done = sum(len(v) for v in result.values())
    per_window = elapsed / max(1, n_wins_done)

    logger.info("=" * 70)
    logger.info(
        "BENCHMARK RESULT: %d windows in %.1f s -> %.2f s/window",
        n_wins_done, elapsed, per_window,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stage E Phase 4 -- Effective TE (Section 10.5)"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--benchmark", type=str, metavar="SUBJECT_ID",
                      help="Time N windows for one subject and extrapolate.")
    grp.add_argument("--subject", type=str, metavar="SUBJECT_ID",
                      help="Run Phase 4 ETE for one subject.")
    grp.add_argument("--batch", action="store_true",
                      help="Run Phase 4 ETE for all subjects in config.")
    parser.add_argument("--n-windows", type=int, default=1,
                         help="Number of windows to time in --benchmark mode (default 1).")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    _cfg = _load_config_if_none(None)

    if args.benchmark:
        _run_benchmark(args.benchmark.strip(), args.n_windows, _cfg)
    elif args.subject:
        _res = process_subject_amplitude_ete(
            subject_id=args.subject.strip(), config=_cfg, overwrite=args.overwrite,
        )
        _wins = sum(len(v) for v in _res.values())
        logger.info("subject=%s  trials=%d  windows=%d", args.subject, len(_res), _wins)
    elif args.batch:
        _log = run_batch_amplitude_ete(config=_cfg, overwrite=args.overwrite)
        logger.info("batch done: %s", _log.get("summary", {}))
