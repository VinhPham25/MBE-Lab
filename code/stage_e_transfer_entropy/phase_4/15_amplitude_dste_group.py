"""
Stage E — Phase 4: Group Aggregation for Amplitude dSTE (Section 9.7).

Loads per-subject Phase 4 ``.npz`` files produced by
``process_subject_amplitude_dste()`` in ``14_amplitude_dste.py`` and
assembles group-level arrays for the Expert and Novice cohorts.

Three aggregation strategies are provided, matching the Phase 1/2/3
group-array schema so downstream statistics (Section 9.8) and
visualisation (Section 9.9) work without modification:

``build_p4_group_arrays_all_windows()``
    ``{trial_id: (n_subs, 20, max_windows)}`` — all windows retained.
    Used for H0 (within-trial stability) and HIV (across-trial stability).

``build_p4_group_arrays_subject_mean()``
    ``(n_subs, 20, 5)`` — per-trial window mean collapsed to one scalar
    per connection per trial.  Primary input for H1 Expert vs Novice test.

``build_p4_group_arrays_first_window()``
    ``(n_subs, 20, 5)`` — first window only per trial.  Sensitivity check.

``build_all_p4_group_arrays()``
    Orchestrator that calls all three strategies and writes
    ``amplitude_group_summary.json``.

Output directory:  ``phase_4_group/``  (resolved via ``get_te_output_dir``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_N_TRIALS_CANONICAL = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _p4_group_dir(config: Optional[dict] = None) -> Path:
    """
    Return the Phase 4 group output directory, creating it if needed.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        ``outputs/stage_e_transfer_entropy/phase_4_group/`` (or equivalent).
    """
    from utils_te import get_te_output_dir

    d = get_te_output_dir("phase_4_group", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p4_conn_dir(config: Optional[dict] = None) -> Path:
    """
    Return the Phase 4 per-subject connectivity directory.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        ``outputs/stage_e_transfer_entropy/phase_4_connectivity/``.
    """
    from utils_te import get_te_output_dir

    return get_te_output_dir("phase_4_connectivity", config=config)


def _connection_labels() -> list[str]:
    """Return canonical 20 directed ROI connection labels."""
    from utils_te import get_connection_labels

    return get_connection_labels()


def _load_subject_npz_lists(
    expert_ids: list[str],
    novice_ids: list[str],
    conn_dir: Path,
) -> tuple[list[Optional[dict]], list[Optional[dict]]]:
    """
    Load per-subject Phase 4 ``.npz`` dicts for Expert and Novice groups.

    Parameters
    ----------
    expert_ids : list[str]
        Expert subject IDs.
    novice_ids : list[str]
        Novice subject IDs.
    conn_dir : Path
        Phase 4 connectivity directory containing
        ``sub-*_amplitude_dste_connectivity.npz`` files.

    Returns
    -------
    tuple[list[Optional[dict]], list[Optional[dict]]]
        ``(expert_list, novice_list)``; missing files are represented as
        ``None``.
    """

    def _load(subject_ids: list[str]) -> list[Optional[dict]]:
        results: list[Optional[dict]] = []
        for sid in subject_ids:
            fpath = conn_dir / f"sub-{sid}_amplitude_dste_connectivity.npz"
            if not fpath.exists():
                logger.warning(
                    "_load_subject_npz_lists: missing file for subject %s (%s)",
                    sid, fpath.name,
                )
                results.append(None)
            else:
                results.append(dict(np.load(fpath, allow_pickle=True)))
        return results

    return _load(expert_ids), _load(novice_ids)


def _window_summary(
    subject_ids: list[str],
    npz_list: list[Optional[dict]],
) -> dict:
    """
    Build per-subject trial/window count summary for provenance JSON.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs aligned with ``npz_list``.
    npz_list : list[Optional[dict]]
        Loaded ``.npz`` dicts or ``None`` for missing files.

    Returns
    -------
    dict
        ``{subject_id: {status, n_trials, trial_window_counts, ...}}``.
    """
    out: dict = {}
    for sid, d in zip(subject_ids, npz_list):
        if d is None:
            out[sid] = {"status": "missing", "trial_window_counts": {}}
            continue
        trial_ids_s = [str(t) for t in d["trial_ids"]]
        window_counts_s = [int(x) for x in d["window_counts"]]
        out[sid] = {
            "status": "ok",
            "n_trials": int(d["n_trials"]),
            "trial_window_counts": dict(zip(trial_ids_s, window_counts_s)),
            "total_windows": int(sum(window_counts_s)),
            "group": str(d.get("group", "Unknown")),
            "roi_names_used": [str(x) for x in d.get("roi_names_used", [])],
        }
    return out


# ---------------------------------------------------------------------------
# Run-metadata consolidation (params-caption feature, 2026-08-28)
#
# Each per-subject Phase 4 .npz already embeds a ``params_json`` field
# (written by ``process_subject_amplitude_dste()`` in 14_amplitude_dste.py)
# capturing the config.yaml values that were active when THAT subject was
# processed. Because subjects can be reprocessed independently (e.g. only
# a few subjects re-run after a config change), the same "params" field can
# differ across subjects on disk. This section consolidates all subjects'
# snapshots into one canonical file so downstream visualization never has
# to read config.yaml directly (which may have since been edited) — it
# reads what was ACTUALLY used to produce the connectivity data on disk.
# ---------------------------------------------------------------------------

# Curated fields shown in plot captions — the parameters most likely to
# change between runs and materially affect the resulting TE values.
# NOTE (2026-08-28): embedding_dimension and narrowband_filter were
# originally omitted here — they are legitimate, important parameters
# (embedding_dimension in particular directly affects the TE estimate)
# and are now included.
_CAPTION_FIELDS = [
    "k0_max",
    "ndis_threshold",
    "conditional_pca_enabled",
    "conditional_pca_n_components",
    "sparsity_fallback_threshold",
    "window_size_samples",
    "u_max",
    "envelope_normalize",
    "embedding_dimension",
    "narrowband_filter",
]


def _extract_subject_params(npz_data: dict) -> Optional[dict]:
    """Parse the ``params_json`` field out of a loaded per-subject .npz dict."""
    if "params_json" not in npz_data:
        return None
    try:
        return json.loads(str(npz_data["params_json"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def consolidate_p4_run_metadata(
    study: str = "expert_novice",
    config: Optional[dict] = None,
) -> dict:
    """
    Consolidate per-subject Phase 4 params snapshots into one canonical file.

    Reads every available subject's embedded ``params_json`` (written at
    processing time in ``14_amplitude_dste.py``) and, for each field in
    ``_CAPTION_FIELDS``, determines the consensus (most common) value. Any
    field where subjects disagree — e.g. because only some subjects were
    reprocessed after a ``config.yaml`` change — is flagged in
    ``mismatched_fields`` rather than silently resolved, so downstream
    visualization can warn the user instead of showing a misleading caption.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration (used only to locate subject IDs and
        the connectivity directory — NOT read for parameter values, since
        the whole point of this function is to report what was actually
        run rather than what config.yaml currently says).

    Returns
    -------
    dict
        The consolidated metadata dict (also written to
        ``phase_4_group/run_metadata.json``):
        ``{created_at, phase, n_subjects_total, n_subjects_with_params,
        params, mismatched_fields, has_mismatch, per_subject_params}``.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))
    all_ids = expert_ids + novice_ids

    conn_dir = _p4_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir
    )
    all_npz = expert_npz_list + novice_npz_list

    per_subject_params: dict = {}
    for sid, npz_data in zip(all_ids, all_npz):
        if npz_data is None:
            continue
        parsed = _extract_subject_params(npz_data)
        if parsed is not None:
            per_subject_params[sid] = parsed

    consensus_params: dict = {}
    mismatched_fields: dict = {}
    for field in _CAPTION_FIELDS:
        values_by_subject: dict = {
            sid: p[field] for sid, p in per_subject_params.items() if field in p
        }
        if not values_by_subject:
            continue
        distinct_values = sorted(
            {json.dumps(v, sort_keys=True) for v in values_by_subject.values()}
        )
        if len(distinct_values) == 1:
            consensus_params[field] = next(iter(values_by_subject.values()))
        else:
            # Majority vote for the "consensus" display value; full
            # breakdown recorded in mismatched_fields for transparency.
            counts: dict = {}
            for v in values_by_subject.values():
                key = json.dumps(v, sort_keys=True)
                counts[key] = counts.get(key, 0) + 1
            majority_key = max(counts, key=counts.get)
            consensus_params[field] = json.loads(majority_key)
            mismatched_fields[field] = {
                json.loads(k): [
                    sid for sid, v in values_by_subject.items()
                    if json.dumps(v, sort_keys=True) == k
                ]
                for k in counts
            }

    metadata = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_4_amplitude_dste",
        "study": study,
        "n_subjects_total": len(all_ids),
        "n_subjects_with_params": len(per_subject_params),
        "params": consensus_params,
        "mismatched_fields": mismatched_fields,
        "has_mismatch": len(mismatched_fields) > 0,
        "per_subject_params": per_subject_params,
    }

    group_dir = _p4_group_dir(config)
    out_path = group_dir / "run_metadata.json"
    with open(out_path, "w") as fh:
        json.dump(metadata, fh, indent=2, cls=_NumpyEncoder)

    if metadata["has_mismatch"]:
        logger.warning(
            "consolidate_p4_run_metadata: %d field(s) disagree across subjects "
            "(likely partial reprocessing after a config.yaml change): %s",
            len(mismatched_fields), sorted(mismatched_fields.keys()),
        )
    logger.info(
        "consolidate_p4_run_metadata: saved %s (%d/%d subjects had params)",
        out_path, metadata["n_subjects_with_params"], metadata["n_subjects_total"],
    )
    return metadata


def _load_group_npz_as_dict(path: Path) -> dict:
    """
    Load trial arrays from a saved group ``.npz``.

    Parameters
    ----------
    path : Path
        Group ``.npz`` file path.

    Returns
    -------
    dict
        ``{trial_id: np.ndarray}`` (keys starting with ``Tr_``).
    """
    data = dict(np.load(path, allow_pickle=True))
    return {k: v for k, v in data.items() if k.startswith("Tr_")}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_p4_group_arrays_all_windows(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[dict, dict]:
    """
    Build Phase 4 all-window group arrays from amplitude connectivity files.

    Each subject's ``(20, n_wins)`` connectivity arrays are stacked into a
    ``(n_subs, 20, max_wins_for_trial)`` group array, with NaN-padding for
    subjects that have fewer windows than the maximum across the cohort.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached group arrays when available.

    Returns
    -------
    tuple[dict, dict]
        ``(B_Exp, B_Nov)`` where each value is a dict
        ``{trial_id: np.ndarray(n_subs, 20, max_wins)}``.

    Notes
    -----
    Saved files (in ``phase_4_group/``):
    - ``B_Exp_amplitude_all.npz``
    - ``B_Nov_amplitude_all.npz``
    - ``amplitude_group_summary.json``
    """
    if config is None:
        config = load_config()

    group_dir = _p4_group_dir(config)
    exp_path = group_dir / "B_Exp_amplitude_all.npz"
    nov_path = group_dir / "B_Nov_amplitude_all.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p4_group_arrays_all_windows: cache hit.")
        return _load_group_npz_as_dict(exp_path), _load_group_npz_as_dict(nov_path)

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p4_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir
    )

    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]
    if not all_npz:
        raise RuntimeError(
            "build_p4_group_arrays_all_windows: no Phase 4 subject files found. "
            "Run run_batch_amplitude_dste() first."
        )

    max_trials = max(int(d["n_trials"]) for d in all_npz)
    max_wins_per_trial: dict[str, int] = {}
    for trial_idx in range(max_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        max_w = 0
        for d in all_npz:
            key = f"{trial_id}_connectivity"
            if key in d and d[key].ndim == 2 and d[key].shape[1] > 0:
                max_w = max(max_w, int(d[key].shape[1]))
        max_wins_per_trial[trial_id] = max_w

    n_conn = len(_connection_labels())
    B_Exp: dict[str, np.ndarray] = {}
    B_Nov: dict[str, np.ndarray] = {}

    for trial_id, max_w in max_wins_per_trial.items():
        if max_w == 0:
            continue
        B_Exp[trial_id] = np.full((len(expert_ids), n_conn, max_w), np.nan)
        B_Nov[trial_id] = np.full((len(novice_ids), n_conn, max_w), np.nan)

        for sub_idx, npz_data in enumerate(expert_npz_list):
            if npz_data is None:
                continue
            key = f"{trial_id}_connectivity"
            if key in npz_data:
                arr = npz_data[key]
                if arr.ndim == 2 and arr.shape[1] > 0:
                    B_Exp[trial_id][sub_idx, :, : arr.shape[1]] = arr

        for sub_idx, npz_data in enumerate(novice_npz_list):
            if npz_data is None:
                continue
            key = f"{trial_id}_connectivity"
            if key in npz_data:
                arr = npz_data[key]
                if arr.ndim == 2 and arr.shape[1] > 0:
                    B_Nov[trial_id][sub_idx, :, : arr.shape[1]] = arr

    exp_save = dict(B_Exp)
    exp_save["subject_ids"] = np.array(expert_ids)
    exp_save["strategy"] = np.array("all_windows")
    np.savez(exp_path, **exp_save)

    nov_save = dict(B_Nov)
    nov_save["subject_ids"] = np.array(novice_ids)
    nov_save["strategy"] = np.array("all_windows")
    np.savez(nov_path, **nov_save)

    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_4_amplitude_dste",
        "study": study,
        "strategy": "all_windows",
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "max_windows_per_trial": {k: int(v) for k, v in max_wins_per_trial.items()},
        "n_connections": n_conn,
        "connection_labels": _connection_labels(),
        "expert_subjects": _window_summary(expert_ids, expert_npz_list),
        "novice_subjects": _window_summary(novice_ids, novice_npz_list),
    }
    with open(group_dir / "amplitude_group_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    logger.info(
        "build_p4_group_arrays_all_windows: saved  Exp=%s  Nov=%s  (trials=%s)",
        {k: v.shape for k, v in B_Exp.items()},
        {k: v.shape for k, v in B_Nov.items()},
        sorted(B_Exp.keys()),
    )
    return B_Exp, B_Nov


def build_p4_group_arrays_subject_mean(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 4 subject-mean group arrays for H1 Expert vs Novice testing.

    For each subject and trial the windows are averaged to produce one
    scalar per connection, yielding arrays of shape ``(n_subs, 20, 5)``.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached arrays when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(B_Exp_mean, B_Nov_mean)`` each with shape ``(n_subs, 20, 5)``.

    Notes
    -----
    Saved files: ``B_Exp_amplitude_mean.npz``, ``B_Nov_amplitude_mean.npz``.
    """
    if config is None:
        config = load_config()

    group_dir = _p4_group_dir(config)
    exp_path = group_dir / "B_Exp_amplitude_mean.npz"
    nov_path = group_dir / "B_Nov_amplitude_mean.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p4_group_arrays_subject_mean: cache hit.")
        return (
            np.load(exp_path, allow_pickle=True)["data"],
            np.load(nov_path, allow_pickle=True)["data"],
        )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p4_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir
    )

    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]]) -> np.ndarray:
        arr = np.full((len(subject_ids), n_conn, _N_TRIALS_CANONICAL), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                continue
            for trial_idx in range(_N_TRIALS_CANONICAL):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]
                if conn.ndim == 2 and conn.shape[1] > 0:
                    arr[sub_idx, :, trial_idx] = np.nanmean(conn, axis=1)
        return arr

    B_Exp_mean = _build(expert_ids, expert_npz_list)
    B_Nov_mean = _build(novice_ids, novice_npz_list)

    np.savez(
        exp_path,
        data=B_Exp_mean,
        subject_ids=np.array(expert_ids),
        strategy=np.array("subject_mean"),
    )
    np.savez(
        nov_path,
        data=B_Nov_mean,
        subject_ids=np.array(novice_ids),
        strategy=np.array("subject_mean"),
    )

    logger.info(
        "build_p4_group_arrays_subject_mean: saved  Exp=%s  Nov=%s",
        B_Exp_mean.shape, B_Nov_mean.shape,
    )
    return B_Exp_mean, B_Nov_mean


def build_p4_group_arrays_first_window(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 4 first-window group arrays for sensitivity analysis.

    Uses only the first window of each trial per subject, yielding arrays
    of shape ``(n_subs, 20, 5)``.  Serves as a sensitivity check to
    confirm results are stable across window-averaging strategies.

    Parameters
    ----------
    study : str
        Study key.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached arrays when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(B_Exp_first, B_Nov_first)`` each with shape ``(n_subs, 20, 5)``.

    Notes
    -----
    Saved files:
    ``B_Exp_amplitude_first_window.npz``, ``B_Nov_amplitude_first_window.npz``.
    """
    if config is None:
        config = load_config()

    group_dir = _p4_group_dir(config)
    exp_path = group_dir / "B_Exp_amplitude_first_window.npz"
    nov_path = group_dir / "B_Nov_amplitude_first_window.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p4_group_arrays_first_window: cache hit.")
        return (
            np.load(exp_path, allow_pickle=True)["data"],
            np.load(nov_path, allow_pickle=True)["data"],
        )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p4_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir
    )

    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]]) -> np.ndarray:
        arr = np.full((len(subject_ids), n_conn, _N_TRIALS_CANONICAL), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                continue
            for trial_idx in range(_N_TRIALS_CANONICAL):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]
                if conn.ndim == 2 and conn.shape[1] > 0:
                    arr[sub_idx, :, trial_idx] = conn[:, 0]
        return arr

    B_Exp_first = _build(expert_ids, expert_npz_list)
    B_Nov_first = _build(novice_ids, novice_npz_list)

    np.savez(
        exp_path,
        data=B_Exp_first,
        subject_ids=np.array(expert_ids),
        strategy=np.array("first_window"),
    )
    np.savez(
        nov_path,
        data=B_Nov_first,
        subject_ids=np.array(novice_ids),
        strategy=np.array("first_window"),
    )

    logger.info(
        "build_p4_group_arrays_first_window: saved  Exp=%s  Nov=%s",
        B_Exp_first.shape, B_Nov_first.shape,
    )
    return B_Exp_first, B_Nov_first


def build_all_p4_group_arrays(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Orchestrate all three Phase 4 group-array aggregation strategies.

    Calls ``build_p4_group_arrays_all_windows()``,
    ``build_p4_group_arrays_subject_mean()``, and
    ``build_p4_group_arrays_first_window()`` in sequence.

    Parameters
    ----------
    study : str
        Study key.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, skip strategies whose output files already exist.

    Returns
    -------
    dict
        ``{strategy: result}`` with keys
        ``'all_windows'``, ``'subject_mean'``, ``'first_window'``.
        Each value is the tuple returned by the corresponding builder.

    Notes
    -----
    Also writes (or updates) ``amplitude_group_summary.json`` in
    ``phase_4_group/`` with cross-strategy metadata.
    """
    if config is None:
        config = load_config()

    results: dict = {}

    logger.info("build_all_p4_group_arrays: strategy=all_windows …")
    B_Exp_all, B_Nov_all = build_p4_group_arrays_all_windows(
        study=study, config=config, overwrite=overwrite
    )
    results["all_windows"] = (B_Exp_all, B_Nov_all)

    logger.info("build_all_p4_group_arrays: strategy=subject_mean …")
    B_Exp_mean, B_Nov_mean = build_p4_group_arrays_subject_mean(
        study=study, config=config, overwrite=overwrite
    )
    results["subject_mean"] = (B_Exp_mean, B_Nov_mean)

    logger.info("build_all_p4_group_arrays: strategy=first_window …")
    B_Exp_first, B_Nov_first = build_p4_group_arrays_first_window(
        study=study, config=config, overwrite=overwrite
    )
    results["first_window"] = (B_Exp_first, B_Nov_first)

    logger.info("build_all_p4_group_arrays: consolidating run_metadata.json …")
    run_metadata = consolidate_p4_run_metadata(study=study, config=config)
    results["run_metadata"] = run_metadata

    group_dir = _p4_group_dir(config)
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))
    n_conn = len(_connection_labels())

    summary_path = group_dir / "amplitude_group_summary.json"
    summary: dict = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_4_amplitude_dste",
        "study": study,
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "n_connections": n_conn,
        "connection_labels": _connection_labels(),
        "strategies": {
            "all_windows": {
                "B_Exp_shape": {k: list(v.shape) for k, v in B_Exp_all.items()},
                "B_Nov_shape": {k: list(v.shape) for k, v in B_Nov_all.items()},
            },
            "subject_mean": {
                "B_Exp_shape": list(B_Exp_mean.shape),
                "B_Nov_shape": list(B_Nov_mean.shape),
            },
            "first_window": {
                "B_Exp_shape": list(B_Exp_first.shape),
                "B_Nov_shape": list(B_Nov_first.shape),
            },
        },
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    logger.info(
        "build_all_p4_group_arrays: DONE — saved 6 group array files + summary → %s",
        group_dir,
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """CLI for Phase 4 group aggregation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stage E Phase 4 — Group Array Aggregation (Section 9.7)"
    )
    parser.add_argument(
        "--strategy",
        choices=["all_windows", "subject_mean", "first_window", "metadata", "all"],
        default="all",
        help="Which aggregation strategy to run (default: all)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help=(
            "Use cached group arrays instead of rebuilding (default: "
            "rebuild/overwrite every run so run_metadata.json always "
            "reflects the latest per-subject params_json)."
        ),
    )
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    cfg = load_config()

    if args.strategy == "all":
        result = build_all_p4_group_arrays(config=cfg, overwrite=args.overwrite)
        for k, v in result.items():
            if k == "run_metadata":
                logger.info(
                    "[run_metadata] has_mismatch=%s  n_subjects_with_params=%d/%d",
                    v.get("has_mismatch"), v.get("n_subjects_with_params", 0),
                    v.get("n_subjects_total", 0),
                )
                continue
            B_E, B_N = v
            if isinstance(B_E, dict):
                shapes = {t: v2.shape for t, v2 in B_E.items()}
            else:
                shapes = B_E.shape
            logger.info("[%s] B_Exp: %s", k, shapes)
    elif args.strategy == "metadata":
        meta = consolidate_p4_run_metadata(config=cfg)
        logger.info(
            "metadata: has_mismatch=%s  n_subjects_with_params=%d/%d  params=%s",
            meta.get("has_mismatch"), meta.get("n_subjects_with_params", 0),
            meta.get("n_subjects_total", 0), meta.get("params"),
        )
    elif args.strategy == "all_windows":
        B_E, B_N = build_p4_group_arrays_all_windows(
            config=cfg, overwrite=args.overwrite
        )
        logger.info("all_windows: Exp trials=%s  Nov trials=%s", list(B_E.keys()), list(B_N.keys()))
    elif args.strategy == "subject_mean":
        B_E, B_N = build_p4_group_arrays_subject_mean(
            config=cfg, overwrite=args.overwrite
        )
        logger.info("subject_mean: Exp=%s  Nov=%s", B_E.shape, B_N.shape)
    elif args.strategy == "first_window":
        B_E, B_N = build_p4_group_arrays_first_window(
            config=cfg, overwrite=args.overwrite
        )
        logger.info("first_window: Exp=%s  Nov=%s", B_E.shape, B_N.shape)


if __name__ == "__main__":
    _cli_main()
