"""
Stage E — Step E8: Phase 2 Multivariate dSPTE Statistical Tests  (Section 7.3.1)

Implements the Phase 2 equivalent of ``04_bivariate_dspte_statistics.py``.  The design
principle from the PRD is: **re-use ``04_bivariate_dspte_statistics.py`` functions
directly — do not duplicate test logic.**  This module:

  1. Builds Phase 2 group arrays from the per-subject
     ``sub-{id}_multivariate_dspte_connectivity.npz`` files in
     ``phase_2_multivariate_dspte/te_connectivity/``.

     Three builders mirroring the Phase 1 equivalents in
     ``03_bivariate_dspte_batch.py``:

       build_p2_group_arrays_all_windows()   → B_Exp_multivariate_all.npz   / B_Nov_multivariate_all.npz
       build_p2_group_arrays_subject_mean()  → B_Exp_multivariate_mean.npz  / B_Nov_multivariate_mean.npz
       build_p2_group_arrays_first_window()  → B_Exp_multivariate_fw.npz    / B_Nov_multivariate_fw.npz

  2. Runs the same 4 statistical tests by calling Phase 1 stat functions with
     Phase 2 arrays as inputs.  Output CSVs go to
     ``phase_2_multivariate_dspte/statistics/``.

       run_normality_tests_p2()           → normality_test.csv
       test_expert_vs_novice_p2()         → H1_expert_vs_novice.csv
       test_within_trial_stability_p2()   → H0_within_trial_stability.csv
       test_across_trial_stability_p2()   → HIV_across_trial_stability.csv

  3. Master runner: ``run_all_statistics_p2(config, overwrite)`` — calls all
     four tests, prints summary, returns dict of DataFrames.

Section 7.3.1 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
-----------------------------------------------------------
7.3.1 — Run same 4 statistical tests as Phase 1 on multivariate group arrays using
         the three-strategy builders (subject_mean for H1, all_windows for H0
         and HIV).  Save to phase_2_multivariate_dspte/statistics/.  Re-use
         04_bivariate_dspte_statistics.py functions directly.

MATLAB equivalents (inherited from Phase 1)
-------------------------------------------
  swtest(x)           → scipy.stats.shapiro(x)
  anova1(X)           → scipy.stats.f_oneway(*groups)
  mafdr(p)            → statsmodels.stats.multitest.fdrcorrection(p)

NaN handling (same as Phase 1)
-------------------------------
All-windows arrays: NaN where subject has fewer windows than the max for that
trial.  Subject-mean arrays: NaN where subject has no data for that trial.
Zero-filled ROI connections (R11 LM1/SMA, R2 SMA) remain 0.0 — valid
observations (true absent connectivity), not missing data.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE_ROOT    = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent          # stage_e_transfer_entropy/
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_TRIALS_CANONICAL = 5   # canonical trial count (R4 has 4; R9 has 2 — padded with NaN)


# ===========================================================================
# Phase 1 stat function importer (avoids circular imports)
# ===========================================================================

def _import_p1_stats():
    """Return the 04_bivariate_dspte_statistics module via importlib."""
    p = _STAGE_E_ROOT / "phase_1" / "04_bivariate_dspte_statistics.py"
    spec = importlib.util.spec_from_file_location("bivariate_dspte_statistics_p1", p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Path helpers
# ===========================================================================

def _p2_stats_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 2 statistics output directory (create if needed)."""
    from utils_te import get_te_output_dir  # noqa
    d = get_te_output_dir("phase_2_statistics", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p2_group_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 2 group array directory (create if needed)."""
    from utils_te import get_te_output_dir  # noqa
    d = get_te_output_dir("phase_2_group", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p2_conn_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 2 per-subject connectivity directory."""
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_2_connectivity", config=config)


def _stats_cfg(config: Optional[dict] = None) -> dict:
    """Return the statistics sub-config from config.yaml."""
    if config is None:
        config = load_config()
    return config["stage_e_transfer_entropy"]["statistics"]


def _load_connection_labels(config: Optional[dict] = None) -> list:
    """Return canonical 20 directed connection labels."""
    from utils_te import get_connection_labels  # noqa
    return get_connection_labels()


# ===========================================================================
# Per-subject .npz loader helper
# ===========================================================================

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """Convert NumPy scalar/array objects into JSON-serialisable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_p2_npz_lists(
    expert_ids: list,
    novice_ids: list,
    conn_dir: Path,
    caller: str,
) -> tuple[list, list]:
    """
    Load per-subject Phase 2 .npz files; return None for missing files.

    Mirrors ``_load_npz_lists()`` from ``03_bivariate_dspte_batch.py``
    but looks for ``sub-{id}_multivariate_dspte_connectivity.npz``.
    """
    def _load(subject_ids):
        """Load one list of subject `.npz` files, returning `None` when missing."""
        results = []
        for sid in subject_ids:
            fpath = conn_dir / f"sub-{sid}_multivariate_dspte_connectivity.npz"
            if not fpath.exists():
                logger.warning(
                    "%s: missing Phase 2 .npz for %s: %s", caller, sid, fpath.name
                )
                results.append(None)
            else:
                results.append(dict(np.load(fpath, allow_pickle=True)))
        return results

    return _load(expert_ids), _load(novice_ids)


def _load_group_p2_npz_as_dict(path: Path) -> dict:
    """Load a Phase 2 group .npz and return as a plain dict."""
    data = dict(np.load(path, allow_pickle=True))
    return {k: v for k, v in data.items() if k.startswith("Tr_")}


def _p2_window_summary(subject_ids, npz_list):
    """
    Summarise available trial/window counts per subject.

    Parameters
    ----------
    subject_ids : list[str]
        Subjects corresponding to `npz_list`.
    npz_list : list[dict | None]
        Loaded per-subject dicts or `None` for missing files.

    Returns
    -------
    dict
        Mapping subject -> status/shape summary used in `multivariate_group_summary.json`.
    """
    out = {}
    for sid, d in zip(subject_ids, npz_list):
        if d is None:
            out[sid] = {"status": "missing", "trial_window_counts": {}}
            continue
        trial_ids_s = list(d["trial_ids"])
        window_counts_s = list(d["window_counts"].astype(int))
        out[sid] = {
            "status": "ok",
            "n_trials": int(d["n_trials"]),
            "trial_window_counts": dict(zip(trial_ids_s, window_counts_s)),
            "total_windows": int(sum(window_counts_s)),
            "roi_names_used": list(d["roi_names_used"]),
            "group": str(d["group"]),
        }
    return out


# ===========================================================================
# Section 7.3.1a — Phase 2 Group Array Builder: all_windows
# ===========================================================================

def build_p2_group_arrays_all_windows(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[dict, dict]:
    """
    Build Phase 2 all-windows group arrays from per-subject multivariate dSPTE .npz.

    Mirrors ``build_group_arrays_all_windows()`` from Phase 1 pipeline.
    Uses for H0 (within-trial stability) and HIV (across-trial stability).

    Returns
    -------
    B_Exp : dict  {trial_id: np.ndarray(n_exp, 20, max_windows)}
    B_Nov : dict  {trial_id: np.ndarray(n_nov, 20, max_windows)}
        NaN where a subject has fewer windows or no data for that trial.

    Saves
    -----
    phase_2_group/B_Exp_multivariate_all.npz
    phase_2_group/B_Nov_multivariate_all.npz
    phase_2_group/multivariate_group_summary.json
    """
    if config is None:
        config = load_config()

    from utils_te import get_connection_labels  # noqa

    group_dir  = _p2_group_dir(config)
    exp_path   = group_dir / "B_Exp_multivariate_all.npz"
    nov_path   = group_dir / "B_Nov_multivariate_all.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p2_group_arrays_all_windows: cached files found — loading.")
        return _load_group_p2_npz_as_dict(exp_path), _load_group_p2_npz_as_dict(nov_path)

    te_cfg     = config.get("stage_e_transfer_entropy", {})
    subj_cfg   = te_cfg.get("subjects", {})
    expert_ids = subj_cfg.get("Expert", [])
    novice_ids = subj_cfg.get("Novice", [])

    logger.info(
        "build_p2_group_arrays_all_windows: Experts n=%d, Novices n=%d",
        len(expert_ids), len(novice_ids),
    )

    conn_dir = _p2_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_p2_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p2_group_arrays_all_windows"
    )

    n_exp_loaded = sum(1 for d in expert_npz_list if d is not None)
    n_nov_loaded = sum(1 for d in novice_npz_list if d is not None)
    logger.info(
        "build_p2_group_arrays_all_windows: loaded %d/%d Expert, %d/%d Novice",
        n_exp_loaded, len(expert_ids), n_nov_loaded, len(novice_ids),
    )

    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]
    if not all_npz:
        raise RuntimeError(
            "build_p2_group_arrays_all_windows: no Phase 2 .npz files found. "
            "Run run_batch_multivariate_dspte() first."
        )

    max_trials = max(int(d["n_trials"]) for d in all_npz)

    max_wins_per_trial: dict = {}
    for trial_idx in range(max_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        max_w = 0
        for d in all_npz:
            key = f"{trial_id}_connectivity"
            if key in d and d[key].ndim == 2 and d[key].shape[1] > 0:
                max_w = max(max_w, int(d[key].shape[1]))
        max_wins_per_trial[trial_id] = max_w

    logger.info(
        "build_p2_group_arrays_all_windows: max_trials=%d  wins_per_trial=%s",
        max_trials, max_wins_per_trial,
    )

    n_conn = len(get_connection_labels())
    n_exp  = len(expert_ids)
    n_nov  = len(novice_ids)

    B_Exp: dict = {}
    B_Nov: dict = {}

    for trial_id, max_w in max_wins_per_trial.items():
        if max_w == 0:
            continue
        B_Exp[trial_id] = np.full((n_exp, n_conn, max_w), np.nan)
        B_Nov[trial_id] = np.full((n_nov, n_conn, max_w), np.nan)

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

        exp_valid = int(np.sum(~np.all(np.isnan(B_Exp[trial_id]), axis=(1, 2))))
        nov_valid = int(np.sum(~np.all(np.isnan(B_Nov[trial_id]), axis=(1, 2))))
        logger.info(
            "  %s: B_Exp shape=%s (%d valid subs)  B_Nov shape=%s (%d valid subs)",
            trial_id, B_Exp[trial_id].shape, exp_valid,
            B_Nov[trial_id].shape, nov_valid,
        )

    exp_save = {tid: arr for tid, arr in B_Exp.items()}
    exp_save["subject_ids"] = np.array(expert_ids)
    exp_save["strategy"]    = np.array("all_windows")
    np.savez(exp_path, **exp_save)
    logger.info("build_p2_group_arrays_all_windows: saved B_Exp_multivariate_all.npz")

    nov_save = {tid: arr for tid, arr in B_Nov.items()}
    nov_save["subject_ids"] = np.array(novice_ids)
    nov_save["strategy"]    = np.array("all_windows")
    np.savez(nov_path, **nov_save)
    logger.info("build_p2_group_arrays_all_windows: saved B_Nov_multivariate_all.npz")

    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_2_multivariate_dspte",
        "strategy": "all_windows",
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "trials": list(max_wins_per_trial.keys()),
        "max_windows_per_trial": {k: int(v) for k, v in max_wins_per_trial.items()},
        "n_connections": n_conn,
        "connection_labels": get_connection_labels(),
        "expert_subjects": _p2_window_summary(expert_ids, expert_npz_list),
        "novice_subjects": _p2_window_summary(novice_ids, novice_npz_list),
    }
    with open(group_dir / "multivariate_group_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info("build_p2_group_arrays_all_windows: saved multivariate_group_summary.json")

    return B_Exp, B_Nov


# ===========================================================================
# Section 7.3.1b — Phase 2 Group Array Builder: subject_mean  (PRIMARY for H1)
# ===========================================================================

def build_p2_group_arrays_subject_mean(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 2 subject-mean group arrays (PRIMARY strategy for H1).

    For each subject × trial, average all windows to produce a single (20,)
    vector.  Stacked into (n_subs, 20, n_trials) with NaN for missing trials.
    Gives every subject equal weight regardless of window count.

    Returns
    -------
    B_Exp_mean : np.ndarray, shape (n_exp, 20, n_trials)
    B_Nov_mean : np.ndarray, shape (n_nov, 20, n_trials)

    Saves
    -----
    phase_2_group/B_Exp_multivariate_mean.npz
    phase_2_group/B_Nov_multivariate_mean.npz
    """
    if config is None:
        config = load_config()

    from utils_te import get_connection_labels  # noqa

    group_dir = _p2_group_dir(config)
    exp_path  = group_dir / "B_Exp_multivariate_mean.npz"
    nov_path  = group_dir / "B_Nov_multivariate_mean.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p2_group_arrays_subject_mean: cached files found — loading.")
        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)
        return exp_npz["data"], nov_npz["data"]

    te_cfg     = config.get("stage_e_transfer_entropy", {})
    subj_cfg   = te_cfg.get("subjects", {})
    expert_ids = subj_cfg.get("Expert", [])
    novice_ids = subj_cfg.get("Novice", [])

    logger.info(
        "build_p2_group_arrays_subject_mean: Experts n=%d, Novices n=%d",
        len(expert_ids), len(novice_ids),
    )

    conn_dir = _p2_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_p2_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p2_group_arrays_subject_mean"
    )

    n_conn   = len(get_connection_labels())
    n_trials = _N_TRIALS_CANONICAL

    def _build_mean_array(subject_ids, npz_list):
        """Build `(n_subjects, 20, n_trials)` array using per-trial window means."""
        n_subs = len(subject_ids)
        arr = np.full((n_subs, n_conn, n_trials), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                logger.warning(
                    "build_p2_group_arrays_subject_mean: skipping %s (missing .npz)",
                    subject_ids[sub_idx],
                )
                continue
            for trial_idx in range(n_trials):
                trial_id = f"Tr_{trial_idx + 1}"
                key      = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]
                if conn.ndim != 2 or conn.shape[1] == 0:
                    continue
                arr[sub_idx, :, trial_idx] = np.nanmean(conn, axis=1)
        return arr

    B_Exp_mean = _build_mean_array(expert_ids, expert_npz_list)
    B_Nov_mean = _build_mean_array(novice_ids, novice_npz_list)

    logger.info(
        "build_p2_group_arrays_subject_mean: B_Exp shape=%s  B_Nov shape=%s",
        B_Exp_mean.shape, B_Nov_mean.shape,
    )

    np.savez(exp_path, data=B_Exp_mean, subject_ids=np.array(expert_ids),
             strategy=np.array("subject_mean"))
    logger.info("build_p2_group_arrays_subject_mean: saved B_Exp_multivariate_mean.npz")

    np.savez(nov_path, data=B_Nov_mean, subject_ids=np.array(novice_ids),
             strategy=np.array("subject_mean"))
    logger.info("build_p2_group_arrays_subject_mean: saved B_Nov_multivariate_mean.npz")

    return B_Exp_mean, B_Nov_mean


# ===========================================================================
# Section 7.3.1c — Phase 2 Group Array Builder: first_window (SENSITIVITY)
# ===========================================================================

def build_p2_group_arrays_first_window(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 2 first-window group arrays (SENSITIVITY CHECK for H1).

    For each subject × trial, take only Win_1 (index [:, 0]).  Most
    conservative de-confounding: every subject contributes one observation
    per trial regardless of recording duration.

    Returns
    -------
    B_Exp_fw : np.ndarray, shape (n_exp, 20, n_trials)
    B_Nov_fw : np.ndarray, shape (n_nov, 20, n_trials)

    Saves
    -----
    phase_2_group/B_Exp_multivariate_fw.npz
    phase_2_group/B_Nov_multivariate_fw.npz
    """
    if config is None:
        config = load_config()

    from utils_te import get_connection_labels  # noqa

    group_dir = _p2_group_dir(config)
    exp_path  = group_dir / "B_Exp_multivariate_fw.npz"
    nov_path  = group_dir / "B_Nov_multivariate_fw.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p2_group_arrays_first_window: cached files found — loading.")
        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)
        return exp_npz["data"], nov_npz["data"]

    te_cfg     = config.get("stage_e_transfer_entropy", {})
    subj_cfg   = te_cfg.get("subjects", {})
    expert_ids = subj_cfg.get("Expert", [])
    novice_ids = subj_cfg.get("Novice", [])

    logger.info(
        "build_p2_group_arrays_first_window: Experts n=%d, Novices n=%d",
        len(expert_ids), len(novice_ids),
    )

    conn_dir = _p2_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_p2_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p2_group_arrays_first_window"
    )

    n_conn   = len(get_connection_labels())
    n_trials = _N_TRIALS_CANONICAL

    def _build_fw_array(subject_ids, npz_list):
        """Build `(n_subjects, 20, n_trials)` array using Win_1 only."""
        n_subs = len(subject_ids)
        arr = np.full((n_subs, n_conn, n_trials), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                logger.warning(
                    "build_p2_group_arrays_first_window: skipping %s (missing .npz)",
                    subject_ids[sub_idx],
                )
                continue
            for trial_idx in range(n_trials):
                trial_id = f"Tr_{trial_idx + 1}"
                key      = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]
                if conn.ndim != 2 or conn.shape[1] == 0:
                    continue
                arr[sub_idx, :, trial_idx] = conn[:, 0]   # Win_1 only
        return arr

    B_Exp_fw = _build_fw_array(expert_ids, expert_npz_list)
    B_Nov_fw = _build_fw_array(novice_ids, novice_npz_list)

    logger.info(
        "build_p2_group_arrays_first_window: B_Exp_fw shape=%s  B_Nov_fw shape=%s",
        B_Exp_fw.shape, B_Nov_fw.shape,
    )

    np.savez(exp_path, data=B_Exp_fw, subject_ids=np.array(expert_ids),
             strategy=np.array("first_window"))
    logger.info("build_p2_group_arrays_first_window: saved B_Exp_multivariate_fw.npz")

    np.savez(nov_path, data=B_Nov_fw, subject_ids=np.array(novice_ids),
             strategy=np.array("first_window"))
    logger.info("build_p2_group_arrays_first_window: saved B_Nov_multivariate_fw.npz")

    return B_Exp_fw, B_Nov_fw


# ===========================================================================
# Data-loading helpers for statistical tests
# ===========================================================================

def _load_p2_mean_arrays(config: Optional[dict] = None):
    """Load Phase 2 subject-mean group arrays from disk."""
    group_dir = _p2_group_dir(config)
    exp_npz   = np.load(group_dir / "B_Exp_multivariate_mean.npz", allow_pickle=True)
    nov_npz   = np.load(group_dir / "B_Nov_multivariate_mean.npz", allow_pickle=True)
    logger.info(
        "_load_p2_mean_arrays: B_Exp shape=%s  B_Nov shape=%s",
        exp_npz["data"].shape, nov_npz["data"].shape,
    )
    return exp_npz["data"], nov_npz["data"], _load_connection_labels(config)


def _load_p2_fw_arrays(config: Optional[dict] = None):
    """Load Phase 2 first-window group arrays from disk."""
    group_dir = _p2_group_dir(config)
    exp_npz   = np.load(group_dir / "B_Exp_multivariate_fw.npz", allow_pickle=True)
    nov_npz   = np.load(group_dir / "B_Nov_multivariate_fw.npz", allow_pickle=True)
    logger.info(
        "_load_p2_fw_arrays: B_Exp_fw shape=%s  B_Nov_fw shape=%s",
        exp_npz["data"].shape, nov_npz["data"].shape,
    )
    return exp_npz["data"], nov_npz["data"]


def _load_p2_all_windows_arrays(config: Optional[dict] = None):
    """
    Load Phase 2 all-windows group arrays from disk.

    Returns
    -------
    B_Exp : dict  {trial_id: np.ndarray(n_exp, 20, n_wins)}
    B_Nov : dict  {trial_id: np.ndarray(n_nov, 20, n_wins)}
    conn_labels : list[str]
    """
    group_dir = _p2_group_dir(config)
    exp_data  = np.load(group_dir / "B_Exp_multivariate_all.npz", allow_pickle=True)
    nov_data  = np.load(group_dir / "B_Nov_multivariate_all.npz", allow_pickle=True)
    B_Exp = {k: exp_data[k] for k in exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: nov_data[k] for k in nov_data.files if k.startswith("Tr_")}
    logger.info(
        "_load_p2_all_windows_arrays: Exp trials=%s  Nov trials=%s",
        sorted(B_Exp.keys()), sorted(B_Nov.keys()),
    )
    return B_Exp, B_Nov, _load_connection_labels(config)


# ===========================================================================
# Section 7.3.1 — Statistical Tests (Phase 2)
# ===========================================================================

def run_normality_tests_p2(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test on Phase 2 subject-mean arrays.

    Directly calls Phase 1's ``run_normality_tests_te()`` with Phase 2
    data arrays.  Saves result to ``phase_2_statistics/normality_test.csv``.

    Parameters
    ----------
    config : dict, optional
    overwrite : bool

    Returns
    -------
    pd.DataFrame
    """
    if config is None:
        config = load_config()

    stats_dir = _p2_stats_dir(config)
    out_path  = stats_dir / "normality_test.csv"

    if not overwrite and out_path.exists():
        logger.info("run_normality_tests_p2: cached CSV found — loading.")
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    B_Exp_mean, B_Nov_mean, conn_labels = _load_p2_mean_arrays(config)

    df = p1_stats.run_normality_tests_te(B_Exp_mean, B_Nov_mean, conn_labels, config)
    df.to_csv(out_path, index=False)

    n_cells      = len(df)
    n_normal     = int(df["is_normal"].sum())
    n_tested     = int(df["p_value"].notna().sum())
    n_non_normal = n_tested - n_normal
    logger.info(
        "Phase 2 Normality (Shapiro-Wilk): %d cells, %d tested, "
        "%d normal (%.1f%%), %d non-normal",
        n_cells, n_tested, n_normal,
        100 * n_normal / max(n_tested, 1), n_non_normal,
    )
    return df


def test_expert_vs_novice_p2(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    H1: Expert vs. Novice ANOVA on Phase 2 multivariate dSPTE connectivity.

    PRIMARY: subject_mean strategy.
    SENSITIVITY: first_window strategy.

    Re-uses Phase 1's ``_run_h1_anova()`` directly — no test logic duplication.
    Saves to ``phase_2_statistics/H1_expert_vs_novice.csv``.

    Returns
    -------
    pd.DataFrame  (same columns as Phase 1 H1 CSV)
    """
    if config is None:
        config = load_config()

    stats_dir = _p2_stats_dir(config)
    out_path  = stats_dir / "H1_expert_vs_novice.csv"

    if not overwrite and out_path.exists():
        logger.info("test_expert_vs_novice_p2: cached CSV found — loading.")
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    cfg      = _stats_cfg(config)
    alpha    = cfg["alpha"]
    fdr_method = cfg["fdr_method"]

    B_Exp_mean, B_Nov_mean, conn_labels = _load_p2_mean_arrays(config)
    B_Exp_fw,   B_Nov_fw               = _load_p2_fw_arrays(config)

    logger.info("Phase 2 H1: running primary ANOVA (subject_mean)...")
    df_primary = p1_stats._run_h1_anova(B_Exp_mean, B_Nov_mean, conn_labels, cfg)

    logger.info("Phase 2 H1: running sensitivity ANOVA (first_window)...")
    df_fw = p1_stats._run_h1_anova(B_Exp_fw, B_Nov_fw, conn_labels, cfg)

    merge_key = ["trial", "connection"]
    fw_cols = df_fw[merge_key + ["p_raw", "p_fdr", "significant"]].rename(
        columns={
            "p_raw": "p_fw",
            "p_fdr": "p_fdr_fw",
            "significant": "significant_fw",
        }
    )
    df = df_primary.merge(fw_cols, on=merge_key, how="left")
    df["strategy_agreement"] = df["significant"] == df["significant_fw"]

    df.to_csv(out_path, index=False)

    n_sig     = df["significant"].sum()
    n_sig_fw  = df["significant_fw"].sum()
    n_agree   = df["strategy_agreement"].sum()
    logger.info(
        "Phase 2 H1 Expert vs Novice: %d cells, "
        "primary sig=%d  fw_sig=%d  strategy_agree=%d/%d",
        len(df), n_sig, n_sig_fw, n_agree, len(df),
    )
    return df


def test_within_trial_stability_p2(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    H0: Within-trial stability ANOVA on Phase 2 multivariate dSPTE.

    Tests whether conditional dSPTE is stable across windows within a trial.
    Uses all-windows arrays; re-uses Phase 1 ANOVA logic via the imported
    Phase 1 statistics module.

    Saves to ``phase_2_statistics/H0_within_trial_stability.csv``.
    """
    if config is None:
        config = load_config()

    stats_dir = _p2_stats_dir(config)
    out_path  = stats_dir / "H0_within_trial_stability.csv"

    if not overwrite and out_path.exists():
        logger.info("test_within_trial_stability_p2: cached CSV found — loading.")
        return pd.read_csv(out_path)

    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    cfg        = _stats_cfg(config)
    min_n      = cfg["min_subjects_anova"]
    min_groups = cfg["min_groups_anova"]
    alpha      = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_nan    = cfg["h0_max_nan_subjects"]
    check_conn = cfg["h0_check_connection_idx"]

    B_Exp, B_Nov, conn_labels = _load_p2_all_windows_arrays(config)
    n_conn = len(conn_labels)

    def _round(v, n=6):
        """Round numeric values for CSV output while preserving NaNs."""
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values):
        """Apply FDR correction on valid p-values; keep NaN slots unchanged."""
        p_fdr = np.full_like(p_values, np.nan)
        q_vals = np.full_like(p_values, np.nan)
        valid = ~np.isnan(p_values)
        if valid.sum() > 0:
            _, p_corr = fdrcorrection(p_values[valid], alpha=alpha, method=fdr_method)
            p_fdr[valid] = p_corr
            q_vals[valid] = p_corr
        return p_fdr, q_vals

    rows = []
    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for trial_id, arr in sorted(B_group.items()):
            n_subs, n_c, n_wins_total = arr.shape

            n_valid_wins = n_wins_total
            for win_idx in range(n_wins_total):
                col = arr[:, check_conn, win_idx]
                if int(np.sum(np.isnan(col))) > max_nan:
                    n_valid_wins = win_idx
                    break

            if n_valid_wins < min_groups:
                logger.warning(
                    "Phase 2 H0: %s %s — only %d valid window(s) after NaN "
                    "filter (need >= %d). Skipping ANOVA.",
                    group_name, trial_id, n_valid_wins, min_groups,
                )

            win_means = {ci: [] for ci in range(n_conn)}
            win_stds  = {ci: [] for ci in range(n_conn)}
            win_ns    = {ci: [] for ci in range(n_conn)}

            for win_idx in range(n_valid_wins):
                for ci in range(n_conn):
                    x = arr[:, ci, win_idx]
                    x = x[~np.isnan(x)]
                    win_means[ci].append(float(np.nanmean(x)) if len(x) > 0 else np.nan)
                    win_stds[ci].append(float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan)
                    win_ns[ci].append(len(x))

            p_raw_vec  = np.full(n_conn, np.nan)
            f_stat_vec = np.full(n_conn, np.nan)

            for ci in range(n_conn):
                if n_valid_wins < min_groups:
                    continue
                groups = []
                for win_idx in range(n_valid_wins):
                    x = arr[:, ci, win_idx]
                    x = x[~np.isnan(x)]
                    if len(x) >= min_n:
                        groups.append(x)
                if len(groups) >= min_groups:
                    f_stat, p_raw = f_oneway(*groups)
                    p_raw_vec[ci]  = float(p_raw)
                    f_stat_vec[ci] = float(f_stat)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec)

            for ci in range(n_conn):
                row = {
                    "group":           group_name,
                    "trial":           trial_id,
                    "connection":      conn_labels[ci],
                    "n_windows_used":  n_valid_wins,
                    "n_total_windows": n_wins_total,
                    "F_stat":   _round(f_stat_vec[ci]),
                    "p_raw":    _round(p_raw_vec[ci]),
                    "p_fdr":    _round(p_fdr_vec[ci]),
                    "q_value":  _round(q_vals_vec[ci]),
                    "significant": (
                        bool(p_fdr_vec[ci] < alpha)
                        if not np.isnan(p_fdr_vec[ci]) else False
                    ),
                }
                for win_idx in range(n_valid_wins):
                    lbl = f"Win_{win_idx + 1}"
                    row[f"{lbl}_mean"] = _round(win_means[ci][win_idx])
                    row[f"{lbl}_std"]  = _round(win_stds[ci][win_idx])
                    row[f"{lbl}_n"]    = win_ns[ci][win_idx]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    n_sig = df["significant"].sum()
    logger.info(
        "Phase 2 H0 Within-trial stability: %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


def test_across_trial_stability_p2(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    HIV: Across-trial stability ANOVA on Phase 2 multivariate dSPTE.

    Tests whether conditional dSPTE is stable across trial repetitions.
    Uses all-windows arrays — mirrors Phase 1's HIV test logic exactly.

    Saves to ``phase_2_statistics/HIV_across_trial_stability.csv``.
    """
    if config is None:
        config = load_config()

    stats_dir = _p2_stats_dir(config)
    out_path  = stats_dir / "HIV_across_trial_stability.csv"

    if not overwrite and out_path.exists():
        logger.info("test_across_trial_stability_p2: cached CSV found — loading.")
        return pd.read_csv(out_path)

    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    cfg         = _stats_cfg(config)
    min_n       = cfg["min_subjects_anova"]
    min_trials  = cfg["hiv_min_trials"]
    min_groups  = cfg["min_groups_anova"]
    alpha       = cfg["alpha"]
    fdr_method  = cfg["fdr_method"]
    max_windows = cfg["hiv_max_windows"]

    B_Exp, B_Nov, conn_labels = _load_p2_all_windows_arrays(config)
    n_conn = len(conn_labels)
    all_trial_ids = sorted(set(B_Exp.keys()) | set(B_Nov.keys()))

    def _round(v, n=6):
        """Round numeric values for CSV output while preserving NaNs."""
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values):
        """Apply FDR correction on valid p-values; keep NaN slots unchanged."""
        p_fdr = np.full_like(p_values, np.nan)
        q_vals = np.full_like(p_values, np.nan)
        valid = ~np.isnan(p_values)
        if valid.sum() > 0:
            _, p_corr = fdrcorrection(p_values[valid], alpha=alpha, method=fdr_method)
            p_fdr[valid] = p_corr
            q_vals[valid] = p_corr
        return p_fdr, q_vals

    rows = []
    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for win_idx in range(max_windows):
            win_label = f"Win_{win_idx + 1}"

            p_raw_vec  = np.full(n_conn, np.nan)
            f_stat_vec = np.full(n_conn, np.nan)
            per_trial_means = {ci: {} for ci in range(n_conn)}
            per_trial_stds  = {ci: {} for ci in range(n_conn)}
            per_conn_n_trials = np.zeros(n_conn, dtype=int)

            for ci in range(n_conn):
                trial_groups    = []
                included_trials = []

                for trial_id in all_trial_ids:
                    if trial_id not in B_group:
                        continue
                    arr = B_group[trial_id]
                    if win_idx >= arr.shape[2]:
                        continue
                    x = arr[:, ci, win_idx]
                    x_valid = x[~np.isnan(x)]
                    if len(x_valid) >= min_n:
                        trial_groups.append(x_valid)
                        included_trials.append(trial_id)
                        per_trial_means[ci][trial_id] = float(np.mean(x_valid))
                        per_trial_stds[ci][trial_id]  = (
                            float(np.std(x_valid, ddof=1)) if len(x_valid) > 1 else np.nan
                        )

                per_conn_n_trials[ci] = len(trial_groups)

                if len(trial_groups) >= max(min_groups, min_trials):
                    f_stat, p_raw = f_oneway(*trial_groups)
                    f_stat_vec[ci] = float(f_stat)
                    p_raw_vec[ci]  = float(p_raw)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec)

            for ci in range(n_conn):
                row = {
                    "group":        group_name,
                    "window":       win_label,
                    "connection":   conn_labels[ci],
                    "n_trials_used": int(per_conn_n_trials[ci]),
                    "F_stat":   _round(f_stat_vec[ci]),
                    "p_raw":    _round(p_raw_vec[ci]),
                    "p_fdr":    _round(p_fdr_vec[ci]),
                    "q_value":  _round(q_vals_vec[ci]),
                    "significant": (
                        bool(p_fdr_vec[ci] < alpha)
                        if not np.isnan(p_fdr_vec[ci]) else False
                    ),
                }
                for trial_id in all_trial_ids:
                    t_mean = per_trial_means[ci].get(trial_id, np.nan)
                    t_std  = per_trial_stds[ci].get(trial_id, np.nan)
                    row[f"{trial_id}_mean"] = _round(t_mean)
                    row[f"{trial_id}_std"]  = _round(t_std)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    n_sig = df["significant"].sum()
    logger.info(
        "Phase 2 HIV Across-trial stability: %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


# ===========================================================================
# Master runner
# ===========================================================================

def run_all_statistics_p2(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Build Phase 2 group arrays and run all four statistical tests.

    Step 1: Build all three group array files (all_windows, subject_mean,
            first_window) from per-subject Phase 2 .npz files.
    Step 2: Run four tests (Normality, H1, H0, HIV) and save CSVs.
    Step 3: Print summary and return dict of DataFrames.

    Parameters
    ----------
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        Force re-computation even if cached files exist.

    Returns
    -------
    dict with keys: 'normality', 'H1', 'H0', 'HIV' → DataFrames
    """
    if config is None:
        config = load_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn_labels = _load_connection_labels(config)

    # -----------------------------------------------------------------------
    # Step 1: Group arrays
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2 Stats — Step 1: Building group arrays")
    logger.info("=" * 60)

    logger.info("Building all_windows arrays (for H0 + HIV)...")
    build_p2_group_arrays_all_windows(config=config, overwrite=overwrite)

    logger.info("Building subject_mean arrays (primary for H1)...")
    build_p2_group_arrays_subject_mean(config=config, overwrite=overwrite)

    logger.info("Building first_window arrays (sensitivity for H1)...")
    build_p2_group_arrays_first_window(config=config, overwrite=overwrite)

    # -----------------------------------------------------------------------
    # Step 2: Statistical tests
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2 Stats — Step 2: Running statistical tests")
    logger.info("=" * 60)

    results = {}

    logger.info("Running 7.3.1a Normality tests (Shapiro-Wilk, subject_mean)...")
    results["normality"] = run_normality_tests_p2(config=config, overwrite=overwrite)

    logger.info("Running 7.3.1b H1: Expert vs Novice ANOVA (mean + fw)...")
    results["H1"] = test_expert_vs_novice_p2(config=config, overwrite=overwrite)

    logger.info("Running 7.3.1c H0: Within-trial stability ANOVA (all_windows)...")
    results["H0"] = test_within_trial_stability_p2(config=config, overwrite=overwrite)

    logger.info("Running 7.3.1d HIV: Across-trial stability ANOVA (all_windows)...")
    results["HIV"] = test_across_trial_stability_p2(config=config, overwrite=overwrite)

    _print_summary_p2(results, conn_labels, config)
    return results


def _print_summary_p2(
    results: dict,
    conn_labels: list,
    config: Optional[dict] = None,
) -> None:
    """Print a human-readable summary of Phase 2 statistical test results."""
    alpha = _stats_cfg(config)["alpha"]

    print("\n" + "=" * 70)
    print("STATISTICAL TESTING SUMMARY — Phase 2 Multivariate dSPTE")
    print("=" * 70)

    # Normality
    df = results["normality"]
    n_cells      = len(df)
    n_insuf      = int(df["p_value"].isna().sum())
    n_tested     = n_cells - n_insuf
    n_non_normal = int((~df["is_normal"]).sum())
    n_normal     = n_tested - n_non_normal
    print(f"\n7.3.1a Normality (Shapiro-Wilk, subject_mean arrays):")
    print(f"    Total cells    : {n_cells}  ({n_tested} tested, {n_insuf} insufficient n)")
    print(f"    Normal tested  : {n_normal} / {n_tested}  ({100*n_normal/max(n_tested,1):.1f}%)")
    print(f"    Non-normal     : {n_non_normal} ({100*n_non_normal/max(n_tested,1):.1f}%)")
    for grp in ["Expert", "Novice"]:
        sub = df[(df["group"] == grp) & df["p_value"].notna()]
        pct = 100 * sub["is_normal"].mean() if len(sub) > 0 else 0
        print(f"    {grp}: {pct:.1f}% normal (of {len(sub)} tested cells)")

    # H1
    df = results["H1"]
    n_sig    = int(df["significant"].sum())
    n_sig_fw = int(df["significant_fw"].sum()) if "significant_fw" in df else "N/A"
    n_agree  = int(df["strategy_agreement"].sum()) if "strategy_agreement" in df else "N/A"
    print(f"\n7.3.1b H1 — Expert vs. Novice (one-way ANOVA, FDR α={alpha}):")
    print(f"    Total cells    : {len(df)}")
    print(f"    Primary sig    : {n_sig}  (subject_mean)")
    print(f"    Sensitivity sig: {n_sig_fw}  (first_window)")
    print(f"    Strategy agree : {n_agree} / {len(df)}")

    if n_sig > 0:
        sig_df = df[df["significant"]]
        sig_conns = sig_df.groupby("connection")["significant"].sum()
        print(f"    Significant connections (any trial):")
        for conn, cnt in sig_conns.items():
            print(f"      {conn}: {cnt} trial(s)")

    # H0
    df   = results["H0"]
    n_sig = int(df["significant"].sum())
    print(f"\n7.3.1c H0 — Within-trial stability (one-way ANOVA across windows):")
    print(f"    Total cells    : {len(df)}")
    print(f"    FDR-significant: {n_sig}  (unstable connections)")
    pct = 100 * n_sig / max(len(df), 1)
    print(f"    % unstable     : {pct:.1f}%")

    # HIV
    df   = results["HIV"]
    n_sig = int(df["significant"].sum())
    print(f"\n7.3.1d HIV — Across-trial stability (one-way ANOVA across trials):")
    print(f"    Total cells    : {len(df)}")
    print(f"    FDR-significant: {n_sig}  (trial-varying connections)")
    pct = 100 * n_sig / max(len(df), 1)
    print(f"    % trial-varying: {pct:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"Phase 2 statistics saved to: phase_2_multivariate_dspte/statistics/")
    print("=" * 70 + "\n")


# ===========================================================================
# Smoke tests (run as script)
# ===========================================================================

if __name__ == "__main__":
    import time
    import traceback

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("Stage E — Section 7.3.1: Phase 2 Multivariate dSPTE Statistics")
    print("=" * 70 + "\n")

    t_start = time.perf_counter()

    try:
        results = run_all_statistics_p2(overwrite=False)

        # Basic shape assertions
        df_norm = results["normality"]
        df_h1   = results["H1"]
        df_h0   = results["H0"]
        df_hiv  = results["HIV"]

        assert len(df_norm) > 0,  "Normality DataFrame is empty"
        assert "W_stat" in df_norm.columns, "Missing W_stat column"
        assert len(df_h1) > 0,   "H1 DataFrame is empty"
        assert "F_stat" in df_h1.columns, "Missing F_stat in H1"
        assert "significant_fw" in df_h1.columns, "Missing significant_fw in H1"
        assert "strategy_agreement" in df_h1.columns, "Missing strategy_agreement"
        assert len(df_h0) > 0,   "H0 DataFrame is empty"
        assert len(df_hiv) > 0,  "HIV DataFrame is empty"

        # Sanity: n_connections per trial = 20
        assert df_h1.groupby("trial")["connection"].nunique().eq(20).all(), (
            "H1 CSV: expected 20 unique connections per trial"
        )

        t_elapsed = time.perf_counter() - t_start
        print(f"\n[PASS] All assertions passed  ({t_elapsed:.1f} s wall-clock)")
        print("Output CSVs:")
        stats_dir = _p2_stats_dir()
        for fname in ["normality_test.csv", "H1_expert_vs_novice.csv",
                      "H0_within_trial_stability.csv", "HIV_across_trial_stability.csv"]:
            p = stats_dir / fname
            print(f"  {'[OK]' if p.exists() else '[MISSING]'}  {p}")

    except Exception:
        traceback.print_exc()
        print("\n[FAIL] Phase 2 statistics run failed — see traceback above.")
        sys.exit(1)
