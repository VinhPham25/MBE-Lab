"""
Stage E — Section 8.3.1: Phase 3 Bimodal-dSPTE Statistical Tests.

This module mirrors the Phase 1 and Phase 2 statistical workflows for
Phase 3 fused bimodal connectivity outputs.

Implemented tasks
-----------------
- Build Phase 3 group arrays from per-subject fused connectivity files:
  - `build_p3_group_arrays_all_windows()`
  - `build_p3_group_arrays_subject_mean()`
  - `build_p3_group_arrays_first_window()`
- Run all four statistical tests by reusing Phase 1 test logic:
  - `run_normality_tests_p3()`
  - `test_expert_vs_novice_p3()`
  - `test_within_trial_stability_p3()`
  - `test_across_trial_stability_p3()`
- Orchestrate the full Phase 3 stats run:
  - `run_all_statistics_p3()`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_N_TRIALS_CANONICAL = 5


def _import_p1_stats():
    """
    Import the Phase 1 statistics module with importlib.

    Returns
    -------
    module
        Loaded `04_dspte_statistics.py` module object.
    """
    p = _STAGE_E_ROOT / "phase_1" / "04_dspte_statistics.py"
    spec = importlib.util.spec_from_file_location("dspte_statistics_p1_for_p3", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p3_stats_dir(config: Optional[dict] = None) -> Path:
    """
    Return the Phase 3 statistics output directory.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Output path for Phase 3 statistics CSVs.
    """
    from utils_te import get_te_output_dir

    d = get_te_output_dir("phase_3_statistics", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p3_group_dir(config: Optional[dict] = None) -> Path:
    """
    Return the Phase 3 group-output directory.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Output path for Phase 3 group arrays.
    """
    from utils_te import get_te_output_dir

    d = get_te_output_dir("phase_3_group", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p3_conn_dir(config: Optional[dict] = None) -> Path:
    """
    Return the Phase 3 per-subject connectivity directory.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Path containing `sub-*_bimodal_dspte_connectivity.npz` files.
    """
    from utils_te import get_te_output_dir

    return get_te_output_dir("phase_3_connectivity", config=config)


def _stats_cfg(config: Optional[dict] = None) -> dict:
    """
    Return statistical configuration parameters.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    dict
        `stage_e_transfer_entropy.statistics` block.
    """
    if config is None:
        config = load_config()
    return config["stage_e_transfer_entropy"]["statistics"]


def _connection_labels() -> list[str]:
    """
    Return canonical 20 directed ROI connection labels.

    Returns
    -------
    list[str]
        Connection labels in canonical off-diagonal order.
    """
    from utils_te import get_connection_labels

    return get_connection_labels()


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""

    def default(self, obj):
        """Convert numpy values to built-in JSON-serialisable values."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_subject_npz_lists(
    expert_ids: list[str],
    novice_ids: list[str],
    conn_dir: Path,
    caller: str,
) -> tuple[list[Optional[dict]], list[Optional[dict]]]:
    """
    Load per-subject fused Phase 3 `.npz` files for both groups.

    Parameters
    ----------
    expert_ids : list[str]
        Expert subject IDs.
    novice_ids : list[str]
        Novice subject IDs.
    conn_dir : Path
        Phase 3 connectivity directory.
    caller : str
        Caller name for logging context.

    Returns
    -------
    tuple[list[Optional[dict]], list[Optional[dict]]]
        Loaded Expert and Novice `.npz` dict lists; missing files are `None`.
    """

    def _load(subject_ids: list[str]) -> list[Optional[dict]]:
        """Load one group's subject files; preserve index order."""
        results: list[Optional[dict]] = []
        for sid in subject_ids:
            fpath = conn_dir / f"sub-{sid}_bimodal_dspte_connectivity.npz"
            if not fpath.exists():
                logger.warning("%s: missing %s", caller, fpath.name)
                results.append(None)
            else:
                results.append(dict(np.load(fpath, allow_pickle=True)))
        return results

    return _load(expert_ids), _load(novice_ids)


def _load_group_npz_as_dict(path: Path) -> dict:
    """
    Load trial arrays from a saved group `.npz`.

    Parameters
    ----------
    path : Path
        Group `.npz` file path.

    Returns
    -------
    dict
        Trial-key dictionary: `{trial_id: array(n_subs, 20, n_wins)}`.
    """
    data = dict(np.load(path, allow_pickle=True))
    return {k: v for k, v in data.items() if k.startswith("Tr_")}


def _window_summary(subject_ids: list[str], npz_list: list[Optional[dict]]) -> dict:
    """
    Build per-subject trial/window count summary metadata.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs aligned with `npz_list`.
    npz_list : list[Optional[dict]]
        Loaded `.npz` dictionaries or `None` for missing files.

    Returns
    -------
    dict
        Mapping from subject ID to summary fields.
    """
    out = {}
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


def build_p3_group_arrays_all_windows(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[dict, dict]:
    """
    Build Phase 3 all-window group arrays from fused connectivity files.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached arrays when available.

    Returns
    -------
    tuple[dict, dict]
        `(B_Exp, B_Nov)` where each is `{trial_id: (n_subs, 20, max_windows)}`.

    Notes
    -----
    Output files:
    - `B_Exp_bimodal_all.npz`
    - `B_Nov_bimodal_all.npz`
    - `bimodal_group_summary.json`
    """
    if config is None:
        config = load_config()

    group_dir = _p3_group_dir(config)
    exp_path = group_dir / "B_Exp_bimodal_all.npz"
    nov_path = group_dir / "B_Nov_bimodal_all.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p3_group_arrays_all_windows: cache hit.")
        return _load_group_npz_as_dict(exp_path), _load_group_npz_as_dict(nov_path)

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p3_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p3_group_arrays_all_windows"
    )

    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]
    if not all_npz:
        raise RuntimeError(
            "build_p3_group_arrays_all_windows: no Phase 3 fused subject files found. "
            "Run run_batch_bimodal_dspte() first."
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

    exp_save = {tid: arr for tid, arr in B_Exp.items()}
    exp_save["subject_ids"] = np.array(expert_ids)
    exp_save["strategy"] = np.array("all_windows")
    np.savez(exp_path, **exp_save)

    nov_save = {tid: arr for tid, arr in B_Nov.items()}
    nov_save["subject_ids"] = np.array(novice_ids)
    nov_save["strategy"] = np.array("all_windows")
    np.savez(nov_path, **nov_save)

    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_3_bimodal_dspte",
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
    with open(group_dir / "bimodal_group_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    logger.info(
        "build_p3_group_arrays_all_windows: saved group arrays (trials=%s)",
        sorted(B_Exp.keys()),
    )
    return B_Exp, B_Nov


def build_p3_group_arrays_subject_mean(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 3 subject-mean group arrays for H1 testing.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached arrays when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(B_Exp_mean, B_Nov_mean)` each with shape `(n_subs, 20, 5)`.

    Notes
    -----
    Per subject and trial, windows are averaged to one scalar per connection.
    """
    if config is None:
        config = load_config()

    group_dir = _p3_group_dir(config)
    exp_path = group_dir / "B_Exp_bimodal_mean.npz"
    nov_path = group_dir / "B_Nov_bimodal_mean.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p3_group_arrays_subject_mean: cache hit.")
        return (
            np.load(exp_path, allow_pickle=True)["data"],
            np.load(nov_path, allow_pickle=True)["data"],
        )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p3_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p3_group_arrays_subject_mean"
    )

    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]]) -> np.ndarray:
        """Build one group's `(n_subs, 20, 5)` subject-mean array."""
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
        "build_p3_group_arrays_subject_mean: saved arrays Exp=%s Nov=%s",
        B_Exp_mean.shape,
        B_Nov_mean.shape,
    )
    return B_Exp_mean, B_Nov_mean


def build_p3_group_arrays_first_window(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Phase 3 first-window group arrays for H1 sensitivity testing.

    Parameters
    ----------
    study : str
        Study key for provenance metadata.
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached arrays when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(B_Exp_fw, B_Nov_fw)` each with shape `(n_subs, 20, 5)`.

    Notes
    -----
    Per subject and trial, only `Win_1` is retained.
    """
    if config is None:
        config = load_config()

    group_dir = _p3_group_dir(config)
    exp_path = group_dir / "B_Exp_bimodal_fw.npz"
    nov_path = group_dir / "B_Nov_bimodal_fw.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info("build_p3_group_arrays_first_window: cache hit.")
        return (
            np.load(exp_path, allow_pickle=True)["data"],
            np.load(nov_path, allow_pickle=True)["data"],
        )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _p3_conn_dir(config)
    expert_npz_list, novice_npz_list = _load_subject_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_p3_group_arrays_first_window"
    )

    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]]) -> np.ndarray:
        """Build one group's `(n_subs, 20, 5)` first-window array."""
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

    B_Exp_fw = _build(expert_ids, expert_npz_list)
    B_Nov_fw = _build(novice_ids, novice_npz_list)

    np.savez(
        exp_path,
        data=B_Exp_fw,
        subject_ids=np.array(expert_ids),
        strategy=np.array("first_window"),
    )
    np.savez(
        nov_path,
        data=B_Nov_fw,
        subject_ids=np.array(novice_ids),
        strategy=np.array("first_window"),
    )

    logger.info(
        "build_p3_group_arrays_first_window: saved arrays Exp=%s Nov=%s",
        B_Exp_fw.shape,
        B_Nov_fw.shape,
    )
    return B_Exp_fw, B_Nov_fw


def _load_mean_arrays(config: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load Phase 3 subject-mean group arrays from disk.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Expert mean array, Novice mean array, and connection labels.
    """
    group_dir = _p3_group_dir(config)
    exp_npz = np.load(group_dir / "B_Exp_bimodal_mean.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_bimodal_mean.npz", allow_pickle=True)
    return exp_npz["data"], nov_npz["data"], _connection_labels()


def _load_fw_arrays(config: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Phase 3 first-window group arrays from disk.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Expert and Novice first-window arrays.
    """
    group_dir = _p3_group_dir(config)
    exp_npz = np.load(group_dir / "B_Exp_bimodal_fw.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_bimodal_fw.npz", allow_pickle=True)
    return exp_npz["data"], nov_npz["data"]


def _load_all_arrays(config: Optional[dict] = None) -> tuple[dict, dict, list[str]]:
    """
    Load Phase 3 all-window group arrays from disk.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[dict, dict, list[str]]
        Expert trial dictionary, Novice trial dictionary, and connection labels.
    """
    group_dir = _p3_group_dir(config)
    exp_data = np.load(group_dir / "B_Exp_bimodal_all.npz", allow_pickle=True)
    nov_data = np.load(group_dir / "B_Nov_bimodal_all.npz", allow_pickle=True)
    B_Exp = {k: exp_data[k] for k in exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: nov_data[k] for k in nov_data.files if k.startswith("Tr_")}
    return B_Exp, B_Nov, _connection_labels()


def run_normality_tests_p3(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run Shapiro-Wilk normality tests for Phase 3 bimodal connectivity.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached CSV when present.

    Returns
    -------
    pd.DataFrame
        Normality results with one row per `(group, trial, connection)`.
    """
    if config is None:
        config = load_config()

    out_path = _p3_stats_dir(config) / "normality_test.csv"
    if not overwrite and out_path.exists():
        logger.info("run_normality_tests_p3: cache hit.")
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    B_Exp_mean, B_Nov_mean, conn_labels = _load_mean_arrays(config)
    df = p1_stats.run_normality_tests_te(B_Exp_mean, B_Nov_mean, conn_labels, config)
    df.to_csv(out_path, index=False)
    return df


def test_expert_vs_novice_p3(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run H1 Expert-vs-Novice ANOVA for Phase 3 bimodal connectivity.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached CSV when present.

    Returns
    -------
    pd.DataFrame
        H1 ANOVA table with primary and first-window sensitivity columns.
    """
    if config is None:
        config = load_config()

    out_path = _p3_stats_dir(config) / "H1_expert_vs_novice.csv"
    if not overwrite and out_path.exists():
        logger.info("test_expert_vs_novice_p3: cache hit.")
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    cfg = _stats_cfg(config)

    B_Exp_mean, B_Nov_mean, conn_labels = _load_mean_arrays(config)
    B_Exp_fw, B_Nov_fw = _load_fw_arrays(config)

    df_primary = p1_stats._run_h1_anova(B_Exp_mean, B_Nov_mean, conn_labels, cfg)
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
    return df


def test_within_trial_stability_p3(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run H0 within-trial stability ANOVA for Phase 3 bimodal connectivity.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached CSV when present.

    Returns
    -------
    pd.DataFrame
        H0 stability results over windows.
    """
    if config is None:
        config = load_config()

    out_path = _p3_stats_dir(config) / "H0_within_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("test_within_trial_stability_p3: cache hit.")
        return pd.read_csv(out_path)

    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    cfg = _stats_cfg(config)
    min_n = cfg["min_subjects_anova"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_nan = cfg["h0_max_nan_subjects"]
    check_conn = cfg["h0_check_connection_idx"]

    B_Exp, B_Nov, conn_labels = _load_all_arrays(config)
    n_conn = len(conn_labels)

    def _round(v, n=6):
        """Round numeric value for CSV output while preserving NaN."""
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply FDR correction on non-NaN p-values and preserve NaN slots."""
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
            n_subs, _n_c, n_wins_total = arr.shape

            n_valid_wins = n_wins_total
            for win_idx in range(n_wins_total):
                col = arr[:, check_conn, win_idx]
                if int(np.sum(np.isnan(col))) > max_nan:
                    n_valid_wins = win_idx
                    break

            win_means = {ci: [] for ci in range(n_conn)}
            win_stds = {ci: [] for ci in range(n_conn)}
            win_ns = {ci: [] for ci in range(n_conn)}

            for win_idx in range(n_valid_wins):
                for ci in range(n_conn):
                    x = arr[:, ci, win_idx]
                    x = x[~np.isnan(x)]
                    win_means[ci].append(float(np.nanmean(x)) if len(x) > 0 else np.nan)
                    win_stds[ci].append(float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan)
                    win_ns[ci].append(len(x))

            p_raw_vec = np.full(n_conn, np.nan)
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
                    p_raw_vec[ci] = float(p_raw)
                    f_stat_vec[ci] = float(f_stat)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec)

            for ci in range(n_conn):
                row = {
                    "group": group_name,
                    "trial": trial_id,
                    "connection": conn_labels[ci],
                    "n_windows_used": n_valid_wins,
                    "n_total_windows": n_wins_total,
                    "F_stat": _round(f_stat_vec[ci]),
                    "p_raw": _round(p_raw_vec[ci]),
                    "p_fdr": _round(p_fdr_vec[ci]),
                    "q_value": _round(q_vals_vec[ci]),
                    "significant": bool(p_fdr_vec[ci] < alpha) if not np.isnan(p_fdr_vec[ci]) else False,
                }
                for win_idx in range(n_valid_wins):
                    lbl = f"Win_{win_idx + 1}"
                    row[f"{lbl}_mean"] = _round(win_means[ci][win_idx])
                    row[f"{lbl}_std"] = _round(win_stds[ci][win_idx])
                    row[f"{lbl}_n"] = win_ns[ci][win_idx]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def test_across_trial_stability_p3(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run HIV across-trial stability ANOVA for Phase 3 bimodal connectivity.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `False`, load cached CSV when present.

    Returns
    -------
    pd.DataFrame
        HIV stability results across repeated trials.
    """
    if config is None:
        config = load_config()

    out_path = _p3_stats_dir(config) / "HIV_across_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("test_across_trial_stability_p3: cache hit.")
        return pd.read_csv(out_path)

    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    cfg = _stats_cfg(config)
    min_n = cfg["min_subjects_anova"]
    min_trials = cfg["hiv_min_trials"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_windows = cfg["hiv_max_windows"]

    B_Exp, B_Nov, conn_labels = _load_all_arrays(config)
    n_conn = len(conn_labels)
    all_trial_ids = sorted(set(B_Exp.keys()) | set(B_Nov.keys()))

    def _round(v, n=6):
        """Round numeric value for CSV output while preserving NaN."""
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply FDR correction on non-NaN p-values and preserve NaN slots."""
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

            p_raw_vec = np.full(n_conn, np.nan)
            f_stat_vec = np.full(n_conn, np.nan)
            per_trial_means = {ci: {} for ci in range(n_conn)}
            per_trial_stds = {ci: {} for ci in range(n_conn)}
            per_conn_n_trials = np.zeros(n_conn, dtype=int)

            for ci in range(n_conn):
                trial_groups = []
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
                        per_trial_means[ci][trial_id] = float(np.mean(x_valid))
                        per_trial_stds[ci][trial_id] = float(np.std(x_valid, ddof=1)) if len(x_valid) > 1 else np.nan

                per_conn_n_trials[ci] = len(trial_groups)
                if len(trial_groups) >= max(min_groups, min_trials):
                    f_stat, p_raw = f_oneway(*trial_groups)
                    f_stat_vec[ci] = float(f_stat)
                    p_raw_vec[ci] = float(p_raw)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec)

            for ci in range(n_conn):
                row = {
                    "group": group_name,
                    "window": win_label,
                    "connection": conn_labels[ci],
                    "n_trials_used": int(per_conn_n_trials[ci]),
                    "F_stat": _round(f_stat_vec[ci]),
                    "p_raw": _round(p_raw_vec[ci]),
                    "p_fdr": _round(p_fdr_vec[ci]),
                    "q_value": _round(q_vals_vec[ci]),
                    "significant": bool(p_fdr_vec[ci] < alpha) if not np.isnan(p_fdr_vec[ci]) else False,
                }
                for trial_id in all_trial_ids:
                    row[f"{trial_id}_mean"] = _round(per_trial_means[ci].get(trial_id, np.nan))
                    row[f"{trial_id}_std"] = _round(per_trial_stds[ci].get(trial_id, np.nan))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def run_all_statistics_p3(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Execute full Section 8.3.1 Phase 3 statistics workflow.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If `True`, rebuild group arrays and recompute all CSV outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrames keyed by test name: `normality`, `h1`, `h0`, `hiv`.
    """
    if config is None:
        config = load_config()

    build_p3_group_arrays_all_windows(config=config, overwrite=overwrite)
    build_p3_group_arrays_subject_mean(config=config, overwrite=overwrite)
    build_p3_group_arrays_first_window(config=config, overwrite=overwrite)

    df_normal = run_normality_tests_p3(config=config, overwrite=overwrite)
    df_h1 = test_expert_vs_novice_p3(config=config, overwrite=overwrite)
    df_h0 = test_within_trial_stability_p3(config=config, overwrite=overwrite)
    df_hiv = test_across_trial_stability_p3(config=config, overwrite=overwrite)

    out = {
        "normality": df_normal,
        "h1": df_h1,
        "h0": df_h0,
        "hiv": df_hiv,
    }
    write_statistics_summary_p3(results=out, config=config)
    return out


def write_statistics_summary_p3(
    results: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[dict] = None,
) -> Path:
    """
    Write a JSON summary file across the four Phase 3 statistical tests.

    Parameters
    ----------
    results : dict[str, pd.DataFrame], optional
        Optional in-memory test outputs keyed by `normality`, `h1`, `h0`, `hiv`.
        If omitted, existing CSV files are loaded from disk.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Absolute path to `bimodal_statistics_summary.json`.
    """
    if config is None:
        config = load_config()

    stats_dir = _p3_stats_dir(config)
    csv_paths = {
        "normality": stats_dir / "normality_test.csv",
        "h1": stats_dir / "H1_expert_vs_novice.csv",
        "h0": stats_dir / "H0_within_trial_stability.csv",
        "hiv": stats_dir / "HIV_across_trial_stability.csv",
    }

    loaded: dict[str, pd.DataFrame] = {}
    if results is not None:
        loaded.update(results)
    for key, path in csv_paths.items():
        if key not in loaded and path.exists():
            loaded[key] = pd.read_csv(path)

    def _summary_for(name: str, df: Optional[pd.DataFrame]) -> dict:
        """Build per-test summary metrics for one statistics dataframe."""
        if df is None or df.empty:
            return {"status": "missing", "n_rows": 0}

        out = {
            "status": "ok",
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        }
        if "significant" in df.columns:
            out["n_significant"] = int(df["significant"].fillna(False).sum())
        if "p_raw" in df.columns:
            out["n_p_raw_non_nan"] = int(df["p_raw"].notna().sum())
        if name == "normality":
            out["n_normal"] = int(df["is_normal"].fillna(False).sum()) if "is_normal" in df.columns else 0
            out["n_tested"] = int(df["p_value"].notna().sum()) if "p_value" in df.columns else 0
        return out

    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_3_bimodal_dspte",
        "stats_dir": str(stats_dir),
        "tests": {
            "normality": _summary_for("normality", loaded.get("normality")),
            "h1_expert_vs_novice": _summary_for("h1", loaded.get("h1")),
            "h0_within_trial_stability": _summary_for("h0", loaded.get("h0")),
            "hiv_across_trial_stability": _summary_for("hiv", loaded.get("hiv")),
        },
    }
    summary["tests_available"] = int(
        sum(1 for v in summary["tests"].values() if v.get("status") == "ok")
    )

    out_path = stats_dir / "bimodal_statistics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    logger.info("write_statistics_summary_p3: saved %s", out_path)
    return out_path


def _cli_main() -> None:
    """
    Run Section 8.3.1 workflows from command line.

    Returns
    -------
    None
        Exits after printing high-level run summary to logs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Phase 3 bimodal statistics")
    parser.add_argument(
        "--test",
        choices=["all", "normality", "h1", "h0", "hiv", "build-only"],
        default="all",
        help="Select which Section 8.3.1 task to run.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing outputs (default behavior).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Use cached outputs when present.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.test == "build-only":
        build_p3_group_arrays_all_windows(config=cfg, overwrite=args.overwrite)
        build_p3_group_arrays_subject_mean(config=cfg, overwrite=args.overwrite)
        build_p3_group_arrays_first_window(config=cfg, overwrite=args.overwrite)
        logger.info("Phase 3 group arrays built.")
        return

    if args.test == "all":
        out = run_all_statistics_p3(config=cfg, overwrite=args.overwrite)
        logger.info(
            "Phase 3 stats complete: normality=%d h1=%d h0=%d hiv=%d",
            len(out["normality"]), len(out["h1"]), len(out["h0"]), len(out["hiv"]),
        )
        return

    build_p3_group_arrays_all_windows(config=cfg, overwrite=args.overwrite)
    build_p3_group_arrays_subject_mean(config=cfg, overwrite=args.overwrite)
    build_p3_group_arrays_first_window(config=cfg, overwrite=args.overwrite)

    if args.test == "normality":
        df = run_normality_tests_p3(config=cfg, overwrite=args.overwrite)
    elif args.test == "h1":
        df = test_expert_vs_novice_p3(config=cfg, overwrite=args.overwrite)
    elif args.test == "h0":
        df = test_within_trial_stability_p3(config=cfg, overwrite=args.overwrite)
    else:
        df = test_across_trial_stability_p3(config=cfg, overwrite=args.overwrite)

    write_statistics_summary_p3(config=cfg)
    logger.info("Phase 3 %s rows=%d", args.test, len(df))


if __name__ == "__main__":
    _cli_main()
