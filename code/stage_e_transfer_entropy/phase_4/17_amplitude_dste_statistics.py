"""
Stage E — Phase 4: Statistics for Amplitude dSTE (Section 9.8).

Mirrors ``12_bimodal_dspte_statistics.py`` (Phase 3) for Phase 4 outputs.
Reuses Phase 1's statistical functions (imported via importlib) and Phase 4's
group-array builders from ``15_amplitude_dste_group.py``.

Tests implemented
-----------------
- Normality:  Shapiro-Wilk per (group, trial, connection)
- H1:         Expert vs Novice Mann-Whitney U + Benjamini-Hochberg FDR
              (primary: subject_mean; sensitivity: first_window)
              Extra column ``p3_amp_p_raw`` compares against Phase 3's
              bivariate amplitude branch to quantify indirect-connection
              suppression by multivariate conditioning.
- H0:         Within-trial stability — one-way ANOVA across windows, FDR
- HIV:        Across-trial stability — one-way ANOVA across repeated trials, FDR

Output files
------------
``phase_4_statistics/``
  normality_test.csv
  H1_expert_vs_novice.csv
  H0_within_trial_stability.csv
  HIV_across_trial_stability.csv
  amplitude_statistics_summary.json
  asymmetry_index.csv                 ← Section 9.8.6

``phase_4_amplitude_dste/visualizations/``
  p4_asymmetry_index_vs_gc.png        ← Section 9.8.6

CLI
---
python3 17_amplitude_dste_statistics.py --test all
python3 17_amplitude_dste_statistics.py --test h1
python3 17_amplitude_dste_statistics.py --test asymmetry
python3 17_amplitude_dste_statistics.py --test build-only
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


def _import_p1_stats():
    """
    Import Phase 1's ``04_bivariate_dspte_statistics.py`` module via importlib.

    Returns the loaded module object exposing ``run_normality_tests_te``,
    ``_run_h1_anova``, ``test_within_trial_stability_te``,
    ``test_across_trial_stability_te``, and ``_apply_fdr``.
    """
    p = _STAGE_E_ROOT / "phase_1" / "04_bivariate_dspte_statistics.py"
    spec = importlib.util.spec_from_file_location("bivariate_dspte_statistics_p1_for_p4", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_p4_group():
    """Import ``15_amplitude_dste_group.py`` via importlib."""
    p = Path(__file__).parent / "15_amplitude_dste_group.py"
    spec = importlib.util.spec_from_file_location("amplitude_dste_group_p4", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p4_stats_dir(config: Optional[dict] = None) -> Path:
    """Return (and create) the Phase 4 statistics output directory."""
    from utils_te import get_te_output_dir
    d = get_te_output_dir("phase_4_statistics", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p4_group_dir(config: Optional[dict] = None) -> Path:
    """Return the Phase 4 group output directory."""
    from utils_te import get_te_output_dir
    d = get_te_output_dir("phase_4_group", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p3_conn_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 3 per-subject connectivity directory (for P3 comparison)."""
    from utils_te import get_te_output_dir
    return get_te_output_dir("phase_3_connectivity", config=config)


def _connection_labels() -> list[str]:
    from utils_te import get_connection_labels
    return get_connection_labels()


def _stats_cfg(config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_config()
    return config["stage_e_transfer_entropy"]["statistics"]


# ---------------------------------------------------------------------------
# Group array loaders (thin wrappers over 15_amplitude_dste_group.py)
# ---------------------------------------------------------------------------

def _load_mean_arrays(
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load Phase 4 subject-mean group arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(B_Exp_mean, B_Nov_mean, conn_labels)`` — shapes ``(n_subs, 20, 5)``.
    """
    group_dir = _p4_group_dir(config)
    exp_npz = np.load(group_dir / "B_Exp_amplitude_mean.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_amplitude_mean.npz", allow_pickle=True)
    return exp_npz["data"], nov_npz["data"], _connection_labels()


def _load_fw_arrays(
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Phase 4 first-window group arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(B_Exp_fw, B_Nov_fw)`` — shapes ``(n_subs, 20, 5)``.
    """
    group_dir = _p4_group_dir(config)
    exp_npz = np.load(group_dir / "B_Exp_amplitude_first_window.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_amplitude_first_window.npz", allow_pickle=True)
    return exp_npz["data"], nov_npz["data"]


def _load_all_arrays(
    config: Optional[dict] = None,
) -> tuple[dict, dict, list[str]]:
    """
    Load Phase 4 all-window group arrays.

    Returns
    -------
    tuple[dict, dict, list[str]]
        ``(B_Exp, B_Nov, conn_labels)`` — dicts ``{trial_id: (n_subs, 20, n_wins)}``.
    """
    group_dir = _p4_group_dir(config)
    exp_data = np.load(group_dir / "B_Exp_amplitude_all.npz", allow_pickle=True)
    nov_data = np.load(group_dir / "B_Nov_amplitude_all.npz", allow_pickle=True)
    B_Exp = {k: exp_data[k] for k in exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: nov_data[k] for k in nov_data.files if k.startswith("Tr_")}
    return B_Exp, B_Nov, _connection_labels()


# ---------------------------------------------------------------------------
# Phase 3 amplitude branch loader (for H1 comparison column)
# ---------------------------------------------------------------------------

def _load_p3_amplitude_mean(
    config: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load Phase 3 amplitude branch subject-mean group arrays for comparison.

    Phase 3's amplitude branch is *bivariate* (no conditioning).
    Comparing its H1 p-values against Phase 4's quantifies the extent to
    which multivariate conditioning sharpens Expert vs Novice separation.

    Returns
    -------
    tuple[Optional[np.ndarray], Optional[np.ndarray]]
        ``(B_Exp_p3amp, B_Nov_p3amp)`` each ``(n_subs, 20, 5)``, or
        ``(None, None)`` if Phase 3 group arrays have not been built yet.
    """
    try:
        from utils_te import get_te_output_dir
        p3_group_dir = get_te_output_dir("phase_3_group", config=config)
        exp_path = p3_group_dir / "B_Exp_bimodal_mean.npz"
        nov_path = p3_group_dir / "B_Nov_bimodal_mean.npz"
        if not exp_path.exists() or not nov_path.exists():
            logger.warning(
                "_load_p3_amplitude_mean: Phase 3 group arrays not found "
                "(%s). Skipping comparison column.", p3_group_dir
            )
            return None, None
        # Phase 3 bimodal group file contains the fused EC, not the pure amplitude branch.
        # Phase 3 amplitude branch per-subject files are in phase_3_connectivity.
        # We reconstruct a subject-mean from the amplitude_branch npz files.
        te_cfg = (config or load_config()).get("stage_e_transfer_entropy", {})
        subj_cfg = te_cfg.get("subjects", {})
        expert_ids = list(subj_cfg.get("Expert", []))
        novice_ids = list(subj_cfg.get("Novice", []))
        n_conn = len(_connection_labels())
        conn_dir = get_te_output_dir("phase_3_connectivity", config=config)

        def _build(subject_ids: list[str]) -> np.ndarray:
            arr = np.full((len(subject_ids), n_conn, _N_TRIALS_CANONICAL), np.nan)
            for sub_idx, sid in enumerate(subject_ids):
                fpath = conn_dir / f"sub-{sid}_amplitude_branch_connectivity.npz"
                if not fpath.exists():
                    continue
                npz = dict(np.load(fpath, allow_pickle=True))
                for t_idx in range(_N_TRIALS_CANONICAL):
                    tid = f"Tr_{t_idx + 1}"
                    key = f"{tid}_connectivity"
                    if key not in npz:
                        continue
                    conn = npz[key]
                    if conn.ndim == 2 and conn.shape[1] > 0:
                        arr[sub_idx, :, t_idx] = np.nanmean(conn, axis=1)
            return arr

        return _build(expert_ids), _build(novice_ids)
    except Exception as exc:
        logger.warning("_load_p3_amplitude_mean: failed (%s) — skipping.", exc)
        return None, None


# ---------------------------------------------------------------------------
# 9.8.1 — Normality tests
# ---------------------------------------------------------------------------

def run_normality_tests_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run Shapiro-Wilk normality tests for Phase 4 amplitude connectivity.

    Reuses Phase 1's ``run_normality_tests_te()`` on Phase 4 group arrays.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached CSV when available.

    Returns
    -------
    pd.DataFrame
        Normality results with one row per ``(group, trial, connection)``.
    """
    if config is None:
        config = load_config()

    out_path = _p4_stats_dir(config) / "normality_test.csv"
    if not overwrite and out_path.exists():
        logger.info("run_normality_tests_p4: cache hit.")
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    B_Exp_mean, B_Nov_mean, conn_labels = _load_mean_arrays(config)
    df = p1_stats.run_normality_tests_te(B_Exp_mean, B_Nov_mean, conn_labels, config)
    df.to_csv(out_path, index=False)
    logger.info("run_normality_tests_p4: saved %s (%d rows)", out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# 9.8.2 — H1: Expert vs Novice (Mann-Whitney U)
# ---------------------------------------------------------------------------

def test_expert_vs_novice_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run H1 Expert-vs-Novice test for Phase 4 amplitude connectivity.

    Uses Mann-Whitney U (non-parametric, consistent with Phase 3) with
    Benjamini-Hochberg FDR correction per trial.  Primary strategy uses
    subject-mean arrays; sensitivity check uses first-window arrays.

    An additional comparison column ``p3_amp_p_raw`` records the uncorrected
    Mann-Whitney U p-value for Phase 3's bivariate amplitude branch on the
    same connection.  A lower Phase 4 p-value indicates that multivariate
    conditioning sharpens Expert vs Novice separation.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached CSV when available.

    Returns
    -------
    pd.DataFrame
        H1 results with columns:
        ``trial, connection, stat_U, p_raw, p_fdr, significant,
        p_fw, p_fdr_fw, significant_fw, strategy_agreement,
        p3_amp_p_raw, p3_amp_significant``.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection

    if config is None:
        config = load_config()

    out_path = _p4_stats_dir(config) / "H1_expert_vs_novice.csv"
    if not overwrite and out_path.exists():
        logger.info("test_expert_vs_novice_p4: cache hit.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])

    B_Exp_mean, B_Nov_mean, conn_labels = _load_mean_arrays(config)
    B_Exp_fw, B_Nov_fw = _load_fw_arrays(config)
    B_Exp_p3, B_Nov_p3 = _load_p3_amplitude_mean(config)

    n_conn = len(conn_labels)

    def _run_mwu(
        B_Exp: np.ndarray,
        B_Nov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Mann-Whitney U per (trial, connection) and return p_raw, p_fdr, U_stat.

        Returns arrays of shape (n_trials, n_conn).
        """
        n_trials = B_Exp.shape[2]
        p_raw_mat = np.full((n_trials, n_conn), np.nan)
        p_fdr_mat = np.full((n_trials, n_conn), np.nan)
        u_stat_mat = np.full((n_trials, n_conn), np.nan)

        for t_idx in range(n_trials):
            p_raw_trial = np.full(n_conn, np.nan)
            u_trial = np.full(n_conn, np.nan)

            for ci in range(n_conn):
                exp_vals = B_Exp[:, ci, t_idx]
                nov_vals = B_Nov[:, ci, t_idx]
                exp_vals = exp_vals[~np.isnan(exp_vals)]
                nov_vals = nov_vals[~np.isnan(nov_vals)]
                min_n = int(cfg.get("min_subjects_anova", 3))
                if len(exp_vals) < min_n or len(nov_vals) < min_n:
                    continue
                try:
                    stat, p = mannwhitneyu(
                        exp_vals, nov_vals, alternative="two-sided"
                    )
                    u_trial[ci] = float(stat)
                    p_raw_trial[ci] = float(p)
                except Exception:
                    pass

            # FDR across 20 connections for this trial
            valid = ~np.isnan(p_raw_trial)
            p_fdr_trial = np.full(n_conn, np.nan)
            if valid.sum() > 0:
                _, p_corr = fdrcorrection(
                    p_raw_trial[valid], alpha=alpha, method=fdr_method
                )
                p_fdr_trial[valid] = p_corr

            p_raw_mat[t_idx] = p_raw_trial
            p_fdr_mat[t_idx] = p_fdr_trial
            u_stat_mat[t_idx] = u_trial

        return p_raw_mat, p_fdr_mat, u_stat_mat

    p_raw_mean, p_fdr_mean, u_mean = _run_mwu(B_Exp_mean, B_Nov_mean)
    p_raw_fw, p_fdr_fw, _ = _run_mwu(B_Exp_fw, B_Nov_fw)

    # Phase 3 amplitude branch comparison
    p3_p_raw_mat = np.full_like(p_raw_mean, np.nan)
    if B_Exp_p3 is not None and B_Nov_p3 is not None:
        p3_p_raw_mat, _, _ = _run_mwu(B_Exp_p3, B_Nov_p3)

    rows = []
    for t_idx in range(_N_TRIALS_CANONICAL):
        trial_id = f"Tr_{t_idx + 1}"
        for ci in range(n_conn):
            p_r = p_raw_mean[t_idx, ci]
            p_f = p_fdr_mean[t_idx, ci]
            p_r_fw = p_raw_fw[t_idx, ci]
            p_f_fw = p_fdr_fw[t_idx, ci]
            p3_p_r = p3_p_raw_mat[t_idx, ci]

            rows.append({
                "trial": trial_id,
                "connection": conn_labels[ci],
                "stat_U": float(u_mean[t_idx, ci]) if not np.isnan(u_mean[t_idx, ci]) else np.nan,
                "p_raw": round(float(p_r), 6) if not np.isnan(p_r) else np.nan,
                "p_fdr": round(float(p_f), 6) if not np.isnan(p_f) else np.nan,
                "significant": bool(p_f < alpha) if not np.isnan(p_f) else False,
                "p_fw": round(float(p_r_fw), 6) if not np.isnan(p_r_fw) else np.nan,
                "p_fdr_fw": round(float(p_f_fw), 6) if not np.isnan(p_f_fw) else np.nan,
                "significant_fw": bool(p_f_fw < alpha) if not np.isnan(p_f_fw) else False,
                "strategy_agreement": (
                    (bool(p_f < alpha)) == (bool(p_f_fw < alpha))
                    if not (np.isnan(p_f) or np.isnan(p_f_fw))
                    else np.nan
                ),
                "p3_amp_p_raw": round(float(p3_p_r), 6) if not np.isnan(p3_p_r) else np.nan,
                "p3_amp_significant": (
                    bool(p3_p_r < alpha) if not np.isnan(p3_p_r) else False
                ),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("test_expert_vs_novice_p4: saved %s (%d rows)", out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# 9.8.3 — H0: Within-trial window stability
# ---------------------------------------------------------------------------

def test_within_trial_stability_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run H0 within-trial stability one-way ANOVA for Phase 4 amplitude TE.

    Tests whether TE values are stable across windows within the same trial.
    Reuses Phase 3's identical H0 implementation applied to Phase 4 arrays.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached CSV when available.

    Returns
    -------
    pd.DataFrame
        One row per ``(group, trial, connection)``.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    if config is None:
        config = load_config()

    out_path = _p4_stats_dir(config) / "H0_within_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("test_within_trial_stability_p4: cache hit.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    min_n = int(cfg["min_subjects_anova"])
    min_groups = int(cfg["min_groups_anova"])
    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])
    max_nan = int(cfg["h0_max_nan_subjects"])
    check_conn = int(cfg["h0_check_connection_idx"])

    B_Exp, B_Nov, conn_labels = _load_all_arrays(config)
    n_conn = len(conn_labels)

    def _round(v, n=6):
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                    "significant": (
                        bool(p_fdr_vec[ci] < alpha)
                        if not np.isnan(p_fdr_vec[ci])
                        else False
                    ),
                }
                for win_idx in range(n_valid_wins):
                    lbl = f"Win_{win_idx + 1}"
                    row[f"{lbl}_mean"] = _round(win_means[ci][win_idx])
                    row[f"{lbl}_std"] = _round(win_stds[ci][win_idx])
                    row[f"{lbl}_n"] = win_ns[ci][win_idx]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(
        "test_within_trial_stability_p4: saved %s (%d rows)", out_path.name, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# 9.8.4 — HIV: Across-trial stability
# ---------------------------------------------------------------------------

def test_across_trial_stability_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run HIV across-trial stability one-way ANOVA for Phase 4 amplitude TE.

    Tests whether TE values are consistent across repeated trials.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached CSV when available.

    Returns
    -------
    pd.DataFrame
        One row per ``(group, window, connection)``.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    if config is None:
        config = load_config()

    out_path = _p4_stats_dir(config) / "HIV_across_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("test_across_trial_stability_p4: cache hit.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    min_n = int(cfg["min_subjects_anova"])
    min_trials = int(cfg["hiv_min_trials"])
    min_groups = int(cfg["min_groups_anova"])
    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])
    max_windows = int(cfg["hiv_max_windows"])

    B_Exp, B_Nov, conn_labels = _load_all_arrays(config)
    n_conn = len(conn_labels)
    all_trial_ids = sorted(set(B_Exp.keys()) | set(B_Nov.keys()))

    def _round(v, n=6):
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            per_trial_means: dict = {ci: {} for ci in range(n_conn)}
            per_trial_stds: dict = {ci: {} for ci in range(n_conn)}
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
                        per_trial_stds[ci][trial_id] = (
                            float(np.std(x_valid, ddof=1))
                            if len(x_valid) > 1
                            else np.nan
                        )

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
                    "significant": (
                        bool(p_fdr_vec[ci] < alpha)
                        if not np.isnan(p_fdr_vec[ci])
                        else False
                    ),
                }
                for trial_id in all_trial_ids:
                    row[f"{trial_id}_mean"] = _round(
                        per_trial_means[ci].get(trial_id, np.nan)
                    )
                    row[f"{trial_id}_std"] = _round(
                        per_trial_stds[ci].get(trial_id, np.nan)
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(
        "test_across_trial_stability_p4: saved %s (%d rows)", out_path.name, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# 9.8.5 — Orchestrator and JSON summary
# ---------------------------------------------------------------------------

def run_all_statistics_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Execute the full Phase 4 statistics workflow (Section 9.8).

    Builds group arrays (if needed), then runs normality, H1, H0, and HIV
    tests and writes the JSON summary.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``True``, rebuild group arrays and recompute all CSV outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrames keyed by ``'normality'``, ``'h1'``, ``'h0'``, ``'hiv'``.
    """
    if config is None:
        config = load_config()

    # Build group arrays (idempotent with cache)
    p4_group = _import_p4_group()
    p4_group.build_p4_group_arrays_all_windows(config=config, overwrite=overwrite)
    p4_group.build_p4_group_arrays_subject_mean(config=config, overwrite=overwrite)
    p4_group.build_p4_group_arrays_first_window(config=config, overwrite=overwrite)

    df_normal = run_normality_tests_p4(config=config, overwrite=overwrite)
    df_h1 = test_expert_vs_novice_p4(config=config, overwrite=overwrite)
    df_h0 = test_within_trial_stability_p4(config=config, overwrite=overwrite)
    df_hiv = test_across_trial_stability_p4(config=config, overwrite=overwrite)

    out = {
        "normality": df_normal,
        "h1": df_h1,
        "h0": df_h0,
        "hiv": df_hiv,
    }
    write_statistics_summary_p4(results=out, config=config)

    # Asymmetry index (always generated as part of full run)
    try:
        compute_asymmetry_index_p4(config=config, overwrite=overwrite)
        plot_asymmetry_index_p4(config=config, overwrite=overwrite)
    except Exception as exc:
        logger.warning("run_all_statistics_p4: asymmetry plot failed (%s) — continuing.", exc)

    return out


def write_statistics_summary_p4(
    results: Optional[dict] = None,
    config: Optional[dict] = None,
) -> Path:
    """
    Write ``amplitude_statistics_summary.json`` summarising all Phase 4 tests.

    Parameters
    ----------
    results : dict, optional
        In-memory test DataFrames keyed by ``normality``, ``h1``, ``h0``, ``hiv``.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Absolute path to the JSON summary file.
    """
    if config is None:
        config = load_config()

    stats_dir = _p4_stats_dir(config)
    csv_paths = {
        "normality": stats_dir / "normality_test.csv",
        "h1": stats_dir / "H1_expert_vs_novice.csv",
        "h0": stats_dir / "H0_within_trial_stability.csv",
        "hiv": stats_dir / "HIV_across_trial_stability.csv",
    }

    loaded: dict = {}
    if results is not None:
        loaded.update(results)
    for key, path in csv_paths.items():
        if key not in loaded and path.exists():
            loaded[key] = pd.read_csv(path)

    def _summary_for(name: str, df: Optional[pd.DataFrame]) -> dict:
        if df is None or df.empty:
            return {"status": "missing", "n_rows": 0}
        out: dict = {
            "status": "ok",
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        }
        if "significant" in df.columns:
            out["n_significant"] = int(df["significant"].fillna(False).sum())
        if "p_raw" in df.columns:
            out["n_p_raw_non_nan"] = int(df["p_raw"].notna().sum())
        if name == "normality":
            out["n_normal"] = (
                int(df["is_normal"].fillna(False).sum())
                if "is_normal" in df.columns
                else 0
            )
        if name == "h1" and "p3_amp_p_raw" in df.columns:
            out["n_p3_amp_significant"] = int(
                df["p3_amp_significant"].fillna(False).sum()
            )
            # Count connections where Phase 4 is significant but Phase 3 amp is not
            if "p3_amp_significant" in df.columns and "significant" in df.columns:
                new_sig = (
                    df["significant"].fillna(False)
                    & ~df["p3_amp_significant"].fillna(False)
                )
                out["n_new_significant_over_p3_amp"] = int(new_sig.sum())
        return out

    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_4_amplitude_dste",
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

    out_path = stats_dir / "amplitude_statistics_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info("write_statistics_summary_p4: saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 9.8.6 — Asymmetry Index Analysis and GC comparison
# ---------------------------------------------------------------------------

# ROI and connection index constants (shared across asymmetry helpers)
_AI_ROI_NAMES: list[str] = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_AI_N_ROIS: int = 5

# 20-vector index → (src_roi_idx, tgt_roi_idx), row-major excluding diagonal
_CONN_TO_PAIR: list[tuple[int, int]] = [
    (src, tgt)
    for src in range(_AI_N_ROIS)
    for tgt in range(_AI_N_ROIS)
    if src != tgt
]

# (src, tgt) → index in the 20-vector
_PAIR_TO_CONN: dict[tuple[int, int], int] = {
    p: i for i, p in enumerate(_CONN_TO_PAIR)
}

# 10 undirected pairs (src < tgt), same order for bar charts
_UNIQUE_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(_AI_N_ROIS) for j in range(i + 1, _AI_N_ROIS)
]

_UNIQUE_PAIR_LABELS: list[str] = [
    f"{_AI_ROI_NAMES[i]}↔{_AI_ROI_NAMES[j]}" for i, j in _UNIQUE_PAIRS
]


def _vec_to_5x5(vec_20: np.ndarray) -> np.ndarray:
    """
    Convert a 20-element directed connectivity vector to a 5×5 matrix.

    Diagonal entries (self-connections) are set to NaN.

    Parameters
    ----------
    vec_20 : np.ndarray, shape (20,)
        Directed connectivity values in row-major order (src→tgt, diagonal skipped).

    Returns
    -------
    np.ndarray, shape (5, 5)
        mat[src, tgt] = connectivity from src → tgt; NaN on diagonal.
    """
    mat = np.full((_AI_N_ROIS, _AI_N_ROIS), np.nan)
    for conn_idx, (src, tgt) in enumerate(_CONN_TO_PAIR):
        mat[src, tgt] = vec_20[conn_idx]
    return mat


def _compute_ai_matrix(mat_5x5: np.ndarray) -> np.ndarray:
    """
    Compute the 5×5 Asymmetry Index (AI) matrix from a directed connectivity matrix.

    For each ordered pair (i, j) with i ≠ j:

        AI[i, j] = (C(i→j) − C(j→i)) / (C(i→j) + C(j→i))

    where C(i→j) is the connectivity from ROI i to ROI j.

    Properties:
      - AI ∈ (−1, +1].
      - AI[i, j] > 0 → dominant flow from i → j.
      - AI[i, j] < 0 → dominant flow from j → i.
      - AI[i, j] = −AI[j, i]  (antisymmetric matrix).
      - AI[i, j] = NaN when both C(i→j) and C(j→i) are zero or NaN.

    Parameters
    ----------
    mat_5x5 : np.ndarray, shape (5, 5)
        Directed connectivity matrix; NaN on diagonal.

    Returns
    -------
    np.ndarray, shape (5, 5)
        Antisymmetric AI matrix; NaN on diagonal and for zero-sum pairs.
    """
    ai = np.full((_AI_N_ROIS, _AI_N_ROIS), np.nan)
    for i in range(_AI_N_ROIS):
        for j in range(_AI_N_ROIS):
            if i == j:
                continue
            fwd = mat_5x5[i, j]
            rev = mat_5x5[j, i]
            denom = fwd + rev
            if np.isnan(denom) or denom == 0.0:
                ai[i, j] = np.nan
            else:
                ai[i, j] = (fwd - rev) / denom
    return ai


def _group_mean_vec(B_group: np.ndarray | dict) -> np.ndarray:
    """
    Compute a single 20-element group-mean connectivity vector.

    Accepts either:
      - ``np.ndarray`` of shape ``(n_subs, 20, n_trials)`` (Phase 4 style), or
      - ``dict`` mapping trial_id → ``np.ndarray (n_subs, 20, n_wins)`` (GC style).

    Returns the grand mean across all subjects, trials, and windows,
    ignoring NaN.
    """
    if isinstance(B_group, np.ndarray):
        # Phase 4 subject-mean arrays: (n_subs, 20, n_trials)
        return np.nanmean(B_group, axis=(0, 2))  # shape (20,)
    else:
        # GC all-windows dict: {trial_id: (n_subs, 20, n_wins)}
        stacked = np.concatenate(
            [arr for arr in B_group.values() if isinstance(arr, np.ndarray)],
            axis=2,
        )  # (n_subs, 20, total_wins)
        return np.nanmean(stacked, axis=(0, 2))  # shape (20,)


def _load_gc_group_vecs(
    config: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load Stage D GC group arrays and return per-group mean 20-vectors.

    Returns
    -------
    tuple[Optional[np.ndarray], Optional[np.ndarray]]
        ``(gc_exp_vec, gc_nov_vec)`` each shape ``(20,)``, or ``(None, None)``
        if Stage D outputs are not found.
    """
    if config is None:
        config = load_config()

    try:
        from stage_d_granger.utils_gc import get_gc_output_dir
        gc_group_dir = get_gc_output_dir("group", config=config)
        exp_path = gc_group_dir / "B_Exp.npz"
        nov_path = gc_group_dir / "B_Nov.npz"

        if not exp_path.exists() or not nov_path.exists():
            logger.warning(
                "_load_gc_group_vecs: Stage D group arrays not found at %s",
                gc_group_dir,
            )
            return None, None

        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)

        trial_keys = [k for k in exp_npz.files if k.startswith("Tr_")]
        B_Exp_gc = {k: exp_npz[k] for k in trial_keys}
        B_Nov_gc = {k: nov_npz[k] for k in trial_keys}

        return _group_mean_vec(B_Exp_gc), _group_mean_vec(B_Nov_gc)

    except Exception as exc:
        logger.warning("_load_gc_group_vecs: failed (%s) — skipping GC comparison.", exc)
        return None, None


def compute_asymmetry_index_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute the Asymmetry Index for Phase 4 amplitude dSTE and Stage D GC.

    For each of the 10 unique undirected ROI pairs, computes:

        AI = (C_forward − C_reverse) / (C_forward + C_reverse)

    using group-mean connectivity vectors (averaged across subjects, trials,
    and windows) for both Expert and Novice cohorts, for both Phase 4
    amplitude dSTE and Stage D linear GC.

    Saves ``asymmetry_index.csv`` to the Phase 4 statistics directory.

    Returns
    -------
    pd.DataFrame
        One row per undirected pair × group, with columns:
        ``pair, group, forward_conn, reverse_conn, ai_te, ai_gc``.
    """
    if config is None:
        config = load_config()

    out_path = _p4_stats_dir(config) / "asymmetry_index.csv"
    if not overwrite and out_path.exists():
        logger.info("compute_asymmetry_index_p4: cache hit.")
        return pd.read_csv(out_path)

    # --- Phase 4 TE group mean vectors ---
    B_Exp_mean, B_Nov_mean, _ = _load_mean_arrays(config)
    te_exp_vec = _group_mean_vec(B_Exp_mean)
    te_nov_vec = _group_mean_vec(B_Nov_mean)
    te_exp_mat = _vec_to_5x5(te_exp_vec)
    te_nov_mat = _vec_to_5x5(te_nov_vec)

    # --- Stage D GC group mean vectors (optional) ---
    gc_exp_vec, gc_nov_vec = _load_gc_group_vecs(config)
    gc_exp_mat = _vec_to_5x5(gc_exp_vec) if gc_exp_vec is not None else None
    gc_nov_mat = _vec_to_5x5(gc_nov_vec) if gc_nov_vec is not None else None

    rows = []
    for group_label, te_mat, gc_mat in [
        ("Expert", te_exp_mat, gc_exp_mat),
        ("Novice", te_nov_mat, gc_nov_mat),
    ]:
        for (i, j) in _UNIQUE_PAIRS:
            pair_label = f"{_AI_ROI_NAMES[i]}↔{_AI_ROI_NAMES[j]}"
            fwd_te = float(te_mat[i, j]) if not np.isnan(te_mat[i, j]) else np.nan
            rev_te = float(te_mat[j, i]) if not np.isnan(te_mat[j, i]) else np.nan
            denom_te = fwd_te + rev_te if (not np.isnan(fwd_te) and not np.isnan(rev_te)) else np.nan
            ai_te = (fwd_te - rev_te) / denom_te if (denom_te and denom_te != 0) else np.nan

            if gc_mat is not None:
                fwd_gc = float(gc_mat[i, j])
                rev_gc = float(gc_mat[j, i])
                denom_gc = fwd_gc + rev_gc
                ai_gc = (fwd_gc - rev_gc) / denom_gc if denom_gc != 0 else np.nan
            else:
                fwd_gc = rev_gc = ai_gc = np.nan

            rows.append({
                "pair":         pair_label,
                "direction":    f"{_AI_ROI_NAMES[i]}→{_AI_ROI_NAMES[j]}",
                "group":        group_label,
                "te_forward":   fwd_te,
                "te_reverse":   rev_te,
                "ai_te":        ai_te,
                "gc_forward":   fwd_gc,
                "gc_reverse":   rev_gc,
                "ai_gc":        ai_gc,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("compute_asymmetry_index_p4: saved %s (%d rows)", out_path.name, len(df))
    return df


def plot_asymmetry_index_p4(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Path:
    """
    Generate the asymmetry index comparison plot for Phase 4 amplitude dSTE
    vs Stage D linear GC.

    Figure layout (4 heatmaps + 1 bar chart):
    ┌─────────────────────┬─────────────────────┐
    │  Expert: TE AI      │  Expert: GC AI      │  ← row 0
    ├─────────────────────┼─────────────────────┤
    │  Novice: TE AI      │  Novice: GC AI      │  ← row 1
    ├─────────────────────┴─────────────────────┤
    │  10-pair bar chart: AI per group × method │  ← row 2
    └───────────────────────────────────────────┘

    Heatmaps: 5×5 antisymmetric AI matrices (RdBu colormap, vmin=-1, vmax=1).
      - Red cell (i, j): dominant net flow i→j.
      - Blue cell (i, j): dominant net flow j→i.
      - Diagonal: masked gray.
      - Each cell annotated with AI value (2 d.p.).

    Bar chart: 10 unique undirected pairs along x-axis. Four bar series per pair:
      Expert TE (dark blue), Novice TE (red), Expert GC (light blue), Novice GC (orange).
      Positive AI → forward direction dominates; negative → reverse.

    Saved to ``phase_4_amplitude_dste/visualizations/p4_asymmetry_index_vs_gc.png``.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm

    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir
    viz_dir = get_te_output_dir("phase_4_viz", config=config)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / "p4_asymmetry_index_vs_gc.png"

    if not overwrite and out_path.exists():
        logger.info("plot_asymmetry_index_p4: cache hit → %s", out_path)
        return out_path

    # Compute / load AI data
    df = compute_asymmetry_index_p4(config=config, overwrite=overwrite)

    def _ai_mat_for(group: str, col: str) -> np.ndarray:
        """Build 5×5 AI matrix from the DataFrame for one group and column."""
        sub = df[df["group"] == group]
        mat = np.full((_AI_N_ROIS, _AI_N_ROIS), np.nan)
        for _, row in sub.iterrows():
            i = _AI_ROI_NAMES.index(row["direction"].split("→")[0])
            j = _AI_ROI_NAMES.index(row["direction"].split("→")[1])
            val = row[col]
            if not (isinstance(val, float) and np.isnan(val)):
                mat[i, j] = float(val)
                mat[j, i] = -float(val)   # antisymmetric
        return mat

    te_exp_ai  = _ai_mat_for("Expert", "ai_te")
    te_nov_ai  = _ai_mat_for("Novice", "ai_te")
    gc_exp_ai  = _ai_mat_for("Expert", "ai_gc")
    gc_nov_ai  = _ai_mat_for("Novice", "ai_gc")
    gc_available = not np.all(np.isnan(gc_exp_ai))

    # Build pair AI series for bar chart
    exp_te_ai_pairs  = [float(df[(df["group"] == "Expert") & (df["direction"] == f"{_AI_ROI_NAMES[i]}→{_AI_ROI_NAMES[j]}")]["ai_te"].values[0]) for i, j in _UNIQUE_PAIRS]
    nov_te_ai_pairs  = [float(df[(df["group"] == "Novice") & (df["direction"] == f"{_AI_ROI_NAMES[i]}→{_AI_ROI_NAMES[j]}")]["ai_te"].values[0]) for i, j in _UNIQUE_PAIRS]
    exp_gc_ai_pairs  = [float(df[(df["group"] == "Expert") & (df["direction"] == f"{_AI_ROI_NAMES[i]}→{_AI_ROI_NAMES[j]}")]["ai_gc"].values[0]) for i, j in _UNIQUE_PAIRS]
    nov_gc_ai_pairs  = [float(df[(df["group"] == "Novice") & (df["direction"] == f"{_AI_ROI_NAMES[i]}→{_AI_ROI_NAMES[j]}")]["ai_gc"].values[0]) for i, j in _UNIQUE_PAIRS]

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.38,
        wspace=0.30,
        top=0.93, bottom=0.07, left=0.07, right=0.97,
    )

    cmap = "RdBu_r"
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    diag_color = "#cccccc"
    fontsize_cell = 7.5

    def _draw_heatmap(ax, ai_mat, title, show_cbar=False):
        masked = np.ma.masked_invalid(ai_mat)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        # Grey diagonal
        for d in range(_AI_N_ROIS):
            ax.add_patch(
                plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                               color=diag_color, zorder=2)
            )
        # Cell annotations
        for r in range(_AI_N_ROIS):
            for c in range(_AI_N_ROIS):
                if r == c:
                    continue
                val = ai_mat[r, c]
                if np.isnan(val):
                    ax.text(c, r, "n/a", ha="center", va="center",
                            fontsize=fontsize_cell - 1, color="#888888", zorder=3)
                else:
                    # Choose contrasting text colour
                    txt_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                            fontsize=fontsize_cell, color=txt_color,
                            fontweight="bold" if abs(val) > 0.4 else "normal",
                            zorder=3)
        ax.set_xticks(range(_AI_N_ROIS))
        ax.set_yticks(range(_AI_N_ROIS))
        ax.set_xticklabels(_AI_ROI_NAMES, fontsize=9)
        ax.set_yticklabels(_AI_ROI_NAMES, fontsize=9)
        ax.set_xlabel("Target ROI", fontsize=9)
        ax.set_ylabel("Source ROI", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        if show_cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                              ticks=[-1, -0.5, 0, 0.5, 1])
            cb.set_label("AI  (row→col dominates if +)", fontsize=8)
            cb.ax.tick_params(labelsize=8)
        return im

    # Row 0: Expert
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    _draw_heatmap(ax00, te_exp_ai,
                  "Expert — Phase 4 Amplitude dSTE AI")
    _draw_heatmap(ax01, gc_exp_ai if gc_available else np.full((_AI_N_ROIS, _AI_N_ROIS), np.nan),
                  "Expert — Stage D Linear GC AI",
                  show_cbar=True)

    # Row 1: Novice
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    _draw_heatmap(ax10, te_nov_ai,
                  "Novice — Phase 4 Amplitude dSTE AI")
    _draw_heatmap(ax11, gc_nov_ai if gc_available else np.full((_AI_N_ROIS, _AI_N_ROIS), np.nan),
                  "Novice — Stage D Linear GC AI",
                  show_cbar=True)

    # Row 2: Bar chart
    ax2 = fig.add_subplot(gs[2, :])

    n_pairs = len(_UNIQUE_PAIRS)
    x = np.arange(n_pairs)
    width = 0.20

    colors = {
        "Expert TE":  "#1f77b4",
        "Novice TE":  "#d62728",
        "Expert GC":  "#aec7e8",
        "Novice GC":  "#f5a623",
    }

    bar_data = [
        ("Expert TE", exp_te_ai_pairs,  -1.5 * width),
        ("Novice TE", nov_te_ai_pairs,  -0.5 * width),
        ("Expert GC", exp_gc_ai_pairs,   0.5 * width),
        ("Novice GC", nov_gc_ai_pairs,   1.5 * width),
    ]

    for label, vals, offset in bar_data:
        bar_vals = [v if not np.isnan(v) else 0.0 for v in vals]
        ax2.bar(
            x + offset, bar_vals, width,
            color=colors[label], label=label, alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(_UNIQUE_PAIR_LABELS, fontsize=8.5, rotation=25, ha="right")
    ax2.set_ylabel("Asymmetry Index (AI)", fontsize=10)
    ax2.set_xlabel("ROI Pair  (positive → first-named ROI leads)", fontsize=9)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_title(
        "Asymmetry Index per ROI Pair — Phase 4 Amplitude dSTE vs Stage D Linear GC",
        fontsize=10, fontweight="bold",
    )
    ax2.legend(fontsize=9, loc="upper right", ncol=4, framealpha=0.85)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Main title
    fig.suptitle(
        "Directional Asymmetry Index — Phase 4 Amplitude dSTE vs Stage D Linear GC\n"
        "AI[i, j] = (C(i→j) − C(j→i)) / (C(i→j) + C(j→i))   "
        "| Red = row dominates col  |  Blue = col dominates row",
        fontsize=10.5, y=0.97,
    )

    if not gc_available:
        fig.text(
            0.75, 0.70,
            "Stage D GC not available\n(run Stage D pipeline first)",
            ha="center", va="center", fontsize=10, color="gray",
            transform=fig.transFigure,
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_asymmetry_index_p4: saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Run Section 9.8 statistics from command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stage E Phase 4 — Amplitude dSTE Statistics (Section 9.8)"
    )
    parser.add_argument(
        "--test",
        choices=["all", "normality", "h1", "h0", "hiv", "asymmetry", "build-only"],
        default="all",
        help="Which statistical test to run (default: all).",
    )
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite cached outputs.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Use cached outputs when present (default).",
    )
    args = parser.parse_args()

    cfg = load_config()
    p4_group = _import_p4_group()

    if args.test == "build-only":
        p4_group.build_p4_group_arrays_all_windows(config=cfg, overwrite=args.overwrite)
        p4_group.build_p4_group_arrays_subject_mean(config=cfg, overwrite=args.overwrite)
        p4_group.build_p4_group_arrays_first_window(config=cfg, overwrite=args.overwrite)
        logger.info("Phase 4 group arrays built.")
        return

    if args.test == "all":
        out = run_all_statistics_p4(config=cfg, overwrite=args.overwrite)
        logger.info(
            "Phase 4 stats complete: normality=%d  h1=%d  h0=%d  hiv=%d",
            len(out["normality"]),
            len(out["h1"]),
            len(out["h0"]),
            len(out["hiv"]),
        )
        return

    p4_group.build_p4_group_arrays_all_windows(config=cfg, overwrite=args.overwrite)
    p4_group.build_p4_group_arrays_subject_mean(config=cfg, overwrite=args.overwrite)
    p4_group.build_p4_group_arrays_first_window(config=cfg, overwrite=args.overwrite)

    if args.test == "normality":
        df = run_normality_tests_p4(config=cfg, overwrite=args.overwrite)
        logger.info("normality rows=%d", len(df))
    elif args.test == "h1":
        df = test_expert_vs_novice_p4(config=cfg, overwrite=args.overwrite)
        n_sig = int(df["significant"].fillna(False).sum()) if "significant" in df.columns else 0
        logger.info("H1 rows=%d  significant=%d", len(df), n_sig)
    elif args.test == "h0":
        df = test_within_trial_stability_p4(config=cfg, overwrite=args.overwrite)
        logger.info("H0 rows=%d", len(df))
    elif args.test == "hiv":
        df = test_across_trial_stability_p4(config=cfg, overwrite=args.overwrite)
        logger.info("HIV rows=%d", len(df))
    elif args.test == "asymmetry":
        df_ai = compute_asymmetry_index_p4(config=cfg, overwrite=args.overwrite)
        out_fig = plot_asymmetry_index_p4(config=cfg, overwrite=args.overwrite)
        logger.info("Asymmetry index: %d rows  figure→%s", len(df_ai), out_fig)
        return
    write_statistics_summary_p4(config=cfg)


if __name__ == "__main__":
    _cli_main()
