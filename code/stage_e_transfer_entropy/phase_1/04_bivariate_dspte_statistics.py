"""
Stage E — Step E4: Phase 1 Bivariate dSPTE Statistical Tests

Mirrors Stage D ``05_gc_statistics.py`` exactly — same four tests (normality,
H1, H0, HIV), same ANOVA + FDR structure, same MATLAB alignment notes.

Differences from Stage D
-------------------------
1. **H1 array routing:** The Expert vs Novice test uses subject-averaged group
   arrays (strategy="subject_mean") as the primary input, and first-window
   arrays (strategy="first_window") as a sensitivity check.  This controls for
   the trial-duration confound: Novice subjects record more windows per trial
   than Experts (~16 vs ~6), so a naive window-stack over-weights late-trial
   Novice dynamics.  Subject averaging gives each subject equal weight.
   Primary results use subject_mean strategy to control for trial-duration
   confound.  fw = first-window-only sensitivity check.

2. **H1 input shape:** Subject_mean/fw arrays are (n_subs, 20, n_trials)
   rather than (n_subs, 20, n_wins).  The "window" axis is the trial axis.
   FDR is applied per trial across 20 connections (same structure as Stage D's
   per-(trial, window) FDR, but here the trial dimension plays the window role).

3. **H0/HIV array routing:** Use all-windows arrays (strategy="all_windows")
   — window-level resolution required for within-trial and across-trial
   stability tests.  These match the Stage D dict format exactly:
   ``{trial_id: np.ndarray(n_subs, 20, n_wins)}``.

4. **H1 extra CSV columns:** ``p_fw``, ``significant_fw``, ``strategy_agreement``.

Section 5 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
-------------------------------------------------------
5.1 — run_normality_tests_te()
5.2 — test_expert_vs_novice_te()       ← H1
5.3 — test_within_trial_stability_te() ← H0
5.4 — test_across_trial_stability_te() ← HIV

MATLAB equivalents
------------------
  swtest(x)            → scipy.stats.shapiro(x)
  anova1(X)            → scipy.stats.f_oneway(*groups)
  mafdr(p)             → statsmodels.stats.multitest.fdrcorrection(p)
  nanmean(X)           → np.nanmean(X)

NaN handling
------------
Subject-mean/fw arrays (n_subs, 20, n_trials): NaN where subject has no data
for that trial (R9: Tr_3–5 NaN; R4: Tr_5 NaN).

All-windows arrays (dict of {trial_id: (n_subs, 20, n_wins)}): NaN where
subject has fewer windows than the trial maximum.

Zero-filled ROI connections (R11 LM1/SMA, R2 SMA) remain 0.0 — valid
observations (true absent connectivity), not missing data.

Output files
------------
statistics/normality_test.csv
statistics/H1_expert_vs_novice.csv
statistics/H0_within_trial_stability.csv
statistics/HIV_across_trial_stability.csv
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro
from statsmodels.stats.multitest import fdrcorrection

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration and data-loading helpers
# ===========================================================================

def _stats_cfg(config: Optional[dict] = None) -> dict:
    """Return the statistics sub-config from config.yaml."""
    if config is None:
        config = load_config()
    return config["stage_e_transfer_entropy"]["statistics"]


def _load_te_stats_dir(config: Optional[dict] = None) -> Path:
    """Return the Phase 1 statistics output directory, creating it if needed."""
    from utils_te import get_te_output_dir  # noqa
    stats_dir = get_te_output_dir("phase_1_statistics", config=config)
    stats_dir.mkdir(parents=True, exist_ok=True)
    return stats_dir


def _load_connection_labels(config: Optional[dict] = None) -> list[str]:
    from utils_te import get_connection_labels  # noqa
    return get_connection_labels()


def _load_mean_arrays(config: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load B_Exp_bivariate_dspte_mean.npz and B_Nov_bivariate_dspte_mean.npz.

    Returns
    -------
    B_Exp_mean : (n_exp, 20, 5)
    B_Nov_mean : (n_nov, 20, 5)
    conn_labels : list[str]
    """
    from utils_te import get_te_output_dir  # noqa
    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_npz = np.load(group_dir / "B_Exp_bivariate_dspte_mean.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_bivariate_dspte_mean.npz", allow_pickle=True)
    logger.info(
        "_load_mean_arrays: B_Exp shape=%s  B_Nov shape=%s",
        exp_npz["data"].shape, nov_npz["data"].shape,
    )
    return exp_npz["data"], nov_npz["data"], _load_connection_labels(config)


def _load_fw_arrays(config: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load B_Exp_bivariate_dspte_fw.npz and B_Nov_bivariate_dspte_fw.npz."""
    from utils_te import get_te_output_dir  # noqa
    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_npz = np.load(group_dir / "B_Exp_bivariate_dspte_fw.npz", allow_pickle=True)
    nov_npz = np.load(group_dir / "B_Nov_bivariate_dspte_fw.npz", allow_pickle=True)
    logger.info(
        "_load_fw_arrays: B_Exp_fw shape=%s  B_Nov_fw shape=%s",
        exp_npz["data"].shape, nov_npz["data"].shape,
    )
    return exp_npz["data"], nov_npz["data"]


def _load_all_windows_arrays(config: Optional[dict] = None) -> tuple[dict, dict, list[str]]:
    """
    Load B_Exp_bivariate_dspte_all.npz and B_Nov_bivariate_dspte_all.npz.

    Returns
    -------
    B_Exp : dict  {trial_id: np.ndarray(n_exp_subs, 20, n_wins)}
    B_Nov : dict  {trial_id: np.ndarray(n_nov_subs, 20, n_wins)}
    conn_labels : list[str]
    """
    from utils_te import get_te_output_dir  # noqa
    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_data = np.load(group_dir / "B_Exp_bivariate_dspte_all.npz", allow_pickle=True)
    nov_data = np.load(group_dir / "B_Nov_bivariate_dspte_all.npz", allow_pickle=True)
    B_Exp = {k: exp_data[k] for k in exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: nov_data[k] for k in nov_data.files if k.startswith("Tr_")}
    logger.info(
        "_load_all_windows_arrays: Exp trials=%s  Nov trials=%s",
        sorted(B_Exp.keys()), sorted(B_Nov.keys()),
    )
    return B_Exp, B_Nov, _load_connection_labels(config)


def _apply_fdr(p_values: np.ndarray, alpha: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction to a 1-D array of p-values.

    NaN entries are excluded and restored as NaN in output.
    Returns (p_fdr, q_values) both length-n, matching input.
    """
    p_fdr = np.full_like(p_values, np.nan)
    q_vals = np.full_like(p_values, np.nan)
    valid_mask = ~np.isnan(p_values)
    if valid_mask.sum() > 0:
        reject, p_corr = fdrcorrection(p_values[valid_mask], alpha=alpha, method=method)
        p_fdr[valid_mask] = p_corr
        q_vals[valid_mask] = p_corr
    return p_fdr, q_vals


def _round(v, n=6):
    """Round float; pass-through NaN."""
    return round(float(v), n) if not np.isnan(v) else np.nan


# ===========================================================================
# Section 5.1 — Normality Tests (Shapiro-Wilk)
# ===========================================================================

def run_normality_tests_te(
    B_Exp: np.ndarray,
    B_Nov: np.ndarray,
    conn_labels: list[str],
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test for each (group, trial, connection) cell.

    Input arrays are the subject-mean arrays (n_subs, 20, n_trials).
    Each cell is the subject-level TE distribution (n_subjects,) for that
    (trial, connection).

    Shapiro-Wilk requires n >= 3; cells with fewer valid subjects get
    W=NaN, p=NaN, is_normal=True (conservative — cannot reject normality).

    Parameters
    ----------
    B_Exp : np.ndarray, shape (n_exp, 20, n_trials)
        Expert group arrays (subject_mean strategy).
    B_Nov : np.ndarray, shape (n_nov, 20, n_trials)
        Novice group arrays (subject_mean strategy).
    conn_labels : list[str]
        20 connection label strings.
    config : dict, optional

    Returns
    -------
    pd.DataFrame  columns: [group, trial, connection,
                             W_stat, p_value, is_normal, n_subjects]
    """
    cfg = _stats_cfg(config)
    min_n = cfg["min_subjects_normality"]
    alpha = cfg["alpha"]

    rows = []
    for group_name, arr in [("Expert", B_Exp), ("Novice", B_Nov)]:
        n_subs, n_conn, n_trials = arr.shape
        for trial_idx in range(n_trials):
            trial_id = f"Tr_{trial_idx + 1}"
            for conn_idx in range(n_conn):
                x = arr[:, conn_idx, trial_idx]
                x_valid = x[~np.isnan(x)]
                n_valid = len(x_valid)

                if n_valid < min_n:
                    w_stat, p_val, is_normal = np.nan, np.nan, True
                else:
                    w_stat, p_val = shapiro(x_valid)
                    is_normal = bool(p_val > alpha)

                rows.append({
                    "group": group_name,
                    "trial": trial_id,
                    "connection": conn_labels[conn_idx],
                    "W_stat": _round(w_stat),
                    "p_value": _round(p_val),
                    "is_normal": is_normal,
                    "n_subjects": n_valid,
                })

    df = pd.DataFrame(rows)
    logger.info(
        "Normality test: %d cells, %d normal (%.1f%%)",
        len(df),
        df["is_normal"].sum(),
        100 * df["is_normal"].mean(),
    )
    return df


# ===========================================================================
# Section 5.2 — H1: Expert vs. Novice (PRIMARY + SENSITIVITY)
# ===========================================================================

def _run_h1_anova(
    B_Exp: np.ndarray,
    B_Nov: np.ndarray,
    conn_labels: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """
    Core ANOVA logic for H1, reused for both primary and sensitivity runs.

    Input shape: (n_subs, 20, n_trials).
    Returns a DataFrame with one row per (trial, connection).
    FDR is applied per trial across 20 connections.
    """
    min_n = cfg["min_subjects_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    primary_trial = cfg["h1_primary_trial"]

    n_exp, n_conn, n_trials = B_Exp.shape
    rows = []

    for trial_idx in range(n_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        arr_exp = B_Exp[:, :, trial_idx]   # (n_exp, 20)
        arr_nov = B_Nov[:, :, trial_idx]   # (n_nov, 20)

        p_raw_vec = np.full(n_conn, np.nan)
        f_stat_vec = np.full(n_conn, np.nan)
        cell_stats = []

        for conn_idx in range(n_conn):
            x_exp = arr_exp[:, conn_idx]
            x_nov = arr_nov[:, conn_idx]
            x_exp = x_exp[~np.isnan(x_exp)]
            x_nov = x_nov[~np.isnan(x_nov)]
            n_e, n_n = len(x_exp), len(x_nov)

            mean_exp = float(np.nanmean(x_exp)) if n_e > 0 else np.nan
            std_exp  = float(np.nanstd(x_exp, ddof=1)) if n_e > 1 else np.nan
            mean_nov = float(np.nanmean(x_nov)) if n_n > 0 else np.nan
            std_nov  = float(np.nanstd(x_nov, ddof=1)) if n_n > 1 else np.nan

            if n_e >= min_n and n_n >= min_n:
                f_stat, p_raw = f_oneway(x_exp, x_nov)
                p_raw_vec[conn_idx] = float(p_raw)
                f_stat_vec[conn_idx] = float(f_stat)
            else:
                f_stat, p_raw = np.nan, np.nan

            cell_stats.append({
                "trial": trial_id,
                "connection": conn_labels[conn_idx],
                "mean_Exp": _round(mean_exp),
                "std_Exp":  _round(std_exp),
                "n_Exp": n_e,
                "mean_Nov": _round(mean_nov),
                "std_Nov":  _round(std_nov),
                "n_Nov": n_n,
                "F_stat": _round(f_stat),
                "p_raw":  _round(p_raw),
                "primary_trial": (trial_id == primary_trial),
            })

        p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

        #Post process the cell stats ajd k0ve t0 rows dataframe
        for conn_idx, row in enumerate(cell_stats):
            row["p_fdr"] = _round(p_fdr_vec[conn_idx])
            row["q_value"] = _round(q_vals_vec[conn_idx])
            row["significant"] = (
                bool(p_fdr_vec[conn_idx] < alpha)
                if not np.isnan(p_fdr_vec[conn_idx]) else False
            )
            rows.append(row)

    return pd.DataFrame(rows)


def test_expert_vs_novice_te(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    H1: Test whether Expert and Novice groups differ in dSPTE connectivity.

    PRIMARY input:
        B_Exp_bivariate_dspte_mean.npz / B_Nov_bivariate_dspte_mean.npz (subject_mean strategy).
        Each subject contributes one value per trial regardless of how many
        windows they recorded — controls for the Novice/Expert window-count
        confound.

    SENSITIVITY CHECK:
        B_Exp_bivariate_dspte_fw.npz / B_Nov_bivariate_dspte_fw.npz (first_window strategy).
        Same ANOVA run on Win_1-only values.  Most conservative de-confounding.

    Primary results use subject_mean strategy to control for trial-duration
    confound.  fw = first-window-only sensitivity check.

    Extra CSV columns vs Stage D H1:
        p_fw             : p_raw from first-window sensitivity ANOVA
        p_fdr_fw         : FDR-corrected p from sensitivity ANOVA (BH)
        significant_fw   : bool, FDR-significant in sensitivity check
        strategy_agreement : True if ``significant`` == ``significant_fw``

    Saves to ``phase_1_bivariate_dspte/statistics/H1_expert_vs_novice.csv``.

    Parameters
    ----------
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        If False, skip if output CSV already exists.

    Returns
    -------
    pd.DataFrame
        H1 test results with primary + sensitivity columns.
    """
    if config is None:
        config = load_config()

    stats_dir = _load_te_stats_dir(config)
    out_path = stats_dir / "H1_expert_vs_novice.csv"

    if not overwrite and out_path.exists():
        logger.info("test_expert_vs_novice_te: cached CSV found — loading.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]

    B_Exp_mean, B_Nov_mean, conn_labels = _load_mean_arrays(config)
    B_Exp_fw, B_Nov_fw = _load_fw_arrays(config)

    # --- PRIMARY ANOVA (subject_mean) ---
    logger.info("H1: running primary ANOVA (subject_mean strategy)...")
    df_primary = _run_h1_anova(B_Exp_mean, B_Nov_mean, conn_labels, cfg)

    # --- SENSITIVITY ANOVA (first_window) ---
    logger.info("H1: running sensitivity ANOVA (first_window strategy)...")
    df_fw = _run_h1_anova(B_Exp_fw, B_Nov_fw, conn_labels, cfg)

    # Merge sensitivity columns into primary
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

    n_sig = df["significant"].sum()
    n_sig_fw = df["significant_fw"].sum()
    n_agree = df["strategy_agreement"].sum()
    logger.info(
        "H1 Expert vs Novice (TE): %d cells, "
        "primary sig=%d  fw_sig=%d  strategy_agree=%d/%d",
        len(df), n_sig, n_sig_fw, n_agree, len(df),
    )
    return df


# ===========================================================================
# Section 5.3 — H0: Within-Trial Stability
# ===========================================================================

def test_within_trial_stability_te(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    H0: Test whether dSPTE connectivity is stable across windows within a trial.

    Uses all-windows arrays — window-level resolution required for within-trial
    stability test.

    Mirrors Stage D ``test_within_trial_stability()`` exactly.  ANOVA tests
    equality of window-level means per (group, trial, connection).  FDR is
    applied per (group, trial) across 20 connections.

    Saves to ``phase_1_bivariate_dspte/statistics/H0_within_trial_stability.csv``.

    Parameters
    ----------
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        If False, skip if output CSV already exists.

    Returns
    -------
    pd.DataFrame  columns: [group, trial, connection,
                             n_windows_used, n_total_windows,
                             F_stat, p_raw, p_fdr, q_value, significant,
                             Win_{k}_mean, Win_{k}_std, Win_{k}_n  (per valid window)]
    """
    if config is None:
        config = load_config()

    stats_dir = _load_te_stats_dir(config)
    out_path = stats_dir / "H0_within_trial_stability.csv"

    if not overwrite and out_path.exists():
        logger.info("test_within_trial_stability_te: cached CSV found — loading.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    min_n = cfg["min_subjects_anova"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_nan = cfg["h0_max_nan_subjects"]
    check_conn = cfg["h0_check_connection_idx"]

    # Uses all-windows arrays — window-level resolution required for
    # within-trial stability test.
    B_Exp, B_Nov, conn_labels = _load_all_windows_arrays(config)
    n_conn = len(conn_labels)

    rows = []
    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for trial_id, arr in sorted(B_group.items()):
            n_subs, n_c, n_wins_total = arr.shape

            # NaN filter: find last valid window (mirrors MATLAB zer_col > 2 logic)
            # If more than max_nan subjects have NaN in the check connection, truncate the window
            n_valid_wins = n_wins_total
            for win_idx in range(n_wins_total):
                col = arr[:, check_conn, win_idx]
                nan_count = int(np.sum(np.isnan(col)))
                if nan_count > max_nan:
                    n_valid_wins = win_idx
                    break

            if n_valid_wins < min_groups:
                logger.warning(
                    "H0: %s %s — only %d valid window(s) after NaN filter "
                    "(need >= %d). Skipping ANOVA.",
                    group_name, trial_id, n_valid_wins, min_groups,
                )

            # Per-window descriptive stats
            win_means = {ci: [] for ci in range(n_conn)}
            win_stds  = {ci: [] for ci in range(n_conn)}
            win_ns    = {ci: [] for ci in range(n_conn)}

            # Collect the descriptive stats for each connection for each window
            for win_idx in range(n_valid_wins):
                for ci in range(n_conn):
                    x = arr[:, ci, win_idx]
                    x = x[~np.isnan(x)]
                    win_means[ci].append(float(np.nanmean(x)) if len(x) > 0 else np.nan)
                    win_stds[ci].append(float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan)
                    win_ns[ci].append(len(x))

            # ANOVA across windows per connection
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
                    p_raw_vec[ci] = float(p_raw)
                    f_stat_vec[ci] = float(f_stat)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

            for ci in range(n_conn):
                row = {
                    "group": group_name,
                    "trial": trial_id,
                    "connection": conn_labels[ci],
                    "n_windows_used": n_valid_wins,
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
        "H0 Within-trial stability (TE): %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


# ===========================================================================
# Section 5.4 — HIV: Across-Trial Stability
# ===========================================================================

def test_across_trial_stability_te(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    HIV: Test whether dSPTE connectivity is stable across trial repetitions.

    Uses all-windows arrays — window-level resolution required for across-trial
    stability test.

    Mirrors Stage D ``test_across_trial_stability()`` exactly.  For each
    (group, window_idx, connection), collects connectivity from that window
    across all trials and runs ANOVA.  Only the first ``hiv_max_windows``
    windows are tested (mirrors MATLAB j=1:3).

    Saves to ``phase_1_bivariate_dspte/statistics/HIV_across_trial_stability.csv``.

    Parameters
    ----------
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        If False, skip if output CSV already exists.

    Returns
    -------
    pd.DataFrame  columns: [group, window, connection,
                             n_trials_used, F_stat, p_raw, p_fdr, q_value,
                             significant,
                             Tr_{k}_mean, Tr_{k}_std  (per included trial)]
    """
    if config is None:
        config = load_config()

    stats_dir = _load_te_stats_dir(config)
    out_path = stats_dir / "HIV_across_trial_stability.csv"

    if not overwrite and out_path.exists():
        logger.info("test_across_trial_stability_te: cached CSV found — loading.")
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    min_n = cfg["min_subjects_anova"]
    min_trials = cfg["hiv_min_trials"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_windows = cfg["hiv_max_windows"]

    # Uses all-windows arrays — window-level resolution required for
    # across-trial stability test.
    B_Exp, B_Nov, conn_labels = _load_all_windows_arrays(config)
    n_conn = len(conn_labels)
    all_trial_ids = sorted(set(B_Exp.keys()) | set(B_Nov.keys()))

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
                trial_groups = []
                included_trials = []

                for trial_id in all_trial_ids:
                    if trial_id not in B_group:
                        continue
                    arr = B_group[trial_id]   # (n_subs, 20, n_wins)
                    if win_idx >= arr.shape[2]:
                        continue
                    x = arr[:, ci, win_idx]
                    x_valid = x[~np.isnan(x)]
                    if len(x_valid) >= min_n:
                        trial_groups.append(x_valid)
                        included_trials.append(trial_id)
                        per_trial_means[ci][trial_id] = float(np.mean(x_valid))
                        per_trial_stds[ci][trial_id] = (
                            float(np.std(x_valid, ddof=1)) if len(x_valid) > 1 else np.nan
                        )

                per_conn_n_trials[ci] = len(trial_groups)

                if len(trial_groups) >= max(min_groups, min_trials):
                    f_stat, p_raw = f_oneway(*trial_groups)
                    f_stat_vec[ci] = float(f_stat)
                    p_raw_vec[ci] = float(p_raw)

            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

            for ci in range(n_conn):
                row = {
                    "group": group_name,
                    "window": win_label,
                    "connection": conn_labels[ci],
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
        "HIV Across-trial stability (TE): %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


# ===========================================================================
# Main runner — run all four tests and save CSVs
# ===========================================================================

def run_all_statistics_te(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run all four dSPTE statistical tests and save CSVs.

    Returns
    -------
    dict with keys: 'normality', 'H1', 'H0', 'HIV' → DataFrames
    """
    if config is None:
        config = load_config()

    stats_dir = _load_te_stats_dir(config)
    conn_labels = _load_connection_labels(config)
    results = {}

    # 5.1 Normality
    logger.info("Running 5.1 Normality tests (Shapiro-Wilk, subject_mean arrays)...")
    B_Exp_mean, B_Nov_mean, _ = _load_mean_arrays(config)
    df_norm = run_normality_tests_te(B_Exp_mean, B_Nov_mean, conn_labels, config)
    out = stats_dir / "normality_test.csv"
    df_norm.to_csv(out, index=False)
    logger.info("Saved %s  (%d rows)", out.name, len(df_norm))
    results["normality"] = df_norm

    # 5.2 H1: Expert vs Novice
    logger.info("Running 5.2 H1: Expert vs Novice ANOVA (mean + fw sensitivity)...")
    df_h1 = test_expert_vs_novice_te(config=config, overwrite=overwrite)
    logger.info("H1 CSV: %d rows", len(df_h1))
    results["H1"] = df_h1

    # 5.3 H0: Within-trial stability
    logger.info("Running 5.3 H0: Within-trial stability ANOVA (all_windows)...")
    df_h0 = test_within_trial_stability_te(config=config, overwrite=overwrite)
    logger.info("H0 CSV: %d rows", len(df_h0))
    results["H0"] = df_h0

    # 5.4 HIV: Across-trial stability
    logger.info("Running 5.4 HIV: Across-trial stability ANOVA (all_windows)...")
    df_hiv = test_across_trial_stability_te(config=config, overwrite=overwrite)
    logger.info("HIV CSV: %d rows", len(df_hiv))
    results["HIV"] = df_hiv

    _print_summary_te(results, conn_labels, config)
    return results


def _print_summary_te(
    results: dict,
    conn_labels: list[str],
    config: Optional[dict] = None,
) -> None:
    """Print a human-readable summary of all TE statistical test results."""
    alpha = _stats_cfg(config)["alpha"]

    print("\n" + "=" * 70)
    print("STATISTICAL TESTING SUMMARY — dSPTE Transfer Entropy Pipeline")
    print("=" * 70)

    # Normality
    df = results["normality"]
    n_cells = len(df)
    n_insufficient = int(df["p_value"].isna().sum())
    n_tested = n_cells - n_insufficient
    n_non_normal = int((~df["is_normal"]).sum())
    n_normal_tested = n_tested - n_non_normal
    print(f"\n5.1 Normality (Shapiro-Wilk, subject_mean arrays):")
    print(f"    Total cells    : {n_cells}  ({n_tested} tested, {n_insufficient} insufficient n)")
    print(f"    Normal tested  : {n_normal_tested} / {n_tested}  ({100*n_normal_tested/max(n_tested,1):.1f}%)")
    print(f"    Non-normal     : {n_non_normal} ({100*n_non_normal/max(n_tested,1):.1f}%)")
    for grp in ["Expert", "Novice"]:
        sub = df[(df["group"] == grp) & df["p_value"].notna()]
        pct = 100 * sub["is_normal"].mean() if len(sub) > 0 else 0
        print(f"    {grp}: {pct:.1f}% normal (of {len(sub)} tested cells)")

    # H1
    df = results["H1"]
    n_sig = df["significant"].sum()
    n_sig_fw = df["significant_fw"].sum() if "significant_fw" in df else "N/A"
    n_agree = df["strategy_agreement"].sum() if "strategy_agreement" in df else "N/A"
    print(f"\n5.2 H1 — Expert vs. Novice (one-way ANOVA, FDR α={alpha}):")
    print(f"    Strategy       : PRIMARY=subject_mean | SENSITIVITY=first_window")
    print(f"    Total cells    : {len(df)}")
    print(f"    Primary sig    : {n_sig}  (subject_mean)")
    print(f"    Sensitivity sig: {n_sig_fw}  (first_window)")
    print(f"    Strategy agree : {n_agree} / {len(df)}")
    df_primary = df[df["primary_trial"]]
    n_sig_primary = df_primary["significant"].sum()
    print(f"    Tr_1 (primary) sig: {n_sig_primary} / {len(df_primary)}")
    if n_sig > 0:
        sig_df = df[df["significant"]][["connection", "trial",
                                        "mean_Exp", "mean_Nov", "p_fdr",
                                        "p_fdr_fw", "strategy_agreement"]]
        print("    FDR-significant connections (primary):")
        for _, r in sig_df.iterrows():
            agree_str = "✓" if r.get("strategy_agreement", False) else "✗"
            print(f"      {r['connection']:20s}  {r['trial']}  "
                  f"Exp={r['mean_Exp']:.3f}  Nov={r['mean_Nov']:.3f}  "
                  f"p_fdr={r['p_fdr']:.4f}  p_fdr_fw={r.get('p_fdr_fw', float('nan')):.4f}  "
                  f"agree={agree_str}")

    # H0
    df = results["H0"]
    n_sig = df["significant"].sum()
    print(f"\n5.3 H0 — Within-trial stability (one-way ANOVA, FDR α={alpha}):")
    print(f"    Input          : all_windows arrays")
    print(f"    Total cells    : {len(df)}")
    print(f"    FDR-significant: {n_sig}  (significant = TE NOT stable within trial)")
    if n_sig > 0:
        sig_df = df[df["significant"]][["group", "trial", "connection", "p_fdr"]]
        for _, r in sig_df.iterrows():
            print(f"      {r['group']:8s}  {r['trial']}  {r['connection']:20s}  p_fdr={r['p_fdr']:.4f}")

    # HIV
    df = results["HIV"]
    n_sig = df["significant"].sum()
    print(f"\n5.4 HIV — Across-trial stability (one-way ANOVA, FDR α={alpha}):")
    print(f"    Input          : all_windows arrays")
    print(f"    Total cells    : {len(df)}")
    print(f"    FDR-significant: {n_sig}  (significant = TE NOT stable across trials)")
    if n_sig > 0:
        sig_df = df[df["significant"]][["group", "window", "connection", "p_fdr"]]
        for _, r in sig_df.iterrows():
            print(f"      {r['group']:8s}  {r['window']}  {r['connection']:20s}  p_fdr={r['p_fdr']:.4f}")

    print("\n" + "=" * 70 + "\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Stage E Phase 1 — dSPTE statistical tests."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if output CSVs already exist.",
    )
    p.add_argument(
        "--test",
        choices=["h1", "h0", "hiv", "normality", "all"],
        default="all",
        help="Which test to run (default: all).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()

    if args.test == "all":
        run_all_statistics_te(config=config, overwrite=args.overwrite)
    elif args.test == "normality":
        B_Exp_mean, B_Nov_mean, labels = _load_mean_arrays(config)
        df = run_normality_tests_te(B_Exp_mean, B_Nov_mean, labels, config)
        out = _load_te_stats_dir(config) / "normality_test.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}  ({len(df)} rows)")
    elif args.test == "h1":
        df = test_expert_vs_novice_te(config=config, overwrite=args.overwrite)
        print(df[["trial", "connection", "p_fdr", "significant",
                   "p_fdr_fw", "significant_fw", "strategy_agreement"]].to_string())
    elif args.test == "h0":
        df = test_within_trial_stability_te(config=config, overwrite=args.overwrite)
        print(df[["group", "trial", "connection", "n_windows_used",
                   "F_stat", "p_fdr", "significant"]].to_string())
    elif args.test == "hiv":
        df = test_across_trial_stability_te(config=config, overwrite=args.overwrite)
        print(df[["group", "window", "connection", "n_trials_used",
                   "F_stat", "p_fdr", "significant"]].to_string())
