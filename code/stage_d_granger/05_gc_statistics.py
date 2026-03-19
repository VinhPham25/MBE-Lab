"""
Stage D — Step D6: Statistical Tests for Linear GC Pipeline

Implements the four statistical tests from GC_freq_main_TRGC_NovExp.m,
adapted for the ASTaUND dataset and Python.

Section 6 task coverage (LINEAR_GC_PROGRESS.md)
------------------------------------------------
6.1 — Normality testing (Shapiro-Wilk)            run_normality_tests()
6.2 — H1: Expert vs. Novice ANOVA + FDR           test_expert_vs_novice()
6.3 — H0: Within-trial stability ANOVA + FDR      test_within_trial_stability()
6.4 — HIV: Across-trial stability ANOVA + FDR     test_across_trial_stability()

MATLAB equivalents
------------------
  swtest(x)            → scipy.stats.shapiro(x)
  anova1(X)            → scipy.stats.f_oneway(*[X[:,c] for c in range(n)])
  mafdr(p)             → statsmodels.stats.multitest.fdrcorrection(p)
  nanmean(X)           → np.nanmean(X)

NaN handling
------------
B_Exp[trial] and B_Nov[trial] are (n_subs, 20, n_wins) arrays with NaN
padding for subjects missing that trial (R9: Tr_3–5, R4: Tr_5).

Subjects with zero-filled ROI connections (R11 for LM1/SMA connections,
R2 for SMA connections) keep 0.0 values — they are valid observations (true
absent connectivity), not missing data.

The 'zero/NaN filter' from MATLAB H0/HIV (zer_col > 2 → truncate windows)
is translated as: if more than ``h0_max_nan_subjects`` subjects have NaN in
connection[0] of window j, stop at window j-1.

MATLAB H1 alignment
--------------------
MATLAB runs H1 only on Tr_1 (``for k = 1:1``).  This module runs H1 for all
trials; the primary trial is recorded in the output column ``primary_trial``.

MATLAB HIV alignment
---------------------
MATLAB loops ``for j = 1:3`` (first 3 windows only) and includes trials where
n >= 5 (Expert) or n == 6 (Novice).  Since the NaN-padded group arrays always
have 7 Expert rows and 6 Novice rows (subject axis constant), the MATLAB
condition is equivalent to "at least 5/6 subjects are non-NaN for this window".
This module uses a configurable ``hiv_min_subjects`` threshold (default 3).

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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro
from statsmodels.stats.multitest import fdrcorrection

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config
from stage_d_granger.utils_gc import get_connection_labels, get_gc_output_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_stats_config() -> dict:
    """Return the statistics sub-config from config.yaml."""
    cfg = load_config()
    return cfg["stage_d_granger"]["linear_gc"]["statistics"]


def _load_group_arrays(study: str = "expert_novice") -> tuple[dict, dict, list[str]]:
    """
    Load B_Exp and B_Nov from the group .npz files and return connection labels.

    Returns
    -------
    B_Exp : dict  {trial_id: np.ndarray(n_exp_subs, 20, n_wins)}
    B_Nov : dict  {trial_id: np.ndarray(n_nov_subs, 20, n_wins)}
    conn_labels : list[str]  20 directed connection strings
    """
    group_dir = get_gc_output_dir("group")
    B_Exp_data = np.load(group_dir / "B_Exp.npz", allow_pickle=True)
    B_Nov_data = np.load(group_dir / "B_Nov.npz", allow_pickle=True)

    # Build dict {trial_id: array} — skip 'subject_ids' key
    B_Exp = {k: B_Exp_data[k] for k in B_Exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: B_Nov_data[k] for k in B_Nov_data.files if k.startswith("Tr_")}

    # Load summary for connection labels
    summary_path = group_dir / "gc_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        conn_labels = summary.get("connection_labels", get_connection_labels())
    else:
        conn_labels = get_connection_labels()

    logger.info(
        "Loaded B_Exp (%d trials) and B_Nov (%d trials) from %s",
        len(B_Exp), len(B_Nov), group_dir,
    )
    return B_Exp, B_Nov, conn_labels


def _apply_fdr(p_values: np.ndarray, alpha: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction to a 1-D array of p-values.

    NaN entries are excluded from correction and restored as NaN in output.
    Returns (p_fdr, q_values) — both length-n arrays matching input.
    """
    p_fdr = np.full_like(p_values, np.nan)
    q_vals = np.full_like(p_values, np.nan)
    valid_mask = ~np.isnan(p_values)
    if valid_mask.sum() > 0:
        reject, p_corr = fdrcorrection(p_values[valid_mask], alpha=alpha, method=method)
        p_fdr[valid_mask] = p_corr
        # q-values: BH-corrected p-values (mafdr convention)
        q_vals[valid_mask] = p_corr
    return p_fdr, q_vals


# ---------------------------------------------------------------------------
# 6.1 Normality Testing (Shapiro-Wilk)
# ---------------------------------------------------------------------------

def run_normality_tests(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test for each (group, trial, window, connection) cell.

    Mirrors MATLAB:
        for k = 1:nnn        % trials
            for j = 1:nn     % windows
                for i = 1:20 % connections
                    [~, NORM_p.(trial_temp).ExpFLS(i,1,j)] = swtest(
                        B_Exp.(trial_temp).Exp(:,i,j));

    The input vector for each cell is the subject-level GC distribution (n_subjects,).
    Shapiro-Wilk requires n >= 3; cells with fewer valid subjects get W=NaN, p=NaN,
    is_normal=True (conservative: cannot reject normality).

    Parameters
    ----------
    B_Exp, B_Nov : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels  : list[str]  20 connection labels

    Returns
    -------
    pd.DataFrame  columns: [group, trial, window, connection,
                             W_stat, p_value, is_normal, n_subjects]
    """
    cfg = _load_stats_config()
    min_n = cfg["min_subjects_normality"]
    alpha = cfg["alpha"]

    rows = []
    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for trial_id, arr in sorted(B_group.items()):
            # arr: (n_subs, 20, n_wins)
            n_subs, n_conn, n_wins = arr.shape
            for win_idx in range(n_wins):
                win_label = f"Win_{win_idx + 1}"
                for conn_idx in range(n_conn):
                    x = arr[:, conn_idx, win_idx]
                    # Drop NaN subjects
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
                        "window": win_label,
                        "connection": conn_labels[conn_idx],
                        "W_stat": round(float(w_stat), 6) if not np.isnan(w_stat) else np.nan,
                        "p_value": round(float(p_val), 6) if not np.isnan(p_val) else np.nan,
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


# ---------------------------------------------------------------------------
# 6.2 H1 — Expert vs. Novice (Primary Hypothesis)
# ---------------------------------------------------------------------------

def test_expert_vs_novice(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
) -> pd.DataFrame:
    """
    One-way ANOVA comparing Expert vs. Novice GC for each (trial, window, connection).

    Mirrors MATLAB H1 section::

        for k = 1:1     % only Tr_1 in MATLAB; here we run all trials
            nn = min(n_Nov_wins, n_Exp_wins)
            for j = 1:nn        % windows
                for i = 1:20    % connections
                    n = min(n_Exp_subs, n_Nov_subs)
                    X = [B_Exp(1:n,i,j), B_Nov(1:n,i,j)]
                    [p1(i,6,j), tbl, ~] = anova1(X)
                    p1(i,1:4,j) = [mean_Exp, std_Exp, mean_Nov, std_Nov]
                    p1(i,5,j)   = F-statistic
                [FDR, Q] = mafdr(p1(:,6,j))

    Python adaptation: NaN subjects are dropped per cell rather than clipping
    to min(n_Exp, n_Nov) rows.  FDR is applied across the 20 connections
    independently per (trial, window) pair — mirroring the MATLAB per-window
    ``mafdr`` call.

    Parameters
    ----------
    B_Exp, B_Nov : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels  : list[str]  20 connection labels

    Returns
    -------
    pd.DataFrame  columns: [trial, window, connection,
                             mean_Exp, std_Exp, n_Exp,
                             mean_Nov, std_Nov, n_Nov,
                             F_stat, p_raw, p_fdr, q_value,
                             significant, primary_trial]
    """
    cfg = _load_stats_config()
    min_n = cfg["min_subjects_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    primary_trial = cfg["h1_primary_trial"]

    trial_ids = sorted(set(B_Exp.keys()) & set(B_Nov.keys()))
    rows = []

    for trial_id in trial_ids:
        arr_exp = B_Exp[trial_id]   # (n_exp, 20, n_wins)
        arr_nov = B_Nov[trial_id]   # (n_nov, 20, n_wins)
        n_wins = min(arr_exp.shape[2], arr_nov.shape[2])

        for win_idx in range(n_wins):
            win_label = f"Win_{win_idx + 1}"

            # Collect raw stats per connection before FDR
            p_raw_vec = np.full(len(conn_labels), np.nan)
            f_stat_vec = np.full(len(conn_labels), np.nan)
            cell_stats = []

            for conn_idx in range(len(conn_labels)):
                x_exp = arr_exp[:, conn_idx, win_idx]
                x_nov = arr_nov[:, conn_idx, win_idx]
                x_exp = x_exp[~np.isnan(x_exp)]
                x_nov = x_nov[~np.isnan(x_nov)]
                n_exp, n_nov = len(x_exp), len(x_nov)

                mean_exp = float(np.nanmean(x_exp)) if n_exp > 0 else np.nan
                std_exp  = float(np.nanstd(x_exp, ddof=1)) if n_exp > 1 else np.nan
                mean_nov = float(np.nanmean(x_nov)) if n_nov > 0 else np.nan
                std_nov  = float(np.nanstd(x_nov, ddof=1)) if n_nov > 1 else np.nan

                if n_exp >= min_n and n_nov >= min_n:
                    f_stat, p_raw = f_oneway(x_exp, x_nov)
                    p_raw_vec[conn_idx] = float(p_raw)
                    f_stat_vec[conn_idx] = float(f_stat)
                else:
                    f_stat, p_raw = np.nan, np.nan

                cell_stats.append({
                    "trial": trial_id,
                    "window": win_label,
                    "connection": conn_labels[conn_idx],
                    "mean_Exp": round(mean_exp, 6) if not np.isnan(mean_exp) else np.nan,
                    "std_Exp":  round(std_exp, 6)  if not np.isnan(std_exp)  else np.nan,
                    "n_Exp": n_exp,
                    "mean_Nov": round(mean_nov, 6) if not np.isnan(mean_nov) else np.nan,
                    "std_Nov":  round(std_nov, 6)  if not np.isnan(std_nov)  else np.nan,
                    "n_Nov": n_nov,
                    "F_stat": round(float(f_stat), 6) if not np.isnan(f_stat) else np.nan,
                    "p_raw":  round(float(p_raw), 6)  if not np.isnan(p_raw)  else np.nan,
                    "primary_trial": (trial_id == primary_trial),
                })

            # FDR correction across 20 connections for this (trial, window)
            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

            for conn_idx, row in enumerate(cell_stats):
                row["p_fdr"] = (
                    round(float(p_fdr_vec[conn_idx]), 6)
                    if not np.isnan(p_fdr_vec[conn_idx]) else np.nan
                )
                row["q_value"] = (
                    round(float(q_vals_vec[conn_idx]), 6)
                    if not np.isnan(q_vals_vec[conn_idx]) else np.nan
                )
                row["significant"] = (
                    bool(p_fdr_vec[conn_idx] < alpha)
                    if not np.isnan(p_fdr_vec[conn_idx]) else False
                )
                rows.append(row)

    df = pd.DataFrame(rows)
    n_sig = df["significant"].sum()
    logger.info(
        "H1 Expert vs Novice: %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    if n_sig > 0:
        sig_conns = df[df["significant"]]["connection"].value_counts().head(5)
        logger.info("Top significant connections:\n%s", sig_conns.to_string())
    return df


# ---------------------------------------------------------------------------
# 6.3 H0 — Within-Trial Stability
# ---------------------------------------------------------------------------

def test_within_trial_stability(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
) -> pd.DataFrame:
    """
    One-way ANOVA testing whether connectivity is stable across windows within a trial.

    Mirrors MATLAB H0 section::

        for m = 1:2        % Expert, Novice
            for k = 1:nnn  % trials
                % NaN/zero filter: truncate nn if connection[0] has >2 missing subs
                for j = 1:nn
                    zer_col = sum(B(1:n, 1, j) == 0)
                    if zer_col > 2: nn = j-1; break

                for i = 1:20    % connections
                    X = squeeze(B(1:n, i, 1:nn))   % (n_subs × n_valid_wins)
                    [P_inTrial(i, ...), tbl, ~] = anova1(X)
                [FDR, Q] = mafdr(P_inTrial(:, p_col))

    Python: NaN subjects dropped per window; anova1(X) on a matrix tests
    equality of column means → f_oneway(*[X[:,w] for w in range(n_valid_wins)]).
    The NaN filter uses connection[0] NaN count (not zeros) since Python uses
    NaN for missing data.  FDR is applied per (group, trial) across 20 connections.

    Parameters
    ----------
    B_Exp, B_Nov : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels  : list[str]  20 connection labels

    Returns
    -------
    pd.DataFrame  columns: [group, trial, connection,
                             n_windows_used, n_total_windows,
                             F_stat, p_raw, p_fdr, q_value, significant,
                             win_{k}_mean, win_{k}_std  (per valid window)]
    """
    cfg = _load_stats_config()
    min_n = cfg["min_subjects_anova"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_nan = cfg["h0_max_nan_subjects"]
    check_conn = cfg["h0_check_connection_idx"]

    rows = []
    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for trial_id, arr in sorted(B_group.items()):
            n_subs, n_conn, n_wins_total = arr.shape

            # ---- NaN filter: find last valid window (MATLAB zer_col > 2 logic) ----
            n_valid_wins = n_wins_total
            for win_idx in range(n_wins_total):
                col = arr[:, check_conn, win_idx]
                nan_count = int(np.sum(np.isnan(col)))
                if nan_count > max_nan:
                    n_valid_wins = win_idx  # stop before this window
                    break

            if n_valid_wins < min_groups:
                logger.warning(
                    "H0: %s %s — only %d valid window(s) after NaN filter, "
                    "need >= %d to run ANOVA. Skipping.",
                    group_name, trial_id, n_valid_wins, min_groups,
                )

            # ---- Collect per-window descriptive stats for all connections ----
            win_means = {conn_idx: [] for conn_idx in range(n_conn)}
            win_stds  = {conn_idx: [] for conn_idx in range(n_conn)}
            win_ns    = {conn_idx: [] for conn_idx in range(n_conn)}

            for win_idx in range(n_valid_wins):
                for conn_idx in range(n_conn):
                    x = arr[:, conn_idx, win_idx]
                    x = x[~np.isnan(x)]
                    win_means[conn_idx].append(
                        float(np.nanmean(x)) if len(x) > 0 else np.nan
                    )
                    win_stds[conn_idx].append(
                        float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan
                    )
                    win_ns[conn_idx].append(len(x))

            # ---- ANOVA across windows per connection ----
            p_raw_vec = np.full(n_conn, np.nan)
            f_stat_vec = np.full(n_conn, np.nan)

            for conn_idx in range(n_conn):
                if n_valid_wins < min_groups:
                    continue
                groups = []
                for win_idx in range(n_valid_wins):
                    x = arr[:, conn_idx, win_idx]
                    x = x[~np.isnan(x)]
                    if len(x) >= min_n:
                        groups.append(x)
                if len(groups) >= min_groups:
                    f_stat, p_raw = f_oneway(*groups)
                    p_raw_vec[conn_idx] = float(p_raw)
                    f_stat_vec[conn_idx] = float(f_stat)

            # ---- FDR correction across 20 connections for (group, trial) ----
            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

            for conn_idx in range(n_conn):
                row = {
                    "group": group_name,
                    "trial": trial_id,
                    "connection": conn_labels[conn_idx],
                    "n_windows_used": n_valid_wins,
                    "n_total_windows": n_wins_total,
                    "F_stat": (
                        round(f_stat_vec[conn_idx], 6)
                        if not np.isnan(f_stat_vec[conn_idx]) else np.nan
                    ),
                    "p_raw": (
                        round(p_raw_vec[conn_idx], 6)
                        if not np.isnan(p_raw_vec[conn_idx]) else np.nan
                    ),
                    "p_fdr": (
                        round(p_fdr_vec[conn_idx], 6)
                        if not np.isnan(p_fdr_vec[conn_idx]) else np.nan
                    ),
                    "q_value": (
                        round(q_vals_vec[conn_idx], 6)
                        if not np.isnan(q_vals_vec[conn_idx]) else np.nan
                    ),
                    "significant": (
                        bool(p_fdr_vec[conn_idx] < alpha)
                        if not np.isnan(p_fdr_vec[conn_idx]) else False
                    ),
                }
                # Append per-window descriptive stats as dynamic columns
                for win_idx in range(n_valid_wins):
                    win_lbl = f"Win_{win_idx + 1}"
                    row[f"{win_lbl}_mean"] = (
                        round(win_means[conn_idx][win_idx], 6)
                        if not np.isnan(win_means[conn_idx][win_idx]) else np.nan
                    )
                    row[f"{win_lbl}_std"] = (
                        round(win_stds[conn_idx][win_idx], 6)
                        if not np.isnan(win_stds[conn_idx][win_idx]) else np.nan
                    )
                    row[f"{win_lbl}_n"] = win_ns[conn_idx][win_idx]
                rows.append(row)

    df = pd.DataFrame(rows)
    n_sig = df["significant"].sum()
    logger.info(
        "H0 Within-trial stability: %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


# ---------------------------------------------------------------------------
# 6.4 HIV — Across-Trial Stability
# ---------------------------------------------------------------------------

def test_across_trial_stability(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
) -> pd.DataFrame:
    """
    One-way ANOVA testing whether connectivity is stable across trial repetitions.

    Mirrors MATLAB HIV section::

        for m = 1:2        % Expert, Novice
            for j = 1:3    % first 3 windows only
                for i = 1:20   % connections
                    for k = 1:nnn  % trials
                        if n >= 5  % Expert; if n == 6 % Novice
                            X(:,k) = B(1:n, i, j)
                    [P_AcrossTrial_1(i,j), ~, ~] = anova1(X)
                    % store means and stds per trial
                [FDR, Q] = mafdr(P_AcrossTrial(Win)(:, p_col))

    For each (group, window_idx, connection): collect GC value for that window
    from all trials, building a (n_subs × n_trials) matrix X.  Then run ANOVA
    across trials (testing if GC is stable across task repetitions).

    Python adaptation:
    - Only windows 0..(hiv_max_windows-1) are tested, matching MATLAB's j=1:3.
    - Trials without sufficient valid subjects (NaN count > threshold) are
      excluded from the across-trial stack.
    - FDR applied per (group, window) across 20 connections.

    Parameters
    ----------
    B_Exp, B_Nov : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels  : list[str]  20 connection labels

    Returns
    -------
    pd.DataFrame  columns: [group, window, connection,
                             n_trials_used, F_stat, p_raw, p_fdr, q_value,
                             significant,
                             Tr_{k}_mean, Tr_{k}_std  (per included trial)]
    """
    cfg = _load_stats_config()
    min_n = cfg["min_subjects_anova"]
    min_trials = cfg["hiv_min_trials"]
    min_groups = cfg["min_groups_anova"]
    alpha = cfg["alpha"]
    fdr_method = cfg["fdr_method"]
    max_windows = cfg["hiv_max_windows"]

    # Sorted trial IDs present in both groups
    all_trial_ids = sorted(set(B_Exp.keys()) | set(B_Nov.keys()))
    n_conn = len(conn_labels)
    rows = []

    for group_name, B_group in [("Expert", B_Exp), ("Novice", B_Nov)]:
        for win_idx in range(max_windows):
            win_label = f"Win_{win_idx + 1}"

            # FDR will be applied per window across 20 connections
            p_raw_vec   = np.full(n_conn, np.nan)
            f_stat_vec  = np.full(n_conn, np.nan)

            # Per-connection, per-trial descriptive stats for the output
            per_trial_means = {conn_idx: {} for conn_idx in range(n_conn)}
            per_trial_stds  = {conn_idx: {} for conn_idx in range(n_conn)}
            per_conn_n_trials_used = np.zeros(n_conn, dtype=int)

            for conn_idx in range(n_conn):
                trial_groups = []    # list of 1-D arrays, one per valid trial
                included_trials = []

                for trial_id in all_trial_ids:
                    if trial_id not in B_group:
                        continue
                    arr = B_group[trial_id]  # (n_subs, 20, n_wins)
                    # Check this window exists for this trial
                    if win_idx >= arr.shape[2]:
                        continue
                    x = arr[:, conn_idx, win_idx]
                    x_valid = x[~np.isnan(x)]
                    if len(x_valid) >= min_n:
                        trial_groups.append(x_valid)
                        included_trials.append(trial_id)
                        per_trial_means[conn_idx][trial_id] = float(np.mean(x_valid))
                        per_trial_stds[conn_idx][trial_id] = (
                            float(np.std(x_valid, ddof=1)) if len(x_valid) > 1 else np.nan
                        )

                per_conn_n_trials_used[conn_idx] = len(trial_groups)

                if len(trial_groups) >= max(min_groups, min_trials):
                    f_stat, p_raw = f_oneway(*trial_groups)
                    f_stat_vec[conn_idx] = float(f_stat)
                    p_raw_vec[conn_idx] = float(p_raw)

            # FDR across 20 connections for this (group, window)
            p_fdr_vec, q_vals_vec = _apply_fdr(p_raw_vec, alpha, fdr_method)

            for conn_idx in range(n_conn):
                row = {
                    "group": group_name,
                    "window": win_label,
                    "connection": conn_labels[conn_idx],
                    "n_trials_used": int(per_conn_n_trials_used[conn_idx]),
                    "F_stat": (
                        round(f_stat_vec[conn_idx], 6)
                        if not np.isnan(f_stat_vec[conn_idx]) else np.nan
                    ),
                    "p_raw": (
                        round(p_raw_vec[conn_idx], 6)
                        if not np.isnan(p_raw_vec[conn_idx]) else np.nan
                    ),
                    "p_fdr": (
                        round(p_fdr_vec[conn_idx], 6)
                        if not np.isnan(p_fdr_vec[conn_idx]) else np.nan
                    ),
                    "q_value": (
                        round(q_vals_vec[conn_idx], 6)
                        if not np.isnan(q_vals_vec[conn_idx]) else np.nan
                    ),
                    "significant": (
                        bool(p_fdr_vec[conn_idx] < alpha)
                        if not np.isnan(p_fdr_vec[conn_idx]) else False
                    ),
                }
                # Per-trial descriptive stats
                for trial_id in all_trial_ids:
                    t_mean = per_trial_means[conn_idx].get(trial_id, np.nan)
                    t_std  = per_trial_stds[conn_idx].get(trial_id, np.nan)
                    row[f"{trial_id}_mean"] = (
                        round(t_mean, 6) if not np.isnan(t_mean) else np.nan
                    )
                    row[f"{trial_id}_std"] = (
                        round(t_std, 6) if not np.isnan(t_std) else np.nan
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    n_sig = df["significant"].sum()
    logger.info(
        "HIV Across-trial stability: %d cells, %d FDR-significant (alpha=%.2f)",
        len(df), n_sig, alpha,
    )
    return df


# ---------------------------------------------------------------------------
# Main runner — loads group arrays, runs all tests, saves CSVs
# ---------------------------------------------------------------------------

def run_all_statistics(study: str = "expert_novice") -> dict[str, pd.DataFrame]:
    """
    Run all four statistical tests and save results to the statistics/ directory.

    Returns
    -------
    dict with keys: 'normality', 'H1', 'H0', 'HIV' → DataFrames
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    stats_dir = get_gc_output_dir("statistics")
    stats_dir.mkdir(parents=True, exist_ok=True)

    B_Exp, B_Nov, conn_labels = _load_group_arrays(study)

    results = {}

    # ---- 6.1 Normality ----
    logger.info("Running 6.1 Normality tests (Shapiro-Wilk)...")
    df_norm = run_normality_tests(B_Exp, B_Nov, conn_labels)
    out_norm = stats_dir / "normality_test.csv"
    df_norm.to_csv(out_norm, index=False)
    logger.info("Saved %s  (%d rows)", out_norm.name, len(df_norm))
    results["normality"] = df_norm

    # ---- 6.2 H1: Expert vs Novice ----
    logger.info("Running 6.2 H1: Expert vs. Novice ANOVA...")
    df_h1 = test_expert_vs_novice(B_Exp, B_Nov, conn_labels)
    out_h1 = stats_dir / "H1_expert_vs_novice.csv"
    df_h1.to_csv(out_h1, index=False)
    logger.info("Saved %s  (%d rows)", out_h1.name, len(df_h1))
    results["H1"] = df_h1

    # ---- 6.3 H0: Within-trial stability ----
    logger.info("Running 6.3 H0: Within-trial stability ANOVA...")
    df_h0 = test_within_trial_stability(B_Exp, B_Nov, conn_labels)
    out_h0 = stats_dir / "H0_within_trial_stability.csv"
    df_h0.to_csv(out_h0, index=False)
    logger.info("Saved %s  (%d rows)", out_h0.name, len(df_h0))
    results["H0"] = df_h0

    # ---- 6.4 HIV: Across-trial stability ----
    logger.info("Running 6.4 HIV: Across-trial stability ANOVA...")
    df_hiv = test_across_trial_stability(B_Exp, B_Nov, conn_labels)
    out_hiv = stats_dir / "HIV_across_trial_stability.csv"
    df_hiv.to_csv(out_hiv, index=False)
    logger.info("Saved %s  (%d rows)", out_hiv.name, len(df_hiv))
    results["HIV"] = df_hiv

    # ---- Summary printout ----
    _print_summary(results, conn_labels)

    return results


def _print_summary(results: dict, conn_labels: list[str]) -> None:
    """Print a human-readable summary of all statistical test results."""
    alpha = _load_stats_config()["alpha"]

    print("\n" + "=" * 70)
    print("STATISTICAL TESTING SUMMARY — Linear GC Pipeline")
    print("=" * 70)

    # Normality
    df = results["normality"]
    n_cells = len(df)
    n_insufficient = int(df["p_value"].isna().sum())  # cells with n < min_subjects
    n_tested = n_cells - n_insufficient
    n_non_normal = int((~df["is_normal"]).sum())       # is_normal=False only for tested cells
    n_normal_tested = n_tested - n_non_normal
    print(f"\n6.1 Normality (Shapiro-Wilk):")
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
    print(f"\n6.2 H1 — Expert vs. Novice (one-way ANOVA, FDR α={alpha}):")
    print(f"    Total cells        : {len(df)}")
    print(f"    FDR-significant    : {n_sig}")
    # Primary trial breakdown
    df_primary = df[df["primary_trial"]]
    n_sig_primary = df_primary["significant"].sum()
    print(f"    Tr_1 (primary) sig : {n_sig_primary} / {len(df_primary)}")
    if n_sig > 0:
        sig_df = df[df["significant"]][["connection", "trial", "window",
                                        "mean_Exp", "mean_Nov", "p_fdr"]]
        print("    FDR-significant connections:")
        for _, r in sig_df.iterrows():
            print(f"      {r['connection']:20s}  {r['trial']} {r['window']}  "
                  f"Exp={r['mean_Exp']:.3f}  Nov={r['mean_Nov']:.3f}  p_fdr={r['p_fdr']:.4f}")

    # H0
    df = results["H0"]
    n_sig = df["significant"].sum()
    print(f"\n6.3 H0 — Within-trial stability (one-way ANOVA, FDR α={alpha}):")
    print(f"    Total cells     : {len(df)}")
    print(f"    FDR-significant : {n_sig}  (significant = GC NOT stable within trial)")
    if n_sig > 0:
        sig_df = df[df["significant"]][["group", "trial", "connection", "p_fdr"]]
        for _, r in sig_df.iterrows():
            print(f"      {r['group']:8s}  {r['trial']}  {r['connection']:20s}  p_fdr={r['p_fdr']:.4f}")

    # HIV
    df = results["HIV"]
    n_sig = df["significant"].sum()
    print(f"\n6.4 HIV — Across-trial stability (one-way ANOVA, FDR α={alpha}):")
    print(f"    Total cells     : {len(df)}")
    print(f"    FDR-significant : {n_sig}  (significant = GC NOT stable across trials)")
    if n_sig > 0:
        sig_df = df[df["significant"]][["group", "window", "connection", "p_fdr"]]
        for _, r in sig_df.iterrows():
            print(f"      {r['group']:8s}  {r['window']}  {r['connection']:20s}  p_fdr={r['p_fdr']:.4f}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_statistics(study="expert_novice")
