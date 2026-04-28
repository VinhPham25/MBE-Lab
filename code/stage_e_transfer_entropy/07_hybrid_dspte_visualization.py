"""
Stage E — Step E7: Phase 2 Hybrid-dSPTE Visualization (Section 7.3.2 + 7.3.3)

Implements the two diagnostic visualizations required by Section 7.3 of
TRANSFER_ENTROPY_PROGRESS.md:

  7.3.2 — plot_phase1_vs_phase2_comparison()
      Paired bar chart (Phase 1 bivariate vs Phase 2 conditional) for each of
      the 20 directed connections.  Connections are highlighted by change size
      using both a relative threshold (>20%) and an absolute threshold
      (>0.005 nat).  Connections meeting only the relative threshold are
      annotated with † to flag scientifically borderline changes.

  7.3.3 — plot_lag_profile_phase1_vs_phase2()
      Dual-panel plot for a specific subject / trial / window / connection.
      Top panel: Phase 1 (blue) and Phase 2 (orange) lag-TE profiles on the
      same axes.  Bottom panel: difference curve TE_phase2(u) - TE_phase1(u)
      to diagnose shared-noise suppression.  A flat or negative difference at
      low lags with a peak at intermediate lags indicates successful
      conditioning.  The primary diagnostic run generates all 20 connections
      for subject A1, Trial 1, Window 1.

Output paths (phase_2_hybrid_dspte/visualizations/)
----------------------------------------------------
  phase1_vs_phase2_comparison.png
  lag_profile_comparison/sub-{id}_Tr{N}_Win{M}_{src}_to_{tgt}_lag_comparison.png

Style conventions
-----------------
  Phase 1 : cornflower blue  #5B9BD5
  Phase 2 : darkorange       #E87722
  Difference curve: dimgray  #696969
  Highlighted bars: transparent gold overlay + asterisk annotation
  Figure DPI: 300
  Spine style: top / right hidden (same as Phase 1 visualization)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent
_STAGE_E_ROOT = Path(__file__).parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants (Phase 2 palette extends Phase 1 conventions)
# ---------------------------------------------------------------------------

_ROI_NAMES  = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_DIAG_IDX   = [0, 6, 12, 18, 24]
_OFF_DIAG   = [i for i in range(25) if i not in _DIAG_IDX]

_P1_COLOR   = "#5B9BD5"   # cornflower blue — Phase 1 bivariate
_P2_COLOR   = "#E87722"   # dark orange     — Phase 2 conditional
_DIFF_COLOR = "#696969"   # dim gray        — difference curve
_HIGHLIGHT  = "#F4D03F"   # gold            — transparent highlight box

_REL_THRESHOLD = 0.20     # |Δ| / P1 > 20% → candidate for indirect connection
_ABS_THRESHOLD = 0.005    # |Δ| > 0.005 nat → scientifically meaningful magnitude


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _p2_viz_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 2 visualization output directory (create if missing)."""
    from utils_te import get_te_output_dir  # noqa
    d = get_te_output_dir("phase_2_viz", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p2_conn_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 2 per-subject connectivity directory."""
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_2_connectivity", config=config)


def _p1_conn_dir(config: Optional[dict] = None) -> Path:
    """Return Phase 1 per-subject connectivity directory."""
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_1_connectivity", config=config)


def _lag_profile_dir(config: Optional[dict] = None) -> Path:
    """Return lag-profile comparison subdirectory under Phase 2 visualizations."""
    d = _p2_viz_dir(config) / "lag_profile_comparison"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_batch_log(config: Optional[dict] = None) -> dict:
    """Load hybrid_te_batch_log.json from the Phase 2 connectivity directory."""
    conn_dir = _p2_conn_dir(config)
    te_cfg = (config or load_config()).get("stage_e_transfer_entropy", {})
    batch_log_name = te_cfg.get("phase_2", {}).get(
        "batch_log_filename", "hybrid_te_batch_log.json"
    )
    log_path = conn_dir / batch_log_name
    if not log_path.exists():
        raise FileNotFoundError(
            f"Phase 2 batch log not found: {log_path}\n"
            "Run run_batch_hybrid_dspte() first."
        )
    with open(log_path) as f:
        return json.load(f)


def _load_subject_p2_npz(
    subject_id: str,
    config: Optional[dict] = None,
) -> dict:
    """Load Phase 2 .npz for one subject, returning all arrays keyed by trial."""
    conn_dir = _p2_conn_dir(config)
    npz_path = conn_dir / f"sub-{subject_id}_hybrid_dspte_connectivity.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Phase 2 npz not found: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def _load_subject_p1_npz(
    subject_id: str,
    config: Optional[dict] = None,
) -> dict:
    """Load Phase 1 .npz for one subject."""
    conn_dir = _p1_conn_dir(config)
    npz_path = conn_dir / f"sub-{subject_id}_dspte_connectivity.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Phase 1 npz not found: {npz_path}\n"
            "Run the Phase 1 batch pipeline first."
        )
    return np.load(npz_path, allow_pickle=True)


def _subject_mean_per_connection(
    npz,
    connectivity_key_suffix: str = "_connectivity",
) -> np.ndarray:
    """
    Return per-connection mean across all trials and windows for one subject.

    Parameters
    ----------
    npz : NpzFile
        Loaded npz file (Phase 1 or Phase 2).
    connectivity_key_suffix : str
        Suffix identifying connectivity arrays (e.g. '_connectivity').

    Returns
    -------
    mean_vec : np.ndarray, shape (20,)
        Subject-level mean TE per connection.
    """
    trial_ids = list(npz["trial_ids"])
    arrays = []
    for tr_id in trial_ids:
        key = f"{tr_id}{connectivity_key_suffix}"
        if key in npz:
            arr = npz[key]   # shape (20, n_wins)
            arrays.append(arr.mean(axis=1))   # (20,)
    if not arrays:
        return np.full(20, np.nan)
    return np.nanmean(np.stack(arrays, axis=0), axis=0)   # (20,)


def _get_connection_labels(config: Optional[dict] = None) -> list[str]:
    """Return canonical 20 directed connection labels."""
    from utils_te import get_connection_labels  # noqa
    return get_connection_labels()


def _get_subject_group(subject_id: str, config: Optional[dict] = None) -> str:
    """Return subject group label ('Expert' or 'Novice')."""
    from utils_te import get_subject_group  # noqa
    return get_subject_group(subject_id, config=config)


# ---------------------------------------------------------------------------
# Section 7.3.2 — Phase 1 vs Phase 2 paired bar chart
# ---------------------------------------------------------------------------

def plot_phase1_vs_phase2_comparison(
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Paired bar chart comparing Phase 1 (bivariate dSPTE) vs Phase 2
    (conditional hybrid-dSPTE) for all 20 directed connections.

    **Highlighting logic (PRD 7.3.2):**

    A connection is a candidate for an indirect/spurious Phase 1 connection if
    it meets BOTH thresholds simultaneously:
      - Relative: ``|P1 − P2| / P1 > 0.20``  (>20% relative change)
      - Absolute: ``|P1 − P2| > 0.005``       (>0.005 nat absolute change)

    Connections meeting only the relative threshold (large % change on a very
    small TE value) are annotated with ``†`` to flag the change as potentially
    not scientifically meaningful.

    The mean is computed as: per-subject mean across all trials and windows →
    group mean (Expert + Novice together, unweighted average of subject means).

    Parameters
    ----------
    save   : bool  — save PNG to phase_2_viz directory
    config : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    if config is None:
        config = load_config()

    conn_labels = _get_connection_labels(config)
    n_conn = len(conn_labels)

    # ------------------------------------------------------------------
    # Gather per-subject mean vectors from Phase 1 and Phase 2 npz files
    # ------------------------------------------------------------------
    conn_dir_p1 = _p1_conn_dir(config)
    conn_dir_p2 = _p2_conn_dir(config)

    p1_subject_means: list[np.ndarray] = []
    p2_subject_means: list[np.ndarray] = []
    subjects_used: list[str] = []

    p1_npz_files = sorted(conn_dir_p1.glob("sub-*_dspte_connectivity.npz"))
    if not p1_npz_files:
        raise FileNotFoundError(
            f"No Phase 1 npz files found in {conn_dir_p1}. "
            "Run the Phase 1 batch pipeline first."
        )

    for p1_path in p1_npz_files:
        subject_id = p1_path.stem.replace("sub-", "").replace("_dspte_connectivity", "")
        p2_path = conn_dir_p2 / f"sub-{subject_id}_hybrid_dspte_connectivity.npz"
        if not p2_path.exists():
            logger.warning(
                "plot_phase1_vs_phase2_comparison: no Phase 2 file for %s — skipping.",
                subject_id,
            )
            continue

        p1_npz = np.load(p1_path, allow_pickle=True)
        p2_npz = np.load(p2_path, allow_pickle=True)

        p1_mean = _subject_mean_per_connection(p1_npz, "_connectivity")
        p2_mean = _subject_mean_per_connection(p2_npz, "_connectivity")

        p1_subject_means.append(p1_mean)
        p2_subject_means.append(p2_mean)
        subjects_used.append(subject_id)

    if not subjects_used:
        raise RuntimeError("No matched Phase 1 + Phase 2 npz files found.")

    n_subjects = len(subjects_used)
    P1 = np.stack(p1_subject_means, axis=0)   # (n_sub, 20)
    P2 = np.stack(p2_subject_means, axis=0)   # (n_sub, 20)

    # Group means and SEM (across subjects)
    p1_mean  = np.nanmean(P1, axis=0)    # (20,)
    p2_mean  = np.nanmean(P2, axis=0)
    p1_sem   = np.nanstd(P1, axis=0, ddof=1) / max(np.sqrt(n_subjects), 1)
    p2_sem   = np.nanstd(P2, axis=0, ddof=1) / max(np.sqrt(n_subjects), 1)

    # ------------------------------------------------------------------
    # Classify connections by threshold
    # ------------------------------------------------------------------
    abs_change  = np.abs(p1_mean - p2_mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_change = np.where(p1_mean > 0, abs_change / p1_mean, 0.0)

    both_thresholds = (rel_change > _REL_THRESHOLD) & (abs_change > _ABS_THRESHOLD)
    rel_only        = (rel_change > _REL_THRESHOLD) & ~both_thresholds

    n_highlighted   = int(both_thresholds.sum())
    n_dagger        = int(rel_only.sum())

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    x      = np.arange(n_conn)
    bar_w  = 0.38
    y_all  = np.concatenate([p1_mean + p1_sem, p2_mean + p2_sem])
    y_max  = float(np.nanmax(y_all)) if len(y_all) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(16, 6))

    bars_p1 = ax.bar(
        x - bar_w / 2, p1_mean, bar_w,
        color=_P1_COLOR, alpha=0.88,
        label="Phase 1 — bivariate dSPTE",
        yerr=p1_sem, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#3A6B9A"},
    )
    bars_p2 = ax.bar(
        x + bar_w / 2, p2_mean, bar_w,
        color=_P2_COLOR, alpha=0.88,
        label="Phase 2 — conditional hybrid-dSPTE",
        yerr=p2_sem, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#A0540F"},
    )

    # Highlight boxes + annotations
    for i in range(n_conn):
        top = max(p1_mean[i] + p1_sem[i], p2_mean[i] + p2_sem[i])
        ann_y = top + 0.05 * y_max

        if both_thresholds[i]:
            # Gold highlight background spanning both bars
            ax.add_patch(plt.Rectangle(
                (x[i] - bar_w - 0.05, 0),
                bar_w * 2 + 0.10,
                top + 0.10 * y_max,
                facecolor=_HIGHLIGHT, alpha=0.25, zorder=0,
            ))
            direction = "↓" if p2_mean[i] < p1_mean[i] else "↑"
            ax.text(
                x[i], ann_y,
                f"*{direction}\n{rel_change[i]:.0%}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#8B6914", zorder=5,
            )
        elif rel_only[i]:
            ax.text(
                x[i], ann_y,
                "†",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color="#888888", zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("dSPTE (nats)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.6, n_conn - 0.4)

    # Legend
    patch_highlight = mpatches.Patch(
        facecolor=_HIGHLIGHT, alpha=0.5,
        label=f"* = both thresholds (n={n_highlighted}): |Δ|>20% AND |Δ|>0.005 nat",
    )
    patch_dagger = mpatches.Patch(
        facecolor="#CCCCCC", alpha=0.8,
        label=f"† = relative only (n={n_dagger}): |Δ|>20% but |Δ|≤0.005 nat",
    )
    ax.legend(
        handles=[bars_p1, bars_p2, patch_highlight, patch_dagger],
        fontsize=9, framealpha=0.9, loc="upper right",
    )

    ax.set_title(
        f"Phase 1 (bivariate) vs Phase 2 (conditional) dSPTE — All Connections\n"
        f"Group mean ± SEM across {n_subjects} subjects  "
        f"(* both thresholds: rel>20% AND abs>0.005;  † relative only)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    logger.info(
        "plot_phase1_vs_phase2_comparison: n_subjects=%d  "
        "highlighted(both)=%d  annotated(rel_only)=%d",
        n_subjects, n_highlighted, n_dagger,
    )

    if save:
        out = _p2_viz_dir(config) / "phase1_vs_phase2_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out)

    return fig


# ---------------------------------------------------------------------------
# Section 7.3.3 — Lag profile comparison (Phase 1 vs Phase 2)
# ---------------------------------------------------------------------------

def plot_lag_profile_phase1_vs_phase2(
    subject_id: str,
    trial: str,
    window: int,
    src_roi: str,
    tgt_roi: str,
    sfreq: Optional[float] = None,
    save: bool = True,
    ax_top: Optional[plt.Axes] = None,
    ax_bot: Optional[plt.Axes] = None,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Dual-panel lag-TE profile comparison for a single (subject, trial, window,
    src→tgt) tuple.

    **Panel layout:**

    Top panel — dSPTE vs lag
        Phase 1 (blue) and Phase 2 (orange) profiles on the same axes.
        Solid filled area under each curve.  Phase 2 optimal delay is marked
        with a vertical dashed orange line.

    Bottom panel — difference curve
        ``TE_phase2(u) − TE_phase1(u)`` as a gray line with zero-reference.
        A flat or negative difference at low lags with a positive peak at
        intermediate lags indicates successful shared-noise suppression by
        multivariate conditioning.

    Parameters
    ----------
    subject_id : str   e.g. 'A1'
    trial      : str   e.g. 'Tr_1'
    window     : int   0-indexed window within the trial
    src_roi    : str   source ROI name e.g. 'CPFC'
    tgt_roi    : str   target ROI name e.g. 'LM1'
    sfreq      : float or None — sampling frequency in Hz.  If None, read from
                 ``stage_e_transfer_entropy.signal.sfreq`` in config.yaml.
    save       : bool
    ax_top, ax_bot : plt.Axes, optional  — draw into existing axes (grid use)
    config     : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    if config is None:
        config = load_config()

    # Read sfreq from config; caller override wins if explicitly supplied.
    if sfreq is None:
        sfreq = float(
            config.get("stage_e_transfer_entropy", {})
            .get("signal", {})
            .get("sfreq", 25.0)
        )

    conn_labels = _get_connection_labels(config)

    # ------------------------------------------------------------------
    # Resolve connection index in the canonical 20-element order
    # ------------------------------------------------------------------
    conn_label = f"{src_roi}→{tgt_roi}"
    if conn_label not in conn_labels:
        raise ValueError(
            f"Connection '{conn_label}' not in canonical label list. "
            f"Available: {conn_labels}"
        )
    conn_idx = conn_labels.index(conn_label)

    # ------------------------------------------------------------------
    # Load Phase 1 lag profiles
    # ------------------------------------------------------------------
    p1_npz  = _load_subject_p1_npz(subject_id, config)
    p1_key  = f"{trial}_te_profiles"
    if p1_key not in p1_npz:
        raise KeyError(
            f"Key '{p1_key}' not found in Phase 1 npz for {subject_id}. "
            f"Available keys: {list(p1_npz.keys())}"
        )
    p1_profiles = p1_npz[p1_key]   # shape (20, u_max_p1, n_wins)
    n_wins_p1   = p1_profiles.shape[2]
    if window >= n_wins_p1:
        raise IndexError(
            f"Window index {window} out of range for {subject_id} {trial} "
            f"Phase 1 (n_wins={n_wins_p1})."
        )
    p1_profile = p1_profiles[conn_idx, :, window]   # (u_max_p1,)

    # ------------------------------------------------------------------
    # Load Phase 2 lag profiles
    # ------------------------------------------------------------------
    p2_npz  = _load_subject_p2_npz(subject_id, config)
    p2_key  = f"{trial}_conditional_te_profiles"
    if p2_key not in p2_npz:
        raise KeyError(
            f"Key '{p2_key}' not found in Phase 2 npz for {subject_id}. "
            f"Available keys: {list(p2_npz.keys())}"
        )
    p2_profiles = p2_npz[p2_key]   # shape (20, u_max_p2, n_wins)
    n_wins_p2   = p2_profiles.shape[2]
    if window >= n_wins_p2:
        raise IndexError(
            f"Window index {window} out of range for {subject_id} {trial} "
            f"Phase 2 (n_wins={n_wins_p2})."
        )
    p2_profile = p2_profiles[conn_idx, :, window]   # (u_max_p2,)

    # ------------------------------------------------------------------
    # Reconcile mismatched u_max between Phase 1 and Phase 2 files.
    # This occurs when the pipeline was re-run with a different u_max in
    # config.yaml before both phases were regenerated.  Trim both profiles
    # to the shorter length and emit a warning so the user is aware.
    # ------------------------------------------------------------------
    u_max_p1 = len(p1_profile)
    u_max_p2 = len(p2_profile)
    u_max    = min(u_max_p1, u_max_p2)
    if u_max_p1 != u_max_p2:
        logger.warning(
            "plot_lag_profile_phase1_vs_phase2 [%s %s Win%d %s→%s]: "
            "Phase 1 u_max=%d ≠ Phase 2 u_max=%d. "
            "Trimming both profiles to the shorter length (%d). "
            "Re-run both batch pipelines with the same u_max to eliminate this mismatch.",
            subject_id, trial, window + 1, src_roi, tgt_roi,
            u_max_p1, u_max_p2, u_max,
        )
    p1_profile = p1_profile[:u_max]
    p2_profile = p2_profile[:u_max]
    lags = np.arange(1, u_max + 1)

    # Phase 2 optimal delay: argmax of the (trimmed) conditional TE profile
    p2_delay_key = f"{trial}_delays"
    if p2_delay_key in p2_npz:
        p2_optimal_delay = int(p2_npz[p2_delay_key][conn_idx, window])
        # Clamp to the effective u_max after trimming
        p2_optimal_delay = min(p2_optimal_delay, u_max)
    else:
        p2_optimal_delay = int(np.argmax(p2_profile)) + 1   # 1-indexed fallback

    diff = p2_profile - p1_profile

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    standalone = (ax_top is None) and (ax_bot is None)
    if standalone:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 6.5),
            gridspec_kw={"height_ratios": [3, 1.5]},
            sharex=True,
        )
    else:
        fig = ax_top.get_figure()

    # ---- Top panel: overlaid profiles ----
    ax_top.plot(lags, p1_profile, color=_P1_COLOR, lw=2.0, alpha=0.9,
                label="Phase 1 (bivariate dSPTE)")
    ax_top.fill_between(lags, 0, p1_profile, color=_P1_COLOR, alpha=0.10)

    ax_top.plot(lags, p2_profile, color=_P2_COLOR, lw=2.0, alpha=0.9,
                label="Phase 2 (conditional hybrid-dSPTE)")
    ax_top.fill_between(lags, 0, p2_profile, color=_P2_COLOR, alpha=0.10)

    if 1 <= p2_optimal_delay <= u_max:
        delay_ms = p2_optimal_delay * (1000.0 / sfreq)
        ax_top.axvline(
            x=p2_optimal_delay, color=_P2_COLOR,
            lw=1.8, linestyle="--", alpha=0.75,
            label=f"Phase 2 δ̂ = {p2_optimal_delay} smp ({delay_ms:.0f} ms)",
        )

    ax_top.set_ylabel("dSPTE (nats)", fontsize=10)
    ax_top.set_ylim(bottom=0)
    ax_top.set_title(
        f"Lag Profile: {src_roi}→{tgt_roi}  —  sub-{subject_id}  {trial}  Win {window + 1}",
        fontsize=11, fontweight="bold", pad=6,
    )
    ax_top.legend(fontsize=9, framealpha=0.9)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.yaxis.grid(True, alpha=0.3, linewidth=0.7)
    ax_top.set_axisbelow(True)

    # ---- Bottom panel: difference curve ----
    ax_bot.plot(lags, diff, color=_DIFF_COLOR, lw=1.8, alpha=0.85,
                label="Phase 2 − Phase 1")
    ax_bot.axhline(0, color="black", lw=0.9, linestyle="-", alpha=0.5)
    ax_bot.fill_between(lags, 0, diff,
                        where=(diff > 0), color=_P2_COLOR, alpha=0.15,
                        label="Phase 2 > Phase 1 (positive suppression)")
    ax_bot.fill_between(lags, 0, diff,
                        where=(diff < 0), color=_P1_COLOR, alpha=0.15,
                        label="Phase 2 < Phase 1 (over-suppression)")

    if 1 <= p2_optimal_delay <= u_max:
        ax_bot.axvline(x=p2_optimal_delay, color=_P2_COLOR,
                       lw=1.8, linestyle="--", alpha=0.75)

    ax_bot.set_xlabel("Lag u (samples)", fontsize=10)
    ax_bot.set_ylabel("ΔdSPTE (nats)", fontsize=10)
    ax_bot.legend(fontsize=8, framealpha=0.9)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.yaxis.grid(True, alpha=0.3, linewidth=0.7)
    ax_bot.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        if save:
            src_s = src_roi.replace(" ", "_")
            tgt_s = tgt_roi.replace(" ", "_")
            tr_num  = trial.replace("Tr_", "") if trial else "x"
            win_num = window + 1
            fname = (
                f"sub-{subject_id}_Tr{tr_num}_Win{win_num}"
                f"_{src_s}_to_{tgt_s}_lag_comparison.png"
            )
            out = _lag_profile_dir(config) / fname
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("Saved %s", out)

    return fig


# ---------------------------------------------------------------------------
# Batch runner for 7.3.3 — all 20 connections for a given subject/trial/window
# ---------------------------------------------------------------------------

def plot_all_lag_profiles_phase1_vs_phase2(
    subject_id: str = "A1",
    trial: str = "Tr_1",
    window: int = 0,
    save: bool = True,
    config: Optional[dict] = None,
) -> list[plt.Figure]:
    """
    Generate lag profile comparison plots for all 20 connections for a given
    subject / trial / window combination.

    This is the primary diagnostic run specified in PRD 7.3.3:
    subject A1, Trial 1, Window 1 (0-indexed: window=0).

    ``sfreq`` and ``u_max`` are read from ``stage_e_transfer_entropy.signal``
    in config.yaml — no hardcoded values.

    Parameters
    ----------
    subject_id : str   default 'A1'
    trial      : str   default 'Tr_1'
    window     : int   0-indexed window index (default 0 → Window 1)
    save       : bool  save each PNG to lag_profile_comparison/
    config     : dict, optional

    Returns
    -------
    figs : list of plt.Figure  — one per connection (20 total)
    """
    if config is None:
        config = load_config()

    conn_labels = _get_connection_labels(config)
    figs: list[plt.Figure] = []
    n_ok, n_err = 0, 0

    for lbl in conn_labels:
        src_roi, tgt_roi = lbl.split("→")
        try:
            fig = plot_lag_profile_phase1_vs_phase2(
                subject_id=subject_id,
                trial=trial,
                window=window,
                src_roi=src_roi,
                tgt_roi=tgt_roi,
                sfreq=None,   # always derive from config
                save=save,
                config=config,
            )
            figs.append(fig)
            plt.close(fig)
            n_ok += 1
        except Exception as exc:
            logger.warning(
                "plot_all_lag_profiles_phase1_vs_phase2: failed for %s → %s: %s",
                src_roi, tgt_roi, exc,
            )
            n_err += 1

    logger.info(
        "plot_all_lag_profiles_phase1_vs_phase2 [sub-%s %s Win%d]: "
        "%d/%d saved  %d errors",
        subject_id, trial, window + 1, n_ok, len(conn_labels), n_err,
    )
    return figs


# ---------------------------------------------------------------------------
# Phase 2 group array helpers (shared by the two new group plots)
# ---------------------------------------------------------------------------

def _load_p2_group_mean_arrays(
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """
    Load Phase 2 subject-mean group arrays built by 08_hybrid_dspte_statistics.py.

    Returns
    -------
    B_Exp_mean : (n_exp, 20, n_trials)
    B_Nov_mean : (n_nov, 20, n_trials)
    conn_labels : list[str]
    expert_ids  : list[str]
    novice_ids  : list[str]
    """
    from utils_te import get_te_output_dir, get_connection_labels  # noqa
    grp_dir = get_te_output_dir("phase_2_group", config=config)

    exp_path = grp_dir / "B_Exp_hybrid_mean.npz"
    nov_path = grp_dir / "B_Nov_hybrid_mean.npz"
    if not exp_path.exists() or not nov_path.exists():
        raise FileNotFoundError(
            f"Phase 2 group arrays not found in {grp_dir}.\n"
            "Run run_all_statistics_p2() from 08_hybrid_dspte_statistics.py first."
        )

    exp_npz = np.load(exp_path, allow_pickle=True)
    nov_npz = np.load(nov_path, allow_pickle=True)
    return (
        exp_npz["data"],
        nov_npz["data"],
        get_connection_labels(),
        list(exp_npz["subject_ids"]),
        list(nov_npz["subject_ids"]),
    )


def _p2_group_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mean_20, sem_20, n) across subject axis, ignoring NaN."""
    mean = np.nanmean(arr, axis=0)
    n    = int(np.sum(~np.isnan(arr[:, 0])))
    sem  = np.nanstd(arr, axis=0, ddof=1) / max(np.sqrt(n), 1)
    return mean, sem, n


def _vec20_to_mat5_p2(vec: np.ndarray) -> np.ndarray:
    """Reconstruct 5×5 directed-connectivity matrix from 20-element off-diagonal vector."""
    _off_diag = [i for i in range(25) if i not in [0, 6, 12, 18, 24]]
    flat = np.zeros(25)
    flat[_off_diag] = vec
    return flat.reshape(5, 5)


# ---------------------------------------------------------------------------
# Group connectivity heatmap (Expert | Novice | Expert−Novice) — Phase 2
# ---------------------------------------------------------------------------

def plot_group_connectivity_matrix_p2(
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Side-by-side 5×5 heatmaps of group-mean hybrid-dSPTE (Expert | Novice | Difference).

    Mirrors Phase 1's ``plot_group_connectivity_matrix_te_averaged()`` exactly but
    loads Phase 2 subject-mean arrays (``B_Exp_hybrid_mean.npz`` /
    ``B_Nov_hybrid_mean.npz``).

    Layout
    ------
    Three panels:
      Left   — Expert mean conditional dSPTE (viridis)
      Centre — Novice mean conditional dSPTE (viridis, same colour scale)
      Right  — Expert − Novice difference (RdBu_r, centred at 0)

    Each cell is annotated with the numeric value.  Diagonal is masked (grey '—').

    Saves to ``phase_2_hybrid_dspte/visualizations/group_hybrid_te_connectivity_matrix.png``.

    Parameters
    ----------
    save   : bool
    config : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    if config is None:
        config = load_config()

    _EXPERT_COLOR = "#1F618D"
    _NOVICE_COLOR = "#C0392B"
    _CMAP_TE  = "viridis"

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_p2_group_mean_arrays(config)

    # Average over subject AND trial axes → (20,)
    exp_mean = np.nanmean(B_Exp_mean, axis=(0, 2))
    nov_mean = np.nanmean(B_Nov_mean, axis=(0, 2))
    n_exp = int(np.sum(~np.isnan(B_Exp_mean[:, 0, 0])))
    n_nov = int(np.sum(~np.isnan(B_Nov_mean[:, 0, 0])))

    exp_mat  = _vec20_to_mat5_p2(exp_mean)
    nov_mat  = _vec20_to_mat5_p2(nov_mean)
    diff_mat = exp_mat - nov_mat

    diag_mask = np.eye(5, dtype=bool)
    exp_d  = np.where(diag_mask, np.nan, exp_mat)
    nov_d  = np.where(diag_mask, np.nan, nov_mat)
    diff_d = np.where(diag_mask, np.nan, diff_mat)

    vmax_abs  = max(float(np.nanmax([np.nanmax(np.abs(exp_d)), np.nanmax(np.abs(nov_d))])), 1e-6)
    diff_abs  = max(float(np.nanmax(np.abs(diff_d))), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    panels = [
        (axes[0], exp_d,  0,         vmax_abs, _CMAP_TE,  f"Expert (n={n_exp})",  _EXPERT_COLOR, "cdSPTE"),
        (axes[1], nov_d,  0,         vmax_abs, _CMAP_TE,  f"Novice (n={n_nov})",  _NOVICE_COLOR, "cdSPTE"),
        (axes[2], diff_d, -diff_abs, diff_abs, "RdBu_r",  "Expert − Novice",      "#555555",     "Δ cdSPTE"),
    ]

    for ax, mat, vmin, vmax, cmap, title, title_color, cbar_lbl in panels:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        for r in range(5):
            for c in range(5):
                if r == c:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor="#CCCCCC", edgecolor="none", zorder=2,
                    ))
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=8, color="#888888", zorder=3)
                else:
                    val = mat[r, c]
                    if not np.isnan(val):
                        if cmap == "RdBu_r":
                            txt_color = "white" if abs(val) > 0.55 * diff_abs else "black"
                        else:
                            txt_color = "white" if val > 0.65 * vmax else "black"
                        ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                                fontsize=7.5, color=txt_color, zorder=3, fontweight="bold")

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(_ROI_NAMES, fontsize=9)
        ax.set_yticklabels(_ROI_NAMES, fontsize=9)
        ax.set_xlabel("Target ROI", fontsize=9)
        ax.set_ylabel("Source ROI", fontsize=9)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2, zorder=4)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=6)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cb.set_label(cbar_lbl, fontsize=8)

    fig.suptitle(
        "Group-Mean Conditional dSPTE Connectivity Matrix — Phase 2 Hybrid  [All-Trial Average]\n"
        "(row = Source, col = Target;  diagonal masked;  subject_mean strategy)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    if save:
        out = _p2_viz_dir(config) / "group_hybrid_te_connectivity_matrix.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ---------------------------------------------------------------------------
# Expert vs Novice bar chart — Phase 2
# ---------------------------------------------------------------------------

def plot_group_te_comparison_p2(
    h1_df=None,
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Expert vs Novice bar chart of mean conditional dSPTE per connection (Phase 2).

    Mirrors Phase 1's ``plot_group_te_comparison_averaged()`` but loads Phase 2
    group arrays.  Three visual layers are stacked for comparison:

    1. **Paired bars** — Expert (steel blue) and Novice (brick red), height =
       subject-mean TE averaged across all trials, error bars = SEM across subjects.

    2. **FDR significance markers** — ``*`` for connections that are FDR-significant
       under the primary (subject_mean) strategy from the Phase 2 H1 CSV.  ``†`` for
       connections significant only in the sensitivity (first_window) strategy.
       Loaded from ``phase_2_hybrid_dspte/statistics/H1_expert_vs_novice.csv``
       automatically if ``h1_df=None``.

    3. **Phase 1 reference dots** — small dots above each group's bar showing the
       corresponding Phase 1 bivariate dSPTE value, so the viewer can see how much
       the conditioning suppressed each connection.

    Saves to ``phase_2_hybrid_dspte/visualizations/group_hybrid_te_comparison.png``.

    Parameters
    ----------
    h1_df  : pd.DataFrame or None  — Phase 2 H1 CSV; auto-loaded if None
    save   : bool
    config : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    import pandas as pd
    from matplotlib.lines import Line2D

    if config is None:
        config = load_config()

    _EXPERT_COLOR  = "#1F618D"
    _NOVICE_COLOR  = "#C0392B"
    _P1_DOT_EXPERT = "#A9CCE3"   # pale blue — Phase 1 expert reference dot
    _P1_DOT_NOVICE = "#F1948A"   # pale red  — Phase 1 novice reference dot

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_p2_group_mean_arrays(config)

    # Average over trial axis for each subject → (n_subs, 20)
    exp_subj = np.nanmean(B_Exp_mean, axis=2)
    nov_subj = np.nanmean(B_Nov_mean, axis=2)

    exp_mean, exp_sem, n_exp = _p2_group_stats(exp_subj)
    nov_mean, nov_sem, n_nov = _p2_group_stats(nov_subj)

    # ------------------------------------------------------------------
    # Phase 1 reference means (from Phase 1 group arrays, if available)
    # ------------------------------------------------------------------
    p1_exp_mean = np.full(20, np.nan)
    p1_nov_mean = np.full(20, np.nan)
    try:
        from utils_te import get_te_output_dir  # noqa
        grp_dir_p1 = get_te_output_dir("phase_1_group", config=config, create=False)
        p1_exp_npz = np.load(grp_dir_p1 / "B_Exp_dspte_mean.npz", allow_pickle=True)
        p1_nov_npz = np.load(grp_dir_p1 / "B_Nov_dspte_mean.npz", allow_pickle=True)
        p1_exp_mean = np.nanmean(np.nanmean(p1_exp_npz["data"], axis=2), axis=0)
        p1_nov_mean = np.nanmean(np.nanmean(p1_nov_npz["data"], axis=2), axis=0)
        logger.info("Loaded Phase 1 reference means for comparison dots.")
    except Exception as exc:
        logger.warning("Could not load Phase 1 group arrays for reference dots: %s", exc)

    # ------------------------------------------------------------------
    # Load Phase 2 H1 CSV for FDR significance markers
    # ------------------------------------------------------------------
    if h1_df is None:
        from utils_te import get_te_output_dir  # noqa
        stats_dir = get_te_output_dir("phase_2_statistics", config=config, create=False)
        h1_path = stats_dir / "H1_expert_vs_novice.csv"
        if h1_path.exists():
            h1_df = pd.read_csv(h1_path)

    sig_primary: set = set()
    sig_fw_only: set = set()
    if h1_df is not None:
        sig_primary = set(h1_df.loc[h1_df["significant"] == True, "connection"].tolist())
        if "significant_fw" in h1_df.columns:
            sig_fw = set(h1_df.loc[h1_df["significant_fw"] == True, "connection"].tolist())
            sig_fw_only = sig_fw - sig_primary

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_conn = len(conn_labels)
    x      = np.arange(n_conn)
    bar_w  = 0.38

    fig, ax = plt.subplots(figsize=(15, 5.5))

    ax.bar(x - bar_w / 2, exp_mean, bar_w,
           color=_EXPERT_COLOR, alpha=0.88, label=f"Expert (n={n_exp})  Phase 2",
           yerr=exp_sem, capsize=3, error_kw={"elinewidth": 1.2})
    ax.bar(x + bar_w / 2, nov_mean, bar_w,
           color=_NOVICE_COLOR, alpha=0.88, label=f"Novice (n={n_nov})  Phase 2",
           yerr=nov_sem, capsize=3, error_kw={"elinewidth": 1.2})

    # Phase 1 reference dots
    p1_has_data = not (np.all(np.isnan(p1_exp_mean)) and np.all(np.isnan(p1_nov_mean)))
    if p1_has_data:
        ax.scatter(x - bar_w / 2, p1_exp_mean,
                   marker="o", s=22, color=_P1_DOT_EXPERT, zorder=5,
                   label="Expert Phase 1 (reference)", edgecolors="#5B9BD5", linewidths=0.7)
        ax.scatter(x + bar_w / 2, p1_nov_mean,
                   marker="o", s=22, color=_P1_DOT_NOVICE, zorder=5,
                   label="Novice Phase 1 (reference)", edgecolors="#C0392B", linewidths=0.7)

    # FDR significance annotations
    y_max_all = float(np.nanmax(
        np.concatenate([exp_mean + exp_sem, nov_mean + nov_sem])
    )) if n_conn > 0 else 0.05

    for i, lbl in enumerate(conn_labels):
        y_top = max(exp_mean[i] + exp_sem[i], nov_mean[i] + nov_sem[i]) + 0.04 * y_max_all
        if lbl in sig_primary:
            ax.text(i, y_top, "*", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="#E74C3C")
        elif lbl in sig_fw_only:
            ax.text(i, y_top, "†", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#E67E22")

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Conditional dSPTE (nats)", fontsize=11)
    ax.set_title(
        "Expert vs. Novice — Phase 2 Conditional dSPTE  [All-Trial Average, subject_mean strategy]\n"
        "Dots = Phase 1 bivariate reference  |  * FDR-sig primary  |  † FDR-sig sensitivity  |  BH α=0.05",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        out = _p2_viz_dir(config) / "group_hybrid_te_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ---------------------------------------------------------------------------
# Main runner — generate all Section 7.3.2 + 7.3.3 visualizations
# ---------------------------------------------------------------------------

def run_all_phase2_visualizations(
    config: Optional[dict] = None,
    p1_vs_p2_subject_id: str = "A1",
    p1_vs_p2_trial: str = "Tr_1",
    p1_vs_p2_window: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Generate all Section 7.3 Phase 2 visualizations.

    Calls in order:
    1. ``plot_phase1_vs_phase2_comparison()``    → 7.3.2 group bar chart
    2. ``plot_all_lag_profiles_phase1_vs_phase2()``  → 7.3.3 all 20 connections
       for the specified subject / trial / window (default: A1, Tr_1, Win 1)

    Parameters
    ----------
    config            : dict, optional
    p1_vs_p2_subject_id : str  — subject for 7.3.3 diagnostic run
    p1_vs_p2_trial    : str    — trial for 7.3.3 (e.g. 'Tr_1')
    p1_vs_p2_window   : int    — 0-indexed window for 7.3.3 (0 = Win 1)
    overwrite         : bool   — if False, skip existing output files
    """
    if config is None:
        config = load_config()

    viz_dir = _p2_viz_dir(config)
    lag_dir = _lag_profile_dir(config)

    # 7.3.1 — group connectivity matrix (Expert | Novice | Difference)
    matrix_path = viz_dir / "group_hybrid_te_connectivity_matrix.png"
    if overwrite or not matrix_path.exists():
        logger.info("Generating 7.3.1 group matrix: Expert | Novice | Expert−Novice …")
        try:
            fig = plot_group_connectivity_matrix_p2(save=True, config=config)
            plt.close(fig)
            logger.info("Group matrix saved: %s", matrix_path)
        except Exception as exc:
            logger.error("Group matrix failed: %s", exc)
    else:
        logger.info("Group matrix already exists (overwrite=False): %s", matrix_path)

    # 7.3.1 — Expert vs Novice bar chart
    bar_path = viz_dir / "group_hybrid_te_comparison.png"
    if overwrite or not bar_path.exists():
        logger.info("Generating 7.3.1 Expert vs Novice bar chart …")
        try:
            fig = plot_group_te_comparison_p2(save=True, config=config)
            plt.close(fig)
            logger.info("Group bar chart saved: %s", bar_path)
        except Exception as exc:
            logger.error("Group bar chart failed: %s", exc)
    else:
        logger.info("Group bar chart already exists (overwrite=False): %s", bar_path)

    # 7.3.2 — group comparison bar chart (Phase 1 vs Phase 2)
    comparison_path = viz_dir / "phase1_vs_phase2_comparison.png"
    if overwrite or not comparison_path.exists():
        logger.info("Generating 7.3.2: Phase 1 vs Phase 2 comparison bar chart …")
        try:
            fig = plot_phase1_vs_phase2_comparison(save=True, config=config)
            plt.close(fig)
            logger.info("7.3.2 saved: %s", comparison_path)
        except Exception as exc:
            logger.error("7.3.2 failed: %s", exc)
    else:
        logger.info("7.3.2 already exists (overwrite=False): %s", comparison_path)

    # 7.3.3 — lag profile comparison for primary diagnostic subject
    logger.info(
        "Generating 7.3.3: lag profile comparisons for sub-%s %s Win%d …",
        p1_vs_p2_subject_id, p1_vs_p2_trial, p1_vs_p2_window + 1,
    )
    try:
        conn_labels = _get_connection_labels(config)

        for lbl in conn_labels:
            src_roi, tgt_roi = lbl.split("→")
            src_s = src_roi.replace(" ", "_")
            tgt_s = tgt_roi.replace(" ", "_")
            tr_num  = p1_vs_p2_trial.replace("Tr_", "")
            win_num = p1_vs_p2_window + 1
            fname   = (
                f"sub-{p1_vs_p2_subject_id}_Tr{tr_num}_Win{win_num}"
                f"_{src_s}_to_{tgt_s}_lag_comparison.png"
            )
            out_path = lag_dir / fname

            if not overwrite and out_path.exists():
                logger.info("7.3.3 skip (exists): %s", fname)
                continue

            try:
                fig = plot_lag_profile_phase1_vs_phase2(
                    subject_id=p1_vs_p2_subject_id,
                    trial=p1_vs_p2_trial,
                    window=p1_vs_p2_window,
                    src_roi=src_roi,
                    tgt_roi=tgt_roi,
                    sfreq=None,   # always derive from config
                    save=True,
                    config=config,
                )
                plt.close(fig)
            except Exception as exc:
                logger.warning("7.3.3 failed for %s: %s", lbl, exc)

        logger.info("7.3.3 complete — files saved to %s", lag_dir)
    except Exception as exc:
        logger.error("7.3.3 outer failure: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """
    Command-line interface for Stage E Phase 2 visualizations.

    Usage examples
    --------------
    # 7.3.2 group comparison bar chart (all subjects):
    python3 07_hybrid_dspte_visualization.py --comparison

    # 7.3.3 lag profiles — all 20 connections for A1 Tr_1 Win 1:
    python3 07_hybrid_dspte_visualization.py --lag-profiles --subject A1 --trial Tr_1 --window 1

    # 7.3.3 single connection:
    python3 07_hybrid_dspte_visualization.py --lag-profiles --subject A1 --trial Tr_1 --window 1 --connection "CPFC→LM1"

    # Full run (7.3.2 + all 20 lag profiles for A1 Tr_1 Win 1):
    python3 07_hybrid_dspte_visualization.py --batch

    # Unit smoke tests (synthetic data, no real subjects required):
    python3 07_hybrid_dspte_visualization.py --test
    """
    import argparse
    import time as _time

    p = argparse.ArgumentParser(
        description="Stage E Phase 2 — Hybrid-dSPTE Visualization (Sections 7.3.2 + 7.3.3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 7.3.2 group comparison bar chart:
  python3 code/stage_e_transfer_entropy/07_hybrid_dspte_visualization.py --comparison

  # 7.3.3 all 20 lag profiles for A1, Tr_1, Window 1:
  python3 code/stage_e_transfer_entropy/07_hybrid_dspte_visualization.py --lag-profiles --subject R7 --trial Tr_2 --window 1

  # 7.3.3 single connection:
  python3 code/stage_e_transfer_entropy/07_hybrid_dspte_visualization.py --lag-profiles --subject A1 --trial Tr_1 --window 1 --connection "CPFC→LM1"

  # Full batch (7.3.2 bar chart + all 20 lag profiles for A1 Tr_1 Win 1):
  python3 code/stage_e_transfer_entropy/07_hybrid_dspte_visualization.py --batch

  # Unit smoke tests:
  python3 code/stage_e_transfer_entropy/07_hybrid_dspte_visualization.py --test
        """,
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run all Phase 2 visualizations: group matrix, group bar chart, "
             "7.3.2 comparison chart + 20 lag profiles for --subject/--trial/--window.",
    )
    p.add_argument(
        "--group-matrix",
        dest="group_matrix",
        action="store_true",
        help="Generate Phase 2 group connectivity matrix (Expert | Novice | Difference).",
    )
    p.add_argument(
        "--group-bar",
        dest="group_bar",
        action="store_true",
        help="Generate Phase 2 Expert vs Novice bar chart with Phase 1 reference dots.",
    )
    p.add_argument(
        "--comparison",
        action="store_true",
        help="Generate 7.3.2 Phase 1 vs Phase 2 group comparison bar chart only.",
    )
    p.add_argument(
        "--lag-profiles",
        dest="lag_profiles",
        action="store_true",
        help="Generate 7.3.3 lag profile comparison plots. "
             "Requires --subject, --trial, --window. "
             "Optionally --connection to plot a single pair.",
    )
    p.add_argument(
        "--subject",
        type=str,
        default="A1",
        metavar="ID",
        help="Subject ID for lag profile plots (default: A1).",
    )
    p.add_argument(
        "--trial",
        type=str,
        default="Tr_1",
        metavar="TR",
        help="Trial ID for lag profile plots, e.g. Tr_1 (default: Tr_1).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=1,
        metavar="N",
        help="1-indexed window number for lag profile plots (default: 1).",
    )
    p.add_argument(
        "--connection",
        type=str,
        default=None,
        metavar="SRC→TGT",
        help="Optional single connection to plot, e.g. 'CPFC→LM1'. "
             "If omitted, all 20 connections are plotted.",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Skip plots whose output file already exists.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run unit smoke tests (synthetic data only, no real subjects required).",
    )

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    cfg = load_config()

    # Normalise window to 0-indexed internally
    window_0idx = args.window - 1
    if window_0idx < 0:
        p.error("--window must be ≥ 1")

    # ------------------------------------------------------------------
    # --test  — unit smoke tests (no real data)
    # ------------------------------------------------------------------
    if args.test:
        import time as _t
        print("\n=== 07_hybrid_dspte_visualization — unit smoke tests ===\n")

        print("Test 1: Path helpers …")
        try:
            viz = _p2_viz_dir(cfg)
            lag = _lag_profile_dir(cfg)
            print(f"  phase_2_viz : {viz}")
            print(f"  lag_profile : {lag}")
            print("  [PASS]\n")
        except Exception as e:
            print(f"  [FAIL] {e}\n")

        print("Test 2: Connection labels …")
        try:
            labels = _get_connection_labels(cfg)
            assert len(labels) == 20
            assert "CPFC→LM1" in labels
            print(f"  20 labels OK  ({labels[0]} … {labels[-1]})")
            print("  [PASS]\n")
        except Exception as e:
            print(f"  [FAIL] {e}\n")

        print("Test 3: _subject_mean_per_connection (synthetic) …")
        try:
            rng = np.random.default_rng(42)
            synthetic = {
                "trial_ids": np.array(["Tr_1", "Tr_2"]),
                "Tr_1_connectivity": rng.random((20, 3)) * 0.02,
                "Tr_2_connectivity": rng.random((20, 2)) * 0.02,
            }

            class _FakeNpz:
                def __init__(self, d):
                    """Store dict-like arrays for smoke-test compatibility."""
                    self._d = d
                def __getitem__(self, k):
                    """Return array for key `k`."""
                    return self._d[k]
                def __contains__(self, k):
                    """Return True when key `k` exists."""
                    return k in self._d

            mv = _subject_mean_per_connection(_FakeNpz(synthetic))
            assert mv.shape == (20,)
            assert np.all(mv >= 0)
            print(f"  shape={mv.shape}  range=[{mv.min():.4f}, {mv.max():.4f}]")
            print("  [PASS]\n")
        except Exception as e:
            print(f"  [FAIL] {e}\n")

        print("Test 4: sfreq read from config (no hardcode) …")
        try:
            sfreq_cfg = float(
                cfg.get("stage_e_transfer_entropy", {})
                .get("signal", {})
                .get("sfreq", None)
            )
            assert sfreq_cfg is not None, "sfreq not found in config"
            u_max_cfg = int(
                cfg.get("stage_e_transfer_entropy", {})
                .get("signal", {})
                .get("u_max", None)
            )
            assert u_max_cfg is not None, "u_max not found in config"
            print(f"  sfreq={sfreq_cfg} Hz  u_max={u_max_cfg}")
            print("  [PASS]\n")
        except Exception as e:
            print(f"  [FAIL] {e}\n")

        print("=== Unit smoke tests complete ===\n")
        return

    if not any([args.batch, args.comparison, args.lag_profiles,
                args.group_matrix, args.group_bar]):
        p.print_help()
        return

    # ------------------------------------------------------------------
    # --group-matrix  — Phase 2 group connectivity heatmap
    # ------------------------------------------------------------------
    if args.group_matrix or args.batch:
        print("\n--- Phase 2 Group Connectivity Matrix (Expert | Novice | Difference) ---")
        t0 = _time.perf_counter()
        try:
            fig = plot_group_connectivity_matrix_p2(save=True, config=cfg)
            plt.close(fig)
            print(f"  Done in {_time.perf_counter() - t0:.1f} s")
        except FileNotFoundError as e:
            print(f"  [SKIP — data not found] {e}")
        except Exception as e:
            logger.error("Group matrix failed: %s", e)

    # ------------------------------------------------------------------
    # --group-bar  — Phase 2 Expert vs Novice bar chart
    # ------------------------------------------------------------------
    if args.group_bar or args.batch:
        print("\n--- Phase 2 Expert vs Novice Bar Chart ---")
        t0 = _time.perf_counter()
        try:
            fig = plot_group_te_comparison_p2(save=True, config=cfg)
            plt.close(fig)
            print(f"  Done in {_time.perf_counter() - t0:.1f} s")
        except FileNotFoundError as e:
            print(f"  [SKIP — data not found] {e}")
        except Exception as e:
            logger.error("Group bar chart failed: %s", e)

    # ------------------------------------------------------------------
    # --comparison  — 7.3.2 Phase 1 vs Phase 2 comparison bar chart
    # ------------------------------------------------------------------
    if args.comparison or args.batch:
        print("\n--- 7.3.2: Phase 1 vs Phase 2 comparison bar chart ---")
        t0 = _time.perf_counter()
        try:
            fig = plot_phase1_vs_phase2_comparison(save=True, config=cfg)
            plt.close(fig)
            print(f"  Done in {_time.perf_counter() - t0:.1f} s")
        except FileNotFoundError as e:
            print(f"  [SKIP — data not found] {e}")
        except Exception as e:
            logger.error("7.3.2 failed: %s", e)

    # ------------------------------------------------------------------
    # --lag-profiles  — 7.3.3 lag profile plots
    # ------------------------------------------------------------------
    if args.lag_profiles or args.batch:
        subject_id = args.subject
        trial      = args.trial
        window_idx = window_0idx

        # Normalise trial format: accept "1" → "Tr_1"
        if not trial.startswith("Tr_"):
            trial = f"Tr_{trial}"

        if args.connection:
            # Single connection
            conn = args.connection
            if "→" not in conn:
                p.error(f"--connection must contain '→', e.g. 'CPFC→LM1'. Got: {conn!r}")
            src_roi, tgt_roi = conn.split("→", 1)
            print(f"\n--- 7.3.3: Lag profile — sub-{subject_id} {trial} Win{args.window} {src_roi}→{tgt_roi} ---")
            t0 = _time.perf_counter()
            try:
                fig = plot_lag_profile_phase1_vs_phase2(
                    subject_id=subject_id,
                    trial=trial,
                    window=window_idx,
                    src_roi=src_roi.strip(),
                    tgt_roi=tgt_roi.strip(),
                    sfreq=None,
                    save=True,
                    config=cfg,
                )
                plt.close(fig)
                print(f"  Done in {_time.perf_counter() - t0:.1f} s")
            except FileNotFoundError as e:
                print(f"  [SKIP — data not found] {e}")
            except Exception as e:
                logger.error("7.3.3 single-connection failed: %s", e)
        else:
            # All 20 connections
            print(
                f"\n--- 7.3.3: Lag profiles — sub-{subject_id} {trial} Win{args.window} "
                f"(all 20 connections) ---"
            )
            t0 = _time.perf_counter()
            try:
                plot_all_lag_profiles_phase1_vs_phase2(
                    subject_id=subject_id,
                    trial=trial,
                    window=window_idx,
                    save=True,
                    config=cfg,
                )
                print(f"  Done in {_time.perf_counter() - t0:.1f} s")
            except FileNotFoundError as e:
                print(f"  [SKIP — data not found] {e}")
            except Exception as e:
                logger.error("7.3.3 batch failed: %s", e)


if __name__ == "__main__":
    _cli_main()
