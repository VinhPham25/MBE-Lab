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
    from utils_te import get_te_output_dir  # noqa
    d = get_te_output_dir("phase_2_viz", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p2_conn_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_2_connectivity", config=config)


def _p1_conn_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_1_connectivity", config=config)


def _lag_profile_dir(config: Optional[dict] = None) -> Path:
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
    from utils_te import get_connection_labels  # noqa
    return get_connection_labels()


def _get_subject_group(subject_id: str, config: Optional[dict] = None) -> str:
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

    # 7.3.2 — group comparison bar chart
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
        help="Run all Phase 2 visualizations: 7.3.2 comparison chart + 20 lag "
             "profiles for the subject/trial/window specified by --subject/--trial/--window.",
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
                    self._d = d
                def __getitem__(self, k):
                    return self._d[k]
                def __contains__(self, k):
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

    if not any([args.batch, args.comparison, args.lag_profiles]):
        p.print_help()
        return

    # ------------------------------------------------------------------
    # --comparison  — 7.3.2 group bar chart
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
