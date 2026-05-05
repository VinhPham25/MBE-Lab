"""
Stage E — Step E5: Phase 1 dSPTE Visualization

Mirrors Stage D ``06_gc_visualization.py`` in structure and naming conventions.
Reuses shared helpers (ROI positions, arrow drawing, 20→5×5 reconstruction)
directly from Stage D where possible.

Section 6 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
-------------------------------------------------------
6.1 — plot_connectivity_matrix_te()         per-subject 5×5 dSPTE heatmap (viridis)
6.2 — plot_optimal_delay_matrix()           5×5 delay heatmap (samples + ms)
6.3 — plot_gc_vs_te_comparison()            GC vs TE side-by-side bar chart (Pearson r, Spearman ρ)
6.4 — plot_group_topology_te()              Brain-topology diagram (Tr_1 Win_1 + subject-averaged)
6.5 — plot_lag_te_profile()                 TE vs lag line plot with optimal-delay marker

Differences from Stage D visualization
---------------------------------------
1. Colormap: ``viridis`` (sequential) instead of ``YlOrRd``.  TE values can
   exceed 1.0 (unlike GC which is bounded near 0–0.1); a perceptually uniform
   sequential map reads better across this range.
2. Subject-averaged group arrays come from ``B_Exp_dspte_mean.npz``
   (strategy="subject_mean") rather than recomputing from per-window stacks.
   This is consistent with the H1 analysis and avoids re-introducing the
   trial-duration confound in the visualization.
3. ``plot_gc_vs_te_comparison()`` is TE-specific: loads both Stage D and Stage E
   .npz files for the same subject/trial/window and produces a side-by-side
   bar chart.
4. ``plot_optimal_delay_matrix()`` and ``plot_lag_te_profile()`` are new (no
   Stage D equivalents).

Conventions (identical to Stage D)
-----------------------------------
  TE matrix convention: mat[src, tgt] = TE from src→tgt
  Row = Source (y-axis), Column = Target (x-axis)
  20-vector → 5×5 matrix: zeros at diagonal [0, 6, 12, 18, 24], reshape (5,5)

Output paths (phase_1_dspte/visualizations/)
--------------------------------------------
  sub-{id}_{trial}_{window}_te_connectivity.png
  sub-{id}_{trial}_{window}_delay_matrix.png
  sub-{id}_{trial}_{window}_gc_vs_te.png
  group_te_topology_Tr1_Win1.png
  group_te_topology_Expert_averaged.png
  group_te_topology_Novice_averaged.png
  group_te_topology_averaged.png
  group_te_comparison_averaged.png
  group_te_connectivity_matrix_averaged.png
  sub-{id}_{trial}_lag_te_profile_{src}_to_{tgt}.png
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr

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

# ---------------------------------------------------------------------------
# Constants and style
# ---------------------------------------------------------------------------

_ROI_NAMES = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_DIAG_IDX = [0, 6, 12, 18, 24]
_OFF_DIAG_IDX = [i for i in range(25) if i not in _DIAG_IDX]

_EXPERT_COLOR = "#1F618D"    # steel blue  (same as Stage D)
_NOVICE_COLOR = "#C0392B"    # brick red   (same as Stage D)
_CMAP_TE      = "viridis"    # sequential, perceptually uniform (TE unbounded above)
_CMAP_DELAY   = "YlOrRd"     # delay magnitude — orange/red for long delays
_CMAP_TOPO    = "viridis"    # topology arrows

# Approximate 2-D anatomical positions — identical to Stage D
_ROI_POSITIONS = {
    "LPFC": (-0.38, 0.72),
    "CPFC": ( 0.00, 0.82),
    "RPFC": ( 0.38, 0.72),
    "LM1":  (-0.65, 0.15),
    "SMA":  ( 0.00, 0.30),
}


# ---------------------------------------------------------------------------
# Shared helper — 20-vector → 5×5 matrix
# ---------------------------------------------------------------------------

def _vec20_to_mat5(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 5×5 matrix from a 20-element off-diagonal vector.

    Diagonal indices [0, 6, 12, 18, 24] are zero-filled.

    Parameters
    ----------
    vec : np.ndarray  shape (20,) or (20, n_windows)

    Returns
    -------
    mat : np.ndarray  shape (5, 5) or (5, 5, n_windows)
    """
    if vec.ndim == 1:
        flat = np.zeros(25)
        flat[_OFF_DIAG_IDX] = vec
        return flat.reshape(5, 5)
    n_wins = vec.shape[1]
    flat = np.zeros((25, n_wins))
    flat[_OFF_DIAG_IDX, :] = vec
    return flat.reshape(5, 5, n_wins)


# ---------------------------------------------------------------------------
# Shared helper — output directory
# ---------------------------------------------------------------------------

def _viz_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir  # noqa
    d = get_te_output_dir("phase_1_viz", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conn_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_1_connectivity", config=config)


def _group_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir  # noqa
    return get_te_output_dir("phase_1_group", config=config)


# ---------------------------------------------------------------------------
# Shared helper — curved directed arrow (identical to Stage D)
# ---------------------------------------------------------------------------

def _draw_curved_arrow(
    ax: plt.Axes,
    src_xy: tuple[float, float],
    tgt_xy: tuple[float, float],
    weight: float,
    color,
    alpha: float,
    lw_max: float = 6.0,
    node_radius: float = 0.08,
) -> None:
    sx, sy = src_xy
    tx, ty = tgt_xy
    dx, dy = tx - sx, ty - sy
    dist = np.hypot(dx, dy)
    if dist < 1e-6 or weight < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    x0, y0 = sx + ux * node_radius, sy + uy * node_radius
    x1, y1 = tx - ux * node_radius, ty - uy * node_radius
    lw = max(0.4, lw_max * weight)
    arrow = mpatches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle="arc3,rad=0.22",
        arrowstyle=mpatches.ArrowStyle("-|>", head_length=0.06, head_width=0.04),
        lw=lw, color=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Shared helper — load subject-mean group arrays
# ---------------------------------------------------------------------------

def _load_group_mean_arrays(
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """
    Load B_Exp_dspte_mean / B_Nov_dspte_mean and return along with subject IDs.

    Returns
    -------
    B_Exp_mean : (n_exp, 20, 5)
    B_Nov_mean : (n_nov, 20, 5)
    conn_labels : list[str]
    expert_ids  : list[str]
    novice_ids  : list[str]
    """
    from utils_te import get_connection_labels  # noqa
    gd = _group_dir(config)
    exp_npz = np.load(gd / "B_Exp_dspte_mean.npz", allow_pickle=True)
    nov_npz = np.load(gd / "B_Nov_dspte_mean.npz", allow_pickle=True)
    return (
        exp_npz["data"],
        nov_npz["data"],
        get_connection_labels(),
        list(exp_npz["subject_ids"]),
        list(nov_npz["subject_ids"]),
    )


def _group_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mean, sem, n) across subject axis (axis=0), ignoring NaN."""
    mean = np.nanmean(arr, axis=0)
    n = int(np.sum(~np.isnan(arr[:, 0])))
    sem = np.nanstd(arr, axis=0, ddof=1) / max(np.sqrt(n), 1)
    return mean, sem, n


# ===========================================================================
# Section 6.1 — Per-subject dSPTE connectivity heatmap
# ===========================================================================

def plot_connectivity_matrix_te(
    te_vec: np.ndarray,
    subject_id: str,
    trial: str,
    window: str,
    group: str,
    vmax: Optional[float] = None,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Render a 5×5 dSPTE connectivity heatmap for one subject/trial/window.

    Identical call signature to Stage D ``plot_connectivity_matrix()``.
    Key difference: colormap is ``viridis`` (TE values are unbounded above,
    unlike GC; a sequential perceptually-uniform map reads better).

    Parameters
    ----------
    te_vec    : np.ndarray  shape (20,)  — off-diagonal TE values
    subject_id: str
    trial     : str   e.g. 'Tr_1'
    window    : str   e.g. 'Win_1'
    group     : str   'Expert' or 'Novice'
    vmax      : float or None  — color scale max; auto if None
    save      : bool
    ax        : plt.Axes or None  — draw into existing axes (multi-panel use)
    config    : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    mat = _vec20_to_mat5(te_vec)
    mat_display = mat.astype(float)
    np.fill_diagonal(mat_display, np.nan)

    if vmax is None:
        vmax = float(np.nanmax(mat_display)) if not np.all(np.isnan(mat_display)) else 1.0
    vmax = max(vmax, 1e-6)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(mat_display, cmap=_CMAP_TE, vmin=0, vmax=vmax, aspect="auto")

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
                val = mat_display[r, c]
                txt_color = "white" if val > 0.65 * vmax else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, zorder=3, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(_ROI_NAMES, fontsize=9)
    ax.set_yticklabels(_ROI_NAMES, fontsize=9)
    ax.set_xlabel("Target ROI", fontsize=10)
    ax.set_ylabel("Source ROI", fontsize=10)

    group_color = _EXPERT_COLOR if group == "Expert" else _NOVICE_COLOR
    ax.set_title(
        f"sub-{subject_id}  {trial} {window}  [{group}]",
        fontsize=10, fontweight="bold", color=group_color, pad=6,
    )

    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2, zorder=4)
    ax.tick_params(which="minor", length=0)

    if standalone:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("dSPTE", fontsize=8)
        fig.tight_layout()
        if save:
            out = _viz_dir(config) / f"sub-{subject_id}_{trial}_{window}_te_connectivity.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("Saved %s", out.name)
    return fig


def plot_connectivity_evolution_te(
    subject_id: str = "A1",
    trial: str = "Tr_1",
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Multi-panel figure showing dSPTE connectivity matrix for each window of a trial.

    Parameters
    ----------
    subject_id : str   default 'A1'
    trial      : str   default 'Tr_1'
    save       : bool
    config     : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    from utils_te import get_subject_group  # noqa
    npz_path = _conn_dir(config) / f"sub-{subject_id}_dspte_connectivity.npz"
    data = np.load(npz_path, allow_pickle=True)

    key = f"{trial}_connectivity"
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path.name}")

    arr = data[key]        # (20, n_windows)
    group = str(data["group"])
    n_wins = arr.shape[1]

    mats = [_vec20_to_mat5(arr[:, w]) for w in range(n_wins)]
    all_vals = np.concatenate([m[~np.eye(5, dtype=bool)] for m in mats])
    vmax = float(np.nanmax(all_vals)) if len(all_vals) > 0 else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, n_wins, figsize=(4.8 * n_wins, 4.8))
    if n_wins == 1:
        axes = [axes]

    for w_idx, ax in enumerate(axes):
        plot_connectivity_matrix_te(
            arr[:, w_idx],
            subject_id=subject_id,
            trial=trial,
            window=f"Win_{w_idx + 1}",
            group=group,
            vmax=vmax,
            save=False,
            ax=ax,
            config=config,
        )

    norm = Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(cmap=_CMAP_TE, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.03, shrink=0.8)
    cbar.set_label("dSPTE", fontsize=9)

    fig.suptitle(
        f"sub-{subject_id}  [{group}]  {trial} — dSPTE Connectivity Evolution",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        tr_num = trial.replace("Tr_", "")
        out = _viz_dir(config) / f"sub-{subject_id}_Tr{tr_num}_te_evolution.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ===========================================================================
# Section 6.2 — Optimal delay matrix heatmap
# ===========================================================================

def plot_optimal_delay_matrix(
    delay_vec: np.ndarray,
    subject_id: str,
    trial: str,
    window: str,
    group: str,
    sfreq: float = 25.0,
    vmax_samples: Optional[float] = None,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    5×5 heatmap of optimal transmission delays δ̂_{src→tgt} in samples and ms.

    Each off-diagonal cell is annotated with the delay in milliseconds.
    The diagonal is marked ``—``.  Color encodes delay duration (longer = warmer).

    Parameters
    ----------
    delay_vec       : np.ndarray  shape (20,)  — optimal delay per connection (samples)
    subject_id      : str
    trial           : str   e.g. 'Tr_1'
    window          : str   e.g. 'Win_1'
    group           : str   'Expert' or 'Novice'
    sfreq           : float  sampling frequency in Hz (default 25.0)
    vmax_samples    : float or None  — colorscale max; auto if None
    save            : bool
    ax              : plt.Axes or None
    config          : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    mat = _vec20_to_mat5(delay_vec)   # (5, 5) in samples
    mat_display = mat.astype(float)
    np.fill_diagonal(mat_display, np.nan)

    mat_ms = mat_display * (1000.0 / sfreq)   # samples → milliseconds

    if vmax_samples is None:
        vmax_samples = float(np.nanmax(mat_display)) if not np.all(np.isnan(mat_display)) else 50.0
    vmax_samples = max(vmax_samples, 1.0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
    else:
        fig = ax.get_figure()

    im = ax.imshow(mat_display, cmap=_CMAP_DELAY, vmin=0, vmax=vmax_samples, aspect="auto")

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
                val_s = mat_display[r, c]
                val_ms = mat_ms[r, c]
                if not np.isnan(val_s):
                    txt_color = "white" if val_s > 0.65 * vmax_samples else "black"
                    ax.text(
                        c, r,
                        f"{int(val_s)} smp\n{val_ms:.0f} ms",
                        ha="center", va="center",
                        fontsize=7, color=txt_color, zorder=3,
                    )

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(_ROI_NAMES, fontsize=9)
    ax.set_yticklabels(_ROI_NAMES, fontsize=9)
    ax.set_xlabel("Target ROI", fontsize=10)
    ax.set_ylabel("Source ROI", fontsize=10)

    group_color = _EXPERT_COLOR if group == "Expert" else _NOVICE_COLOR
    ax.set_title(
        f"Optimal Delay  sub-{subject_id}  {trial} {window}  [{group}]",
        fontsize=10, fontweight="bold", color=group_color, pad=6,
    )

    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2, zorder=4)
    ax.tick_params(which="minor", length=0)

    if standalone:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Optimal delay (samples @ {sfreq:.0f} Hz)", fontsize=8)
        fig.tight_layout()
        if save:
            out = _viz_dir(config) / f"sub-{subject_id}_{trial}_{window}_delay_matrix.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("Saved %s", out.name)
    return fig


# ===========================================================================
# Section 6.3 — GC vs TE comparison bar chart
# ===========================================================================

def plot_gc_vs_te_comparison(
    subject_id: str,
    trial: str = "Tr_1",
    window_idx: int = 0,
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing GC and dSPTE for the same subject/trial/window.

    Loads Stage D ``sub-{id}_gc_connectivity.npz`` and Stage E
    ``sub-{id}_dspte_connectivity.npz`` for the given subject.  Bar chart shows
    all 20 connections with GC in steel blue and dSPTE in teal.  Title reports
    Pearson r and Spearman ρ between the two vectors.

    Parameters
    ----------
    subject_id  : str
    trial       : str   e.g. 'Tr_1'
    window_idx  : int   0-indexed window (default 0 → Win_1)
    save        : bool
    config      : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    if config is None:
        config = load_config()

    win_label = f"Win_{window_idx + 1}"

    # --- Load Stage D GC vector ---
    gc_output_root = _PROJECT_ROOT / "outputs" / "stage_d_granger" / "linear_gc" / "expert-novice"
    gc_npz_path = gc_output_root / "gc_connectivity" / f"sub-{subject_id}_gc_connectivity.npz"
    if not gc_npz_path.exists():
        raise FileNotFoundError(f"GC .npz not found: {gc_npz_path}")
    gc_data = np.load(gc_npz_path, allow_pickle=True)
    gc_key = f"{trial}_connectivity"
    if gc_key not in gc_data:
        raise KeyError(f"Key '{gc_key}' not in GC .npz for {subject_id}")
    gc_arr = gc_data[gc_key]   # (20, n_wins_gc)
    if window_idx >= gc_arr.shape[1]:
        raise IndexError(
            f"{subject_id} GC {trial} has only {gc_arr.shape[1]} windows; "
            f"window_idx={window_idx} out of range."
        )
    gc_vec = gc_arr[:, window_idx]

    # --- Load Stage E dSPTE vector ---
    te_npz_path = _conn_dir(config) / f"sub-{subject_id}_dspte_connectivity.npz"
    if not te_npz_path.exists():
        raise FileNotFoundError(f"dSPTE .npz not found: {te_npz_path}")
    te_data = np.load(te_npz_path, allow_pickle=True)
    te_key = f"{trial}_connectivity"
    if te_key not in te_data:
        raise KeyError(f"Key '{te_key}' not in dSPTE .npz for {subject_id}")
    te_arr = te_data[te_key]   # (20, n_wins_te)
    if window_idx >= te_arr.shape[1]:
        raise IndexError(
            f"{subject_id} dSPTE {trial} has only {te_arr.shape[1]} windows; "
            f"window_idx={window_idx} out of range."
        )
    te_vec = te_arr[:, window_idx]

    group = str(te_data.get("group", np.array("Unknown")))

    # --- Correlation statistics ---
    from utils_te import get_connection_labels  # noqa
    conn_labels = get_connection_labels()

    valid_mask = ~(np.isnan(gc_vec) | np.isnan(te_vec))
    if valid_mask.sum() >= 3:
        r_val, r_p = pearsonr(gc_vec[valid_mask], te_vec[valid_mask])
        rho_val, rho_p = spearmanr(gc_vec[valid_mask], te_vec[valid_mask])
        corr_str = f"Pearson r={r_val:.3f} (p={r_p:.3f})   Spearman ρ={rho_val:.3f} (p={rho_p:.3f})"
    else:
        corr_str = "Pearson r=N/A   Spearman ρ=N/A  (insufficient valid connections)"

    # --- Plot ---
    n_conn = len(conn_labels)
    x = np.arange(n_conn)
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.bar(x - bar_w / 2, gc_vec, bar_w,
           color=_EXPERT_COLOR, alpha=0.85, label="GC (linear, Stage D)")
    ax.bar(x + bar_w / 2, te_vec, bar_w,
           color="#1ABC9C",    alpha=0.85, label="dSPTE (Stage E)")

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Connectivity magnitude", fontsize=11)
    ax.set_title(
        f"GC vs dSPTE — sub-{subject_id}  [{group}]  {trial} {win_label}\n"
        f"{corr_str}",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save:
        win_num = window_idx + 1
        tr_num = trial.replace("Tr_", "")
        out = _viz_dir(config) / f"sub-{subject_id}_Tr{tr_num}_Win{win_num}_gc_vs_te.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ===========================================================================
# Section 6.4 — Group topology diagram
# ===========================================================================

def _draw_topology_panel(
    ax: plt.Axes,
    te_mean: np.ndarray,
    group_name: str,
    group_color: str,
    title: str,
    te_max: float,
    threshold: float,
) -> int:
    """Draw one brain-topology panel onto ax. Returns n_connections_shown."""
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    head_circle = plt.Circle((0, 0.35), 1.0, fill=False, ec="#CCCCCC", lw=2.0, zorder=0)
    ax.add_patch(head_circle)
    ax.plot([0], [1.36], "^", ms=8, color="#CCCCCC", zorder=1)

    te_mat = _vec20_to_mat5(te_mean)
    arrow_cmap = plt.get_cmap(_CMAP_TOPO)

    for src_idx, src_roi in enumerate(_ROI_NAMES):
        for tgt_idx, tgt_roi in enumerate(_ROI_NAMES):
            if src_idx == tgt_idx:
                continue
            w = te_mat[src_idx, tgt_idx]
            if np.isnan(w) or w < threshold:
                continue
            norm_w = w / te_max
            rgba = arrow_cmap(0.25 + 0.75 * norm_w)
            _draw_curved_arrow(
                ax,
                src_xy=_ROI_POSITIONS[src_roi],
                tgt_xy=_ROI_POSITIONS[tgt_roi],
                weight=norm_w,
                color=rgba,
                alpha=0.85,
                lw_max=5.5,
                node_radius=0.09,
            )

    for roi, (px, py) in _ROI_POSITIONS.items():
        circle = plt.Circle((px, py), 0.09, color=group_color,
                             zorder=5, alpha=0.95, ec="white", lw=1.8)
        ax.add_patch(circle)
        ax.text(px, py, roi, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=6)

    n_shown = int(np.sum((te_mean >= threshold) & ~np.isnan(te_mean)))
    ax.set_title(title, fontsize=11, fontweight="bold", color=group_color, pad=10)
    return n_shown


def plot_group_topology_te(
    trial: str = "Tr_1",
    window_idx: int = 0,
    threshold_pct: float = 40.0,
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Brain-topology directed-arrow diagrams for Expert and Novice (side-by-side).

    Generates two variants:
    1. Snapshot: uses ``B_Exp_dspte_mean[:, :, trial_idx]`` (group mean of
       subject averages for the specified trial).
    2. Averaged: calls ``plot_group_topology_te_averaged()`` internally.

    ROI nodes at anatomical positions; directed arrows for connections above
    the ``threshold_pct`` percentile, scaled by magnitude.

    Parameters
    ----------
    trial         : str   trial to show as snapshot (default 'Tr_1')
    window_idx    : int   kept for API symmetry; mean arrays have no window axis
    threshold_pct : float percentile threshold for showing arrows (default 40)
    save          : bool
    config        : dict, optional

    Returns
    -------
    fig : plt.Figure  (snapshot panel)
    """
    if config is None:
        config = load_config()

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_group_mean_arrays(config)

    trial_idx = int(trial.split("_")[1]) - 1
    exp_mean = np.nanmean(B_Exp_mean[:, :, trial_idx], axis=0)   # (20,)
    nov_mean = np.nanmean(B_Nov_mean[:, :, trial_idx], axis=0)

    combined = np.concatenate([exp_mean, nov_mean])
    te_max = float(np.nanmax(combined)) if np.any(~np.isnan(combined)) else 1.0
    te_max = max(te_max, 1e-6)
    threshold = float(np.nanpercentile(combined, threshold_pct))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))

    for ax, te_mean, g_name, g_color in [
        (axes[0], exp_mean, "Expert", _EXPERT_COLOR),
        (axes[1], nov_mean, "Novice", _NOVICE_COLOR),
    ]:
        n_exp_subs = int(np.sum(~np.isnan(B_Exp_mean[:, 0, trial_idx])))
        n_nov_subs = int(np.sum(~np.isnan(B_Nov_mean[:, 0, trial_idx])))
        n_subs = n_exp_subs if g_name == "Expert" else n_nov_subs
        n_shown = _draw_topology_panel(
            ax, te_mean, g_name, g_color,
            title=(
                f"{g_name} (n={n_subs})  [{trial}  subject_mean avg]\n"
                f"≥{threshold_pct:.0f}th pctl shown"
            ),
            te_max=te_max,
            threshold=threshold,
        )

    sm = ScalarMappable(
        cmap=plt.get_cmap(_CMAP_TOPO),
        norm=Normalize(vmin=0.25 * te_max, vmax=te_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.03, shrink=0.55)
    cbar.set_label("dSPTE (subject_mean avg)", fontsize=9)

    fig.suptitle(
        f"Group dSPTE Connectivity Topology — {trial}\n"
        "Nodes: anatomical ROIs  |  Arrows: directed causal influence  |  "
        "Arrow weight ∝ dSPTE magnitude",
        fontsize=11, y=1.00,
    )

    if save:
        tr_num = trial.replace("Tr_", "")
        out = _viz_dir(config) / f"group_te_topology_Tr{tr_num}_Win1.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


def plot_group_topology_te_averaged(
    threshold_pct: float = 40.0,
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Brain-topology diagram using subject-mean TE averaged across all trials.

    Uses ``B_Exp_dspte_mean`` and ``B_Nov_dspte_mean`` — averages over the
    trial axis to produce a per-group, all-trial mean connectivity vector.
    Saves Expert, Novice, and combined panel PNGs.

    Parameters
    ----------
    threshold_pct : float  percentile threshold (default 40)
    save          : bool
    config        : dict, optional

    Returns
    -------
    fig : plt.Figure  (combined two-panel)
    """
    if config is None:
        config = load_config()

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_group_mean_arrays(config)

    # Average over subject AND trial axes → (20,)
    exp_mean = np.nanmean(B_Exp_mean, axis=(0, 2))
    nov_mean = np.nanmean(B_Nov_mean, axis=(0, 2))
    n_exp = int(np.sum(~np.isnan(B_Exp_mean[:, 0, 0])))
    n_nov = int(np.sum(~np.isnan(B_Nov_mean[:, 0, 0])))

    combined = np.concatenate([exp_mean, nov_mean])
    te_max = float(np.nanmax(combined)) if np.any(~np.isnan(combined)) else 1.0
    te_max = max(te_max, 1e-6)
    threshold = float(np.nanpercentile(combined, threshold_pct))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))

    for ax, te_mean, g_name, g_color, n_subs in [
        (axes[0], exp_mean, "Expert", _EXPERT_COLOR, n_exp),
        (axes[1], nov_mean, "Novice", _NOVICE_COLOR, n_nov),
    ]:
        _draw_topology_panel(
            ax, te_mean, g_name, g_color,
            title=(
                f"{g_name} (n={n_subs})  [All trials averaged]\n"
                f"≥{threshold_pct:.0f}th pctl shown"
            ),
            te_max=te_max,
            threshold=threshold,
        )

    sm = ScalarMappable(
        cmap=plt.get_cmap(_CMAP_TOPO),
        norm=Normalize(vmin=0.25 * te_max, vmax=te_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.03, shrink=0.55)
    cbar.set_label("dSPTE (all-trial average)", fontsize=9)

    fig.suptitle(
        "Group dSPTE Connectivity Topology — All-Trial Average\n"
        "Nodes: anatomical ROIs  |  Arrows: directed causal influence",
        fontsize=11, y=1.00,
    )

    if save:
        vd = _viz_dir(config)
        out = vd / "group_te_topology_averaged.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

        for te_mean, g, g_color, n_subs in [
            (exp_mean, "Expert", _EXPERT_COLOR, n_exp),
            (nov_mean, "Novice", _NOVICE_COLOR, n_nov),
        ]:
            fig_s, ax_s = plt.subplots(figsize=(7, 7.5))
            _draw_topology_panel(
                ax_s, te_mean, g, g_color,
                title=(
                    f"{g} (n={n_subs})  [All trials averaged]\n"
                    f"≥{threshold_pct:.0f}th pctl shown"
                ),
                te_max=te_max,
                threshold=threshold,
            )
            cbar_s = fig_s.colorbar(
                ScalarMappable(
                    cmap=plt.get_cmap(_CMAP_TOPO),
                    norm=Normalize(vmin=0.25 * te_max, vmax=te_max),
                ),
                ax=ax_s, fraction=0.04, pad=0.02, shrink=0.6,
            )
            cbar_s.set_label("dSPTE", fontsize=9)
            fig_s.tight_layout()
            out_s = vd / f"group_te_topology_{g}_averaged.png"
            fig_s.savefig(out_s, dpi=300, bbox_inches="tight")
            plt.close(fig_s)
            logger.info("Saved %s", out_s.name)

    return fig


# ===========================================================================
# Section 6.4 extra — Group comparison bar chart (averaged)
# ===========================================================================

def plot_group_te_comparison_averaged(
    h1_df: Optional[pd.DataFrame] = None,
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Bar chart comparing Expert vs Novice mean dSPTE per connection.

    Uses subject-mean group arrays (strategy="subject_mean") averaged across
    all trials.  Error bars are SEM across subjects.  FDR-significant
    connections from H1 are marked with ``*`` (primary strategy) and ``†``
    (sensitivity fw strategy if different).

    Parameters
    ----------
    h1_df : pd.DataFrame or None  — H1 CSV; loaded from disk if None
    save  : bool
    config: dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    if config is None:
        config = load_config()

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_group_mean_arrays(config)

    # Average over trial axis for each subject → (n_subs, 20)
    exp_subj = np.nanmean(B_Exp_mean, axis=2)   # (n_exp, 20)
    nov_subj = np.nanmean(B_Nov_mean, axis=2)   # (n_nov, 20)

    exp_mean, exp_sem, n_exp = _group_stats(exp_subj)
    nov_mean, nov_sem, n_nov = _group_stats(nov_subj)

    # Load H1 CSV if not provided
    if h1_df is None:
        stats_dir = _PROJECT_ROOT / "outputs" / "stage_e_transfer_entropy" / \
                    "expert-novice" / "phase_1_dspte" / "statistics"
        h1_path = stats_dir / "H1_expert_vs_novice.csv"
        if h1_path.exists():
            h1_df = pd.read_csv(h1_path)

    sig_primary: set[str] = set()
    sig_fw_only: set[str] = set()
    if h1_df is not None:
        sig_primary = set(h1_df.loc[h1_df["significant"] == True, "connection"].tolist())
        if "significant_fw" in h1_df.columns:
            sig_fw = set(h1_df.loc[h1_df["significant_fw"] == True, "connection"].tolist())
            sig_fw_only = sig_fw - sig_primary

    n_conn = len(conn_labels)
    x = np.arange(n_conn)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - bar_w / 2, exp_mean, bar_w,
           color=_EXPERT_COLOR, alpha=0.88, label=f"Expert (n={n_exp})",
           yerr=exp_sem, capsize=3, error_kw={"elinewidth": 1.2})
    ax.bar(x + bar_w / 2, nov_mean, bar_w,
           color=_NOVICE_COLOR, alpha=0.88, label=f"Novice (n={n_nov})",
           yerr=nov_sem, capsize=3, error_kw={"elinewidth": 1.2})

    y_max_all = float(np.nanmax(
        np.concatenate([exp_mean + exp_sem, nov_mean + nov_sem])
    )) if len(exp_mean) > 0 else 1.0

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
    ax.set_ylabel("dSPTE", fontsize=11)
    ax.set_title(
        "Expert vs. Novice dSPTE Connectivity — All-Trial Average (subject_mean strategy)\n"
        "(* = FDR-sig primary;  † = FDR-sig sensitivity only;  BH α=0.05)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        out = _viz_dir(config) / "group_te_comparison_averaged.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


def plot_group_connectivity_matrix_te_averaged(
    save: bool = True,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Side-by-side 5×5 heatmaps of group-mean dSPTE (Expert | Novice | Difference).

    Uses subject-mean arrays averaged across all trials.  Third panel shows
    Expert − Novice with a diverging colormap.

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

    B_Exp_mean, B_Nov_mean, conn_labels, exp_ids, nov_ids = _load_group_mean_arrays(config)

    exp_mean = np.nanmean(B_Exp_mean, axis=(0, 2))
    nov_mean = np.nanmean(B_Nov_mean, axis=(0, 2))
    n_exp = int(np.sum(~np.isnan(B_Exp_mean[:, 0, 0])))
    n_nov = int(np.sum(~np.isnan(B_Nov_mean[:, 0, 0])))

    exp_mat  = _vec20_to_mat5(exp_mean)
    nov_mat  = _vec20_to_mat5(nov_mean)
    diff_mat = exp_mat - nov_mat

    diag_mask = np.eye(5, dtype=bool)
    exp_d  = np.where(diag_mask, np.nan, exp_mat)
    nov_d  = np.where(diag_mask, np.nan, nov_mat)
    diff_d = np.where(diag_mask, np.nan, diff_mat)

    vmax_abs  = max(float(np.nanmax([np.nanmax(np.abs(exp_d)), np.nanmax(np.abs(nov_d))])), 1e-6)
    diff_abs  = max(float(np.nanmax(np.abs(diff_d))), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    panels = [
        (axes[0], exp_d,  0,         vmax_abs, _CMAP_TE,  f"Expert (n={n_exp})",  _EXPERT_COLOR, "dSPTE"),
        (axes[1], nov_d,  0,         vmax_abs, _CMAP_TE,  f"Novice (n={n_nov})",  _NOVICE_COLOR, "dSPTE"),
        (axes[2], diff_d, -diff_abs, diff_abs, "RdBu_r",  "Expert − Novice",      "#555555",     "Δ dSPTE"),
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
        "Group-Mean dSPTE Connectivity Matrix — All-Trial Average (subject_mean strategy)\n"
        "(row = Source, col = Target;  diagonal masked)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    if save:
        out = _viz_dir(config) / "group_te_connectivity_matrix_averaged.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ===========================================================================
# Section 6.5 — Lag-TE profile
# ===========================================================================

def plot_lag_te_profile(
    te_vs_lag: np.ndarray,
    src_roi: str,
    tgt_roi: str,
    optimal_delay: int,
    subject_id: str = "",
    trial: str = "",
    window: str = "",
    sfreq: float = 25.0,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
    config: Optional[dict] = None,
) -> plt.Figure:
    """
    Line plot of ``dSPTE_{X→Y, u}`` vs lag ``u`` with an optimal-delay marker.

    Shows whether the multi-delay peak is sharp (confident delay estimate)
    or broad (uncertain).  The optimal delay is marked with a vertical dashed
    line and annotated with the delay in both samples and milliseconds.

    Parameters
    ----------
    te_vs_lag     : np.ndarray  shape (u_max,)  — TE value at each lag u=1..u_max
    src_roi       : str   source ROI name (e.g. 'LPFC')
    tgt_roi       : str   target ROI name (e.g. 'SMA')
    optimal_delay : int   optimal lag in samples, 1-indexed (e.g. 10 = lag 10)
    subject_id    : str   for title
    trial         : str   for title
    window        : str   for title
    sfreq         : float sampling frequency Hz (default 25.0)
    save          : bool
    ax            : plt.Axes or None
    config        : dict, optional

    Returns
    -------
    fig : plt.Figure
    """
    u_max = len(te_vs_lag)
    lags = np.arange(1, u_max + 1)
    # optimal_delay is 1-indexed (lag value, not array index)
    delay_ms = optimal_delay * (1000.0 / sfreq)
    # 0-based index into te_vs_lag
    delay_idx = optimal_delay - 1

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    ax.plot(lags, te_vs_lag, color="#1ABC9C", lw=2.0, alpha=0.9, label=f"{src_roi}→{tgt_roi}")
    ax.fill_between(lags, 0, te_vs_lag, alpha=0.12, color="#1ABC9C")

    if 1 <= optimal_delay <= u_max:
        ax.axvline(x=optimal_delay, color="#E74C3C", lw=1.8, linestyle="--",
                   label=f"δ̂ = {optimal_delay} smp  ({delay_ms:.0f} ms)")
        peak_val = te_vs_lag[delay_idx]
        ax.annotate(
            f"δ̂={optimal_delay}\n({delay_ms:.0f} ms)",
            xy=(optimal_delay, peak_val),
            xytext=(optimal_delay + max(1, u_max * 0.04), peak_val * 1.05),
            fontsize=8, color="#E74C3C",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2),
        )

    ax.set_xlabel("Lag u (samples)", fontsize=10)
    ax.set_ylabel("dSPTE", fontsize=10)
    title_parts = []
    if subject_id:
        title_parts.append(f"sub-{subject_id}")
    if trial:
        title_parts.append(trial)
    if window:
        title_parts.append(window)
    title_parts.append(f"{src_roi}→{tgt_roi}  Lag-TE Profile")
    ax.set_title("  ".join(title_parts), fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)

    if standalone:
        fig.tight_layout()
        if save and subject_id:
            src_s = src_roi.replace(" ", "_")
            tgt_s = tgt_roi.replace(" ", "_")
            tr_num = trial.replace("Tr_", "") if trial else "x"
            win_num = window.replace("Win_", "") if window else "x"
            fname = (
                f"sub-{subject_id}_Tr{tr_num}_Win{win_num}"
                f"_lag_te_profile_{src_s}_to_{tgt_s}.png"
            )
            out = _viz_dir(config) / fname
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("Saved %s", out.name)

    return fig


# ===========================================================================
# Main runner — generate all Section 6 visualizations
# ===========================================================================

def run_all_visualizations_te(
    config: Optional[dict] = None,
    overwrite: bool = True,
) -> None:
    """
    Generate all Section 6 visualizations and save to phase_1_dspte/visualizations/.

    Skips individual files that already exist if ``overwrite=True``.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids: list = subj_cfg.get("Expert", [])
    novice_ids: list = subj_cfg.get("Novice", [])
    sfreq = float(te_cfg.get("signal", {}).get("sfreq", 25.0))

    vd = _viz_dir(config)
    cd = _conn_dir(config)

    # Load H1 CSV for FDR markers
    stats_path = (
        _PROJECT_ROOT / "outputs" / "stage_e_transfer_entropy"
        / "expert-novice" / "phase_1_dspte" / "statistics"
        / "H1_expert_vs_novice.csv"
    )
    h1_df = pd.read_csv(stats_path) if stats_path.exists() else None
    if h1_df is None:
        logger.warning("H1 CSV not found — bar chart will have no FDR markers")

    # ---- 6.1: Per-subject connectivity heatmaps (A1 Tr_1 Win_1 + evolution) ----
    logger.info("6.1 — Per-subject connectivity heatmaps...")
    for subject_id in expert_ids + novice_ids:
        npz_path = cd / f"sub-{subject_id}_dspte_connectivity.npz"
        if not npz_path.exists():
            logger.warning("  %s: .npz not found, skipping 6.1", subject_id)
            continue
        data = np.load(npz_path, allow_pickle=True)
        group = str(data["group"])

        out_single = vd / f"sub-{subject_id}_Tr_1_Win_1_te_connectivity.png"
        if overwrite or not out_single.exists():
            if "Tr_1_connectivity" in data and data["Tr_1_connectivity"].shape[1] > 0:
                fig = plot_connectivity_matrix_te(
                    data["Tr_1_connectivity"][:, 0],
                    subject_id=subject_id, trial="Tr_1", window="Win_1",
                    group=group, save=True, config=config,
                )
                plt.close(fig)

        # Evolution plot for A1 only (can be slow for all subjects)
        if subject_id == "A1":
            out_evo = vd / "sub-A1_Tr1_te_evolution.png"
            if overwrite or not out_evo.exists():
                try:
                    fig = plot_connectivity_evolution_te("A1", "Tr_1", save=True, config=config)
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("  A1 evolution: %s", exc)

    # ---- 6.2: Optimal delay matrix (A1 Tr_1 Win_1) ----
    logger.info("6.2 — Optimal delay matrix (A1 Tr_1 Win_1)...")
    a1_path = cd / "sub-A1_dspte_connectivity.npz"
    if a1_path.exists():
        a1_data = np.load(a1_path, allow_pickle=True)
        if "Tr_1_delays" in a1_data and a1_data["Tr_1_delays"].shape[1] > 0:
            out_delay = vd / "sub-A1_Tr_1_Win_1_delay_matrix.png"
            if overwrite or not out_delay.exists():
                fig = plot_optimal_delay_matrix(
                    a1_data["Tr_1_delays"][:, 0],
                    subject_id="A1", trial="Tr_1", window="Win_1",
                    group="Expert", sfreq=sfreq, save=True, config=config,
                )
                plt.close(fig)

    # ---- 6.3: GC vs TE comparison (A1 Tr_1 Win_1) ----
    logger.info("6.3 — GC vs TE comparison (A1 Tr_1 Win_1)...")
    gc_path = (
        _PROJECT_ROOT / "outputs" / "stage_d_granger" / "linear_gc"
        / "expert-novice" / "gc_connectivity" / "sub-A1_gc_connectivity.npz"
    )
    if gc_path.exists() and a1_path.exists():
        out_gcte = vd / "sub-A1_Tr1_Win1_gc_vs_te.png"
        if overwrite or not out_gcte.exists():
            try:
                fig = plot_gc_vs_te_comparison(
                    "A1", trial="Tr_1", window_idx=0, save=True, config=config
                )
                plt.close(fig)
            except Exception as exc:
                logger.warning("  GC vs TE comparison failed: %s", exc)
    else:
        logger.warning("6.3 skipped — GC or TE .npz not found for A1")

    # ---- 6.4: Group topology (Tr_1 snapshot + all-trial averaged) ----
    logger.info("6.4 — Group topology diagrams...")
    try:
        out_topo_tr1 = vd / "group_te_topology_Tr1_Win1.png"
        if overwrite or not out_topo_tr1.exists():
            fig = plot_group_topology_te(trial="Tr_1", save=True, config=config)
            plt.close(fig)

        out_topo_avg = vd / "group_te_topology_averaged.png"
        if overwrite or not out_topo_avg.exists():
            fig = plot_group_topology_te_averaged(save=True, config=config)
            plt.close(fig)

        out_bar_avg = vd / "group_te_comparison_averaged.png"
        if overwrite or not out_bar_avg.exists():
            fig = plot_group_te_comparison_averaged(h1_df=h1_df, save=True, config=config)
            plt.close(fig)

        out_mat_avg = vd / "group_te_connectivity_matrix_averaged.png"
        if overwrite or not out_mat_avg.exists():
            fig = plot_group_connectivity_matrix_te_averaged(save=True, config=config)
            plt.close(fig)

    except Exception as exc:
        logger.warning("  Group topology/bar chart failed: %s", exc)

    # ---- 6.5: Lag-TE profile (A1 Tr_1 Win_1, LPFC→SMA as example pair) ----
    logger.info("6.5 — Lag-TE profile (A1 Tr_1 Win_1, LPFC→SMA)...")
    te_raw_dir = (
        _PROJECT_ROOT / "outputs" / "stage_e_transfer_entropy"
        / "expert-novice" / "phase_1_dspte" / "te_raw"
    )
    # te_raw stores lag profiles per subject; if not available, generate
    # a synthetic example from the batch log delay data
    _generate_lag_profile_example(
        "A1", "Tr_1", 0, "LPFC", "SMA",
        sfreq=sfreq, save=True, config=config, overwrite=overwrite,
    )

    # Summary
    pngs = sorted(vd.glob("*.png"))
    logger.info("Section 6 complete. %d PNG files saved:", len(pngs))
    for f in pngs:
        logger.info("  %s  (%.0f KB)", f.name, f.stat().st_size / 1024)


def _generate_lag_profile_example(
    subject_id: str,
    trial: str,
    window_idx: int,
    src_roi: str,
    tgt_roi: str,
    sfreq: float = 25.0,
    save: bool = True,
    config: Optional[dict] = None,
    overwrite: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate a lag-TE profile for a single connection from the per-subject .npz.

    Reads the ``{trial}_te_profiles`` array (shape 20 × u_max × n_wins) that is
    stored by the pipeline since the Phase 3 audit fix.  If the key is absent
    (old .npz files), falls back to a synthesised bell-curve with an
    "illustrative" watermark so legacy files remain renderable.
    """
    cd = _conn_dir(config)
    npz_path = cd / f"sub-{subject_id}_dspte_connectivity.npz"
    if not npz_path.exists():
        logger.warning("_generate_lag_profile_example: %s .npz not found", subject_id)
        return None

    data = np.load(npz_path, allow_pickle=True)
    delay_key   = f"{trial}_delays"
    conn_key    = f"{trial}_connectivity"
    profile_key = f"{trial}_te_profiles"

    if delay_key not in data or data[delay_key].ndim != 2 or data[delay_key].shape[1] <= window_idx:
        logger.warning("_generate_lag_profile_example: delay data not available for %s %s Win_%d",
                       subject_id, trial, window_idx + 1)
        return None

    delay_vec = data[delay_key][:, window_idx]   # (20,)
    conn_vec  = data[conn_key][:, window_idx] if conn_key in data else np.zeros(20)

    # Find the connection index for src_roi→tgt_roi
    from utils_te import get_connection_labels  # noqa
    conn_labels = get_connection_labels()
    pair_label = f"{src_roi}→{tgt_roi}"
    if pair_label not in conn_labels:
        logger.warning("_generate_lag_profile_example: '%s' not in connection labels", pair_label)
        return None

    conn_idx = conn_labels.index(pair_label)
    optimal_delay_smp = int(delay_vec[conn_idx])
    te_peak = float(conn_vec[conn_idx])

    te_cfg = (config or load_config()).get("stage_e_transfer_entropy", {})
    u_max = int(te_cfg.get("signal", {}).get("u_max", 50))

    illustrative = False
    if profile_key in data:
        profiles_arr = data[profile_key]  # (20, u_max, n_wins)
        if profiles_arr.ndim == 3 and profiles_arr.shape[2] > window_idx:
            profile = profiles_arr[conn_idx, :, window_idx].astype(float)
        else:
            illustrative = True
    else:
        illustrative = True

    if illustrative:
        # Legacy fallback: synthesise a bell curve centred on the stored argmax.
        lags = np.arange(1, u_max + 1)
        sigma = max(u_max * 0.08, 2.0)
        profile = te_peak * np.exp(-0.5 * ((lags - optimal_delay_smp) / sigma) ** 2)
        profile = np.clip(profile, 0, None)
        logger.warning(
            "_generate_lag_profile_example: te_profiles not in .npz for %s — "
            "using synthesised bell curve. Re-run pipeline with overwrite=True.",
            subject_id,
        )

    win_label = f"Win_{window_idx + 1}"
    vd = _viz_dir(config)
    out_fname = (
        vd / f"sub-{subject_id}_Tr{trial.replace('Tr_', '')}_Win{window_idx + 1}"
              f"_lag_te_profile_{src_roi}_to_{tgt_roi}.png"
    )

    if not overwrite and out_fname.exists():
        logger.info("_generate_lag_profile_example: cached PNG found, skipping.")
        return None

    fig = plot_lag_te_profile(
        profile,
        src_roi=src_roi,
        tgt_roi=tgt_roi,
        optimal_delay=optimal_delay_smp,
        subject_id=subject_id,
        trial=trial,
        window=win_label,
        sfreq=sfreq,
        save=False,
        config=config,
    )

    if illustrative:
        fig.text(
            0.99, 0.02,
            "Note: lag profile is illustrative (synthesised bell curve centred on δ̂).\n"
            "Re-run pipeline with overwrite=True to generate real per-lag TE values.",
            ha="right", va="bottom", fontsize=6, color="gray", style="italic",
        )

    if save:
        fig.savefig(out_fname, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out_fname.name)

    return fig


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Stage E Phase 1 — dSPTE visualization."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if PNGs already exist.",
    )
    p.add_argument(
        "--plot",
        choices=["all", "heatmap", "delay", "gcte", "topology", "lag"],
        default="all",
        help="Which plot group to generate (default: all).",
    )
    p.add_argument(
        "--subject",
        default="A1",
        help="Subject ID for per-subject plots (default: A1).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()

    if args.plot == "all":
        run_all_visualizations_te(config=config, overwrite=args.overwrite)
    elif args.plot == "heatmap":
        cd = _conn_dir(config)
        npz = np.load(cd / f"sub-{args.subject}_dspte_connectivity.npz", allow_pickle=True)
        fig = plot_connectivity_matrix_te(
            npz["Tr_1_connectivity"][:, 0],
            subject_id=args.subject, trial="Tr_1", window="Win_1",
            group=str(npz["group"]), save=True, config=config,
        )
        plt.close(fig)
    elif args.plot == "delay":
        cd = _conn_dir(config)
        npz = np.load(cd / f"sub-{args.subject}_dspte_connectivity.npz", allow_pickle=True)
        fig = plot_optimal_delay_matrix(
            npz["Tr_1_delays"][:, 0],
            subject_id=args.subject, trial="Tr_1", window="Win_1",
            group=str(npz["group"]), save=True, config=config,
        )
        plt.close(fig)
    elif args.plot == "gcte":
        fig = plot_gc_vs_te_comparison(
            args.subject, trial="Tr_1", window_idx=0, save=True, config=config
        )
        plt.close(fig)
    elif args.plot == "topology":
        fig = plot_group_topology_te(trial="Tr_1", save=True, config=config)
        plt.close(fig)
        fig = plot_group_topology_te_averaged(save=True, config=config)
        plt.close(fig)
    elif args.plot == "lag":
        te_cfg = config.get("stage_e_transfer_entropy", {})
        sfreq = float(te_cfg.get("signal", {}).get("sfreq", 25.0))
        _generate_lag_profile_example(
            args.subject, "Tr_1", 0, "LPFC", "SMA",
            sfreq=sfreq, save=True, config=config, overwrite=args.overwrite,
        )
