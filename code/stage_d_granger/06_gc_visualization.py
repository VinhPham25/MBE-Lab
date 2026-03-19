"""
Stage D — Step D7: Visualization for Linear GC Pipeline

Implements all Section 7 visualizations from LINEAR_GC_PROGRESS.md.

Section 7 task coverage
-----------------------
7.1.1 — plot_connectivity_matrix()      per-subject 5×5 GC heatmap
7.1.2 — plot_connectivity_evolution()   A1 multi-panel across windows of Tr_1
7.2.1 — plot_group_gc_comparison()      Expert vs Novice bar chart with FDR markers
7.2.2 — plot_group_topology()           Brain-topology directed-arrow diagrams

Conventions
-----------
  GC matrix convention: mat[src, tgt] = GC from src→tgt
  Row = Source (y-axis), Column = Target (x-axis)
  20-vector → 5×5 matrix: insert zeros at diagonal indices [0,6,12,18,24]
  and reshape to (5,5).

Output paths
------------
  visualizations/sub-{id}_Tr{n}_Win{n}_connectivity.png
  visualizations/sub-A1_Tr1_connectivity_evolution.png
  visualizations/group_gc_comparison_Tr1_Win1.png
  visualizations/group_gc_topology_Expert.png
  visualizations/group_gc_topology_Novice.png
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
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config
from stage_d_granger.utils_gc import ROI_NAMES, get_connection_labels, get_gc_output_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and style
# ---------------------------------------------------------------------------

_DIAG_IDX = [0, 6, 12, 18, 24]
_OFF_DIAG_IDX = [i for i in range(25) if i not in _DIAG_IDX]

_EXPERT_COLOR = "#1F618D"    # steel blue
_NOVICE_COLOR = "#C0392B"    # brick red
_CMAP_HEATMAP = "YlOrRd"     # sequential, perceptually uniform for GC magnitude
_CMAP_TOPO    = "YlOrRd"     # same colormap for topology arrows

# Approximate 2-D anatomical positions on a unit-radius head (x, y)
# LPFC/CPFC/RPFC = anterior row; LM1 = left lateral; SMA = medial motor
_ROI_POSITIONS = {
    "LPFC": (-0.38, 0.72),
    "CPFC": ( 0.00, 0.82),
    "RPFC": ( 0.38, 0.72),
    "LM1":  (-0.65, 0.15),
    "SMA":  ( 0.00, 0.30),
}

# ---------------------------------------------------------------------------
# Helper: reconstruct 5×5 matrix from 20-vector
# ---------------------------------------------------------------------------

def _vec20_to_mat5(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 5×5 GC matrix from a 20-element off-diagonal vector.

    The 20 elements correspond to the 25 entries of a 5×5 matrix with diagonal
    indices [0, 6, 12, 18, 24] removed (in row-major order).  Diagonal entries
    are zero-filled.

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
    # 2-D: (20, n_windows) → (5, 5, n_windows)
    n_wins = vec.shape[1]
    flat = np.zeros((25, n_wins))
    flat[_OFF_DIAG_IDX, :] = vec
    return flat.reshape(5, 5, n_wins)


def _load_group_arrays() -> tuple[dict, dict, list[str]]:
    """Load B_Exp, B_Nov, and connection labels from group .npz files."""
    group_dir = get_gc_output_dir("group")
    B_Exp_data = np.load(group_dir / "B_Exp.npz", allow_pickle=True)
    B_Nov_data = np.load(group_dir / "B_Nov.npz", allow_pickle=True)
    B_Exp = {k: B_Exp_data[k] for k in B_Exp_data.files if k.startswith("Tr_")}
    B_Nov = {k: B_Nov_data[k] for k in B_Nov_data.files if k.startswith("Tr_")}
    summary_path = group_dir / "gc_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            conn_labels = json.load(f).get("connection_labels", get_connection_labels())
    else:
        conn_labels = get_connection_labels()
    return B_Exp, B_Nov, conn_labels


# ---------------------------------------------------------------------------
# 7.1.1 — Per-subject connectivity heatmap
# ---------------------------------------------------------------------------

def plot_connectivity_matrix(
    gc_vec: np.ndarray,
    subject_id: str,
    trial: str,
    window: str,
    group: str,
    vmax: Optional[float] = None,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Render a 5×5 GC connectivity heatmap for a single subject/trial/window.

    Parameters
    ----------
    gc_vec    : np.ndarray  shape (20,)  — off-diagonal GC values
    subject_id: str
    trial     : str   e.g. 'Tr_1'
    window    : str   e.g. 'Win_1'
    group     : str   'Expert' or 'Novice'
    vmax      : float or None  — color scale max; auto if None
    save      : bool  — save PNG to visualizations/
    ax        : plt.Axes or None  — draw into existing axes (for multi-panel)

    Returns
    -------
    fig : plt.Figure
    """
    mat = _vec20_to_mat5(gc_vec)  # (5, 5)

    # Mask diagonal with NaN for display
    mat_display = mat.astype(float)
    np.fill_diagonal(mat_display, np.nan)

    if vmax is None:
        vmax = float(np.nanmax(mat_display)) if not np.all(np.isnan(mat_display)) else 1.0
    vmax = max(vmax, 0.01)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    else:
        fig = ax.get_figure()

    # Heatmap
    im = ax.imshow(mat_display, cmap=_CMAP_HEATMAP, vmin=0, vmax=vmax, aspect="auto")

    # Annotations
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
                txt_color = "white" if val > 0.55 * vmax else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, zorder=3, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(ROI_NAMES, fontsize=9)
    ax.set_yticklabels(ROI_NAMES, fontsize=9)
    ax.set_xlabel("Target ROI", fontsize=10)
    ax.set_ylabel("Source ROI", fontsize=10)

    group_color = _EXPERT_COLOR if group == "Expert" else _NOVICE_COLOR
    ax.set_title(
        f"sub-{subject_id}  {trial} {window}  [{group}]",
        fontsize=10, fontweight="bold", color=group_color, pad=6,
    )

    # Add minor grid lines separating cells
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2, zorder=4)
    ax.tick_params(which="minor", length=0)

    if standalone:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("GC (0.01–0.07 Hz band avg)", fontsize=8)
        fig.tight_layout()
        if save:
            vis_dir = get_gc_output_dir("visualizations")
            vis_dir.mkdir(parents=True, exist_ok=True)
            out = vis_dir / f"sub-{subject_id}_{trial}_{window}_connectivity.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("Saved %s", out.name)
    return fig


# ---------------------------------------------------------------------------
# 7.1.2 — Multi-panel connectivity evolution for A1 Tr_1
# ---------------------------------------------------------------------------

def plot_connectivity_evolution(
    subject_id: str = "A1",
    trial: str = "Tr_1",
    save: bool = True,
) -> plt.Figure:
    """
    Multi-panel figure showing GC connectivity matrix for each window of a trial.

    Loads the per-subject .npz, reconstructs the 5×5 matrix per window, and
    renders them side-by-side with a shared color scale.

    Parameters
    ----------
    subject_id : str   default 'A1'
    trial      : str   default 'Tr_1'
    save       : bool

    Returns
    -------
    fig : plt.Figure
    """
    gc_dir = get_gc_output_dir("gc_connectivity")
    npz_path = gc_dir / f"sub-{subject_id}_gc_connectivity.npz"
    data = np.load(npz_path, allow_pickle=True)

    key = f"{trial}_connectivity"
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path.name}")

    arr = data[key]          # (20, n_windows)
    group = str(data["group"])
    n_wins = arr.shape[1]

    # Shared vmax across all windows for consistent color scale
    mats = [_vec20_to_mat5(arr[:, w]) for w in range(n_wins)]
    all_vals = [m[~np.eye(5, dtype=bool)] for m in mats]
    vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals else 1.0
    vmax = max(vmax, 0.01)

    fig, axes = plt.subplots(1, n_wins, figsize=(4.8 * n_wins, 4.8))
    if n_wins == 1:
        axes = [axes]

    for w_idx, ax in enumerate(axes):
        gc_vec = arr[:, w_idx]
        plot_connectivity_matrix(
            gc_vec,
            subject_id=subject_id,
            trial=trial,
            window=f"Win_{w_idx + 1}",
            group=group,
            vmax=vmax,
            save=False,
            ax=ax,
        )

    # Shared colorbar
    norm = Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(cmap=_CMAP_HEATMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.03, shrink=0.8)
    cbar.set_label("GC magnitude (0.01–0.07 Hz band avg)", fontsize=9)

    fig.suptitle(
        f"sub-{subject_id}  [{group}]  {trial} — GC Connectivity Evolution",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if save:
        vis_dir = get_gc_output_dir("visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        tr_num = trial.replace("Tr_", "")
        out = vis_dir / f"sub-{subject_id}_Tr{tr_num}_connectivity_evolution.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ---------------------------------------------------------------------------
# 7.2.1 — Group GC bar chart (Expert vs Novice)
# ---------------------------------------------------------------------------

def plot_group_gc_comparison(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
    trial: str = "Tr_1",
    window_idx: int = 0,
    h1_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing Expert vs Novice mean GC for each of the 20 connections.

    Parameters
    ----------
    B_Exp, B_Nov : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels  : list[str]  20 connection labels
    trial        : str  which trial to plot (default 'Tr_1')
    window_idx   : int  which window to plot (0-indexed, default 0)
    h1_df        : pd.DataFrame or None  H1 results; if provided, marks FDR-sig
    save         : bool

    Returns
    -------
    fig : plt.Figure
    """
    win_label = f"Win_{window_idx + 1}"

    exp_arr = B_Exp[trial][:, :, window_idx]   # (n_exp, 20)
    nov_arr = B_Nov[trial][:, :, window_idx]   # (n_nov, 20)

    exp_mean = np.nanmean(exp_arr, axis=0)   # (20,)
    nov_mean = np.nanmean(nov_arr, axis=0)
    exp_sem  = np.nanstd(exp_arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(exp_arr), axis=0))
    nov_sem  = np.nanstd(nov_arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(nov_arr), axis=0))

    # FDR-significant connections from H1
    sig_conns: set[str] = set()
    if h1_df is not None:
        mask = (
            (h1_df["trial"] == trial) &
            (h1_df["window"] == win_label) &
            (h1_df["significant"] == True)
        )
        sig_conns = set(h1_df.loc[mask, "connection"].tolist())

    # --- Layout ---
    n_conn = len(conn_labels)
    x = np.arange(n_conn)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(15, 5))

    b_exp = ax.bar(x - bar_w / 2, exp_mean, bar_w,
                   color=_EXPERT_COLOR, alpha=0.88, label="Expert",
                   yerr=exp_sem, capsize=3, error_kw={"elinewidth": 1.2})
    b_nov = ax.bar(x + bar_w / 2, nov_mean, bar_w,
                   color=_NOVICE_COLOR, alpha=0.88, label="Novice",
                   yerr=nov_sem, capsize=3, error_kw={"elinewidth": 1.2})

    # Mark FDR-significant connections with asterisk above the taller bar
    y_max_all = max(np.nanmax(exp_mean + exp_sem), np.nanmax(nov_mean + nov_sem)) * 1.0
    for i, lbl in enumerate(conn_labels):
        if lbl in sig_conns:
            y_star = max(exp_mean[i] + exp_sem[i], nov_mean[i] + nov_sem[i]) + 0.05 * y_max_all
            ax.text(i, y_star, "*", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="#E74C3C")

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("GC magnitude (0.01–0.07 Hz)", fontsize=11)
    ax.set_title(
        f"Expert vs. Novice GC Connectivity — {trial} {win_label}\n"
        f"(* = FDR-significant, BH α=0.05)",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)

    # Add expert n and novice n as subtitle
    n_exp = int(np.sum(~np.isnan(exp_arr[:, 0])))
    n_nov = int(np.sum(~np.isnan(nov_arr[:, 0])))
    fig.text(0.5, -0.01, f"Expert n={n_exp}  |  Novice n={n_nov}",
             ha="center", fontsize=9, color="gray")

    fig.tight_layout()

    if save:
        vis_dir = get_gc_output_dir("visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        tr_num = trial.replace("Tr_", "")
        out = vis_dir / f"group_gc_comparison_Tr{tr_num}_Win{window_idx + 1}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)

    return fig


# ---------------------------------------------------------------------------
# 7.2.2 — Brain-topology connection diagram
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
    """
    Draw a curved directed arrow from src_xy to tgt_xy with thickness ∝ weight.
    Uses a quadratic Bezier via FancyArrowPatch with connectionstyle='arc3,rad=0.2'.
    Arrow is clipped so it starts/ends at the node edge (not center).
    """
    sx, sy = src_xy
    tx, ty = tgt_xy
    dx, dy = tx - sx, ty - sy
    dist = np.hypot(dx, dy)
    if dist < 1e-6 or weight < 1e-6:
        return

    # Offset start/end by node_radius along the connecting vector
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


def plot_group_topology(
    B_Exp: dict,
    B_Nov: dict,
    conn_labels: list[str],
    trial: str = "Tr_1",
    window_idx: int = 0,
    threshold_pct: float = 40.0,
    save: bool = True,
) -> plt.Figure:
    """
    Brain-topology directed-arrow diagrams for Expert and Novice side-by-side.

    ROI nodes are placed at anatomical positions on a 2-D head outline.
    Directed arrows are drawn for connections with magnitude above the
    ``threshold_pct`` percentile (of all 20 connections), scaled by magnitude.
    The left panel shows Expert connectivity, the right shows Novice.

    Parameters
    ----------
    B_Exp, B_Nov    : dict  {trial_id: array(n_subs, 20, n_wins)}
    conn_labels     : list[str]  20 connection labels
    trial           : str   default 'Tr_1'
    window_idx      : int   0-indexed, default 0 → Win_1
    threshold_pct   : float percentile threshold for displaying arrows (default 40)
    save            : bool

    Returns
    -------
    fig : plt.Figure
    """
    win_label = f"Win_{window_idx + 1}"

    exp_arr = B_Exp[trial][:, :, window_idx]   # (n_exp, 20)
    nov_arr = B_Nov[trial][:, :, window_idx]   # (n_nov, 20)
    exp_mean = np.nanmean(exp_arr, axis=0)     # (20,)
    nov_mean = np.nanmean(nov_arr, axis=0)

    # Shared normalization across both groups for comparable arrow weights
    combined = np.concatenate([exp_mean, nov_mean])
    gc_max = float(np.nanmax(combined)) if np.any(~np.isnan(combined)) else 1.0
    gc_max = max(gc_max, 0.01)

    # Threshold: show only connections above percentile
    threshold = float(np.nanpercentile(combined, threshold_pct))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))

    for ax, gc_mean, group_name, group_color in [
        (axes[0], exp_mean, "Expert", _EXPERT_COLOR),
        (axes[1], nov_mean, "Novice", _NOVICE_COLOR),
    ]:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.5, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")

        # Head outline
        head_circle = plt.Circle((0, 0.35), 1.0, fill=False, ec="#CCCCCC",
                                  lw=2.0, zorder=0)
        ax.add_patch(head_circle)
        # Nose marker
        ax.plot([0], [1.36], "^", ms=8, color="#CCCCCC", zorder=1)

        # Draw arrows for all connections above threshold
        gc_mat = _vec20_to_mat5(gc_mean)   # (5, 5)
        arrow_cmap = plt.get_cmap(_CMAP_TOPO)

        for src_idx, src_roi in enumerate(ROI_NAMES):
            for tgt_idx, tgt_roi in enumerate(ROI_NAMES):
                if src_idx == tgt_idx:
                    continue
                w = gc_mat[src_idx, tgt_idx]
                if w < threshold or np.isnan(w):
                    continue
                norm_w = w / gc_max
                rgba = arrow_cmap(0.3 + 0.7 * norm_w)  # avoid too-light colors
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

        # Draw nodes on top
        for roi, (px, py) in _ROI_POSITIONS.items():
            circle = plt.Circle((px, py), 0.09, color=group_color,
                                  zorder=5, alpha=0.95, ec="white", lw=1.8)
            ax.add_patch(circle)
            ax.text(px, py, roi, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=6)

        n_shown = int(np.sum((gc_mean >= threshold) & ~np.isnan(gc_mean)))
        ax.set_title(
            f"{group_name}  [{trial} {win_label}]\n"
            f"{n_shown}/20 connections shown (≥{threshold_pct:.0f}th pctl)",
            fontsize=11, fontweight="bold", color=group_color, pad=10,
        )

    # Shared colorbar for arrow color scale
    norm = Normalize(vmin=0, vmax=gc_max)
    sm = ScalarMappable(
        cmap=plt.get_cmap(_CMAP_TOPO),
        norm=Normalize(vmin=0.3 * gc_max, vmax=gc_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.03, shrink=0.55)
    cbar.set_label("GC magnitude (0.01–0.07 Hz)", fontsize=9)

    fig.suptitle(
        "Group GC Connectivity Topology\n"
        "Nodes: anatomical ROIs  |  Arrows: directed causal influence  |  "
        "Arrow weight ∝ GC magnitude",
        fontsize=11, y=1.00,
    )

    if save:
        vis_dir = get_gc_output_dir("visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        tr_num = trial.replace("Tr_", "")
        out = vis_dir / f"group_gc_topology_Tr{tr_num}_Win{window_idx + 1}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out.name)
        # Also save separate Expert / Novice pngs
        for i, g in enumerate(["Expert", "Novice"]):
            fig_single, ax_single = plt.subplots(figsize=(7, 7.5))
            ax_single.set_xlim(-1.1, 1.1)
            ax_single.set_ylim(-0.5, 1.25)
            ax_single.set_aspect("equal")
            ax_single.axis("off")

            gc_mean = exp_mean if g == "Expert" else nov_mean
            group_color = _EXPERT_COLOR if g == "Expert" else _NOVICE_COLOR
            gc_mat = _vec20_to_mat5(gc_mean)
            arrow_cmap = plt.get_cmap(_CMAP_TOPO)

            head_circle = plt.Circle((0, 0.35), 1.0, fill=False, ec="#CCCCCC",
                                      lw=2.0, zorder=0)
            ax_single.add_patch(head_circle)
            ax_single.plot([0], [1.36], "^", ms=8, color="#CCCCCC", zorder=1)

            for src_idx, src_roi in enumerate(ROI_NAMES):
                for tgt_idx, tgt_roi in enumerate(ROI_NAMES):
                    if src_idx == tgt_idx:
                        continue
                    w = gc_mat[src_idx, tgt_idx]
                    if w < threshold or np.isnan(w):
                        continue
                    norm_w = w / gc_max
                    rgba = arrow_cmap(0.3 + 0.7 * norm_w)
                    _draw_curved_arrow(
                        ax_single,
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
                ax_single.add_patch(circle)
                ax_single.text(px, py, roi, ha="center", va="center",
                               fontsize=8.5, fontweight="bold", color="white", zorder=6)

            n_shown = int(np.sum((gc_mean >= threshold) & ~np.isnan(gc_mean)))
            ax_single.set_title(
                f"{g}  [{trial} {win_label}]\n"
                f"{n_shown}/20 connections shown (≥{threshold_pct:.0f}th pctl)",
                fontsize=12, fontweight="bold", color=group_color, pad=12,
            )
            cbar_s = fig_single.colorbar(
                ScalarMappable(
                    cmap=plt.get_cmap(_CMAP_TOPO),
                    norm=Normalize(vmin=0.3 * gc_max, vmax=gc_max),
                ),
                ax=ax_single, fraction=0.04, pad=0.02, shrink=0.6,
            )
            cbar_s.set_label("GC magnitude", fontsize=9)
            fig_single.tight_layout()
            out_s = vis_dir / f"group_gc_topology_{g}_Tr{tr_num}_Win{window_idx + 1}.png"
            fig_single.savefig(out_s, dpi=300, bbox_inches="tight")
            plt.close(fig_single)
            logger.info("Saved %s", out_s.name)

    return fig


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_visualizations(study: str = "expert_novice") -> None:
    """Generate all Section 7 visualizations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    vis_dir = get_gc_output_dir("visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    B_Exp, B_Nov, conn_labels = _load_group_arrays()

    # Load H1 results for FDR markers
    stats_dir = get_gc_output_dir("statistics")
    h1_df = None
    h1_path = stats_dir / "H1_expert_vs_novice.csv"
    if h1_path.exists():
        h1_df = pd.read_csv(h1_path)
        logger.info("Loaded H1 results from %s", h1_path.name)
    else:
        logger.warning("H1 results not found — bar chart will have no FDR markers")

    # ---- 7.1.1: A1 Tr_1 Win_1 single connectivity matrix ----
    logger.info("7.1.1 — Generating A1 Tr_1 Win_1 connectivity heatmap...")
    gc_dir = get_gc_output_dir("gc_connectivity")
    a1_data = np.load(gc_dir / "sub-A1_gc_connectivity.npz", allow_pickle=True)
    a1_vec = a1_data["Tr_1_connectivity"][:, 0]   # Win_1
    fig = plot_connectivity_matrix(
        a1_vec, subject_id="A1", trial="Tr_1", window="Win_1", group="Expert",
    )
    plt.close(fig)

    # ---- 7.1.2: A1 Tr_1 connectivity evolution (all windows) ----
    logger.info("7.1.2 — Generating A1 Tr_1 connectivity evolution...")
    fig = plot_connectivity_evolution(subject_id="A1", trial="Tr_1")
    plt.close(fig)

    # ---- 7.2.1: Group GC bar chart — Tr_1 Win_1 ----
    logger.info("7.2.1 — Generating group GC comparison bar chart (Tr_1 Win_1)...")
    fig = plot_group_gc_comparison(
        B_Exp, B_Nov, conn_labels,
        trial="Tr_1", window_idx=0, h1_df=h1_df,
    )
    plt.close(fig)

    # Also generate for Tr_5 Win_1 (contains the single significant connection)
    logger.info("7.2.1 — Generating group GC comparison bar chart (Tr_5 Win_1)...")
    fig = plot_group_gc_comparison(
        B_Exp, B_Nov, conn_labels,
        trial="Tr_5", window_idx=0, h1_df=h1_df,
    )
    plt.close(fig)

    # ---- 7.2.2: Brain topology diagram — Tr_1 Win_1 ----
    logger.info("7.2.2 — Generating brain topology diagram (Tr_1 Win_1)...")
    fig = plot_group_topology(
        B_Exp, B_Nov, conn_labels,
        trial="Tr_1", window_idx=0,
    )
    plt.close(fig)

    # List all generated visualizations
    vis_files = sorted(vis_dir.glob("*.png"))
    logger.info("Section 7 complete. %d PNG files in visualizations/:", len(vis_files))
    for f in vis_files:
        logger.info("  %s", f.name)


if __name__ == "__main__":
    run_all_visualizations()
