"""
Stage E — Phase 4: Visualization for Amplitude dSTE (Section 9.9).

Four plot families are implemented:

1. ``plot_connectivity_heatmaps_p4()``
   Expert-mean and Novice-mean 5×5 directed connectivity matrices rendered as
   annotated heatmaps, one figure per trial (× 2 groups).  Uses the
   subject-mean group arrays for a fair per-subject comparison.

2. ``plot_expert_novice_delta_p4()``
   Bar chart of ``Expert_mean − Novice_mean`` across the 20 canonical
   directed connections, averaged over trials.  Analogous to Stage D's GC
   figure.  Bars are coloured by sign.

3. ``plot_bivariate_vs_multivariate_p4()``
   Side-by-side overlay of Phase 3's bivariate amplitude branch TE (no
   conditioning) versus Phase 4's multivariate conditional amplitude TE for
   the same 20 connections.  Quantifies indirect-connection suppression by
   multivariate conditioning — expected ~40–50% TE reduction on spurious
   connections (consistent with Phase 1 → Phase 2 suppression ratio).

4. ``plot_expert_novice_grouped_p4()``
   Paired Expert / Novice bars per connection (subject_mean strategy, SEM
   error bars), mirroring the Phase 2 ``plot_group_te_comparison_p2()`` style.
   FDR significance markers loaded from Phase 4 statistics CSV when available.

All figures are saved to ``phase_4_amplitude_dste/visualizations/`` using
the naming convention in ``docs/OUTPUTS.md``:

- ``group_te_heatmap_expert_vs_novice_Tr{N}.png``
- ``group_te_delta_expert_vs_novice.png``
- ``group_te_bar_bivariate_vs_conditional.png``
- ``group_te_bar_expert_vs_novice.png``

**Params caption (2026-08-28, extended 2026-08-28):** every figure gets a
small caption below the plot summarising the parameters that produced it
(k0_max, nDis threshold, PCA conditioning, sparsity threshold, window size,
u_max, envelope normalization, embedding dimension, narrowband filter
on/off). This is read from ``phase_4_group/run_metadata.json``
(written by ``15_amplitude_dste_group.py::consolidate_p4_run_metadata()``)
rather than from ``config.yaml`` directly, so the caption always reflects
what actually produced the connectivity data on disk — even if
``config.yaml`` has since been edited without re-running the pipeline. If
subjects were processed under different parameter values (e.g. only some
were reprocessed after a config change), the caption shows a red "MIXED
PARAMS" warning instead of silently picking one.

CLI
---
python3 16_amplitude_dste_visualization.py --plot all
python3 16_amplitude_dste_visualization.py --plot heatmaps
python3 16_amplitude_dste_visualization.py --plot delta
python3 16_amplitude_dste_visualization.py --plot bivariate-comparison
python3 16_amplitude_dste_visualization.py --plot group-te
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_ROI_NAMES = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_N_ROIS = 5
_EXPERT_COLOR = "#1F618D"
_NOVICE_COLOR = "#C0392B"
_P4_COLOR = "#1A5276"
_P3_COLOR = "#B7950B"


# ---------------------------------------------------------------------------
# Shared helpers
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


def _p4_viz_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir
    d = get_te_output_dir("phase_4_viz", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _p4_group_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir
    return get_te_output_dir("phase_4_group", config=config)


def _p3_conn_dir(config: Optional[dict] = None) -> Path:
    from utils_te import get_te_output_dir
    return get_te_output_dir("phase_3_connectivity", config=config)


# ---------------------------------------------------------------------------
# Params caption (2026-08-28)
#
# Rather than reading config.yaml directly — which may have been edited
# since the data on disk was actually produced — every Phase 4 figure reads
# the consolidated ``run_metadata.json`` written by
# ``15_amplitude_dste_group.py::consolidate_p4_run_metadata()``. That file
# is built from the ``params_json`` embedded in each subject's .npz at
# processing time, so the caption always reflects what actually generated
# the plotted numbers.
# ---------------------------------------------------------------------------

_CAPTION_FIELD_LABELS = [
    ("k0_max", "k0_max"),
    ("ndis_threshold", "nDis_thr"),
    ("conditional_pca_enabled", "PCA"),
    ("conditional_pca_n_components", "PCA_ncomp"),
    ("sparsity_fallback_threshold", "sparsity_thr"),
    ("window_size_samples", "window_samp"),
    ("u_max", "u_max"),
    ("envelope_normalize", "envelope_norm"),
    ("embedding_dimension", "embed_dim"),
    ("narrowband_filter", "narrowband"),
]

# Fields rendered as on/off rather than their raw (typically boolean) value.
_CAPTION_BOOLEAN_ONOFF_FIELDS = {
    "conditional_pca_enabled",
    "narrowband_filter",
}


def _load_p4_run_metadata(config: Optional[dict] = None) -> Optional[dict]:
    """
    Load the consolidated Phase 4 run-metadata JSON, if it exists.

    Returns ``None`` (with a logged warning) if
    ``phase_4_group/run_metadata.json`` has not yet been generated — the
    caller should render a helpful placeholder rather than silently
    omitting the caption.
    """
    gdir = _p4_group_dir(config)
    path = gdir / "run_metadata.json"
    if not path.exists():
        logger.warning(
            "_load_p4_run_metadata: %s not found — run "
            "`python3 15_amplitude_dste_group.py --strategy metadata` (or "
            "`--strategy all`) to generate it. Plots will show a placeholder "
            "caption instead of real parameters.", path,
        )
        return None
    with open(path) as fh:
        return json.load(fh)


def _format_params_caption(run_meta: Optional[dict]) -> str:
    """
    Build a one-line caption string of the parameters that produced the
    plotted Phase 4 data, for display below the figure.
    """
    if run_meta is None:
        return (
            "⚠ Params unavailable — run 15_amplitude_dste_group.py "
            "(--strategy metadata) to generate run_metadata.json"
        )

    params = run_meta.get("params", {})
    parts = []
    for key, label in _CAPTION_FIELD_LABELS:
        if key not in params:
            continue
        val = params[key]
        if key in _CAPTION_BOOLEAN_ONOFF_FIELDS:
            parts.append(f"{label}={'on' if val else 'off'}")
        else:
            parts.append(f"{label}={val}")

    n_subj = run_meta.get("n_subjects_with_params", "?")
    n_total = run_meta.get("n_subjects_total", "?")
    caption = f"Phase 4 params (n={n_subj}/{n_total} subjects): " + " | ".join(parts)

    if run_meta.get("has_mismatch"):
        mismatched = sorted(run_meta.get("mismatched_fields", {}).keys())
        caption += f"  ⚠ MIXED PARAMS across subjects: {', '.join(mismatched)}"

    return caption


def _add_params_caption(fig: plt.Figure, config: Optional[dict] = None) -> str:
    """
    Render the Phase 4 params caption at the bottom of a figure.

    Uses ``fig.text()`` (figure-relative coordinates, not axes-relative) so
    it appears below all subplots regardless of layout. Returns the caption
    string that was rendered (useful for also writing it into the
    accompanying summary JSON).
    """
    run_meta = _load_p4_run_metadata(config)
    caption = _format_params_caption(run_meta)
    color = "#943126" if (run_meta is None or run_meta.get("has_mismatch")) else "#5D6D7E"

    # Remove any prior caption before adding a new one, mirroring
    # utils.add_plot_timestamp()'s gid-tagging convention — prevents
    # stacked/duplicate captions if a plot function is re-invoked on the
    # same figure object.
    for txt in list(getattr(fig, "texts", [])):
        if getattr(txt, "get_gid", lambda: None)() == "autogen_p4_params_caption":
            txt.remove()

    t = fig.text(
        0.5, -0.02, caption,
        ha="center", va="top", fontsize=7.5, color=color,
        transform=fig.transFigure, wrap=True,
    )
    t.set_gid("autogen_p4_params_caption")
    return caption


def _connection_labels() -> list[str]:
    from utils_te import get_connection_labels
    return get_connection_labels()


def _subject_ids(config: dict) -> list[str]:
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    return list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))


def _load_p4_mean_arrays(config: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Phase 4 subject-mean group arrays ``(n_subs, 20, 5)``."""
    gdir = _p4_group_dir(config)
    exp = np.load(gdir / "B_Exp_amplitude_mean.npz", allow_pickle=True)["data"]
    nov = np.load(gdir / "B_Nov_amplitude_mean.npz", allow_pickle=True)["data"]
    return exp, nov, _connection_labels()


def _vec20_to_mat5(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 5×5 directed connectivity matrix from a 20-element vector.

    The canonical ordering removes the diagonal, so 25 − 5 = 20 elements
    map to the 20 off-diagonal cells in row-major order.

    Parameters
    ----------
    vec : np.ndarray, shape (20,)

    Returns
    -------
    np.ndarray, shape (5, 5)
        mat[src, tgt]; diagonal is zero.
    """
    mat = np.zeros((_N_ROIS, _N_ROIS), dtype=np.float64)
    idx = 0
    for r in range(_N_ROIS):
        for c in range(_N_ROIS):
            if r != c:
                mat[r, c] = vec[idx]
                idx += 1
    return mat


def _group_trial_mean_mat(
    B: np.ndarray,
    trial_idx: int,
) -> np.ndarray:
    """
    Compute the group-mean 5×5 matrix for one trial from a (n_subs, 20, 5) array.

    Parameters
    ----------
    B : np.ndarray, shape (n_subs, 20, 5)
    trial_idx : int

    Returns
    -------
    np.ndarray, shape (5, 5)
    """
    vec_mean = np.nanmean(B[:, :, trial_idx], axis=0)
    return _vec20_to_mat5(vec_mean)


# ---------------------------------------------------------------------------
# Plot 1 — Connectivity heatmaps
# ---------------------------------------------------------------------------

def plot_connectivity_heatmaps_p4(
    save: bool = True,
    config: Optional[dict] = None,
) -> list[plt.Figure]:
    """
    Plot Expert-mean and Novice-mean 5×5 amplitude dSTE heatmaps per trial.

    One figure is produced per trial (5 trials = 5 figures), each containing
    two subplots (Expert | Novice).

    Parameters
    ----------
    save : bool
        If ``True``, save PNG files to ``phase_4_visualizations/``.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure per trial.
    """
    if config is None:
        config = load_config()

    B_Exp, B_Nov, conn_labels = _load_p4_mean_arrays(config)
    n_trials = B_Exp.shape[2]
    viz_dir = _p4_viz_dir(config) if save else None
    figs: list[plt.Figure] = []

    global_max = max(
        float(np.nanmax(np.abs(B_Exp))),
        float(np.nanmax(np.abs(B_Nov))),
    )
    vmax = global_max if global_max > 0 else 1.0

    for t_idx in range(n_trials):
        trial_id = f"Tr_{t_idx + 1}"
        exp_mat = _group_trial_mean_mat(B_Exp, t_idx)
        nov_mat = _group_trial_mean_mat(B_Nov, t_idx)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, mat, group_name, color in [
            (axes[0], exp_mat, "Expert", _EXPERT_COLOR),
            (axes[1], nov_mat, "Novice", _NOVICE_COLOR),
        ]:
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")

            for r in range(_N_ROIS):
                for c in range(_N_ROIS):
                    if r == c:
                        continue
                    val = mat[r, c]
                    text_color = "white" if val > 0.5 * vmax else "black"
                    ax.text(
                        c, r, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold",
                    )

            ax.set_xticks(range(_N_ROIS))
            ax.set_yticks(range(_N_ROIS))
            ax.set_xticklabels(_ROI_NAMES, fontsize=9)
            ax.set_yticklabels(_ROI_NAMES, fontsize=9)
            ax.set_xlabel("Target ROI", fontsize=9)
            ax.set_ylabel("Source ROI", fontsize=9)
            ax.set_title(
                f"{group_name} — {trial_id}\nAmplitude dSTE (multivariate conditional)",
                fontsize=10, fontweight="bold", color=color,
            )

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Mean TE (nats)")

        fig.suptitle(
            f"Phase 4 Amplitude dSTE Connectivity — {trial_id}\n"
            f"Group means (n_Exp={B_Exp.shape[0]}, n_Nov={B_Nov.shape[0]})",
            fontsize=11, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _add_params_caption(fig, config)

        if save and viz_dir is not None:
            out = viz_dir / f"group_te_heatmap_expert_vs_novice_Tr{t_idx + 1}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            logger.info("plot_connectivity_heatmaps_p4: saved %s", out.name)

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Plot 2 — Expert vs Novice delta bar chart
# ---------------------------------------------------------------------------

def plot_expert_novice_delta_p4(
    save: bool = True,
    config: Optional[dict] = None,
) -> tuple[plt.Figure, dict]:
    """
    Plot Expert-mean minus Novice-mean amplitude dSTE across 20 connections.

    Each bar represents ``(Expert_mean − Novice_mean)`` averaged over all
    five trials.  Positive bars = Expert > Novice; negative = Novice > Expert.
    Bars are colour-coded by sign.

    Parameters
    ----------
    save : bool
        If ``True``, save figure + JSON to ``phase_4_visualizations/``.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict]
        Figure and metadata dictionary.
    """
    if config is None:
        config = load_config()

    B_Exp, B_Nov, conn_labels = _load_p4_mean_arrays(config)
    n_conn = len(conn_labels)

    exp_mean_all = np.nanmean(B_Exp, axis=(0, 2))   # (20,)
    nov_mean_all = np.nanmean(B_Nov, axis=(0, 2))   # (20,)
    delta = exp_mean_all - nov_mean_all              # (20,)

    colors = [_EXPERT_COLOR if d >= 0 else _NOVICE_COLOR for d in delta]
    x = np.arange(n_conn)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    bars = ax.bar(x, delta, color=colors, alpha=0.85, edgecolor="#1B2631", linewidth=0.4)

    ax.axhline(0, color="#1B2631", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Expert − Novice mean TE (nats)", fontsize=11)
    ax.set_xlabel("Directed connection", fontsize=10)
    ax.set_title(
        "Phase 4 Amplitude dSTE: Expert vs Novice Delta\n"
        f"(n_Exp={B_Exp.shape[0]}, n_Nov={B_Nov.shape[0]}, mean over all trials)",
        fontsize=11, fontweight="bold",
    )

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_EXPERT_COLOR, alpha=0.85, label="Expert > Novice"),
        Patch(facecolor=_NOVICE_COLOR, alpha=0.85, label="Novice > Expert"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    params_caption = _add_params_caption(fig, config)

    metadata = {
        "connection_labels": conn_labels,
        "expert_mean": exp_mean_all.tolist(),
        "novice_mean": nov_mean_all.tolist(),
        "delta": delta.tolist(),
        "n_exp_gt_nov": int(np.sum(delta > 0)),
        "n_nov_gt_exp": int(np.sum(delta < 0)),
        "n_equal": int(np.sum(delta == 0)),
        "params_caption": params_caption,
    }

    if save:
        viz_dir = _p4_viz_dir(config)
        fig_path = viz_dir / "group_te_delta_expert_vs_novice.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        with open(viz_dir / "group_te_delta_expert_vs_novice_summary.json", "w") as fh:
            json.dump(metadata, fh, indent=2, cls=_NumpyEncoder)
        logger.info("plot_expert_novice_delta_p4: saved %s", fig_path.name)

    return fig, metadata


# ---------------------------------------------------------------------------
# Plot 3 — Bivariate vs multivariate conditional comparison
# ---------------------------------------------------------------------------

def _load_p3_amplitude_branch_mean(
    config: dict,
) -> Optional[np.ndarray]:
    """
    Load Phase 3 amplitude-branch per-subject .npz files and return the
    cohort-mean connectivity vector ``(20,)`` averaged over subjects and trials.

    Phase 3's amplitude branch is *bivariate* (no multivariate conditioning).
    This is the baseline against which Phase 4's multivariate conditioning is compared.

    Returns ``None`` if Phase 3 files are not found.
    """
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    all_ids = list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))
    conn_dir = _p3_conn_dir(config)
    n_conn = len(_connection_labels())
    _N_TRIALS = 5

    vectors: list[np.ndarray] = []
    for sid in all_ids:
        fpath = conn_dir / f"sub-{sid}_amplitude_branch_connectivity.npz"
        if not fpath.exists():
            continue
        npz = dict(np.load(fpath, allow_pickle=True))
        for t_idx in range(_N_TRIALS):
            tid = f"Tr_{t_idx + 1}"
            key = f"{tid}_connectivity"
            if key not in npz:
                continue
            conn = npz[key]
            if conn.ndim == 2 and conn.shape[1] > 0:
                vectors.append(np.nanmean(conn, axis=1))

    if not vectors:
        logger.warning(
            "_load_p3_amplitude_branch_mean: no Phase 3 amplitude-branch files "
            "found in %s — skipping bivariate comparison.", conn_dir
        )
        return None

    return np.nanmean(np.stack(vectors, axis=0), axis=0)


def _load_p4_amplitude_cohort_mean(config: dict) -> np.ndarray:
    """
    Compute the Phase 4 cohort-mean connectivity vector ``(20,)`` from
    the subject-mean group arrays, averaged over groups and trials.
    """
    B_Exp, B_Nov, _ = _load_p4_mean_arrays(config)
    all_means = np.concatenate(
        [np.nanmean(B_Exp, axis=2), np.nanmean(B_Nov, axis=2)], axis=0
    )
    return np.nanmean(all_means, axis=0)


def plot_bivariate_vs_multivariate_p4(
    save: bool = True,
    config: Optional[dict] = None,
) -> tuple[plt.Figure, dict]:
    """
    Plot Phase 3 bivariate vs Phase 4 multivariate amplitude TE per connection.

    Shows cohort-mean TE for:
    - Phase 3 amplitude branch  (bivariate, no conditioning)
    - Phase 4 amplitude dSTE   (multivariate conditional)

    A reduction in TE from Phase 3 → Phase 4 on a connection indicates that
    the connection was at least partially driven by indirect paths that the
    multivariate conditioning suppresses.  Expected suppression ~40–50% for
    indirect connections (Phase 1 → Phase 2 ratio).

    Parameters
    ----------
    save : bool
        If ``True``, save figure + JSON to ``phase_4_visualizations/``.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict]
        Figure and metadata dictionary, or an empty dict if Phase 3 data
        is unavailable.
    """
    if config is None:
        config = load_config()

    conn_labels = _connection_labels()
    n_conn = len(conn_labels)

    p4_mean = _load_p4_amplitude_cohort_mean(config)
    p3_mean = _load_p3_amplitude_branch_mean(config)

    if p3_mean is None:
        logger.warning(
            "plot_bivariate_vs_multivariate_p4: Phase 3 amplitude branch data "
            "unavailable — generating Phase 4 cohort-mean bar chart only."
        )
        p3_available = False
    else:
        p3_available = True

    x = np.arange(n_conn)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.bar(
        x - bar_w / 2 if p3_available else x,
        np.nan_to_num(p4_mean),
        width=bar_w if p3_available else 0.7,
        color=_P4_COLOR,
        alpha=0.85,
        label="Phase 4 — multivariate conditional amplitude dSTE",
        edgecolor="#1B2631",
        linewidth=0.35,
    )

    if p3_available and p3_mean is not None:
        ax.bar(
            x + bar_w / 2,
            np.nan_to_num(p3_mean),
            width=bar_w,
            color=_P3_COLOR,
            alpha=0.75,
            hatch="///",
            label="Phase 3 — bivariate amplitude dSTE (no conditioning)",
            edgecolor="#1B2631",
            linewidth=0.35,
        )

        # Suppression ratio annotation line
        ratio = np.where(
            p3_mean > 0,
            (p3_mean - p4_mean) / p3_mean * 100,
            np.nan,
        )
        mean_suppression = float(np.nanmean(ratio[ratio > 0])) if np.any(ratio > 0) else 0.0
        ax.text(
            0.98, 0.97,
            f"Mean suppression (P3→P4): {mean_suppression:.1f}%\n"
            "(connections where P4 < P3)",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox={"facecolor": "#EBF5FB", "edgecolor": "#5DADE2",
                  "alpha": 0.9, "boxstyle": "round,pad=0.3"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Mean TE (nats)", fontsize=11)
    ax.set_xlabel("Directed connection", fontsize=10)

    title = (
        "Phase 4 vs Phase 3: Multivariate Conditioning Effect on Amplitude TE\n"
        "Blue = Phase 4 (multivariate conditional),  "
        "Hatched = Phase 3 (bivariate, no conditioning)"
    )
    if not p3_available:
        title = "Phase 4 Amplitude dSTE — Cohort Mean per Connection"
    ax.set_title(title, fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    params_caption = _add_params_caption(fig, config)

    metadata: dict = {
        "connection_labels": conn_labels,
        "p4_mean": p4_mean.tolist(),
        "p3_mean": p3_mean.tolist() if p3_available and p3_mean is not None else None,
        "p3_available": p3_available,
        "params_caption": params_caption,
    }
    if p3_available and p3_mean is not None:
        metadata["suppression_pct"] = np.where(
            p3_mean > 0,
            (p3_mean - p4_mean) / p3_mean * 100,
            np.nan,
        ).tolist()
        metadata["mean_suppression_pct"] = float(
            np.nanmean([v for v in metadata["suppression_pct"]
                        if v is not None and not np.isnan(v) and v > 0])
        ) if any(
            v is not None and not np.isnan(v) and v > 0
            for v in metadata["suppression_pct"]
        ) else 0.0

    if save:
        viz_dir = _p4_viz_dir(config)
        fig_path = viz_dir / "group_te_bar_bivariate_vs_conditional.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        with open(viz_dir / "group_te_bar_bivariate_vs_conditional_summary.json", "w") as fh:
            json.dump(metadata, fh, indent=2, cls=_NumpyEncoder)
        logger.info("plot_bivariate_vs_multivariate_p4: saved %s", fig_path.name)

    return fig, metadata


# ---------------------------------------------------------------------------
# Plot 4 — Expert vs Novice grouped bar chart (mirrors Phase 2 style)
# ---------------------------------------------------------------------------

def _p4_group_stats(
    subj_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute mean and SEM across subjects for each of the 20 connections.

    Parameters
    ----------
    subj_matrix : np.ndarray, shape (n_subs, 20)
        Per-subject trial-averaged TE vectors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(mean_vec, sem_vec, n_subjects)`` each of length 20.
    """
    n = subj_matrix.shape[0]
    mean_vec = np.nanmean(subj_matrix, axis=0)
    std_vec  = np.nanstd(subj_matrix, axis=0, ddof=1)
    sem_vec  = std_vec / np.sqrt(n) if n > 1 else np.zeros_like(std_vec)
    return mean_vec, sem_vec, n


def plot_expert_novice_grouped_p4(
    save: bool = True,
    config: Optional[dict] = None,
) -> tuple[plt.Figure, dict]:
    """
    Paired Expert / Novice bar chart of mean amplitude dSTE per connection.

    Mirrors Phase 2's ``plot_group_te_comparison_p2()`` — two bars per
    connection (Expert in steel blue, Novice in brick red) with SEM error
    bars across subjects, using the subject_mean strategy (trial-averaged per
    subject so every subject contributes equally regardless of trial count).

    FDR significance markers are overlaid when
    ``phase_4_amplitude_dste/statistics/H1_expert_vs_novice.csv`` exists:
    - ``*``  — FDR-significant under primary (subject_mean) strategy
    - ``†``  — significant under sensitivity (first_window) only

    Parameters
    ----------
    save : bool
        If ``True``, save figure + JSON to ``phase_4_visualizations/``.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict]
        Figure and metadata dictionary.
    """
    if config is None:
        config = load_config()

    _EXP_COL = "#1F618D"
    _NOV_COL = "#C0392B"

    B_Exp, B_Nov, conn_labels = _load_p4_mean_arrays(config)
    n_conn = len(conn_labels)

    # Subject-mean strategy: average over trial axis → (n_subs, 20)
    exp_subj = np.nanmean(B_Exp, axis=2)
    nov_subj = np.nanmean(B_Nov, axis=2)

    exp_mean, exp_sem, n_exp = _p4_group_stats(exp_subj)
    nov_mean, nov_sem, n_nov = _p4_group_stats(nov_subj)

    # ------------------------------------------------------------------
    # Optional FDR significance markers from Phase 4 statistics CSV
    # ------------------------------------------------------------------
    sig_primary: set = set()
    sig_fw_only: set = set()
    stats_loaded = False
    try:
        import pandas as pd
        from utils_te import get_te_output_dir
        stats_dir = get_te_output_dir("phase_4_statistics", config=config, create=False)
        h1_path = stats_dir / "H1_expert_vs_novice.csv"
        if h1_path.exists():
            h1_df = pd.read_csv(h1_path)
            sig_primary = set(
                h1_df.loc[h1_df["significant"] == True, "connection"].tolist()
            )
            if "significant_fw" in h1_df.columns:
                sig_fw = set(
                    h1_df.loc[h1_df["significant_fw"] == True, "connection"].tolist()
                )
                sig_fw_only = sig_fw - sig_primary
            stats_loaded = True
            logger.info(
                "plot_expert_novice_grouped_p4: loaded %d primary / %d FW-only "
                "significant connections from %s",
                len(sig_primary), len(sig_fw_only), h1_path.name,
            )
    except Exception as exc:
        logger.debug(
            "plot_expert_novice_grouped_p4: Phase 4 stats not loaded (%s). "
            "Significance markers will be omitted.", exc,
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    x     = np.arange(n_conn)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(16, 5.5))

    ax.bar(
        x - bar_w / 2, exp_mean, bar_w,
        color=_EXP_COL, alpha=0.88,
        label=f"Expert (n={n_exp})",
        yerr=exp_sem, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#1B2631"},
        edgecolor="#1B2631", linewidth=0.35,
    )
    ax.bar(
        x + bar_w / 2, nov_mean, bar_w,
        color=_NOV_COL, alpha=0.88,
        label=f"Novice (n={n_nov})",
        yerr=nov_sem, capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#1B2631"},
        edgecolor="#1B2631", linewidth=0.35,
    )

    # FDR significance annotations
    y_max_all = float(np.nanmax(
        np.concatenate([exp_mean + exp_sem, nov_mean + nov_sem])
    )) if n_conn > 0 else 0.05
    y_pad = 0.04 * y_max_all

    for i, lbl in enumerate(conn_labels):
        y_top = max(exp_mean[i] + exp_sem[i], nov_mean[i] + nov_sem[i]) + y_pad
        if lbl in sig_primary:
            ax.text(i, y_top, "*", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="#E74C3C")
        elif lbl in sig_fw_only:
            ax.text(i, y_top, "†", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#E67E22")

    ax.axhline(0, color="#1B2631", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Amplitude dSTE (nats)", fontsize=11)
    ax.set_xlabel("Directed connection", fontsize=10)
    ax.set_ylim(bottom=0)

    sig_note = (
        "  |  * FDR-sig primary  |  † FDR-sig sensitivity  |  BH α=0.05"
        if stats_loaded else ""
    )
    ax.set_title(
        f"Expert vs. Novice — Phase 4 Amplitude dSTE  "
        f"[All-Trial Average, subject_mean strategy]{sig_note}",
        fontsize=11, fontweight="bold", pad=8,
    )

    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    params_caption = _add_params_caption(fig, config)

    metadata = {
        "connection_labels": conn_labels,
        "expert_mean": exp_mean.tolist(),
        "expert_sem":  exp_sem.tolist(),
        "novice_mean": nov_mean.tolist(),
        "novice_sem":  nov_sem.tolist(),
        "n_exp": n_exp,
        "n_nov": n_nov,
        "stats_loaded": stats_loaded,
        "n_sig_primary": len(sig_primary),
        "n_sig_fw_only": len(sig_fw_only),
        "sig_primary_connections": sorted(sig_primary),
        "sig_fw_only_connections": sorted(sig_fw_only),
        "params_caption": params_caption,
    }

    if save:
        viz_dir = _p4_viz_dir(config)
        fig_path = viz_dir / "group_te_bar_expert_vs_novice.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        with open(viz_dir / "group_te_bar_expert_vs_novice_summary.json", "w") as fh:
            json.dump(metadata, fh, indent=2, cls=_NumpyEncoder)
        logger.info("plot_expert_novice_grouped_p4: saved %s", fig_path.name)

    return fig, metadata


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_all_p4(
    save: bool = True,
    config: Optional[dict] = None,
) -> dict:
    """
    Run all three Phase 4 visualization plots.

    Parameters
    ----------
    save : bool
        If ``True``, save all figures and JSON summaries.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    dict
        ``{plot_name: (fig_or_figs, metadata)}`` for each plot family.
    """
    if config is None:
        config = load_config()

    results: dict = {}

    logger.info("plot_all_p4: generating connectivity heatmaps …")
    figs_hm = plot_connectivity_heatmaps_p4(save=save, config=config)
    for fig in figs_hm:
        plt.close(fig)
    results["heatmaps"] = figs_hm
    logger.info("plot_all_p4: heatmaps done (%d figures).", len(figs_hm))

    logger.info("plot_all_p4: generating Expert vs Novice delta bar chart …")
    fig_delta, meta_delta = plot_expert_novice_delta_p4(save=save, config=config)
    plt.close(fig_delta)
    results["delta"] = (fig_delta, meta_delta)
    logger.info(
        "plot_all_p4: delta done — Expert>Novice=%d  Novice>Expert=%d.",
        meta_delta.get("n_exp_gt_nov", 0), meta_delta.get("n_nov_gt_exp", 0),
    )

    logger.info("plot_all_p4: generating bivariate vs multivariate comparison …")
    fig_cmp, meta_cmp = plot_bivariate_vs_multivariate_p4(save=save, config=config)
    plt.close(fig_cmp)
    results["bivariate_comparison"] = (fig_cmp, meta_cmp)
    if meta_cmp.get("p3_available"):
        logger.info(
            "plot_all_p4: comparison done — mean suppression=%.1f%%.",
            meta_cmp.get("mean_suppression_pct", 0.0),
        )
    else:
        logger.info("plot_all_p4: comparison done (Phase 3 data unavailable).")

    logger.info("plot_all_p4: generating Expert vs Novice grouped bar chart …")
    fig_grp, meta_grp = plot_expert_novice_grouped_p4(save=save, config=config)
    plt.close(fig_grp)
    results["group_te"] = (fig_grp, meta_grp)
    logger.info(
        "plot_all_p4: grouped done — n_exp=%d  n_nov=%d.",
        meta_grp.get("n_exp", 0), meta_grp.get("n_nov", 0),
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Run Section 9.9 visualization from command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stage E Phase 4 — Amplitude dSTE Visualization (Section 9.9)"
    )
    parser.add_argument(
        "--plot",
        choices=["all", "heatmaps", "delta", "bivariate-comparison", "group-te"],
        default="all",
        help="Which plot(s) to generate (default: all).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save figures or JSON to disk (display-only).",
    )
    args = parser.parse_args()

    cfg = load_config()
    do_save = not args.no_save

    if args.plot in {"all", "heatmaps"}:
        figs = plot_connectivity_heatmaps_p4(save=do_save, config=cfg)
        logger.info("heatmaps: %d figures generated.", len(figs))
        for fig in figs:
            plt.close(fig)

    if args.plot in {"all", "delta"}:
        fig, meta = plot_expert_novice_delta_p4(save=do_save, config=cfg)
        logger.info(
            "delta: Expert>Novice=%d  Novice>Expert=%d",
            meta.get("n_exp_gt_nov", 0), meta.get("n_nov_gt_exp", 0),
        )
        plt.close(fig)

    if args.plot in {"all", "bivariate-comparison"}:
        fig, meta = plot_bivariate_vs_multivariate_p4(save=do_save, config=cfg)
        if meta.get("p3_available"):
            logger.info(
                "bivariate-comparison: mean suppression=%.1f%%",
                meta.get("mean_suppression_pct", 0.0),
            )
        else:
            logger.info("bivariate-comparison: Phase 3 data not available.")
        plt.close(fig)

    if args.plot in {"all", "group-te"}:
        fig, meta = plot_expert_novice_grouped_p4(save=do_save, config=cfg)
        logger.info(
            "group-te: n_exp=%d  n_nov=%d  stats_loaded=%s",
            meta.get("n_exp", 0), meta.get("n_nov", 0), meta.get("stats_loaded"),
        )
        plt.close(fig)


if __name__ == "__main__":
    _cli_main()
