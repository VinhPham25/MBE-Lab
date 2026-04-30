"""
Stage E — Section 8.3.2: Bimodal Contribution Visualizations.

Implements two Phase 3 contribution plots:

1) Overall contribution plot
   `plot_phase_vs_amplitude_contribution()`
   Stacked bars across all 20 directed connections, averaged over all included
   subjects and windows.

2) Group-classified contribution plot
   `plot_phase_vs_amplitude_contribution_by_group()`
   For each connection, two bars are shown (Expert vs Novice), each stacked by
   weighted phase and weighted amplitude contributions. Amplitude segments use
   diagonal hatching for visual separation.
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
import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
_STAGE_E_ROOT = Path(__file__).parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_PHASE_COLOR = "#2E86C1"
_AMP_COLOR = "#27AE60"

_EXPERT_COLOR = "#1F618D"
_NOVICE_COLOR = "#C0392B"


def _p3_conn_dir(config: Optional[dict] = None) -> Path:
    """
    Return Phase 3 per-subject connectivity directory.

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


def _p3_viz_dir(config: Optional[dict] = None) -> Path:
    """
    Return Phase 3 visualization output directory.

    Parameters
    ----------
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    Path
        Output directory where Phase 3 visualizations are saved.
    """
    from utils_te import get_te_output_dir

    d = get_te_output_dir("phase_3_viz", config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _connection_labels() -> list[str]:
    """
    Return canonical 20 directed ROI connection labels.

    Returns
    -------
    list[str]
        Connection labels in off-diagonal canonical order.
    """
    from utils_te import get_connection_labels

    return get_connection_labels()


def _subject_ids_from_config(config: dict) -> list[str]:
    """
    Return concatenated Expert+Novice subject list.

    Parameters
    ----------
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    list[str]
        Subject IDs from `stage_e_transfer_entropy.subjects`.
    """
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    return list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))


def _subject_group_map(config: dict) -> dict[str, str]:
    """
    Build subject-to-group mapping from configuration.

    Parameters
    ----------
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    dict[str, str]
        Mapping of subject ID to `Expert` or `Novice`.
    """
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    out: dict[str, str] = {}
    for sid in subj_cfg.get("Expert", []):
        out[str(sid)] = "Expert"
    for sid in subj_cfg.get("Novice", []):
        out[str(sid)] = "Novice"
    return out


def _parse_weight_array(npz_data: dict, config: dict) -> tuple[float, float]:
    """
    Parse fusion weights from a fused subject `.npz`, with config fallback.

    Parameters
    ----------
    npz_data : dict
        Loaded fused subject `.npz` dictionary.
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    tuple[float, float]
        `(w_phase, w_amplitude)` normalized to sum to 1.
    """
    wp_cfg = float(
        config.get("stage_e_transfer_entropy", {})
        .get("phase_3", {})
        .get("w_phase", 0.7)
    )
    wa_cfg = float(
        config.get("stage_e_transfer_entropy", {})
        .get("phase_3", {})
        .get("w_amplitude", 0.3)
    )

    params_raw = npz_data.get("params_json", None)
    if params_raw is None:
        s = wp_cfg + wa_cfg
        return (wp_cfg / s, wa_cfg / s) if s > 0 else (0.5, 0.5)

    try:
        if isinstance(params_raw, np.ndarray):
            params_raw = params_raw.item()
        params = json.loads(str(params_raw))
        wp = float(params.get("w_phase", wp_cfg))
        wa = float(params.get("w_amplitude", wa_cfg))
    except Exception:
        wp, wa = wp_cfg, wa_cfg

    wp = max(0.0, wp)
    wa = max(0.0, wa)
    s = wp + wa
    if s <= 0:
        return 0.5, 0.5
    return wp / s, wa / s


def _load_subject_branch_npz(path: Path) -> Optional[dict]:
    """
    Load one subject branch `.npz` as a plain dictionary.

    Parameters
    ----------
    path : Path
        Branch file path.

    Returns
    -------
    Optional[dict]
        Loaded dictionary, or `None` if file does not exist.
    """
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _collect_weighted_contribution_samples(
    subjects: list[str],
    config: dict,
) -> tuple[list[dict], dict]:
    """
    Collect weighted per-window contribution vectors for all available subjects.

    Parameters
    ----------
    subjects : list[str]
        Subject IDs to include.
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    tuple[list[dict], dict]
        - List of sample dicts with keys:
          `subject_id`, `group`, `phase_weighted_vec`, `amp_weighted_vec`
        - Metadata dictionary containing inclusion counts and per-subject details.
    """
    conn_dir = _p3_conn_dir(config)
    n_conn = len(_connection_labels())
    group_map = _subject_group_map(config)

    samples: list[dict] = []
    subject_meta: dict[str, dict] = {}

    for sid in subjects:
        fused = _load_subject_branch_npz(conn_dir / f"sub-{sid}_bimodal_dspte_connectivity.npz")
        phase = _load_subject_branch_npz(conn_dir / f"sub-{sid}_phase_branch_connectivity.npz")
        amp = _load_subject_branch_npz(conn_dir / f"sub-{sid}_amplitude_branch_connectivity.npz")

        if fused is None or phase is None or amp is None:
            subject_meta[sid] = {"status": "missing_branch_files", "n_windows_used": 0}
            continue

        wp, wa = _parse_weight_array(fused, config)
        trial_ids = [str(t) for t in fused.get("trial_ids", [])]
        group = group_map.get(sid, "Unknown")

        n_windows_sub = 0
        for tr_id in trial_ids:
            key = f"{tr_id}_connectivity"
            if key not in phase or key not in amp:
                continue

            phase_arr = np.asarray(phase[key], dtype=np.float64)
            amp_arr = np.asarray(amp[key], dtype=np.float64)
            if phase_arr.ndim != 2 or amp_arr.ndim != 2:
                continue
            if phase_arr.shape[0] != n_conn or amp_arr.shape[0] != n_conn:
                continue

            n_wins = min(phase_arr.shape[1], amp_arr.shape[1])
            if n_wins <= 0:
                continue

            phase_weighted = wp * phase_arr[:, :n_wins]
            amp_weighted = wa * amp_arr[:, :n_wins]

            for i in range(n_wins):
                samples.append(
                    {
                        "subject_id": sid,
                        "group": group,
                        "phase_weighted_vec": phase_weighted[:, i],
                        "amp_weighted_vec": amp_weighted[:, i],
                    }
                )
            n_windows_sub += n_wins

        subject_meta[sid] = {
            "status": "ok" if n_windows_sub > 0 else "no_valid_windows",
            "group": group,
            "w_phase": float(wp),
            "w_amplitude": float(wa),
            "n_windows_used": int(n_windows_sub),
        }

    if not samples:
        raise RuntimeError(
            "No valid Phase 3 branch windows found for contribution plotting. "
            "Run process_subject_bimodal_dspte() / run_batch_bimodal_dspte() first."
        )

    meta = {
        "n_subjects_requested": len(subjects),
        "n_subjects_with_data": int(sum(1 for v in subject_meta.values() if v.get("status") == "ok")),
        "n_windows_total": int(len(samples)),
        "subjects": subject_meta,
    }
    return samples, meta


def _mean_or_nan(sample_list: list[np.ndarray], n_conn: int) -> np.ndarray:
    """
    Compute mean vector across samples, returning NaN vector for empty inputs.

    Parameters
    ----------
    sample_list : list[np.ndarray]
        List of vectors with shape `(n_conn,)`.
    n_conn : int
        Number of directed connections.

    Returns
    -------
    np.ndarray
        Mean vector of shape `(n_conn,)`, or all-NaN when no samples exist.
    """
    if not sample_list:
        return np.full(n_conn, np.nan, dtype=np.float64)
    return np.nanmean(np.stack(sample_list, axis=0), axis=0)


def _collect_weighted_contributions(
    subjects: list[str],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Collect overall weighted branch contributions from available subject files.

    Parameters
    ----------
    subjects : list[str]
        Subject IDs to include.
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        - Mean weighted phase contribution vector `(20,)`
        - Mean weighted amplitude contribution vector `(20,)`
        - Metadata dict with counts and subject inclusion summary
    """
    n_conn = len(_connection_labels())
    samples, meta = _collect_weighted_contribution_samples(subjects, config)

    phase_vecs = [np.asarray(s["phase_weighted_vec"], dtype=np.float64) for s in samples]
    amp_vecs = [np.asarray(s["amp_weighted_vec"], dtype=np.float64) for s in samples]

    phase_mean = _mean_or_nan(phase_vecs, n_conn)
    amp_mean = _mean_or_nan(amp_vecs, n_conn)
    return phase_mean, amp_mean, meta


def _collect_weighted_contributions_by_group(
    subjects: list[str],
    config: dict,
) -> dict:
    """
    Collect weighted branch contributions separately for Expert and Novice groups.

    Parameters
    ----------
    subjects : list[str]
        Subject IDs to include.
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    dict
        Dictionary containing per-group phase/amplitude/final means and counts.
    """
    n_conn = len(_connection_labels())
    samples, meta = _collect_weighted_contribution_samples(subjects, config)

    grouped = {
        "Expert": {"phase": [], "amp": []},
        "Novice": {"phase": [], "amp": []},
    }

    for s in samples:
        g = s.get("group", "Unknown")
        if g not in grouped:
            continue
        grouped[g]["phase"].append(np.asarray(s["phase_weighted_vec"], dtype=np.float64))
        grouped[g]["amp"].append(np.asarray(s["amp_weighted_vec"], dtype=np.float64))

    out = {
        "meta": meta,
        "Expert": {
            "phase_mean": _mean_or_nan(grouped["Expert"]["phase"], n_conn),
            "amplitude_mean": _mean_or_nan(grouped["Expert"]["amp"], n_conn),
            "n_windows": int(len(grouped["Expert"]["phase"])),
        },
        "Novice": {
            "phase_mean": _mean_or_nan(grouped["Novice"]["phase"], n_conn),
            "amplitude_mean": _mean_or_nan(grouped["Novice"]["amp"], n_conn),
            "n_windows": int(len(grouped["Novice"]["phase"])),
        },
    }
    out["Expert"]["final_mean"] = out["Expert"]["phase_mean"] + out["Expert"]["amplitude_mean"]
    out["Novice"]["final_mean"] = out["Novice"]["phase_mean"] + out["Novice"]["amplitude_mean"]
    return out


def plot_phase_vs_amplitude_contribution(
    subjects: Optional[list[str]] = None,
    save: bool = True,
    config: Optional[dict] = None,
) -> tuple[plt.Figure, dict]:
    """
    Plot weighted Phase vs Amplitude TE contributions for all 20 connections.

    Parameters
    ----------
    subjects : list[str], optional
        Optional subset of subject IDs. If `None`, all configured subjects are used.
    save : bool
        Whether to save the figure and metadata JSON to the Phase 3 viz directory.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict]
        Figure object and metadata dictionary containing data-usage summary.
    """
    if config is None:
        config = load_config()
    if subjects is None:
        subjects = _subject_ids_from_config(config)

    conn_labels = _connection_labels()
    phase_mean, amp_mean, meta = _collect_weighted_contributions(subjects, config)
    final_mean = phase_mean + amp_mean

    x = np.arange(len(conn_labels))
    fig, ax = plt.subplots(figsize=(16, 6.5))

    phase_plot = np.nan_to_num(phase_mean, nan=0.0)
    amp_plot = np.nan_to_num(amp_mean, nan=0.0)
    final_plot = np.nan_to_num(final_mean, nan=0.0)

    ax.bar(
        x,
        phase_plot,
        color=_PHASE_COLOR,
        alpha=0.92,
        label="Weighted phase contribution (w_phase · TE_phase)",
    )
    ax.bar(
        x,
        amp_plot,
        bottom=phase_plot,
        color=_AMP_COLOR,
        alpha=0.92,
        label="Weighted amplitude contribution (w_amplitude · TE_amplitude)",
    )

    ax.plot(x, final_plot, color="#1B2631", linewidth=1.5, marker="o", markersize=3, label="TE_final mean")

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Mean TE contribution (nats)", fontsize=11)
    ax.set_xlabel("Directed connection", fontsize=10)

    n_sub = meta["n_subjects_with_data"]
    n_win = meta["n_windows_total"]
    ax.set_title(
        "Phase 3 Bimodal Contributions per Connection\n"
        f"Averaged across subjects and windows (n_subjects={n_sub}, n_windows={n_win})",
        fontsize=11,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if save:
        viz_dir = _p3_viz_dir(config)
        fig_path = viz_dir / "phase_vs_amplitude_contribution.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        meta_out = {
            **meta,
            "connection_labels": conn_labels,
            "phase_contribution_mean": phase_mean.tolist(),
            "amplitude_contribution_mean": amp_mean.tolist(),
            "final_mean": final_mean.tolist(),
        }
        with open(viz_dir / "phase_vs_amplitude_contribution_summary.json", "w") as f:
            json.dump(meta_out, f, indent=2)

        logger.info("Saved %s", fig_path)

    return fig, {
        **meta,
        "connection_labels": conn_labels,
        "phase_contribution_mean": phase_mean,
        "amplitude_contribution_mean": amp_mean,
        "final_mean": final_mean,
    }


def plot_phase_vs_amplitude_contribution_by_group(
    subjects: Optional[list[str]] = None,
    save: bool = True,
    config: Optional[dict] = None,
) -> tuple[plt.Figure, dict]:
    """
    Plot Expert vs Novice stacked bimodal contributions for each directed connection.

    For each of the 20 directed connections, two bars are rendered:
      - Expert bar (left)
      - Novice bar (right)

    Each bar is stacked by weighted phase and weighted amplitude contributions.
    Amplitude segments use diagonal hatching (`///`) to preserve branch identity
    even when viewed in grayscale.

    Parameters
    ----------
    subjects : list[str], optional
        Optional subset of subject IDs. If `None`, all configured subjects are used.
    save : bool
        Whether to save the figure and metadata JSON to the Phase 3 viz directory.
    config : dict, optional
        Parsed pipeline configuration.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict]
        Figure object and grouped metadata dictionary.
    """
    if config is None:
        config = load_config()
    if subjects is None:
        subjects = _subject_ids_from_config(config)

    conn_labels = _connection_labels()
    grouped = _collect_weighted_contributions_by_group(subjects, config)

    exp_phase = grouped["Expert"]["phase_mean"]
    exp_amp = grouped["Expert"]["amplitude_mean"]
    exp_final = grouped["Expert"]["final_mean"]

    nov_phase = grouped["Novice"]["phase_mean"]
    nov_amp = grouped["Novice"]["amplitude_mean"]
    nov_final = grouped["Novice"]["final_mean"]

    x = np.arange(len(conn_labels))
    bar_w = 0.38

    exp_phase_p = np.nan_to_num(exp_phase, nan=0.0)
    exp_amp_p = np.nan_to_num(exp_amp, nan=0.0)
    exp_final_p = np.nan_to_num(exp_final, nan=0.0)

    nov_phase_p = np.nan_to_num(nov_phase, nan=0.0)
    nov_amp_p = np.nan_to_num(nov_amp, nan=0.0)
    nov_final_p = np.nan_to_num(nov_final, nan=0.0)

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.bar(
        x - bar_w / 2,
        exp_phase_p,
        width=bar_w,
        color=_EXPERT_COLOR,
        alpha=0.90,
        label="Expert phase contribution",
    )
    ax.bar(
        x - bar_w / 2,
        exp_amp_p,
        width=bar_w,
        bottom=exp_phase_p,
        color=_EXPERT_COLOR,
        alpha=0.40,
        hatch="///",
        edgecolor="#1B2631",
        linewidth=0.35,
        label="Expert amplitude contribution",
    )

    ax.bar(
        x + bar_w / 2,
        nov_phase_p,
        width=bar_w,
        color=_NOVICE_COLOR,
        alpha=0.90,
        label="Novice phase contribution",
    )
    ax.bar(
        x + bar_w / 2,
        nov_amp_p,
        width=bar_w,
        bottom=nov_phase_p,
        color=_NOVICE_COLOR,
        alpha=0.40,
        hatch="///",
        edgecolor="#1B2631",
        linewidth=0.35,
        label="Novice amplitude contribution",
    )

    ax.plot(x - bar_w / 2, exp_final_p, color="#154360", linewidth=1.2, marker="o", markersize=2.8)
    ax.plot(x + bar_w / 2, nov_final_p, color="#78281F", linewidth=1.2, marker="s", markersize=2.8)

    ax.set_xticks(x)
    ax.set_xticklabels(conn_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Mean TE contribution (nats)", fontsize=11)
    ax.set_xlabel("Directed connection", fontsize=10)

    n_exp_w = grouped["Expert"]["n_windows"]
    n_nov_w = grouped["Novice"]["n_windows"]
    ax.set_title(
        "Phase 3 Bimodal Contributions by Group (Expert vs Novice)\n"
        "Stacked by branch: solid = phase, hatched = amplitude",
        fontsize=11,
        fontweight="bold",
    )

    if n_exp_w == 0 or n_nov_w == 0:
        ax.text(
            0.01,
            0.98,
            f"Warning: windows used -> Expert={n_exp_w}, Novice={n_nov_w}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#7B241C",
            bbox={"facecolor": "#FDEDEC", "edgecolor": "#F5B7B1", "alpha": 0.9},
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()

    out_meta = {
        **grouped["meta"],
        "connection_labels": conn_labels,
        "expert": {
            "n_windows": int(n_exp_w),
            "phase_contribution_mean": exp_phase.tolist(),
            "amplitude_contribution_mean": exp_amp.tolist(),
            "final_mean": exp_final.tolist(),
        },
        "novice": {
            "n_windows": int(n_nov_w),
            "phase_contribution_mean": nov_phase.tolist(),
            "amplitude_contribution_mean": nov_amp.tolist(),
            "final_mean": nov_final.tolist(),
        },
    }

    if save:
        viz_dir = _p3_viz_dir(config)
        fig_path = viz_dir / "phase_vs_amplitude_contribution_by_group.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        with open(viz_dir / "phase_vs_amplitude_contribution_by_group_summary.json", "w") as f:
            json.dump(out_meta, f, indent=2)

        logger.info("Saved %s", fig_path)

    return fig, out_meta


def _cli_main() -> None:
    """
    Run Section 8.3.2 visualization workflows from command line.

    Returns
    -------
    None
        Writes the plot PNGs and summary JSON files when possible.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 3 bimodal contribution visualization")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional explicit subject list (e.g., A1 R3 R5).",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "overall", "grouped"],
        default="all",
        help="Which plot(s) to generate.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs")
    args = parser.parse_args()

    cfg = load_config()

    if args.plot in {"all", "overall"}:
        fig_all, meta_all = plot_phase_vs_amplitude_contribution(
            subjects=args.subjects,
            save=not args.no_save,
            config=cfg,
        )
        logger.info(
            "Overall contribution plot complete: n_subjects_with_data=%d, n_windows_total=%d",
            meta_all["n_subjects_with_data"],
            meta_all["n_windows_total"],
        )
        plt.close(fig_all)

    if args.plot in {"all", "grouped"}:
        fig_group, meta_group = plot_phase_vs_amplitude_contribution_by_group(
            subjects=args.subjects,
            save=not args.no_save,
            config=cfg,
        )
        logger.info(
            "Grouped contribution plot complete: Expert_windows=%d, Novice_windows=%d",
            meta_group["expert"]["n_windows"],
            meta_group["novice"]["n_windows"],
        )
        plt.close(fig_group)


if __name__ == "__main__":
    _cli_main()
