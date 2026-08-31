"""
Stage E — Phase Variability Bias Diagnostic (Issue 2 Investigation)

Diagnostic question: Does phase variability bias explain why Novice > Expert
on all 20 dSPTE connections after narrowband filtering?

Hypothesis: More variable Novice hemodynamic responses → more variable
narrowband phase trajectories → more diverse symbol sequences → higher
marginal entropy H(φ_Y) → higher TE(X→Y) = H(Y_future|Y_past) − H(Y_future|Y_past, X_past).
If H(Y_future|Y_past) is globally elevated for Novices, TE will be inflated
regardless of actual coupling strength.

Diagnostic approach:
    1. For each subject × trial × ROI, apply the same narrowband filter used
       in the dSPTE pipeline (0.04–0.15 Hz Butterworth), extract Hilbert phase,
       embed with d=2, τ from ACF, and symbolise via Neural-Gas (same k₀ search).
    2. Compute marginal symbol entropy H(S_Y) = −Σ p(s) log p(s) from the
       resulting symbol sequence.
    3. Compare Expert vs Novice H(S_Y) per ROI across all trials.
    4. If Novice H(S_Y) > Expert H(S_Y) systematically → confirms Phase
       Variability Bias as the root cause of Novice > Expert in all connections.

Output:
    - JSON summary: per-subject per-ROI marginal entropy values
    - PNG figure: Expert vs Novice H(S_Y) per ROI (violin + scatter plot)
    - PNG figure: group mean H(S_Y) with individual subject lines
    - Saved to: outputs/stage_e_transfer_entropy/expert-novice/phase_1_bivariate_dspte/
                diagnostics/phase_entropy_bias/

Reference
---------
Mathematically, TE(X→Y) = H(Y_future | Y_past) − H(Y_future | Y_past, X_past).
If H(Y_future | Y_past) is globally higher for Novices (regardless of X),
then TE(X→Y) for Novice targets will be elevated across all source ROIs X.
The marginal entropy H(S_Y) of the symbolised phase sequence approximates
H(φ_Y) and can be computed cheaply before the full TE calculation.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/
_PROJECT_ROOT = _CODE_ROOT.parent

for _p in [str(_CODE_ROOT), str(_STAGE_E_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_ROI_NAMES_CANONICAL = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]

# ---------------------------------------------------------------------------
# Lazy module loaders (mirror 03_bivariate_dspte_batch.py pattern)
# ---------------------------------------------------------------------------

_SEG_MODULE = None
_CORE_MODULE = None


def _get_seg_module():
    global _SEG_MODULE
    if _SEG_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "trial_segmentation",
            _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEG_MODULE = mod
    return _SEG_MODULE


def _get_core_module():
    global _CORE_MODULE
    if _CORE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "bivariate_dspte_core", _STAGE_E_ROOT / "phase_1" / "01_bivariate_dspte_core.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE_MODULE = mod
    return _CORE_MODULE


# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------

def compute_marginal_entropy(symbols: np.ndarray, k0: int) -> float:
    """
    Compute marginal entropy H(S) = −Σ_s p(s) log p(s) from a symbol sequence.

    Parameters
    ----------
    symbols : np.ndarray, shape (T',), dtype int
        Integer symbol sequence with values in {0, ..., k0−1}.
    k0 : int
        Alphabet size (codebook size).

    Returns
    -------
    entropy : float
        Marginal entropy in nats (natural log). Non-negative.
        Maximum value = log(k0) (uniform distribution).
    """
    counts = np.bincount(symbols, minlength=k0).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Shannon entropy, ignoring zero-probability bins
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -float(np.sum(probs * log_p))
    return max(0.0, entropy)


def compute_subject_phase_entropy(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
) -> dict:
    """
    Compute marginal phase symbol entropy H(S_Y) for all ROIs and trials
    for one subject.

    Steps:
    1. Load ROI signals from Stage D cache.
    2. Load .fif for trial segmentation.
    3. For each trial × ROI:
       a. Apply narrowband Butterworth bandpass to full trial signal.
       b. Extract Hilbert instantaneous phase.
       c. Compute embedding delay τ from ACF.
       d. Embed with d=2 (lighter than dSPTE's d=3 for diagnostic speed).
       e. Fit Neural-Gas codebook on single-channel embedded matrix.
       f. Quantise to symbols → compute H(S_Y).

    Parameters
    ----------
    subject_id : str
    study : str
    config : dict, optional

    Returns
    -------
    result : dict
        {
          'subject_id': str,
          'group': str,
          'trials': {
            'Tr_1': {
              'LPFC': {'entropy': float, 'k0': int, 'tau': int, 'n_symbols': int},
              ...
            },
            ...
          }
        }
    """
    if config is None:
        config = load_config()

    from utils_te import (  # noqa: E402
        get_subject_group,
        load_or_extract_roi_signals,
        load_preprocessed_fif,
        get_te_output_dir,
    )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})
    sym_cfg = te_cfg.get("symbolisation", {})
    win_cfg = te_cfg.get("window", {})

    f_lo = float(filt_cfg.get("f_lo", 0.04))
    f_hi = float(filt_cfg.get("f_hi", 0.15))
    filt_order = int(filt_cfg.get("order", 3))
    k0_max = int(sym_cfg.get("k0_max_bivariate", 35))
    ndis_threshold = float(sym_cfg.get("ndis_threshold", 0.05))

    core_mod = _get_core_module()
    seg_mod = _get_seg_module()

    group = get_subject_group(subject_id, config=config)

    logger.info("compute_subject_phase_entropy [%s]: loading ROI signals ...", subject_id)
    Data_Reg, valid_roi_names, sfreq = load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    logger.info(
        "compute_subject_phase_entropy [%s]: Data_Reg shape=%s  ROIs=%s  sfreq=%.1f Hz",
        subject_id, Data_Reg.shape, valid_roi_names, sfreq,
    )

    raw = load_preprocessed_fif(subject_id, study=study, config=config)
    segments = seg_mod.get_trial_segments(raw, config=config)
    logger.info(
        "compute_subject_phase_entropy [%s]: %d trials found.", subject_id, len(segments)
    )

    trials_result: dict = {}

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset : offset + 1]

        roi_result: dict = {}

        for roi_idx, roi_name in enumerate(valid_roi_names):
            row = Y_trial[roi_idx]

            if np.all(np.isnan(row)):
                logger.warning(
                    "compute_subject_phase_entropy [%s %s]: ROI %s all-NaN — skip.",
                    subject_id, trial_id, roi_name,
                )
                roi_result[roi_name] = {
                    "entropy": float("nan"),
                    "k0": 0,
                    "tau": 0,
                    "n_symbols": 0,
                    "skipped": True,
                }
                continue

            # Step (a): narrowband filter on full trial
            try:
                x_filt = core_mod.narrowband_filter(
                    row, f_lo=f_lo, f_hi=f_hi, sfreq=sfreq, order=filt_order
                )
            except Exception as e:
                logger.warning(
                    "compute_subject_phase_entropy [%s %s %s]: filter failed: %s",
                    subject_id, trial_id, roi_name, e,
                )
                roi_result[roi_name] = {
                    "entropy": float("nan"),
                    "k0": 0,
                    "tau": 0,
                    "n_symbols": 0,
                    "skipped": True,
                }
                continue

            # Step (b): Hilbert phase
            ph = core_mod.extract_phase_signal(x_filt)

            # Step (c): embedding delay
            tau = core_mod.compute_embedding_delay(ph)

            # Step (d): embed with d=2 for speed (sufficient for marginal entropy)
            try:
                emb = core_mod.uniform_embed(ph, d=2, tau=tau)   # shape (T', 2)
            except ValueError as e:
                logger.warning(
                    "compute_subject_phase_entropy [%s %s %s]: embed failed: %s",
                    subject_id, trial_id, roi_name, e,
                )
                roi_result[roi_name] = {
                    "entropy": float("nan"),
                    "k0": 0,
                    "tau": int(tau),
                    "n_symbols": 0,
                    "skipped": True,
                }
                continue

            # Step (e): fit Neural-Gas on single-channel embedded matrix
            from utils_te import fit_neural_gas_codebook, quantise_to_symbols  # noqa

            try:
                codebook, k0, ndis = fit_neural_gas_codebook(
                    emb,
                    k0_max=k0_max,
                    ndis_threshold=ndis_threshold,
                    config=config,
                )
            except Exception as e:
                logger.warning(
                    "compute_subject_phase_entropy [%s %s %s]: NG failed: %s",
                    subject_id, trial_id, roi_name, e,
                )
                roi_result[roi_name] = {
                    "entropy": float("nan"),
                    "k0": 0,
                    "tau": int(tau),
                    "n_symbols": 0,
                    "skipped": True,
                }
                continue

            # Step (f): quantise → marginal entropy
            symbols = quantise_to_symbols(emb, codebook)
            h = compute_marginal_entropy(symbols, k0=k0)

            roi_result[roi_name] = {
                "entropy": round(float(h), 6),
                "k0": int(k0),
                "ndis": round(float(ndis), 4),
                "tau": int(tau),
                "n_symbols": int(len(symbols)),
                "skipped": False,
            }

            logger.info(
                "compute_subject_phase_entropy [%s %s %s]: "
                "H=%.4f  k0=%d  nDis=%.3f  tau=%d  T'=%d",
                subject_id, trial_id, roi_name,
                h, k0, ndis, tau, len(symbols),
            )

        trials_result[trial_id] = roi_result

    return {
        "subject_id": subject_id,
        "group": group,
        "trials": trials_result,
        "valid_roi_names": valid_roi_names,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_phase_entropy_diagnostic(
    study: str = "expert_novice",
    subjects: Optional[list] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Run the phase variability bias diagnostic for all subjects.

    Returns
    -------
    results : dict
        Per-subject entropy data + group comparison summary.
    Saves
    -----
    diagnostics/phase_entropy_bias/phase_entropy_results.json
    diagnostics/phase_entropy_bias/phase_entropy_summary.json
    diagnostics/phase_entropy_bias/phase_entropy_bias_violin.png
    diagnostics/phase_entropy_bias/phase_entropy_bias_lines.png
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_subject_group  # noqa

    # Resolve output directory
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})

    if subjects is None:
        experts = subj_cfg.get("Expert", [])
        novices = subj_cfg.get("Novice", [])
        subjects = experts + novices

    # Use the phase_1_viz dir as the base, then add diagnostics subdir
    viz_dir = get_te_output_dir("phase_1_viz", config=config)
    diag_dir = viz_dir.parent / "diagnostics" / "phase_entropy_bias"
    diag_dir.mkdir(parents=True, exist_ok=True)

    results_path = diag_dir / "phase_entropy_results.json"
    if not overwrite and results_path.exists():
        logger.info("run_phase_entropy_diagnostic: loading cached results from %s", results_path)
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
        t_batch_start = time.perf_counter()

        for subject_id in subjects:
            logger.info("=== Phase Entropy Diagnostic: Subject %s ===", subject_id)
            t0 = time.perf_counter()
            try:
                res = compute_subject_phase_entropy(subject_id, study=study, config=config)
                res["elapsed_s"] = round(time.perf_counter() - t0, 1)
                all_results[subject_id] = res
                logger.info(
                    "Subject %s done in %.1f s", subject_id, res["elapsed_s"]
                )
            except Exception as exc:
                logger.error("Subject %s FAILED: %s", subject_id, exc)
                all_results[subject_id] = {"status": "failed", "error": str(exc)}

        logger.info(
            "Batch complete in %.1f s", time.perf_counter() - t_batch_start
        )

        # Save raw results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved per-subject results → %s", results_path)

    # Build group comparison summary
    summary = _build_group_summary(all_results, subjects, subj_cfg)

    summary_path = diag_dir / "phase_entropy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved group summary → %s", summary_path)

    # Generate visualisations
    _plot_violin(all_results, summary, diag_dir)
    _plot_subject_lines(all_results, summary, diag_dir)

    logger.info("Phase entropy diagnostic complete. Outputs → %s", diag_dir)
    return {"results": all_results, "summary": summary, "output_dir": str(diag_dir)}


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_group_summary(all_results: dict, subjects: list, subj_cfg: dict) -> dict:
    """Build per-ROI Expert vs Novice entropy statistics."""
    expert_ids = subj_cfg.get("Expert", [])
    novice_ids = subj_cfg.get("Novice", [])

    roi_names = _ROI_NAMES_CANONICAL
    summary: dict = {
        "created_at": datetime.now().isoformat(),
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "rois": {},
    }

    # Collect entropy values per ROI per group
    exp_by_roi: dict = {roi: [] for roi in roi_names}
    nov_by_roi: dict = {roi: [] for roi in roi_names}

    for sid, res in all_results.items():
        if "trials" not in res:
            continue
        group = res.get("group", "Unknown")
        valid_rois = res.get("valid_roi_names", [])

        subj_roi_entropy: dict = {roi: [] for roi in roi_names}

        for trial_id, trial_data in res["trials"].items():
            for roi_name, roi_data in trial_data.items():
                if isinstance(roi_data, dict) and not roi_data.get("skipped", False):
                    h = roi_data.get("entropy")
                    if h is not None and not np.isnan(h):
                        subj_roi_entropy[roi_name].append(h)

        # Per-subject mean entropy across all trials for each ROI
        for roi_name in roi_names:
            vals = subj_roi_entropy[roi_name]
            if vals:
                mean_h = float(np.mean(vals))
                if group == "Expert":
                    exp_by_roi[roi_name].append({"subject": sid, "mean_entropy": mean_h})
                elif group == "Novice":
                    nov_by_roi[roi_name].append({"subject": sid, "mean_entropy": mean_h})

    # Compute group statistics per ROI
    all_exp_entropy = []
    all_nov_entropy = []

    for roi_name in roi_names:
        exp_vals = [d["mean_entropy"] for d in exp_by_roi[roi_name]]
        nov_vals = [d["mean_entropy"] for d in nov_by_roi[roi_name]]

        all_exp_entropy.extend(exp_vals)
        all_nov_entropy.extend(nov_vals)

        if exp_vals and nov_vals:
            nov_gt_exp = float(np.mean(nov_vals)) > float(np.mean(exp_vals))
            diff = float(np.mean(nov_vals)) - float(np.mean(exp_vals))
        else:
            nov_gt_exp = None
            diff = None

        summary["rois"][roi_name] = {
            "expert": {
                "subjects": exp_by_roi[roi_name],
                "mean": round(float(np.mean(exp_vals)), 6) if exp_vals else None,
                "std": round(float(np.std(exp_vals)), 6) if len(exp_vals) > 1 else None,
                "n": len(exp_vals),
            },
            "novice": {
                "subjects": nov_by_roi[roi_name],
                "mean": round(float(np.mean(nov_vals)), 6) if nov_vals else None,
                "std": round(float(np.std(nov_vals)), 6) if len(nov_vals) > 1 else None,
                "n": len(nov_vals),
            },
            "novice_gt_expert": nov_gt_exp,
            "mean_difference_nov_minus_exp": round(diff, 6) if diff is not None else None,
        }

    # Global summary
    n_rois_nov_gt_exp = sum(
        1 for roi in summary["rois"].values() if roi.get("novice_gt_expert") is True
    )
    n_rois_total = sum(
        1 for roi in summary["rois"].values() if roi.get("novice_gt_expert") is not None
    )

    summary["global"] = {
        "n_rois_novice_gt_expert": n_rois_nov_gt_exp,
        "n_rois_total": n_rois_total,
        "fraction_rois_bias_confirmed": (
            round(n_rois_nov_gt_exp / n_rois_total, 3) if n_rois_total > 0 else None
        ),
        "grand_mean_expert_entropy": round(float(np.mean(all_exp_entropy)), 6) if all_exp_entropy else None,
        "grand_mean_novice_entropy": round(float(np.mean(all_nov_entropy)), 6) if all_nov_entropy else None,
        "grand_mean_difference_nov_minus_exp": (
            round(float(np.mean(all_nov_entropy)) - float(np.mean(all_exp_entropy)), 6)
            if all_exp_entropy and all_nov_entropy else None
        ),
        "bias_confirmed": n_rois_nov_gt_exp == n_rois_total and n_rois_total > 0,
    }

    return summary


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_violin(all_results: dict, summary: dict, diag_dir: Path) -> None:
    """Violin + strip plot: Expert vs Novice H(S_Y) per ROI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    roi_names = _ROI_NAMES_CANONICAL
    n_rois = len(roi_names)

    fig, ax = plt.subplots(figsize=(12, 5))

    positions_exp = np.arange(n_rois) * 3.0 - 0.6
    positions_nov = np.arange(n_rois) * 3.0 + 0.6

    exp_color = "#C0392B"
    nov_color = "#2980B9"

    exp_by_roi_vals: dict = {roi: [] for roi in roi_names}
    nov_by_roi_vals: dict = {roi: [] for roi in roi_names}

    for roi_name in roi_names:
        for entry in summary["rois"].get(roi_name, {}).get("expert", {}).get("subjects", []):
            exp_by_roi_vals[roi_name].append(entry["mean_entropy"])
        for entry in summary["rois"].get(roi_name, {}).get("novice", {}).get("subjects", []):
            nov_by_roi_vals[roi_name].append(entry["mean_entropy"])

    for i, roi_name in enumerate(roi_names):
        exp_vals = exp_by_roi_vals[roi_name]
        nov_vals = nov_by_roi_vals[roi_name]

        if len(exp_vals) >= 2:
            vp_exp = ax.violinplot(exp_vals, positions=[positions_exp[i]],
                                   widths=0.9, showmedians=True)
            for pc in vp_exp["bodies"]:
                pc.set_facecolor(exp_color)
                pc.set_alpha(0.4)
            for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
                if part in vp_exp:
                    vp_exp[part].set_color(exp_color)

        if len(nov_vals) >= 2:
            vp_nov = ax.violinplot(nov_vals, positions=[positions_nov[i]],
                                   widths=0.9, showmedians=True)
            for pc in vp_nov["bodies"]:
                pc.set_facecolor(nov_color)
                pc.set_alpha(0.4)
            for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
                if part in vp_nov:
                    vp_nov[part].set_color(nov_color)

        # Individual subject dots
        rng = np.random.default_rng(42 + i)
        jitter_exp = rng.uniform(-0.15, 0.15, size=len(exp_vals))
        jitter_nov = rng.uniform(-0.15, 0.15, size=len(nov_vals))
        ax.scatter(positions_exp[i] + jitter_exp, exp_vals,
                   color=exp_color, s=40, zorder=5, alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.scatter(positions_nov[i] + jitter_nov, nov_vals,
                   color=nov_color, s=40, zorder=5, alpha=0.8, edgecolors="white", linewidths=0.5)

        # Annotate direction
        roi_sum = summary["rois"].get(roi_name, {})
        nov_gt_exp = roi_sum.get("novice_gt_expert")
        diff = roi_sum.get("mean_difference_nov_minus_exp")
        if nov_gt_exp is not None and diff is not None:
            marker = "▲" if nov_gt_exp else "▼"
            color = "#E74C3C" if nov_gt_exp else "#27AE60"
            ax.text(
                positions_exp[i] + 0.6, max(max(exp_vals) if exp_vals else 0,
                                            max(nov_vals) if nov_vals else 0) + 0.02,
                f"{marker}{abs(diff):.3f}",
                ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold",
            )

    ax.set_xticks(np.arange(n_rois) * 3.0)
    ax.set_xticklabels(roi_names, fontsize=10)
    ax.set_ylabel("Marginal Symbol Entropy H(S_Y) [nats]", fontsize=10)
    ax.set_title(
        "Phase Variability Bias Diagnostic\n"
        "Expert vs Novice Marginal Phase Symbol Entropy H(φ_Y) per ROI\n"
        "(▲ = Novice > Expert; higher H → inflated TE regardless of coupling)",
        fontsize=10,
    )

    exp_patch = mpatches.Patch(color=exp_color, alpha=0.6, label=f"Expert (n={len(summary.get('expert_ids', []))})")
    nov_patch = mpatches.Patch(color=nov_color, alpha=0.6, label=f"Novice (n={len(summary.get('novice_ids', []))})")
    ax.legend(handles=[exp_patch, nov_patch], fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    glob = summary.get("global", {})
    bias_str = (
        f"Bias confirmed on {glob.get('n_rois_novice_gt_expert', '?')}/{glob.get('n_rois_total', '?')} ROIs  |  "
        f"Grand mean ΔH = {glob.get('grand_mean_difference_nov_minus_exp', 0):.4f} nats  |  "
        f"Expert: {glob.get('grand_mean_expert_entropy', 0):.4f}  Novice: {glob.get('grand_mean_novice_entropy', 0):.4f}"
    )
    ax.text(
        0.5, -0.12, bias_str,
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555",
    )

    plt.tight_layout()
    out_path = diag_dir / "phase_entropy_bias_violin.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved violin plot → %s", out_path)


def _plot_subject_lines(all_results: dict, summary: dict, diag_dir: Path) -> None:
    """
    Line plot: each subject's mean H(S_Y) across ROIs, coloured by group.
    Overlaid with group means ± std.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roi_names = _ROI_NAMES_CANONICAL
    x = np.arange(len(roi_names))
    exp_color = "#C0392B"
    nov_color = "#2980B9"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    expert_ids = summary.get("expert_ids", [])
    novice_ids = summary.get("novice_ids", [])

    for ax, group_ids, group_label, color in [
        (axes[0], expert_ids, "Expert", exp_color),
        (axes[1], novice_ids, "Novice", nov_color),
    ]:
        group_lines = []
        for sid in group_ids:
            res = all_results.get(sid)
            if res is None or "trials" not in res:
                continue
            # Per-ROI mean entropy across all trials for this subject
            roi_mean_entropy = []
            for roi_name in roi_names:
                vals = []
                for trial_data in res["trials"].values():
                    roi_data = trial_data.get(roi_name)
                    if isinstance(roi_data, dict) and not roi_data.get("skipped", False):
                        h = roi_data.get("entropy")
                        if h is not None and not np.isnan(h):
                            vals.append(h)
                roi_mean_entropy.append(float(np.mean(vals)) if vals else float("nan"))

            group_lines.append(roi_mean_entropy)
            ax.plot(x, roi_mean_entropy, color=color, alpha=0.3, linewidth=1.2,
                    marker="o", markersize=4)
            ax.text(
                x[-1] + 0.1, roi_mean_entropy[-1], sid,
                fontsize=6.5, va="center", color=color, alpha=0.7,
            )

        # Group mean ± std
        if group_lines:
            arr = np.array(group_lines)
            group_mean = np.nanmean(arr, axis=0)
            group_std = np.nanstd(arr, axis=0)
            ax.plot(x, group_mean, color=color, linewidth=2.5, marker="o",
                    markersize=7, label=f"{group_label} mean", zorder=5)
            ax.fill_between(x, group_mean - group_std, group_mean + group_std,
                            color=color, alpha=0.15, label="±1 SD")

        ax.set_xticks(x)
        ax.set_xticklabels(roi_names, fontsize=10)
        ax.set_title(f"{group_label} Group", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean Phase Symbol Entropy H(S_Y) [nats]", fontsize=10)

    glob = summary.get("global", {})
    fig.suptitle(
        "Phase Variability Bias Diagnostic: Per-Subject Mean H(S_Y) Across ROIs\n"
        f"Expert grand mean: {glob.get('grand_mean_expert_entropy', 0):.4f} nats  |  "
        f"Novice grand mean: {glob.get('grand_mean_novice_entropy', 0):.4f} nats  |  "
        f"ΔH (Nov−Exp): {glob.get('grand_mean_difference_nov_minus_exp', 0):.4f}",
        fontsize=10,
    )

    plt.tight_layout()
    out_path = diag_dir / "phase_entropy_bias_lines.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved subject lines plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """
    CLI usage:
        python3 code/stage_e_transfer_entropy/06_phase_entropy_diagnostic.py
        python3 code/stage_e_transfer_entropy/06_phase_entropy_diagnostic.py --overwrite
        python3 code/stage_e_transfer_entropy/06_phase_entropy_diagnostic.py --subject A1
    """
    import argparse

    p = argparse.ArgumentParser(
        description="Phase variability bias diagnostic for Stage E dSPTE.",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if cached results exist.")
    p.add_argument("--subject", type=str, default=None, metavar="ID",
                   help="Run for one subject only (for testing).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()

    if args.subject:
        res = compute_subject_phase_entropy(args.subject, config=config)
        print(f"\nSubject {args.subject} ({res['group']}):")
        for trial_id, trial_data in res["trials"].items():
            print(f"  {trial_id}:")
            for roi_name, roi_data in trial_data.items():
                if isinstance(roi_data, dict) and not roi_data.get("skipped"):
                    print(
                        f"    {roi_name}: H={roi_data['entropy']:.4f}  "
                        f"k0={roi_data['k0']}  tau={roi_data['tau']}"
                    )
        return

    output = run_phase_entropy_diagnostic(
        study="expert_novice",
        config=config,
        overwrite=args.overwrite,
    )

    summary = output["summary"]
    glob = summary.get("global", {})

    print("\n" + "=" * 70)
    print("PHASE VARIABILITY BIAS DIAGNOSTIC — RESULTS")
    print("=" * 70)
    print(f"\nSubjects: {glob.get('n_rois_total', '?')} ROIs evaluated")
    print(f"Grand mean H(S_Y)  Expert:  {glob.get('grand_mean_expert_entropy', 0):.4f} nats")
    print(f"Grand mean H(S_Y)  Novice:  {glob.get('grand_mean_novice_entropy', 0):.4f} nats")
    print(f"ΔH (Novice − Expert):       {glob.get('grand_mean_difference_nov_minus_exp', 0):.4f} nats")
    print(f"\nNovice > Expert on {glob.get('n_rois_novice_gt_expert', '?')}/{glob.get('n_rois_total', '?')} ROIs")
    print(f"Bias confirmed: {glob.get('bias_confirmed', False)}")

    print("\nPer-ROI breakdown:")
    for roi_name in _ROI_NAMES_CANONICAL:
        roi_data = summary["rois"].get(roi_name, {})
        exp_mean = roi_data.get("expert", {}).get("mean")
        nov_mean = roi_data.get("novice", {}).get("mean")
        diff = roi_data.get("mean_difference_nov_minus_exp")
        nov_gt = roi_data.get("novice_gt_expert")
        marker = "▲ BIAS" if nov_gt else "▼"
        if exp_mean is not None and nov_mean is not None:
            print(
                f"  {roi_name:6s}: Expert={exp_mean:.4f}  Novice={nov_mean:.4f}  "
                f"Δ={diff:+.4f}  {marker}"
            )

    bias_verdict = (
        "CONFIRMED — Novice > Expert on all ROIs: phase variability bias is the "
        "root cause of Novice > Expert dSPTE across all 20 connections."
        if glob.get("bias_confirmed")
        else "PARTIAL — Novice > Expert on some but not all ROIs."
    )
    print(f"\nConclusion: {bias_verdict}")
    print(f"\nOutputs saved to: {output['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    _cli_main()
