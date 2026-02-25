"""
Stage C — Step C2: CNR Calculation and Visualization
Purpose: Compute channel-wise Contrast-to-Noise Ratio (CNR) between two groups
         using per-subject HRF peak amplitudes, then produce a box-plot that
         replicates Figure 4i of the ASTaUND paper.

Task coverage:
  4.4.1 — CNR calculation function  |μ₁−μ₂| / √(σ₁²+σ₂²)  using 5–15 s peak window
  4.4.2 — CNR for Expert vs Novice comparison across all long-separation channels
  4.4.3 — CNR box plot (Figure 4i style); future-ready for multi-study overlay
  4.4.4 — Save per-channel CNR results to CSV

CNR formula
-----------
For each long-separation channel c and chromophore (HbO / HbR):
  1. For every subject s in each group, compute the signed peak amplitude a_s
     within the analysis window [peak_window[0], peak_window[1]] s relative to
     stimulus onset:
       HbO: a_s = max(evoked_s[c, t ∈ peak_window])
       HbR: a_s = min(evoked_s[c, t ∈ peak_window])  (signed — negative expected)
  2. Compute group statistics across subjects:
       μ₁ = mean({a_s | s ∈ group 1})    σ₁ = std({a_s | s ∈ group 1})
       μ₂ = mean({a_s | s ∈ group 2})    σ₂ = std({a_s | s ∈ group 2})
  3. CNR_c = |μ₁ − μ₂| / √(σ₁² + σ₂²)
     (σ denominator clipped at 1e-9 μM to avoid division by zero on flat channels)

Box-plot design (Figure 4i style)
----------------------------------
  • One box per study comparison (x-axis)
  • y-axis: CNR (all long-channel HbO + HbR values pooled per comparison)
  • Blue boxes, red median line, red '+' outliers — matches paper style
  • Horizontal dashed line at CNR = 1.0 (detectable threshold)
  • Current study (Expert vs Novice) always plotted first; other studies are
    added by passing their CNR arrays in the `extra_comparisons` argument of
    `plot_cnr_boxplot()`, or by re-running `run_cnr_analysis()` after extending
    `COMPARISON_REGISTRY` at the bottom of this file.

Outputs (outputs/stage_c_validation/cnr/)
  cnr_Expert-Novice.csv          — per-channel CNR (HbO + HbR rows)
  cnr_boxplot.png                — box plot (Figure 4i style)
  cnr_summary.json               — group stats and pass/fail flags

Upstream : 01_hrf_extraction.py  (run_hrf_extraction, build_region_map,
                                   _is_short_sep, get_long_channel_names)
           02_group_hrf_analysis.py  (_load_or_compute_subject_evoked,
                                      _harmonise_evoked)
Reference : PROGRESS.md Section 4.4; config.yaml stage_c.cnr; paper Figure 4i
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config

# ---------------------------------------------------------------------------
# Import Section 4.1 / 4.2 helpers via importlib
# ---------------------------------------------------------------------------
def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_hrf = _load_module("hrf_extraction", "01_hrf_extraction.py")
_grp = _load_module("group_hrf",      "02_group_hrf_analysis.py")

build_region_map               = _hrf.build_region_map
_is_short_sep                  = _hrf._is_short_sep
get_long_channel_names         = _hrf.get_long_channel_names
_load_or_compute_subject_evoked = _grp._load_or_compute_subject_evoked
_harmonise_evoked              = _grp._harmonise_evoked

# ---------------------------------------------------------------------------
# Study / group definitions
# ---------------------------------------------------------------------------
STUDY  = "expert_novice"
GROUPS: dict[str, list[str]] = {
    "Expert": ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"],
    "Novice": ["R1", "R2", "R3", "R4", "R5", "R6"],
}

# Registry for multi-study box plot.  Each entry:
#   key   = display label (x-axis tick)
#   value = None  → will be populated at runtime from this study
#           list of floats → pre-computed CNR values from another study
#
# When other studies are processed, add their CNR arrays here so they appear
# on the box plot automatically.
COMPARISON_REGISTRY: dict[str, list[float] | None] = {
    "Expert vs Novice":   None,   # computed by this script
    "Trainee vs Control": None,   # Section 5.1 — Learning Curve (pending)
    "Sham vs tDCS":       None,   # Section 5.3 — Long-term tES (pending)
    "Sham vs tRNS":       None,   # Section 5.3 — Long-term tES (pending)
}


# ===========================================================================
# 4.4.1 — CNR calculation helpers
# ===========================================================================

def _extract_peak_amplitudes(
    evoked: mne.Evoked,
    long_channels: list[str],
    chromophore: str,
    peak_window: tuple[float, float],
) -> dict[str, float]:
    """
    For one subject's Evoked, return the signed peak amplitude (μM) within
    `peak_window` for every channel in `long_channels` that matches
    `chromophore` ('hbo' or 'hbr').

    HbO  → max value in window (positive peak expected)
    HbR  → min value in window (negative trough expected)

    Channels missing from the Evoked (dropped due to bad quality) are returned
    as NaN so they can be excluded from group statistics without bias.
    """
    times  = evoked.times
    t_mask = (times >= peak_window[0]) & (times <= peak_window[1])
    data   = evoked.data * 1e6          # mol/L → μM

    peaks: dict[str, float] = {}
    for ch_name in long_channels:
        if ch_name not in evoked.ch_names:
            peaks[ch_name] = np.nan
            continue
        idx   = evoked.ch_names.index(ch_name)
        trace = data[idx, t_mask]
        if chromophore == "hbo":
            peaks[ch_name] = float(np.max(trace))
        else:
            peaks[ch_name] = float(np.min(trace))
    return peaks


def compute_cnr(
    group1_peaks: dict[str, list[float]],
    group2_peaks: dict[str, list[float]],
) -> dict[str, float]:
    """
    Compute CNR = |μ₁ − μ₂| / √(σ₁² + σ₂²) for every channel present in both
    group peak dicts.

    Parameters
    ----------
    group1_peaks, group2_peaks : dict  {channel_name: [peak_per_subject, …]}
        Peak amplitudes (μM) for each subject per channel.

    Returns
    -------
    cnr : dict  {channel_name: cnr_value}
        Channels with fewer than 2 valid subjects in either group get NaN.
    """
    all_channels = set(group1_peaks) | set(group2_peaks)
    cnr: dict[str, float] = {}

    for ch in sorted(all_channels):
        vals1 = np.array([v for v in group1_peaks.get(ch, []) if not np.isnan(v)])
        vals2 = np.array([v for v in group2_peaks.get(ch, []) if not np.isnan(v)])

        if len(vals1) < 2 or len(vals2) < 2:
            cnr[ch] = np.nan
            continue

        mu1, mu2 = vals1.mean(), vals2.mean()
        s1,  s2  = vals1.std(ddof=1), vals2.std(ddof=1)

        denom = np.sqrt(s1**2 + s2**2)
        if denom < 1e-9:           # both groups essentially identical
            cnr[ch] = 0.0
        else:
            cnr[ch] = float(abs(mu1 - mu2) / denom)

    return cnr


def collect_group_peaks(
    subjects: list[str],
    long_channels: list[str],
    chromophore: str,
    peak_window: tuple[float, float],
    out_base: Path,
    study: str,
    target_sfreq: float = 25.0,
) -> dict[str, list[float]]:
    """
    Load (or compute) per-subject Evoked objects for `subjects` and collect
    their peak amplitudes into {channel: [peak_subj1, peak_subj2, …]}.
    """
    group_peaks: dict[str, list[float]] = {ch: [] for ch in long_channels}

    for sid in subjects:
        ev = _load_or_compute_subject_evoked(sid, out_base, study)
        if ev is None:
            print(f"    [{sid}] ⚠ skipped (no data)")
            continue
        ev = _harmonise_evoked(ev, target_sfreq)

        peaks = _extract_peak_amplitudes(ev, long_channels, chromophore, peak_window)
        for ch, val in peaks.items():
            group_peaks[ch].append(val)

    return group_peaks


# ===========================================================================
# 4.4.2 — Expert vs Novice CNR across all channels
# ===========================================================================

def run_expert_novice_cnr(
    groups: dict[str, list[str]],
    out_base: Path,
    stage_b_dir: Path,
    study: str,
    peak_window: tuple[float, float] = (5.0, 15.0),
    target_sfreq: float = 25.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute per-channel CNR for Expert vs Novice (HbO and HbR long channels).

    Returns
    -------
    df_cnr   : DataFrame with columns [channel, chromophore, cnr, group1_mu,
               group2_mu, group1_sigma, group2_sigma]
    cnr_meta : dict  metadata (group sizes, peak window, n valid channels)
    """
    group_names = list(groups.keys())
    g1_name, g2_name = group_names[0], group_names[1]

    # ── Get channel lists from first available subject's preprocessed raw ────
    first_subj = groups[g1_name][0]
    candidates = list(stage_b_dir.glob(f"sub-{first_subj}_*_raw.fif"))
    if not candidates:
        raise FileNotFoundError(
            f"No Stage B .fif found for {first_subj} in {stage_b_dir}"
        )
    raw_ref = mne.io.read_raw_fif(candidates[0], preload=False, verbose=False)
    hbo_long, hbr_long = get_long_channel_names(raw_ref)
    print(f"\n  Long-separation channels: {len(hbo_long)} HbO, {len(hbr_long)} HbR")

    rows: list[dict] = []

    for chromo, long_chs in [("hbo", hbo_long), ("hbr", hbr_long)]:
        label = "HbO" if chromo == "hbo" else "HbR"
        print(f"\n  [CNR — {label}]")

        # Per-subject peaks for each group
        g1_peaks = collect_group_peaks(
            groups[g1_name], long_chs, chromo, peak_window,
            out_base, study, target_sfreq,
        )
        g2_peaks = collect_group_peaks(
            groups[g2_name], long_chs, chromo, peak_window,
            out_base, study, target_sfreq,
        )

        # Channel-wise CNR
        cnr_vals = compute_cnr(g1_peaks, g2_peaks)

        for ch, cnr_val in cnr_vals.items():
            vals1 = np.array([v for v in g1_peaks.get(ch, []) if not np.isnan(v)])
            vals2 = np.array([v for v in g2_peaks.get(ch, []) if not np.isnan(v)])
            rows.append({
                "channel":       ch,
                "chromophore":   label,
                "cnr":           round(cnr_val, 4) if not np.isnan(cnr_val) else np.nan,
                f"{g1_name}_n":  len(vals1),
                f"{g1_name}_mu": round(float(vals1.mean()), 4) if len(vals1) >= 1 else np.nan,
                f"{g1_name}_sigma": round(float(vals1.std(ddof=1)), 4) if len(vals1) >= 2 else np.nan,
                f"{g2_name}_n":  len(vals2),
                f"{g2_name}_mu": round(float(vals2.mean()), 4) if len(vals2) >= 1 else np.nan,
                f"{g2_name}_sigma": round(float(vals2.std(ddof=1)), 4) if len(vals2) >= 2 else np.nan,
            })

        # Quick stats
        valid_cnr = [v for v in cnr_vals.values() if not np.isnan(v)]
        if valid_cnr:
            print(f"    {label}: {len(valid_cnr)} valid channels | "
                  f"median CNR={np.median(valid_cnr):.3f}  "
                  f"mean={np.mean(valid_cnr):.3f}  "
                  f"max={np.max(valid_cnr):.3f}")

    df_cnr = pd.DataFrame(rows)

    all_cnr = df_cnr["cnr"].dropna().values
    cnr_meta = {
        "comparison":        f"{g1_name} vs {g2_name}",
        "group1":            g1_name,
        "group2":            g2_name,
        "group1_n_subjects": len(groups[g1_name]),
        "group2_n_subjects": len(groups[g2_name]),
        "peak_window_s":     list(peak_window),
        "n_channels_hbo":    len(hbo_long),
        "n_channels_hbr":    len(hbr_long),
        "n_valid_cnr":       int(len(all_cnr)),
        "median_cnr":        round(float(np.median(all_cnr)), 4) if len(all_cnr) > 0 else None,
        "mean_cnr":          round(float(np.mean(all_cnr)), 4) if len(all_cnr) > 0 else None,
        "max_cnr":           round(float(np.max(all_cnr)), 4) if len(all_cnr) > 0 else None,
        "pct_above_1.0":     round(float((all_cnr >= 1.0).mean() * 100), 1) if len(all_cnr) > 0 else None,
    }

    return df_cnr, cnr_meta


# ===========================================================================
# 4.4.3 — CNR box-plot (Figure 4i style)
# ===========================================================================

def plot_cnr_boxplot(
    comparison_data: dict[str, list[float]],
    out_path: Path,
    cnr_threshold: float = 1.0,
) -> None:
    """
    Create a box plot of CNR distributions across channels, one box per
    study comparison — replicating ASTaUND Figure 4i style.

    Parameters
    ----------
    comparison_data : dict  {label: [cnr_values]}
        Only entries with non-None values are plotted; None entries are shown
        as greyed-out placeholder boxes with a note.
    out_path : Path
        Output file path (.png).
    cnr_threshold : float
        Horizontal dashed line marking the detectable difference threshold.
    """
    # ── Separate available vs pending comparisons ────────────────────────────
    available  = {k: v for k, v in comparison_data.items() if v is not None}
    pending    = [k for k, v in comparison_data.items() if v is None]

    if not available:
        print("  ⚠ No CNR data available — skipping plot")
        return

    labels  = list(available.keys())
    all_vals = [np.array(v) for v in available.values()]

    # ── Figure layout ─────────────────────────────────────────────────────────
    total_cols = len(available) + len(pending)
    fig_width  = max(6, 2.0 * total_cols + 1.5)
    fig, ax    = plt.subplots(figsize=(fig_width, 5))

    # ── Available-data boxes (blue/white, red median — paper style) ───────────
    bp = ax.boxplot(
        all_vals,
        positions=range(1, len(available) + 1),
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor="white", color="#1F77B4", linewidth=1.5),
        medianprops=dict(color="#D62728", linewidth=2.0),
        whiskerprops=dict(color="#1F77B4", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#1F77B4", linewidth=1.5),
        flierprops=dict(marker="+", color="#D62728", markersize=7, linewidth=1.2),
    )

    # ── Pending placeholders (greyed-out) ─────────────────────────────────────
    for i, label in enumerate(pending, start=len(available) + 1):
        ax.bar(i, 0, color="none")          # invisible bar at correct x pos
        ax.text(
            i, 0.05, "pending",
            ha="center", va="bottom", fontsize=8, color="gray",
            style="italic",
        )

    # ── Threshold line ────────────────────────────────────────────────────────
    all_x = list(range(1, total_cols + 1))
    ax.axhline(
        cnr_threshold, color="gray", lw=1.2, ls="--", zorder=0,
        label=f"CNR = {cnr_threshold:.1f} (detectable threshold)",
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    all_tick_labels = labels + pending
    ax.set_xticks(range(1, total_cols + 1))
    ax.set_xticklabels(all_tick_labels, fontsize=10, rotation=0, ha="center")

    # Add per-box sample-size annotation
    for i, (label, vals) in enumerate(available.items(), start=1):
        n   = len(vals)
        med = float(np.median(vals))
        ax.text(
            i, -0.08,
            f"n={n}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="gray",
        )

    ax.set_ylim(bottom=0)
    ax.set_ylabel("CNR", fontsize=12)
    ax.set_title("Task-Related CNR", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.4, total_cols + 0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ===========================================================================
# 4.4.4 — Save CNR results to CSV
# ===========================================================================

def save_cnr_csv(df_cnr: pd.DataFrame, out_path: Path) -> None:
    """Save per-channel CNR DataFrame to CSV."""
    df_cnr.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(df_cnr)} rows × {len(df_cnr.columns)} cols)")


# ===========================================================================
# Top-level pipeline
# ===========================================================================

def run_cnr_analysis(
    groups: dict[str, list[str]] = GROUPS,
    study: str = STUDY,
    peak_window: tuple[float, float] = (5.0, 15.0),
    comparison_registry: dict[str, list[float] | None] = COMPARISON_REGISTRY,
) -> dict:
    """
    Full Section 4.4 pipeline — CNR calculation, visualization, and CSV export.

    Parameters
    ----------
    groups               : dict {group_name: [subject_ids]}
    study                : config study key
    peak_window          : (start, end) seconds for peak-amplitude extraction
    comparison_registry  : dict label → pre-computed CNR list (or None = pending)

    Returns
    -------
    results : dict  summary metrics and file paths
    """
    config = load_config()
    study_display = Path(config["datasets"][study]["data_path"]).name
    stage_b_name  = study.replace("_", "-")

    # Output directories
    out_dir = Path(config["paths"]["stage_c_outputs"]) / "cnr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # HRF subject evoked cache lives in stage_c/hrf/{study_display}/
    out_base   = Path(config["paths"]["stage_c_outputs"]) / "hrf" / study_display
    stage_b_dir = (Path(config["paths"]["stage_b_outputs"])
                   / "processed_data" / stage_b_name)

    # CNR threshold from config
    cnr_threshold = float(
        config.get("stage_c", {}).get("cnr", {}).get("threshold_detectable", 1.0)
    )

    print(f"\n{'='*60}")
    print(f"  Section 4.4 — CNR Calculation")
    print(f"  Study: {study_display}  |  Comparison: "
          f"{' vs '.join(groups.keys())}")
    print(f"  Peak window: {peak_window[0]}–{peak_window[1]} s")
    print(f"{'='*60}")

    # ── 4.4.1 + 4.4.2 — Compute CNR ──────────────────────────────────────────
    df_cnr, cnr_meta = run_expert_novice_cnr(
        groups, out_base, stage_b_dir, study,
        peak_window=peak_window,
    )

    # ── 4.4.4 — Save CSV ─────────────────────────────────────────────────────
    csv_name = f"cnr_{study_display}.csv"
    csv_path = out_dir / csv_name
    save_cnr_csv(df_cnr, csv_path)

    # ── 4.4.3 — Build comparison dict and plot ───────────────────────────────
    valid_cnr = df_cnr["cnr"].dropna().tolist()
    comparison_label = f"{list(groups.keys())[0]} vs {list(groups.keys())[1]}"

    # Merge current study into registry (deep copy to avoid mutating module-level)
    plot_data: dict[str, list[float] | None] = {}
    for label, val in comparison_registry.items():
        if label == comparison_label:
            plot_data[label] = valid_cnr
        else:
            plot_data[label] = val

    plot_path = out_dir / "cnr_boxplot.png"
    plot_cnr_boxplot(plot_data, plot_path, cnr_threshold=cnr_threshold)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        **cnr_meta,
        "cnr_threshold_detectable": cnr_threshold,
        "outputs": {
            "cnr_csv":    str(csv_path),
            "cnr_boxplot": str(plot_path),
        },
    }
    summary_path = out_dir / "cnr_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: cnr_summary.json")

    # ── Console report ────────────────────────────────────────────────────────
    all_cnr = df_cnr["cnr"].dropna().values
    pct_above = (all_cnr >= cnr_threshold).mean() * 100 if len(all_cnr) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Section 4.4 complete")
    print(f"  Channels: {cnr_meta['n_channels_hbo']} HbO + "
          f"{cnr_meta['n_channels_hbr']} HbR = "
          f"{cnr_meta['n_channels_hbo'] + cnr_meta['n_channels_hbr']} total")
    print(f"  Valid CNR values: {cnr_meta['n_valid_cnr']}")
    print(f"  Median CNR: {cnr_meta['median_cnr']}")
    print(f"  Mean CNR  : {cnr_meta['mean_cnr']}")
    print(f"  Max CNR   : {cnr_meta['max_cnr']}")
    print(f"  % channels CNR ≥ {cnr_threshold}: {pct_above:.1f}%")
    print(f"{'='*60}\n")

    return summary


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_cnr_analysis()
