"""
04_calculate_cnr.py
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
  full_preprocess_cnr_Expert-Novice.csv  — per-channel CNR (HbO + HbR rows)
  full_preprocess_cnr_boxplot.png        — box plot (Figure 4i style)
  cnr_summary.json                       — group stats and pass/fail flags

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
    "Expert vs Novice":                    None,   # computed by 04_technical_validation_cnr.py
    "Fully Preprocessed Expert vs Novice": None,   # computed by this script
    "Trainee vs Control":                  None,   # Section 5.1 — Learning Curve (pending)
    "Sham vs tDCS":                        None,   # Section 5.3 — Long-term tES (pending)
    "Sham vs tRNS":                        None,   # Section 5.3 — Long-term tES (pending)
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

    Used by the legacy peak-based CNR path (kept for comparison in 4.4.2).
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
# 4.4.1 — Group-HRF CNR helpers (paper-faithful method)
# ===========================================================================

def collect_all_evokeds(
    subjects: list[str],
    out_base: Path,
    study: str,
    target_sfreq: float = 25.0,
) -> list[mne.Evoked]:
    """
    Load (or compute) per-subject Evoked objects for all subjects in a group.

    Returns a list of harmonised Evoked objects (NaN channels dropped, sfreq
    aligned to `target_sfreq`).  Subjects with no data are silently skipped.
    """
    evokeds: list[mne.Evoked] = []
    for sid in subjects:
        ev = _load_or_compute_subject_evoked(sid, out_base, study)
        if ev is None:
            print(f"    [{sid}] ⚠ skipped (no data)")
            continue
        ev = _harmonise_evoked(ev, target_sfreq)
        evokeds.append(ev)
    return evokeds


def compute_group_hrf_cnr(
    group1_evokeds: list,
    group2_evokeds: list,
    long_channels: list[str],
    chromophore: str,
    peak_window: tuple[float, float],
) -> tuple[dict[str, float], dict[str, dict]]:
    """
    Paper-faithful group-HRF CNR computation (Task 4.1).

    Algorithm
    ---------
    For each long-separation channel c and chromophore (HbO / HbR):

    1. Compute grand-average HRF for group g by averaging all subject Evokeds
       channel-wise:  grand_avg_g[c] = mean({evoked_s[c] | s ∈ group g})

    2. Extract signed peak from the grand-average trace:
         HbO: μ_g = max(grand_avg_g[c, t ∈ peak_window])
         HbR: μ_g = min(grand_avg_g[c, t ∈ peak_window])

       This is distinct from the legacy mean-of-peaks because averaging
       first suppresses inter-subject noise, yielding a cleaner signal peak.

    3. Compute inter-subject standard deviation of individual peaks:
         σ_g = std({peak(evoked_s[c, t ∈ peak_window]) | s ∈ group g})

    4. CNR_c = |μ_g1 − μ_g2| / √(σ_g1² + σ_g2²)

    Parameters
    ----------
    group1_evokeds, group2_evokeds : list[mne.Evoked]
        Per-subject Evoked objects (already harmonised: NaN-dropped, resampled).
    long_channels : list[str]
        Long-separation channel names for the given chromophore.
    chromophore : str  'hbo' or 'hbr'
    peak_window : tuple[float, float]  (start, end) seconds for peak extraction.

    Returns
    -------
    cnr : dict {channel: cnr_value}
        NaN for channels absent from grand averages or with < 2 valid subjects.
    meta : dict {channel: {mu1, mu2, sigma1, sigma2, n1, n2}}
        Per-channel numerator/denominator breakdown for reporting.
    """
    if not group1_evokeds or not group2_evokeds:
        empty = {ch: np.nan for ch in long_channels}
        return empty, {}

    # Grand averages (MNE drops channels absent from ALL subjects)
    ga1 = mne.grand_average(group1_evokeds, interpolate_bads=False, drop_bads=False)
    ga2 = mne.grand_average(group2_evokeds, interpolate_bads=False, drop_bads=False)

    def _channel_peak(evoked: mne.Evoked, ch_name: str, chromo: str) -> float:
        if ch_name not in evoked.ch_names:
            return np.nan
        idx = evoked.ch_names.index(ch_name)
        t = evoked.times
        t_mask = (t >= peak_window[0]) & (t <= peak_window[1])
        trace = evoked.data[idx, t_mask] * 1e6   # mol/L → μM
        if len(trace) == 0:
            return np.nan
        return float(np.max(trace)) if chromo == "hbo" else float(np.min(trace))

    cnr: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for ch in long_channels:
        # ── Grand-average peaks (numerator) ──────────────────────────────────
        mu1 = _channel_peak(ga1, ch, chromophore)
        mu2 = _channel_peak(ga2, ch, chromophore)

        # ── Individual subject peaks (for σ denominator) ──────────────────────
        indiv1 = [
            v for ev in group1_evokeds
            for v in [_channel_peak(ev, ch, chromophore)]
            if np.isfinite(v)
        ]
        indiv2 = [
            v for ev in group2_evokeds
            for v in [_channel_peak(ev, ch, chromophore)]
            if np.isfinite(v)
        ]

        meta[ch] = {
            "mu1": mu1, "mu2": mu2,
            "n1": len(indiv1), "n2": len(indiv2),
            "sigma1": np.nan, "sigma2": np.nan,
        }

        if np.isnan(mu1) or np.isnan(mu2) or len(indiv1) < 2 or len(indiv2) < 2:
            cnr[ch] = np.nan
            continue

        s1 = float(np.std(indiv1, ddof=1))
        s2 = float(np.std(indiv2, ddof=1))
        meta[ch]["sigma1"] = s1
        meta[ch]["sigma2"] = s2

        denom = np.sqrt(s1**2 + s2**2)
        cnr[ch] = float(abs(mu1 - mu2) / denom) if denom >= 1e-9 else 0.0

    return cnr, meta


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
    Compute per-channel CNR for Expert vs Novice using the paper-faithful
    group-HRF method (primary) and the legacy peak-based method (comparison).

    Primary — group-HRF (paper-faithful, Task 4.1 / 4.4):
      μ_g  = peak of grand-average HRF across subjects in group g
      σ_g  = inter-subject std of individual peak amplitudes in group g
      CNR  = |μ_g1 − μ_g2| / √(σ_g1² + σ_g2²)

    Legacy — peak-based (Task 4.2, kept for comparison):
      μ_g  = mean of per-subject peak amplitudes in group g
      σ_g  = std  of per-subject peak amplitudes in group g
      CNR  = |μ_g1 − μ_g2| / √(σ_g1² + σ_g2²)

    Returns
    -------
    df_cnr   : DataFrame  [channel, chromophore, cnr (group-HRF, primary),
                           {g1}_ga_peak, {g2}_ga_peak, {g1}_n, {g1}_sigma,
                           {g2}_n, {g2}_sigma, legacy_cnr]
    cnr_meta : dict       metadata including method comparison stats (Task 4.3)
    """
    group_names = list(groups.keys())
    g1_name, g2_name = group_names[0], group_names[1]

    # ── Channel list from first available Stage B .fif ────────────────────────
    first_subj = groups[g1_name][0]
    candidates = list(stage_b_dir.glob(f"sub-{first_subj}_*_raw.fif"))
    if not candidates:
        raise FileNotFoundError(
            f"No Stage B .fif found for {first_subj} in {stage_b_dir}"
        )
    raw_ref = mne.io.read_raw_fif(candidates[0], preload=False, verbose=False)
    hbo_long, hbr_long = get_long_channel_names(raw_ref)
    print(f"\n  Long-separation channels: {len(hbo_long)} HbO, {len(hbr_long)} HbR")

    # ── Load all evokeds once for group-HRF CNR (Task 4.1) ───────────────────
    print(f"\n  Loading evokeds for group-HRF CNR …")
    g1_evokeds = collect_all_evokeds(groups[g1_name], out_base, study, target_sfreq)
    g2_evokeds = collect_all_evokeds(groups[g2_name], out_base, study, target_sfreq)
    print(f"  {g1_name}: {len(g1_evokeds)} subjects | "
          f"{g2_name}: {len(g2_evokeds)} subjects")

    rows: list[dict] = []

    for chromo, long_chs in [("hbo", hbo_long), ("hbr", hbr_long)]:
        label = "HbO" if chromo == "hbo" else "HbR"
        print(f"\n  [CNR — {label}]")

        # ── Group-HRF CNR (primary, Task 4.1 / 4.4) ──────────────────────────
        ghrf_cnr, ghrf_meta = compute_group_hrf_cnr(
            g1_evokeds, g2_evokeds, long_chs, chromo, peak_window,
        )

        # ── Legacy peak-based CNR (Task 4.2, comparison only) ────────────────
        g1_peaks = collect_group_peaks(
            groups[g1_name], long_chs, chromo, peak_window,
            out_base, study, target_sfreq,
        )
        g2_peaks = collect_group_peaks(
            groups[g2_name], long_chs, chromo, peak_window,
            out_base, study, target_sfreq,
        )
        legacy_cnr_vals = compute_cnr(g1_peaks, g2_peaks)

        # ── Build per-channel rows ─────────────────────────────────────────────
        all_chs = sorted(set(long_chs))
        for ch in all_chs:
            m = ghrf_meta.get(ch, {})
            cnr_val = ghrf_cnr.get(ch, np.nan)
            leg_val = legacy_cnr_vals.get(ch, np.nan)
            rows.append({
                "channel":              ch,
                "chromophore":          label,
                "cnr":                  round(cnr_val, 4) if np.isfinite(cnr_val) else np.nan,
                f"{g1_name}_ga_peak":   round(m.get("mu1", np.nan), 4) if np.isfinite(m.get("mu1", np.nan)) else np.nan,
                f"{g2_name}_ga_peak":   round(m.get("mu2", np.nan), 4) if np.isfinite(m.get("mu2", np.nan)) else np.nan,
                f"{g1_name}_n":         m.get("n1", 0),
                f"{g1_name}_sigma":     round(m.get("sigma1", np.nan), 4) if np.isfinite(m.get("sigma1", np.nan)) else np.nan,
                f"{g2_name}_n":         m.get("n2", 0),
                f"{g2_name}_sigma":     round(m.get("sigma2", np.nan), 4) if np.isfinite(m.get("sigma2", np.nan)) else np.nan,
                "legacy_cnr":           round(leg_val, 4) if np.isfinite(leg_val) else np.nan,
            })

        # ── Per-chromophore stats ─────────────────────────────────────────────
        valid_ghrf = [v for v in ghrf_cnr.values() if np.isfinite(v)]
        valid_leg  = [v for v in legacy_cnr_vals.values() if np.isfinite(v)]
        if valid_ghrf:
            print(f"    {label} [group-HRF primary] : {len(valid_ghrf)} channels | "
                  f"median={np.median(valid_ghrf):.3f}  max={np.max(valid_ghrf):.3f}")
        if valid_leg:
            print(f"    {label} [legacy peak-based]  : {len(valid_leg)} channels | "
                  f"median={np.median(valid_leg):.3f}  max={np.max(valid_leg):.3f}")

    df_cnr = pd.DataFrame(rows)

    all_ghrf = df_cnr["cnr"].dropna().values
    all_leg  = df_cnr["legacy_cnr"].dropna().values

    cnr_meta = {
        "comparison":            f"Fully Preprocessed {g1_name} vs {g2_name}",
        "group1":                g1_name,
        "group2":                g2_name,
        "group1_n_subjects":     len(groups[g1_name]),
        "group2_n_subjects":     len(groups[g2_name]),
        "peak_window_s":         list(peak_window),
        "n_channels_hbo":        len(hbo_long),
        "n_channels_hbr":        len(hbr_long),
        "cnr_method":            "group_hrf",
        "n_valid_cnr":           int(len(all_ghrf)),
        "median_cnr":            round(float(np.median(all_ghrf)), 4) if len(all_ghrf) > 0 else None,
        "mean_cnr":              round(float(np.mean(all_ghrf)),   4) if len(all_ghrf) > 0 else None,
        "max_cnr":               round(float(np.max(all_ghrf)),    4) if len(all_ghrf) > 0 else None,
        "pct_above_1.0":         round(float((all_ghrf >= 1.0).mean() * 100), 1) if len(all_ghrf) > 0 else None,
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

    # ── 4.4.1 + 4.4.2 — Compute group-HRF CNR (primary) + legacy (comparison) ─
    df_cnr, cnr_meta = run_expert_novice_cnr(
        groups, out_base, stage_b_dir, study,
        peak_window=peak_window,
    )

    # ── 4.4.4 — Save CSV (group-HRF CNR as primary "cnr" column) ─────────────
    csv_name = f"full_preprocess_cnr_{study_display}.csv"
    csv_path = out_dir / csv_name
    save_cnr_csv(df_cnr, csv_path)

    # ── 4.4.3 — Method comparison report ─────────────────────────────────────
    leg = cnr_meta.get("legacy_peak_based", {})
    print(f"\n  Task 4.3 — Method Comparison (Full Preprocessing Pipeline):")
    print(f"  {'Method':<28} {'Median':>8}  {'Mean':>8}  {'Max':>8}  {'n valid':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Group-HRF':<28} "
          f"{str(cnr_meta['median_cnr']):>8}  "
          f"{str(cnr_meta['mean_cnr']):>8}  "
          f"{str(cnr_meta['max_cnr']):>8}  "
          f"{cnr_meta['n_valid_cnr']:>8}")
    print(f"  {'Legacy peak-based':<28} "
          f"{str(leg.get('median_cnr', 'N/A')):>8}  "
          f"{str(leg.get('mean_cnr', 'N/A')):>8}  "
          f"{str(leg.get('max_cnr', 'N/A')):>8}  "
          f"{leg.get('n_valid_cnr', 0):>8}")
    print(f"  {'Paper target (Figure 4i)':<28} {'~0.45':>8}  {'—':>8}  {'>2.25':>8}  {'—':>8}")

    # ── Build plot data — always include current study (4.4.4 primary) ────────
    valid_cnr = df_cnr["cnr"].dropna().tolist()   # group-HRF CNR values
    comparison_label = f"Fully Preprocessed {list(groups.keys())[0]} vs {list(groups.keys())[1]}"

    plot_data: dict[str, list[float] | None] = {comparison_label: valid_cnr}
    for lbl, val in comparison_registry.items():
        if lbl != comparison_label:
            plot_data[lbl] = val

    plot_path = out_dir / "full_preprocess_cnr_boxplot.png"
    plot_cnr_boxplot(plot_data, plot_path, cnr_threshold=cnr_threshold)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        **cnr_meta,
        "cnr_threshold_detectable": cnr_threshold,
        "outputs": {
            "cnr_csv":     str(csv_path),
            "cnr_boxplot": str(plot_path),
        },
    }
    summary_path = out_dir / "full_preprocess_cnr_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: full_preprocess_cnr_summary.json")

    # ── Console report ────────────────────────────────────────────────────────
    all_cnr = df_cnr["cnr"].dropna().values
    pct_above = (all_cnr >= cnr_threshold).mean() * 100 if len(all_cnr) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Section 4.4 complete  (method: group-HRF, paper-faithful)")
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