"""
Stage B — Step B4: Recursive PCA (rPCA) Motion Correction
Purpose: Apply PCA to motion-corrupted segments to remove high-variance motion
         components and preserve the low-variance neural signal.

Task coverage:
  3.4.1 — Implement rPCA algorithm (n_components=0.97)
  [3.4.2–3.4.5 added in subsequent tasks]

Algorithm rationale
-------------------
Upstream Step B3 (03_motion_correct.py) detects motion artifacts using the
Homer3 dual-criterion (AMPthresh=5.0 OD + STDEVthresh=20×std_diff) on filtered
OD data.  For subject A1 this flags a mean of ~17.4% of samples per channel
(union mask: all unique time points where any good channel was flagged).

rPCA motion correction (Homer3: hmrR_PCAfilter):
1.  Compute union motion mask (OR across all good channels) → motion time points
2.  Extract corrupted samples as data matrix X: shape (n_flagged, n_good_channels)
3.  PCA(n_components=0.97): retain the principal components that together explain
    97% of the variance in X.  In motion-corrupted windows these high-variance
    components correspond to the broadband motion artifact.
4.  Reconstruct the motion signal: X_motion = PCA.inverse_transform(PCA.transform(X))
5.  Corrected = original − motion_reconstruction  → keeps the low-variance neural
    residual at the corrupted time points.
6.  All un-flagged time points are unchanged.

Why global (not segment-by-segment)?
With 17.4% flagged at 20 Hz, the union mask covers ~4 600 time points across
60 good channels → matrix is (4 600, 60), well-conditioned for PCA.
Segment-by-segment processing risks short windows (<60 samples) where PCA
is ill-posed.  Global extraction is equivalent and numerically stable.

Reference: PROGRESS.md Section 3.4
Upstream:  03_motion_correct.py (Step B3 motion mask, use_amp_only=False)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

# ---------------------------------------------------------------------------
# Import Step B3 functions via importlib (avoid circular imports)
# ---------------------------------------------------------------------------
import importlib.util

_spec_b3 = importlib.util.spec_from_file_location(
    "motion_correct",
    Path(__file__).parent / "03_motion_correct.py",
)
_b3 = importlib.util.module_from_spec(_spec_b3)
_spec_b3.loader.exec_module(_b3)

detect_motion_artifacts = _b3.detect_motion_artifacts
compute_motion_stats    = _b3.compute_motion_stats
run_bandpass_filter     = _b3.run_bandpass_filter   # re-exported from B2 inside B3


# ===========================================================================
# Task 3.4.1 — rPCA Motion Correction Algorithm
# ===========================================================================

def apply_rpca_correction(
    raw_od_filt,
    motion_results: dict,
    n_components: float = 0.97,
    min_channels_flagged: int = 1,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.4.1: Apply rPCA to motion-corrupted samples to remove high-variance
    motion components and preserve the neural (low-variance) residual.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        Filtered OD data (Step B2 output, 0.01–0.5 Hz).
    motion_results : dict
        Output of detect_motion_artifacts() (Step B3, use_amp_only=False).
        Must contain 'masks' and 'bad_channels'.
    n_components : float
        Variance-explained threshold for PCA.  PCA retains the minimum number
        of components whose cumulative explained variance >= n_components.
        These high-variance components represent the motion artifact subspace.
        Default 0.97 → removes the dominant 97% of variance (motion).
    min_channels_flagged : int
        Minimum number of good channels that must simultaneously flag a time
        point for it to be included in the motion mask.
        1 (default) = OR union (flag if any channel detects motion → 99.9%).
        Higher values = "vote threshold" — only correct time points where
        motion is truly global (flagged by many channels at once).
        Example: 10 → flag only where ≥10/60 channels agree on motion.
    verbose : bool

    Returns
    -------
    raw_od_corrected : mne.io.Raw
        Corrected OD data (same shape; bad channels unchanged).
    correction_summary : dict
        {n_flagged, pct_flagged_union, n_components_used,
         variance_removed_pct, n_good_channels}
    """
    data    = raw_od_filt.get_data().copy()      # (n_channels, n_times)
    ch_names = raw_od_filt.ch_names
    masks    = motion_results['masks']
    bad_set  = set(motion_results['bad_channels'])
    n_times  = data.shape[1]

    good_idx   = [i for i, ch in enumerate(ch_names) if ch not in bad_set]
    good_names = [ch_names[i] for i in good_idx]
    n_good     = len(good_idx)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.1 — rPCA MOTION CORRECTION")
        print("=" * 70)
        print(f"  n_components         : {n_components}  "
              f"(PCA keeps components explaining ≥{n_components*100:.0f}% variance)")
        print(f"  min_channels_flagged : {min_channels_flagged}  "
              f"({'OR union' if min_channels_flagged == 1 else 'vote threshold'})")
        print(f"  Good channels        : {n_good}  |  Bad (skipped): {len(bad_set)}")

    # ------------------------------------------------------------------
    # Step 1: Build motion mask using vote threshold
    # ------------------------------------------------------------------
    # Count how many channels flag each time point simultaneously.
    # min_channels_flagged=1 replicates the original OR union (flag if any
    # channel detects motion). Higher values make the mask more selective,
    # requiring global agreement among multiple channels.
    ch_vote_counts = np.zeros(n_times, dtype=np.int32)
    for ch in good_names:
        if ch in masks:
            ch_vote_counts += masks[ch].astype(np.int32)
    union_mask = ch_vote_counts >= min_channels_flagged

    n_flagged   = int(union_mask.sum())
    pct_flagged = 100.0 * n_flagged / n_times

    if verbose:
        print(f"\n  Motion mask: {n_flagged}/{n_times} samples "
              f"({pct_flagged:.1f}%) flagged  "
              f"[≥{min_channels_flagged} ch voting]")

    if n_flagged == 0:
        if verbose:
            print("  No motion detected — no correction applied.")
        empty_summary = {
            'n_flagged': 0,
            'pct_flagged_union': 0.0,
            'n_components_used': 0,
            'variance_removed_pct': 0.0,
            'n_good_channels': n_good,
        }
        return raw_od_filt.copy(), empty_summary

    # ------------------------------------------------------------------
    # Step 2: Extract corrupted time points
    # ------------------------------------------------------------------
    motion_idx = np.where(union_mask)[0]

    # Shape: (n_flagged, n_good_channels) — sklearn PCA expects (n_samples, n_features)
    # n_flagged is the time points where the union_mask returned True (motion-corrupted time points)
    # n_good: the label of good channels (the channels that actually have data, for subject A1 6 bad channels were removed because they weren't in 80-140 dB)
    X = data[np.ix_(good_idx, motion_idx)].T   # (n_flagged, n_good)

    if verbose:
        print(f"  Data matrix    : {X.shape[0]} samples × {X.shape[1]} channels")

    # ------------------------------------------------------------------
    # Step 3: PCA — retain components explaining n_components of variance
    # ------------------------------------------------------------------
    # PCA(n_components=float) with svd_solver='full' requires n_samples > n_features.
    # Here n_flagged >> n_good (4600 >> 60) so the float threshold works directly.
    pca = PCA(n_components=n_components, svd_solver='full')
    #pca.fit_transform(X) returns k-dimensional principal components, which are mathematical patterns of how the noise in the good channels move together
    X_transformed  = pca.fit_transform(X)            # (n_flagged, k_PCs)
    #use the k-dimensional principal components to reconstruct the motion signal
    X_motion_recon = pca.inverse_transform(X_transformed)  # (n_flagged, n_good)

    n_pc        = int(pca.n_components_)
    var_removed = float(np.sum(pca.explained_variance_ratio_))

    if verbose:
        print(f"\n  PCA result:")
        print(f"    Components retained  : {n_pc}")
        print(f"    Variance explained   : {var_removed*100:.2f}%  (motion subspace)")
        print(f"    Variance remaining   : {(1 - var_removed)*100:.2f}%  (neural residual)")

    # ------------------------------------------------------------------
    # Step 4: Corrected = original − motion reconstruction
    # ------------------------------------------------------------------
    X_corrected = X - X_motion_recon          # (n_flagged, n_good)

    # ------------------------------------------------------------------
    # Step 5: Insert corrected values back at flagged time points
    # ------------------------------------------------------------------
    data[np.ix_(good_idx, motion_idx)] = X_corrected.T   # (n_good, n_flagged)

    raw_od_corrected = raw_od_filt.copy()
    raw_od_corrected._data = data

    summary = {
        'n_flagged':              n_flagged,
        'pct_flagged_union':      round(pct_flagged, 2),
        'n_components_used':      n_pc,
        'variance_removed_pct':   round(var_removed * 100, 2),
        'n_good_channels':        n_good,
        'n_components_param':     n_components,
        'min_channels_flagged':   min_channels_flagged,
    }

    if verbose:
        print(f"\n  ✅ rPCA applied to {n_flagged} samples ({pct_flagged:.1f}%) "
              f"using {n_pc} PCs (≥{n_components*100:.0f}% variance removed)")
        print("=" * 70 + "\n")

    return raw_od_corrected, summary


# ===========================================================================
# Task 3.4.2 — Before/After Motion Correction Comparison Visualization
# ===========================================================================

def plot_before_after_correction(
    raw_od_filt,
    raw_od_corrected,
    motion_results: dict,
    stats_before: pd.DataFrame,
    correction_summary: dict,
    subject_id: str,
    output_dir: Path,
    n_high: int = 2,
    n_low: int = 2,
    t_window: float = 300.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.4.2: Before/after rPCA correction comparison.

    Layout
    ------
    For each selected channel one row with three panels:
      [Before OD | After OD (corrected) | Removed signal (Before − After)]

      Before OD (blue): graph of the filtered OD trace with red shading covering the union motion mask (motion-flagged time pointS)
        - This is the original filtered OD trace before the rPCA correction
      After OD (green): graph of the corrected OD trace with red shading covering the union motion mask (motion-flagged time points)
        - This is the corrected OD trace after the rPCA correction and ideally the actual oxygenation change 
      Removed signal (red): graph of the difference between the before and after OD traces
        - This is the actual noise that was removed by rPCA correction. 

      Overall, Before OD = After OD + Removed signal 
      Blue = Green + Red

    Channel selection: n_high channels with highest motion ratio (most artifact)
    + n_low channels with lowest motion ratio (least artifact / ok).  Union
    motion mask is shaded red in the Before and After panels.

    Parameters
    ----------
    raw_od_filt      : mne.io.Raw  (Step B2 filtered OD — before correction)
    raw_od_corrected : mne.io.Raw  (Step B4 corrected OD — after correction)
    motion_results   : dict        (Step B3 masks + params)
    stats_before     : pd.DataFrame  (per-channel motion stats, pre-correction)
    correction_summary : dict      (rPCA metrics from apply_rpca_correction)
    subject_id       : str
    output_dir       : Path
    n_high           : int   number of high-motion channels to display
    n_low            : int   number of low/ok-motion channels to display
    t_window         : float display window length in seconds (default 300 s)
    verbose          : bool

    Returns
    -------
    fig_path : Path
    """
    sfreq       = raw_od_filt.info['sfreq']
    n_plot_samp = int(t_window * sfreq)
    times       = raw_od_filt.times[:n_plot_samp]
    data_before = raw_od_filt.get_data()
    data_after  = raw_od_corrected.get_data()
    ch_names    = raw_od_filt.ch_names
    bad_set     = set(motion_results['bad_channels'])
    masks       = motion_results['masks']

    # -- Union motion mask for shading (first t_window seconds) ---------------
    n_times    = data_before.shape[1]
    good_names = [ch for ch in ch_names if ch not in bad_set]
    union_mask = np.zeros(n_times, dtype=bool)
    for ch in good_names:
        if ch in masks:
            union_mask |= masks[ch]
    union_seg = union_mask[:n_plot_samp]

    # -- Channel selection: highest and lowest motion ratios ------------------
    good_df = stats_before[stats_before['motion_quality'] != 'bad_channel'].copy()
    sorted_asc  = good_df.sort_values('motion_ratio', ascending=True)
    sorted_desc = good_df.sort_values('motion_ratio', ascending=False)

    low_chs  = sorted_asc['channel'].tolist()[:n_low]
    high_chs = sorted_desc['channel'].tolist()[:n_high]
    plot_chs = high_chs + low_chs      # high-motion first, then low-motion

    n_rows = len(plot_chs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3.2 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Before rPCA (filtered OD)', 'After rPCA (corrected)', 'Removed (Before − After)']
    col_colors = ['#1f77b4', '#2ca02c', '#d62728']

    for row_idx, ch_name in enumerate(plot_chs):
        ch_idx  = ch_names.index(ch_name)
        before  = data_before[ch_idx, :n_plot_samp]
        after   = data_after[ch_idx, :n_plot_samp]
        removed = before - after

        row_stats = stats_before[stats_before['channel'] == ch_name].iloc[0]
        is_high   = row_stats['motion_quality'] in ('elevated_motion', 'high_motion')
        tag       = f"{row_stats['motion_quality']}  {row_stats['pct_corrupted']:.1f}% flagged"
        ch_label  = f"{ch_name}  [{tag}]"

        # Per-channel variance ratio: what fraction was removed
        var_before  = float(np.var(before))
        var_after   = float(np.var(after))
        var_removed = float(np.var(removed))
        var_ratio   = (var_before - var_after) / var_before * 100 if var_before > 0 else 0.0

        for col, (sig, col_color) in enumerate(zip([before, after, removed], col_colors)):
            ax = axes[row_idx, col]

            # Shade union-mask regions
            if col < 2:   # shade Before and After, not Removed
                _shade_union(ax, times, union_seg, color='red', alpha=0.18)

            ax.plot(times, sig, color=col_color, lw=0.65, alpha=0.9)
            ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.3)
            ax.set_ylabel('OD', fontsize=8)

            if row_idx == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold',
                             color=col_color)

            if col == 0:
                ax.set_ylabel(ch_label, fontsize=7.5, labelpad=4)
                ax.yaxis.set_label_position('left')

            if col == 2:
                ax.text(0.98, 0.96, f'Var removed: {var_ratio:.1f}%',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color='#d62728',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        axes[row_idx, -1].set_ylabel('OD\n(removed)', fontsize=7.5)

    axes[-1, 0].set_xlabel('Time (s)', fontsize=9)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=9)
    axes[-1, 2].set_xlabel('Time (s)', fontsize=9)

    # Add legend patch for motion shading
    from matplotlib.patches import Patch
    legend_patch = Patch(color='red', alpha=0.25, label='Motion-flagged (union mask)')
    fig.legend(handles=[legend_patch], loc='upper right', fontsize=8,
               bbox_to_anchor=(1.0, 1.01))

    fig.suptitle(
        f'Subject {subject_id} — rPCA Motion Correction: Before vs After\n'
        f'n_components={correction_summary["n_components_param"]}  |  '
        f'PCs removed={correction_summary["n_components_used"]}  |  '
        f'Variance removed={correction_summary["variance_removed_pct"]:.1f}%  |  '
        f'Union mask={correction_summary["pct_flagged_union"]:.1f}%  |  '
        f'Display window: first {t_window:.0f}s',
        fontsize=10, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_rpca_before_after.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved before/after plot : {fig_path}")

    return fig_path


def _shade_union(ax, times, mask, color='red', alpha=0.18):
    """Shade contiguous True blocks in mask."""
    in_block = False
    t_start  = None
    for i, flagged in enumerate(mask):
        if flagged and not in_block:
            in_block = True
            t_start  = times[i]
        elif not flagged and in_block:
            in_block = False
            ax.axvspan(t_start, times[i - 1], color=color, alpha=alpha)
    if in_block:
        ax.axvspan(t_start, times[-1], color=color, alpha=alpha)


# ===========================================================================
# Tasks 3.4.3 & 3.4.4 — Post-correction motion verification and effectiveness
# ===========================================================================

def verify_motion_reduction(
    raw_od_filt,
    raw_od_corrected,
    motion_results_before: dict,
    stats_before: pd.DataFrame,
    amp_thresh: float = 5.0,
    stdev_thresh: float = 20.0,
    t_motion: float = 0.5,
    t_mask: float = 0.5,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.4.3: Re-run motion detection on rPCA-corrected OD and quantify
    the reduction in motion artifacts vs the pre-correction baseline.

    Both the dual criterion (AMPthresh + STDEVthresh) and the amplitude-only
    criterion are evaluated on the corrected signal to give a complete picture
    of how the correction changed signal statistics.

    Parameters
    ----------
    raw_od_filt      : mne.io.Raw  (before — filtered OD)
    raw_od_corrected : mne.io.Raw  (after  — rPCA corrected OD)
    motion_results_before : dict   (Step B3 masks on pre-correction OD)
    stats_before     : pd.DataFrame
    amp_thresh, stdev_thresh, t_motion, t_mask : float
        Same detection parameters used in Step B3.
    verbose : bool

    Returns
    -------
    stats_after : pd.DataFrame
        Per-channel motion stats on the corrected OD (dual criterion).
    comparison_df : pd.DataFrame
        Per-channel comparison: channel, motion_ratio_before, motion_ratio_after,
        reduction_abs, reduction_pct, quality_before, quality_after.
    reduction_summary : dict
        Aggregate metrics: mean/median motion ratio before and after,
        mean reduction, n_channels_improved.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.3 — POST-CORRECTION MOTION VERIFICATION")
        print("=" * 70)
        print("Re-running Homer3 dual criterion on rPCA-corrected OD...")

    # Re-detect on corrected data (dual criterion, same params as B3)
    motion_results_after = detect_motion_artifacts(
        raw_od_corrected,
        amp_thresh   = amp_thresh,
        stdev_thresh = stdev_thresh,
        t_motion     = t_motion,
        t_mask       = t_mask,
        use_amp_only = False,   # dual criterion — same as before
        verbose      = False,
    )
    stats_after = compute_motion_stats(
        motion_results_after, raw_od_corrected, verbose=False
    )

    # Also run amplitude-only on corrected to see if large spikes remain
    motion_results_amp_only = detect_motion_artifacts(
        raw_od_corrected,
        amp_thresh   = amp_thresh,
        stdev_thresh = stdev_thresh,
        t_motion     = t_motion,
        t_mask       = t_mask,
        use_amp_only = True,
        verbose      = False,
    )
    stats_amp_only = compute_motion_stats(
        motion_results_amp_only, raw_od_corrected, verbose=False
    )

    # Build per-channel comparison DataFrame
    bad_set = set(motion_results_before['bad_channels'])
    records = []
    for ch in raw_od_filt.ch_names:
        if ch in bad_set:
            continue
        row_b = stats_before[stats_before['channel'] == ch].iloc[0]
        row_a = stats_after[stats_after['channel'] == ch].iloc[0]
        row_amp = stats_amp_only[stats_amp_only['channel'] == ch].iloc[0]

        mr_before = float(row_b['motion_ratio'])
        mr_after  = float(row_a['motion_ratio'])
        mr_amp    = float(row_amp['motion_ratio'])
        reduction_abs = mr_before - mr_after
        reduction_pct = (reduction_abs / mr_before * 100) if mr_before > 0 else 0.0

        records.append({
            'channel':            ch,
            'wavelength':         int(row_b['wavelength']),
            'motion_ratio_before': round(mr_before, 4),
            'motion_ratio_after':  round(mr_after, 4),
            'motion_ratio_amp_only': round(mr_amp, 4),
            'reduction_abs':       round(reduction_abs, 4),
            'reduction_pct':       round(reduction_pct, 2),
            'quality_before':      row_b['motion_quality'],
            'quality_after':       row_a['motion_quality'],
        })

    comparison_df = pd.DataFrame(records)

    # Aggregate summary
    good_before = stats_before[stats_before['motion_quality'] != 'bad_channel']
    good_after  = stats_after[stats_after['motion_quality']  != 'bad_channel']
    good_amp    = stats_amp_only[stats_amp_only['motion_quality'] != 'bad_channel']

    mean_before = float(good_before['motion_ratio'].mean())
    mean_after  = float(good_after['motion_ratio'].mean())
    mean_amp    = float(good_amp['motion_ratio'].mean())
    n_improved  = int((comparison_df['reduction_abs'] > 0).sum())
    n_worse     = int((comparison_df['reduction_abs'] < 0).sum())

    reduction_summary = {
        'mean_motion_ratio_before':      round(mean_before, 4),
        'mean_motion_ratio_after_dual':  round(mean_after,  4),
        'mean_motion_ratio_after_amp':   round(mean_amp,    4),
        'mean_reduction_abs':            round(mean_before - mean_after, 4),
        'mean_reduction_pct':            round((mean_before - mean_after) / mean_before * 100, 2) if mean_before > 0 else 0.0,
        'n_channels_improved':           n_improved,
        'n_channels_worse':              n_worse,
        'n_channels_unchanged':          len(comparison_df) - n_improved - n_worse,
    }

    if verbose:
        print(f"\n  {'Criterion':<30} {'Before':>10} {'After':>10} {'Reduction':>12}")
        print(f"  {'-'*64}")
        print(f"  {'Dual (AMP + STDEV)':<30} "
              f"{mean_before*100:>9.1f}%  "
              f"{mean_after*100:>9.1f}%  "
              f"{(mean_before-mean_after)*100:>10.1f}%")
        print(f"  {'Amplitude-only (≥5.0 OD)':<30} "
              f"{'—':>10}  "
              f"{mean_amp*100:>9.1f}%  {'—':>12}")
        print(f"\n  Channels improved : {n_improved} / {len(comparison_df)}")
        print(f"  Channels worse    : {n_worse}")
        print("=" * 70 + "\n")

    return stats_after, comparison_df, reduction_summary


def plot_motion_reduction_summary(
    comparison_df: pd.DataFrame,
    reduction_summary: dict,
    correction_summary: dict,
    subject_id: str,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """
    Task 3.4.4: Visualize per-channel motion reduction (before vs after rPCA).

    Figure layout (2 rows):
      Row 1: Per-channel grouped bar chart — Before (orange) vs After (teal)
             motion ratio, sorted by before-correction ratio.
      Row 2: Per-channel reduction (%) bar chart — green = improved,
             red = worse.

    Parameters
    ----------
    comparison_df    : pd.DataFrame  (output of verify_motion_reduction)
    reduction_summary : dict
    correction_summary : dict        (rPCA metrics)
    subject_id       : str
    output_dir       : Path
    verbose          : bool

    Returns
    -------
    fig_path : Path
    """
    df = comparison_df.copy().sort_values('motion_ratio_before', ascending=False)
    n_ch = len(df)
    x    = np.arange(n_ch)
    w    = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_ch * 0.22), 9),
                             gridspec_kw={'height_ratios': [2, 1]})

    # ---- Row 1: Grouped bars Before / After ---------------------------------
    ax1 = axes[0]
    bars_b = ax1.bar(x - w/2, df['motion_ratio_before'] * 100,
                     width=w, color='#ff7f0e', alpha=0.85, label='Before rPCA')
    bars_a = ax1.bar(x + w/2, df['motion_ratio_after'] * 100,
                     width=w, color='#17becf', alpha=0.85, label='After rPCA (dual)')

    # Add amp-only dots on After bar
    ax1.scatter(x + w/2, df['motion_ratio_amp_only'] * 100,
                color='#2ca02c', s=18, zorder=5, label='After rPCA (amp-only)',
                marker='D')

    ax1.axhline(10, color='gray', lw=0.8, ls='--', alpha=0.6, label='Elevated threshold (10%)')
    ax1.set_ylabel('Motion Ratio (%)', fontsize=10)
    ax1.set_title(
        f'Subject {subject_id} — Per-Channel Motion Ratio: Before vs After rPCA\n'
        f'n_components={correction_summary["n_components_param"]}  |  '
        f'PCs removed={correction_summary["n_components_used"]}  |  '
        f'Variance removed={correction_summary["variance_removed_pct"]:.1f}%',
        fontsize=10, fontweight='bold',
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['channel'], rotation=90, fontsize=5.5)
    ax1.legend(fontsize=8, loc='upper right')

    # Annotate aggregate stats in top-right corner
    stats_text = (
        f"Mean before : {reduction_summary['mean_motion_ratio_before']*100:.1f}%\n"
        f"Mean after  : {reduction_summary['mean_motion_ratio_after_dual']*100:.1f}%  (dual)\n"
        f"Mean after  : {reduction_summary['mean_motion_ratio_after_amp']*100:.1f}%  (amp-only)\n"
        f"Mean Δ      : {reduction_summary['mean_reduction_pct']:.1f}% reduction\n"
        f"Improved    : {reduction_summary['n_channels_improved']}/{n_ch} channels"
    )
    ax1.text(0.01, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    # ---- Row 2: Reduction (%) per channel -----------------------------------
    ax2 = axes[1]
    colors_red = ['#2ca02c' if v >= 0 else '#d62728'
                  for v in df['reduction_pct']]
    ax2.bar(x, df['reduction_pct'], color=colors_red, alpha=0.85, width=0.7)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_ylabel('Reduction (%)', fontsize=10)
    ax2.set_xlabel('Channel', fontsize=10)
    ax2.set_title('Per-Channel Motion Ratio Reduction After rPCA  '
                  '(green = improved, red = worse)', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['channel'], rotation=90, fontsize=5.5)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_rpca_motion_reduction.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved motion reduction plot : {fig_path}")

    return fig_path


def save_motion_comparison(
    comparison_df: pd.DataFrame,
    reduction_summary: dict,
    subject_id: str,
    output_dir: Path,
) -> tuple:
    """Save per-channel comparison CSV and reduction summary JSON."""
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / f'{subject_id}_rpca_motion_comparison.csv'
    json_path = output_dir / f'{subject_id}_rpca_effectiveness.json'

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(reduction_summary, f, indent=2)

    print(f"Saved motion comparison CSV : {csv_path}")
    print(f"Saved effectiveness JSON    : {json_path}")
    return csv_path, json_path


# ===========================================================================
# Task 3.4.5 — Parameter Tuning
# ===========================================================================

def tune_rpca_parameters(
    raw_od_filt,
    motion_results: dict,
    stats_before: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    n_components_grid: list = None,
    min_channels_grid: list = None,
    amp_thresh: float = 5.0,
    stdev_thresh: float = 20.0,
    t_motion: float = 0.5,
    t_mask: float = 0.5,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.4.5: Grid search over n_components × min_channels_flagged to find
    the configuration that maximises post-correction motion reduction.

    Two axes of variation
    ---------------------
    n_components (float):
        PCA variance threshold.  Higher → more PCs removed → less residual
        variance.  Range: 0.90–0.99.

    min_channels_flagged (int):
        Vote threshold for the motion mask.  1 = OR union (current: 99.9%).
        Higher values target only time points where motion is truly global
        (flagged by many channels simultaneously), leaving the rest of the
        signal untouched.

    Evaluation metrics (per configuration)
    ----------------------------------------
    - pct_mask   : how much of the signal is inside the motion mask
    - n_pcs      : number of PCs removed
    - var_removed: % variance removed from mask segment
    - MR_dual    : mean post-correction motion ratio (dual criterion)
    - MR_amp     : mean post-correction motion ratio (amplitude-only ≥5 OD)
    - reduction  : relative reduction in mean MR (dual) vs before

    Parameters
    ----------
    raw_od_filt      : mne.io.Raw
    motion_results   : dict  (Step B3 masks, use_amp_only=False)
    stats_before     : pd.DataFrame
    subject_id       : str
    output_dir       : Path
    n_components_grid : list of float  (default [0.90, 0.95, 0.97, 0.99])
    min_channels_grid : list of int    (default [1, 3, 5, 10, 15])
    amp_thresh etc.   : motion detection params (same as Step B3)
    verbose           : bool

    Returns
    -------
    results_df  : pd.DataFrame  (one row per configuration)
    best_params : dict          (optimal n_components, min_channels_flagged)
    """
    if n_components_grid is None:
        n_components_grid = [0.90, 0.95, 0.97, 0.99]
    if min_channels_grid is None:
        min_channels_grid = [1, 3, 5, 10, 15]

    good_df    = stats_before[stats_before['motion_quality'] != 'bad_channel']
    mr_before  = float(good_df['motion_ratio'].mean())
    n_configs  = len(n_components_grid) * len(min_channels_grid)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.5 — rPCA PARAMETER TUNING SWEEP")
        print("=" * 70)
        print(f"  Baseline mean MR (dual criterion): {mr_before*100:.2f}%")
        print(f"  n_components grid : {n_components_grid}")
        print(f"  min_ch_flagged grid: {min_channels_grid}")
        print(f"  Total configurations: {n_configs}")
        print("-" * 70)
        hdr = (f"{'n_comp':>7} {'min_ch':>6} {'mask%':>7} {'n_PCs':>6} "
               f"{'var_rem':>8} {'MR_dual%':>9} {'MR_amp%':>8} {'reduc%':>8}")
        print(hdr)
        print("-" * 70)

    records = []
    for n_comp in n_components_grid:
        for min_ch in min_channels_grid:
            # Apply rPCA with this configuration
            raw_corr, corr_sum = apply_rpca_correction(
                raw_od_filt,
                motion_results,
                n_components         = n_comp,
                min_channels_flagged = min_ch,
                verbose              = False,
            )

            # Post-correction motion detection (dual criterion)
            mr_after_results = detect_motion_artifacts(
                raw_corr,
                amp_thresh   = amp_thresh,
                stdev_thresh = stdev_thresh,
                t_motion     = t_motion,
                t_mask       = t_mask,
                use_amp_only = False,
                verbose      = False,
            )
            mr_after_stats = compute_motion_stats(
                mr_after_results, raw_corr, verbose=False
            )

            # Amplitude-only check
            mr_amp_results = detect_motion_artifacts(
                raw_corr,
                amp_thresh   = amp_thresh,
                stdev_thresh = stdev_thresh,
                t_motion     = t_motion,
                t_mask       = t_mask,
                use_amp_only = True,
                verbose      = False,
            )
            mr_amp_stats = compute_motion_stats(
                mr_amp_results, raw_corr, verbose=False
            )

            good_after = mr_after_stats[mr_after_stats['motion_quality'] != 'bad_channel']
            good_amp   = mr_amp_stats[mr_amp_stats['motion_quality']  != 'bad_channel']
            mr_dual    = float(good_after['motion_ratio'].mean())
            mr_amp     = float(good_amp['motion_ratio'].mean())
            reduction  = (mr_before - mr_dual) / mr_before * 100 if mr_before > 0 else 0.0

            rec = {
                'n_components':       n_comp,
                'min_channels_flagged': min_ch,
                'pct_mask':           corr_sum['pct_flagged_union'],
                'n_pcs_removed':      corr_sum['n_components_used'],
                'variance_removed_pct': corr_sum['variance_removed_pct'],
                'MR_before':          round(mr_before * 100, 2),
                'MR_after_dual':      round(mr_dual * 100, 2),
                'MR_after_amp':       round(mr_amp * 100, 2),
                'reduction_pct':      round(reduction, 2),
            }
            records.append(rec)

            if verbose:
                print(f"  {n_comp:>7.2f} {min_ch:>6d} {rec['pct_mask']:>7.1f} "
                      f"{rec['n_pcs_removed']:>6d} {rec['variance_removed_pct']:>7.1f}% "
                      f"{rec['MR_after_dual']:>8.2f}% {rec['MR_after_amp']:>7.2f}% "
                      f"{rec['reduction_pct']:>7.1f}%")

    results_df = pd.DataFrame(records)

    # Best = highest reduction_pct with MR_after_amp == 0 (no large spikes)
    # If all have MR_after_amp == 0, sort purely by reduction_pct
    clean = results_df[results_df['MR_after_amp'] == 0.0]
    if clean.empty:
        clean = results_df
    best_row   = clean.sort_values('reduction_pct', ascending=False).iloc[0]
    best_params = {
        'n_components':       float(best_row['n_components']),
        'min_channels_flagged': int(best_row['min_channels_flagged']),
        'MR_after_dual':      float(best_row['MR_after_dual']),
        'MR_after_amp':       float(best_row['MR_after_amp']),
        'reduction_pct':      float(best_row['reduction_pct']),
        'pct_mask':           float(best_row['pct_mask']),
        'n_pcs_removed':      int(best_row['n_pcs_removed']),
        'variance_removed_pct': float(best_row['variance_removed_pct']),
    }

    if verbose:
        print("-" * 70)
        print(f"\n  ★  Optimal configuration:")
        print(f"     n_components         = {best_params['n_components']}")
        print(f"     min_channels_flagged = {best_params['min_channels_flagged']}")
        print(f"     Mask coverage        = {best_params['pct_mask']:.1f}%")
        print(f"     PCs removed          = {best_params['n_pcs_removed']}")
        print(f"     Variance removed     = {best_params['variance_removed_pct']:.1f}%")
        print(f"     MR after (dual)      = {best_params['MR_after_dual']:.2f}%")
        print(f"     MR after (amp-only)  = {best_params['MR_after_amp']:.2f}%")
        print(f"     Reduction            = {best_params['reduction_pct']:.1f}%")
        print("=" * 70 + "\n")

    # Save results table
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{subject_id}_rpca_tuning_results.csv'
    results_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Saved tuning results: {csv_path}")

    return results_df, best_params


def plot_tuning_results(
    results_df: pd.DataFrame,
    best_params: dict,
    subject_id: str,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """
    Task 3.4.5: Visualize the parameter tuning trade-off surface.

    Two panels:
      Left  : Heat-map of post-correction MR_dual (%) over the
               n_components × min_channels_flagged grid.
      Right : Heat-map of motion reduction (%) over the same grid.
    Best configuration is marked with a star on both panels.
    """
    import matplotlib.colors as mcolors

    n_comp_vals = sorted(results_df['n_components'].unique())
    min_ch_vals = sorted(results_df['min_channels_flagged'].unique())

    def _pivot(col):
        return results_df.pivot(
            index='min_channels_flagged',
            columns='n_components',
            values=col,
        )

    mr_grid  = _pivot('MR_after_dual')
    red_grid = _pivot('reduction_pct')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def _heatmap(ax, data, title, cmap, fmt='.1f', vmin=None, vmax=None):
        im = ax.imshow(data.values, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(n_comp_vals)))
        ax.set_xticklabels([str(v) for v in n_comp_vals], fontsize=9)
        ax.set_yticks(range(len(min_ch_vals)))
        ax.set_yticklabels([str(v) for v in min_ch_vals], fontsize=9)
        ax.set_xlabel('n_components', fontsize=10)
        ax.set_ylabel('min_channels_flagged', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)
        # Annotate each cell
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data.values[r, c]
                ax.text(c, r, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=8, color='white' if val > data.values.mean() else 'black')
        # Mark best
        best_r = list(min_ch_vals).index(best_params['min_channels_flagged'])
        best_c = list(n_comp_vals).index(best_params['n_components'])
        ax.plot(best_c, best_r, marker='*', color='gold', ms=18, zorder=5,
                markeredgecolor='k', markeredgewidth=0.8)

    _heatmap(axes[0], mr_grid,
             f'Post-correction MR (dual, %)  ← lower is better',
             'YlOrRd_r', fmt='.1f')
    _heatmap(axes[1], red_grid,
             f'Reduction vs baseline (%)  ← higher is better',
             'YlGn', fmt='.1f')

    fig.suptitle(
        f'Subject {subject_id} — rPCA Parameter Tuning\n'
        f'Baseline mean MR = {results_df["MR_before"].iloc[0]:.1f}%  '
        f'(dual criterion)   ★ = optimal: '
        f'n_comp={best_params["n_components"]}, '
        f'min_ch={best_params["min_channels_flagged"]} '
        f'→ MR={best_params["MR_after_dual"]:.1f}% '
        f'({best_params["reduction_pct"]:.1f}% reduction)',
        fontsize=10, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    fig_path = output_dir / f'{subject_id}_rpca_tuning_heatmap.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved tuning heatmap : {fig_path}")
    return fig_path


def save_corrected_od(
    raw_od_corrected,
    correction_summary: dict,
    subject_id: str,
    output_dir: Path,
) -> tuple:
    """
    Save the rPCA-corrected OD data and correction summary.

    Returns
    -------
    fif_path    : Path  (.fif file)
    summary_path: Path  (JSON summary)
    """
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save corrected OD as .fif
    fif_path = output_dir / f"{subject_id}_od_corrected_rpca_raw.fif"
    raw_od_corrected.save(str(fif_path), overwrite=True, verbose=False)
    print(f"Saved corrected OD : {fif_path}")

    # Save summary JSON
    summary_path = output_dir / f"{subject_id}_rpca_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(correction_summary, f, indent=2)
    print(f"Saved rPCA summary : {summary_path}")

    return fif_path, summary_path


# ===========================================================================
# End-to-end pipeline — Task 3.4.1 (B4 Step 1)
# ===========================================================================

def run_rpca_correction(
    dataset_name: str,
    subject_id: str,
    n_components: float = 0.97,
    min_channels_flagged: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Task 3.4.1–3.4.4 pipeline: Load filtered OD → motion detect → rPCA correct
    → save → before/after visualization → post-correction motion verification
    → effectiveness documentation.

    Motion detection uses the dual criterion (use_amp_only=False):
      AMPthresh=5.0 OD + STDEVthresh=20×std_diff
    This produces ~17.4% flagged samples for subject A1 (mean across good
    channels), forming the correction target for rPCA.

    Parameters
    ----------
    dataset_name : str
    subject_id   : str
    n_components : float  PCA variance threshold (default 0.97)
    min_channels_flagged : int  vote threshold for the motion mask (default 1 = OR union)
    verbose      : bool

    Returns
    -------
    results : dict
        {
          'raw_od_filt':       mne.io.Raw  (filtered OD, input to rPCA),
          'raw_od_corrected':  mne.io.Raw  (rPCA-corrected OD),
          'motion_results':    dict        (Step B3 masks + params),
          'stats_before':      pd.DataFrame (per-channel stats, pre-correction),
          'correction_summary': dict       (rPCA metrics),
          'output_dir':        Path,
        }
    """
    config    = load_config()
    b3_params = get_stage_params('stage_b', config)['motion_detection']

    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B4: rPCA MOTION CORRECTION  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Step B1+B2: filtered OD -------------------------------------------
    print("\n[1/3] Loading filtered OD (Steps B1+B2)...")
    b2_results  = run_bandpass_filter(dataset_name, subject_id, verbose=False)
    raw_od_filt = b2_results['raw_od_filt']

    # ---- Step B3: motion detection (dual criterion) ------------------------
    print("[2/3] Detecting motion artifacts (dual criterion, use_amp_only=False)...")
    motion_results = detect_motion_artifacts(
        raw_od_filt,
        amp_thresh   = float(b3_params.get('AMPthresh', 5.0)),
        stdev_thresh = float(b3_params['tStdev']),    # 20.0
        t_motion     = float(b3_params['tMotion']),   # 0.5
        t_mask       = 0.5,
        use_amp_only = False,   # dual criterion → 17.4% for A1
        verbose      = verbose,
    )
    stats_before = compute_motion_stats(motion_results, raw_od_filt, verbose=verbose)

    # ---- Step B4.1: rPCA correction ----------------------------------------
    print("[3/4] Applying rPCA correction (Task 3.4.1)...")
    raw_od_corrected, correction_summary = apply_rpca_correction(
        raw_od_filt,
        motion_results,
        n_components         = n_components,
        min_channels_flagged = min_channels_flagged,
        verbose              = verbose,
    )

    # ---- Save outputs -------------------------------------------------------
    output_dir = get_output_path('stage_b', f'rPCA_correction/{subject_id}')
    fif_path, summary_path = save_corrected_od(
        raw_od_corrected,
        correction_summary,
        subject_id,
        output_dir,
    )

    # ---- 3.4.2 Before/after visualization ----------------------------------
    print("[4/7] Generating before/after correction comparison plot (Task 3.4.2)...")
    fig_ba_path = plot_before_after_correction(
        raw_od_filt,
        raw_od_corrected,
        motion_results,
        stats_before,
        correction_summary,
        subject_id  = subject_id,
        output_dir  = output_dir,
        n_high      = 2,
        n_low       = 2,
        t_window    = 300.0,
        verbose     = verbose,
    )

    # ---- 3.4.3 Post-correction motion verification -------------------------
    print("[5/7] Verifying motion reduction post-rPCA (Task 3.4.3)...")
    b3_params = get_stage_params('stage_b', config)['motion_detection']
    stats_after, comparison_df, reduction_summary = verify_motion_reduction(
        raw_od_filt,
        raw_od_corrected,
        motion_results,
        stats_before,
        amp_thresh   = float(b3_params.get('AMPthresh', 5.0)),
        stdev_thresh = float(b3_params['tStdev']),
        t_motion     = float(b3_params['tMotion']),
        t_mask       = 0.5,
        verbose      = verbose,
    )

    # ---- 3.4.4 Effectiveness documentation ---------------------------------
    print("[6/7] Documenting rPCA effectiveness (Task 3.4.4)...")
    fig_red_path = plot_motion_reduction_summary(
        comparison_df,
        reduction_summary,
        correction_summary,
        subject_id = subject_id,
        output_dir = output_dir,
        verbose    = verbose,
    )
    csv_path, eff_path = save_motion_comparison(
        comparison_df, reduction_summary, subject_id, output_dir
    )

    # ---- 3.4.5 Parameter tuning --------------------------------------------
    print("[7/7] Running parameter tuning sweep (Task 3.4.5)...")
    tuning_df, best_params = tune_rpca_parameters(
        raw_od_filt,
        motion_results,
        stats_before,
        subject_id = subject_id,
        output_dir = output_dir,
        verbose    = verbose,
    )
    fig_tune_path = plot_tuning_results(
        tuning_df, best_params, subject_id, output_dir, verbose=verbose
    )

    # ---- Console summary ----------------------------------------------------
    rs = reduction_summary
    print("\n" + "=" * 70)
    print(f"✅ Step B4 (Tasks 3.4.1–3.4.5) complete for subject {subject_id}")
    print(f"\n  rPCA Correction:")
    print(f"    Union mask samples  : {correction_summary['n_flagged']} "
          f"({correction_summary['pct_flagged_union']:.1f}%)")
    print(f"    PCs removed         : {correction_summary['n_components_used']}")
    print(f"    Variance removed    : {correction_summary['variance_removed_pct']:.1f}%")
    print(f"\n  Motion Reduction Verification:")
    print(f"    Mean MR before      : {rs['mean_motion_ratio_before']*100:.1f}%  (dual criterion)")
    print(f"    Mean MR after       : {rs['mean_motion_ratio_after_dual']*100:.1f}%  (dual criterion)")
    print(f"    Mean MR after       : {rs['mean_motion_ratio_after_amp']*100:.1f}%  (amp-only)")
    print(f"    Mean reduction      : {rs['mean_reduction_pct']:.1f}%")
    print(f"    Channels improved   : {rs['n_channels_improved']}/{len(comparison_df)}")
    print(f"\n  Optimal parameters (Task 3.4.5):")
    print(f"    n_components         = {best_params['n_components']}")
    print(f"    min_channels_flagged = {best_params['min_channels_flagged']}")
    print(f"    Post-correction MR   = {best_params['MR_after_dual']:.2f}%  (dual)")
    print(f"    Reduction            = {best_params['reduction_pct']:.1f}%")
    print(f"\n  Outputs saved to: {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od_filt':        raw_od_filt,
        'raw_od_corrected':   raw_od_corrected,
        'motion_results':     motion_results,
        'stats_before':       stats_before,
        'stats_after':        stats_after,
        'comparison_df':      comparison_df,
        'reduction_summary':  reduction_summary,
        'correction_summary': correction_summary,
        'tuning_df':          tuning_df,
        'best_params':        best_params,
        'output_dir':         output_dir,
        'fig_before_after':   fig_ba_path,
        'fig_reduction':      fig_red_path,
        'fig_tuning':         fig_tune_path,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    run_rpca_correction(
        dataset_name = 'expert_novice',
        subject_id   = 'A1',
        n_components = 0.97,
        verbose      = True,
    )
