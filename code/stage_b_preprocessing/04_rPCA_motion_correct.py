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
        print(f"  n_components  : {n_components}  "
              f"(PCA keeps components explaining ≥{n_components*100:.0f}% variance)")
        print(f"  Good channels : {n_good}  |  Bad (skipped): {len(bad_set)}")

    # ------------------------------------------------------------------
    # Step 1: Union motion mask across all good channels
    # ------------------------------------------------------------------
    union_mask = np.zeros(n_times, dtype=bool)
    for ch in good_names:
        if ch in masks:
            union_mask |= masks[ch]

    n_flagged  = int(union_mask.sum())
    pct_flagged = 100.0 * n_flagged / n_times

    if verbose:
        print(f"\n  Union motion mask: {n_flagged}/{n_times} samples "
              f"({pct_flagged:.1f}%) flagged")

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
        'n_flagged':           n_flagged,
        'pct_flagged_union':   round(pct_flagged, 2),
        'n_components_used':   n_pc,
        'variance_removed_pct': round(var_removed * 100, 2),
        'n_good_channels':     n_good,
        'n_components_param':  n_components,
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
    fif_path = output_dir / f"{subject_id}_od_corrected_rpca.fif"
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
    verbose: bool = True,
) -> dict:
    """
    Task 3.4.1–3.4.2 pipeline: Load filtered OD → motion detect → rPCA correct
    → before/after visualization → save.

    Motion detection uses the dual criterion (use_amp_only=False):
      AMPthresh=5.0 OD + STDEVthresh=20×std_diff
    This produces ~17.4% flagged samples for subject A1 (mean across good
    channels), forming the correction target for rPCA.

    Parameters
    ----------
    dataset_name : str
    subject_id   : str
    n_components : float  PCA variance threshold (default 0.97)
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
        n_components = n_components,
        verbose      = verbose,
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
    print("[4/4] Generating before/after correction comparison plot (Task 3.4.2)...")
    fig_path = plot_before_after_correction(
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

    # ---- Console summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"✅ Step B4 (Tasks 3.4.1–3.4.2) complete for subject {subject_id}")
    print(f"  Motion samples (union mask) : {correction_summary['n_flagged']} "
          f"({correction_summary['pct_flagged_union']:.1f}%)")
    print(f"  PCA components removed      : {correction_summary['n_components_used']}")
    print(f"  Variance removed            : {correction_summary['variance_removed_pct']:.1f}%")
    print(f"  Before/after plot           : {fig_path.name}")
    print(f"  Corrected OD saved to       : {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od_filt':        raw_od_filt,
        'raw_od_corrected':   raw_od_corrected,
        'motion_results':     motion_results,
        'stats_before':       stats_before,
        'correction_summary': correction_summary,
        'output_dir':         output_dir,
        'fig_path':           fig_path,
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
