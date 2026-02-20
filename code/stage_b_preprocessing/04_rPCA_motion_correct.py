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
    Task 3.4.1 pipeline: Load filtered OD → motion detect → rPCA correct → save.

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
    print("[3/3] Applying rPCA correction (Task 3.4.1)...")
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

    # ---- Console summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"✅ Step B4 (Task 3.4.1) complete for subject {subject_id}")
    print(f"  Motion samples (union mask) : {correction_summary['n_flagged']} "
          f"({correction_summary['pct_flagged_union']:.1f}%)")
    print(f"  PCA components removed      : {correction_summary['n_components_used']}")
    print(f"  Variance removed            : {correction_summary['variance_removed_pct']:.1f}%")
    print(f"  Corrected OD saved to       : {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od_filt':       raw_od_filt,
        'raw_od_corrected':  raw_od_corrected,
        'motion_results':    motion_results,
        'stats_before':      stats_before,
        'correction_summary': correction_summary,
        'output_dir':        output_dir,
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
