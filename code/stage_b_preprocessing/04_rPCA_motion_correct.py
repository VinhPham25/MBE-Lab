"""
Stage B — Step B4: Recursive PCA (rPCA) Motion Correction
Purpose: Apply PCA iteratively to motion-corrupted fNIRS OD data to remove
         high-variance motion components and preserve the low-variance neural signal.

Homer3 Reference: hmrR_MotionCorrectPCArecurse.m + hmrR_MotionCorrectPCA.m

Audit Fixes Implemented (see docs/reproduced_matlab_comparison.md §B4)
-----------------------------------------------------------------------
Discrepancy 4a — ITERATIVE (was single-pass):
    Homer3 loops up to maxIter=5 times.  After each PCA pass, motion detection
    runs again on the corrected signal; if motion is still present another PCA
    step is applied.  The corrected signal becomes the input for the next pass.

Discrepancy 4b — FULL-SIGNAL CORRECTION (was flagged-samples only):
    Homer3's hmrR_MotionCorrectPCA identifies the motion subspace by fitting
    PCA on the motion-flagged segments, then subtracts that projection from the
    entire continuous signal (all time points, not just the flagged ones).  The
    previous Python code replaced only flagged samples with their PCA residuals,
    leaving any global correlated noise in the unflagged portions untouched.

Discrepancy 4c — OR-UNION MASK (was vote-threshold):
    Homer3 calls hmrR_MotionArtifact (not hmrR_MotionArtifactByChannel), which
    returns a single global tInc vector.  If ANY active channel is flagged at
    time t, tInc[t]=0 for all channels.  This is a strict OR-union.  The
    previous Python code used a vote threshold (≥10 channels simultaneously),
    which was far more conservative.  Default is now min_channels_flagged=1
    (OR-union) matching Homer3.

Discrepancy 4d — nSV=0.97 (was 0.99):
    Homer3 default (Yucel et al.) is nSV=0.97.  config.yaml is updated to 0.97.

Algorithm (Homer3-faithful)
---------------------------
1.  Build OR-union motion mask from per-channel B3 masks:
      any good channel flagged at time t → time t enters the mask.
2.  If motion mask is empty or max_iterations reached → stop.
3.  Extract flagged-segment data matrix X: shape (n_flagged, n_good).
4.  Compute cross-product C = X' X  (Homer3: c = y' * y).
5.  Eigendecomposition of C via SVD: U, s, _ = svd(C).
6.  Find minimum k such that Σ(s[0:k]) / Σ(s) ≥ n_components
    (Homer3 nSV logic: find first cumsum position ≥ nSV).
7.  Motion subspace = V_k (first k columns of U).
8.  FULL-SIGNAL projection removal:
      data_good -= V_k @ (V_k.T @ data_good)
      This removes the motion subspace from ALL time points.
9.  Re-detect motion on corrected signal (Homer3: re-call hmrR_MotionArtifact).
10. Repeat from step 2 with updated data and new mask (up to max_iterations).

Reference: docs/reproduced_matlab_comparison.md §B4
Upstream:  03_motion_correct.py (Step B3 motion mask, use_amp_only config)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, load_config

# ---------------------------------------------------------------------------
# Load config once at import time — all tunable constants derive from here
# ---------------------------------------------------------------------------
_CONFIG = load_config()
_RPCA_CFG = _CONFIG.get("stage_b", {}).get("rpca", {})
_B3_CFG   = _CONFIG.get("stage_b", {}).get("motion_detection", {})

# Homer3 defaults (Yucel et al.): nSV=0.97, maxIter=5, OR-union mask
_DEFAULT_N_COMPONENTS    = float(_RPCA_CFG.get("n_components",         0.97))
_DEFAULT_MAX_ITER        = int(  _RPCA_CFG.get("max_iterations",       5))
_DEFAULT_MIN_CHANNELS    = int(  _RPCA_CFG.get("min_channels_flagged", 1))
_DEFAULT_AMP_THRESH      = float(_B3_CFG.get("AMPthresh",  5.0))
_DEFAULT_STDEV_THRESH    = float(_B3_CFG.get("tStdev",    20.0))
_DEFAULT_T_MOTION        = float(_B3_CFG.get("tMotion",    0.5))
_DEFAULT_T_MASK          = float(_B3_CFG.get("t_mask",     0.5))

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
# Private helpers
# ===========================================================================

def _build_motion_mask(masks: dict, good_names: list, n_times: int,
                       min_channels_flagged: int = 1) -> np.ndarray:
    """
    Build a boolean motion mask from per-channel B3 detection masks.

    Homer3 hmrR_MotionArtifact parity (Discrepancy 4c):
    With min_channels_flagged=1 (default), this is a strict OR-union:
    any good channel flagged at time t → that sample enters the mask.
    Higher values apply a vote threshold (non-Homer3 behavior, available
    for experimentation).

    Parameters
    ----------
    masks : dict
        Per-channel boolean arrays {channel_name: bool_array(n_times)}.
    good_names : list of str
        Names of active (non-bad) channels.
    n_times : int
        Total number of time samples.
    min_channels_flagged : int
        1 = OR-union (Homer3 default).
        k>1 = require ≥k channels to simultaneously flag a sample.

    Returns
    -------
    motion_mask : np.ndarray, shape (n_times,), dtype bool
    """
    if min_channels_flagged <= 1:
        # OR-union: flag if ANY good channel detects motion (Homer3 parity)
        union = np.zeros(n_times, dtype=bool)
        for ch in good_names:
            if ch in masks:
                union |= masks[ch]
        return union
    else:
        # Vote threshold (optional, non-Homer3 behavior)
        counts = np.zeros(n_times, dtype=np.int32)
        for ch in good_names:
            if ch in masks:
                counts += masks[ch].astype(np.int32)
        return counts >= min_channels_flagged


def _impute_nan(X: np.ndarray) -> np.ndarray:
    """
    Replace non-finite values in X with per-column mean.
    Used to handle trailing-zero channels (e.g. subject R10) that produce
    NaN/Inf after zero-phase bandpass filtering.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels)
        Data matrix, modified in-place.

    Returns
    -------
    n_replaced : int
        Number of non-finite values replaced.
    """
    import warnings
    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad == 0:
        return 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    col_means[~np.isfinite(col_means)] = 0.0
    nan_mask = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X[~np.isfinite(X)] = 0.0   # any residual → 0
    return n_bad


def _apply_pca_single_pass(
    data: np.ndarray,
    good_idx: list,
    motion_mask: np.ndarray,
    n_sv: float,
    verbose: bool = False,
) -> tuple:
    """
    One PCA motion-correction pass — Homer3 hmrR_MotionCorrectPCA parity.

    Key implementation:
    - PCA is FITTED on motion-flagged segments only (identifies motion subspace).
    - Projection removal is applied to the FULL signal (all time points).
      This is Discrepancy 4b: old code replaced only flagged samples; Homer3
      removes the motion subspace globally.

    MATLAB cross-product SVD (Homer3):
        c = y' * y          (y = flagged segments, shape n_flagged × n_good)
        [V, St, _] = svd(c)
        svs = diag(St) / sum(diag(St))
        find k: first cumsum(svs) index ≥ nSV
        yc = y - y * V_k * V_k'   (remove top-k motion subspace from flagged)

    Python equivalent (extended to full signal):
        C = X.T @ X           (same cross-product)
        U, s, _ = svd(C)      (U = V in MATLAB for symmetric PSD matrix)
        svs = s / s.sum()
        k = first index where cumsum(svs) ≥ n_sv
        V_k = U[:, :k]
        data_good -= V_k @ (V_k.T @ data_good)   ← applied to ALL time points

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Full OD data matrix (in-place safe copy expected by caller).
    good_idx : list of int
        Row indices of active (non-bad) channels in data.
    motion_mask : np.ndarray, shape (n_times,), dtype bool
        OR-union motion mask (True = motion-flagged sample).
    n_sv : float
        Variance threshold in (0, 1): retain k components explaining ≥ n_sv
        of the total variance in the flagged segments.
        (Homer3 nSV parameter; Yucel et al. default = 0.97.)
    verbose : bool

    Returns
    -------
    data_out : np.ndarray, shape (n_channels, n_times)
        Corrected data matrix (new array).
    svs : np.ndarray
        Normalised singular values (explained variance fractions) from the
        cross-product SVD of the flagged segments.
    k : int
        Number of principal components removed (motion subspace dimension).
    """
    motion_idx = np.where(motion_mask)[0]
    n_flagged = len(motion_idx)

    if n_flagged == 0:
        return data.copy(), np.array([]), 0

    # Extract flagged-segment data: (n_flagged, n_good)
    # np.ix_(good_idx, motion_idx) selects the rows indexed by good_idx and columns indexed by motion_idx
    # .T transposes to shape (n_flagged, n_good) for PCA fitting
    X = data[np.ix_(good_idx, motion_idx)].T.copy()

    # NaN/Inf imputation (R10 trailing-zero artefact)
    n_imputed = _impute_nan(X)
    if verbose and n_imputed > 0:
        print(f"      NaN/Inf imputation: replaced {n_imputed} values with "
              f"column mean (trailing-zero channel artefact)")

    # Cross-product matrix  C = X' X  (Homer3: c = y' * y)
    # Shape: (n_good, n_good) — positive semi-definite
    C = X.T @ X

    # Eigendecomposition via SVD  (Homer3: [V, St, _] = svd(c))
    # For symmetric PSD matrix: svd(C) gives C = U S U'  where U = eigenvectors, s = singular values
    # s (singular values) represent the variance explained by each principal component
    U, s, _ = np.linalg.svd(C, full_matrices=False)

    # Normalised explained variance fractions  (Homer3: svs = diag(St)/sum(diag(St)))
    total = s.sum()
    svs = s / total if total > 0 else s

    # Find minimum k such that cumsum(svs)[k-1] >= n_sv  (Homer3 nSV logic)
    if 0.0 < n_sv < 1.0:
        svsc = np.cumsum(svs)
        k = int(np.searchsorted(svsc, n_sv, side='left')) + 1
        k = min(k, len(svs))
    else:
        # n_sv given as integer count of components to remove
        k = max(1, int(round(n_sv)))
        k = min(k, len(svs))

    # Motion subspace eigenvectors: V_k  (Homer3 notation: V[:, 1:k])
    # selects the first k columns of U corresponding to the top-k principal components
    V_k = U[:, :k]   # shape (n_good, k)

    # Full-signal projection removal  (Discrepancy 4b fix)
    # Homer3 parity: remove motion subspace from ALL time points, not just flagged
    # data_good_corrected = data_good - V_k @ (V_k.T @ data_good)
    data_good = data[good_idx, :]   # (n_good, n_times)
    
    # V_k.T @ data_good projects your data onto the motion subspace (finds the component of your data along those axes).
    # V_k @ (...) reconstructs the motion artifact component in the original space.
    motion_projection = V_k @ (V_k.T @ data_good)  # (n_good, n_times)

    data_out = data.copy()
    # data_good - motion_projection subtracts this component from your data, removing the motion artifacts globally (from all time points, not just flagged ones).
    data_out[good_idx, :] = data_good - motion_projection

    if verbose:
        var_removed = float(np.sum(svs[:k]))
        print(f"      PCA pass: k={k} components, "
              f"{var_removed*100:.2f}% variance in flagged segments removed, "
              f"{n_flagged} flagged samples used for fitting")

    return data_out, svs, k


# ===========================================================================
# Task 3.4.1 — rPCA Motion Correction Algorithm (Homer3 parity)
# ===========================================================================

def apply_rpca_correction(
    raw_od_filt,
    motion_results: dict,
    n_components: float = _DEFAULT_N_COMPONENTS,
    min_channels_flagged: int = _DEFAULT_MIN_CHANNELS,
    max_iterations: int = _DEFAULT_MAX_ITER,
    amp_thresh: float = _DEFAULT_AMP_THRESH,
    stdev_thresh: float = _DEFAULT_STDEV_THRESH,
    t_motion: float = _DEFAULT_T_MOTION,
    t_mask: float = _DEFAULT_T_MASK,
    use_amp_only: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Homer3-faithful recursive PCA motion correction (hmrR_MotionCorrectPCArecurse).

    Implements all four B4 audit corrections from docs/reproduced_matlab_comparison.md:
      4a — ITERATIVE: loops up to max_iterations, re-detecting motion after each pass.
      4b — FULL-SIGNAL: motion subspace identified from flagged segments, subtracted
           from the complete continuous signal (all time points).
      4c — OR-UNION MASK: any good channel flagged at time t → t enters the mask
           (Homer3 hmrR_MotionArtifact parity; min_channels_flagged=1).
      4d — nSV=0.97: Homer3/Yucel et al. default variance threshold.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        Filtered OD data (Step B2 output, low-pass 0.5 Hz).
    motion_results : dict
        Output of detect_motion_artifacts() (Step B3).
        Must contain 'masks' {ch_name: bool_array} and 'bad_channels'.
    n_components : float
        Variance threshold nSV in (0, 1).  The minimum number of PCA components
        whose cumulative explained variance ≥ n_components are removed from the
        signal.  Homer3/Yucel et al. default: 0.97.
        Loaded from config.yaml → stage_b.rpca.n_components.
    min_channels_flagged : int
        Controls motion mask construction:
          1 (default) = OR-union — Homer3 hmrR_MotionArtifact parity.
          k>1 = vote threshold (non-Homer3, for experimentation only).
        Loaded from config.yaml → stage_b.rpca.min_channels_flagged.
    max_iterations : int
        Maximum number of PCA correction passes.  Homer3 default: 5 (maxIter).
        Each iteration re-detects motion and re-applies PCA if motion remains.
        Loaded from config.yaml → stage_b.rpca.max_iterations.
    amp_thresh : float
        Amplitude threshold for motion re-detection (OD units).
        Loaded from config.yaml → stage_b.motion_detection.AMPthresh.
    stdev_thresh : float
        STDEV multiplier for motion re-detection.
        Loaded from config.yaml → stage_b.motion_detection.tStdev.
    t_motion : float
        Motion detection window (seconds).
        Loaded from config.yaml → stage_b.motion_detection.tMotion.
    t_mask : float
        Temporal buffer around detected events (seconds).
        Loaded from config.yaml → stage_b.motion_detection.t_mask.
    use_amp_only : bool
        If True, re-detection uses amplitude-only criterion.
        Loaded from config.yaml → stage_b.motion_detection.use_amp_only.
    verbose : bool

    Returns
    -------
    raw_od_corrected : mne.io.Raw
        Corrected OD data (same shape; bad channels unchanged).
    correction_summary : dict
        {
          'n_flagged_initial':    int   — flagged samples in initial mask,
          'pct_flagged_union':    float — % of signal in initial OR-union mask,
          'n_iterations':         int   — actual number of PCA passes applied,
          'n_components_param':   float — n_components config parameter,
          'n_components_used':    int   — PCs removed in final (last) pass,
          'variance_removed_pct': float — % variance removed in final pass,
          'n_good_channels':      int,
          'min_channels_flagged': int,
          'max_iterations':       int,
          'svs_per_iter':         list  — per-iteration singular value arrays,
          'k_per_iter':           list  — per-iteration k (PCs removed),
          'n_flagged':            int   — alias for n_flagged_initial (compat),
        }
    """
    data     = raw_od_filt.get_data().copy()   # (n_channels, n_times)
    ch_names = raw_od_filt.ch_names
    n_times  = data.shape[1]
    bad_set  = set(motion_results['bad_channels'])

    good_idx   = [i for i, ch in enumerate(ch_names) if ch not in bad_set]
    good_names = [ch_names[i] for i in good_idx]
    n_good     = len(good_idx)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.1 — rPCA MOTION CORRECTION  (Homer3 hmrR_MotionCorrectPCArecurse)")
        print("=" * 70)
        print(f"  n_components (nSV)   : {n_components}  "
              f"(remove components explaining ≥{n_components*100:.0f}% variance)")
        print(f"  max_iterations       : {max_iterations}  (Homer3 default: 5)")
        print(f"  min_channels_flagged : {min_channels_flagged}  "
              f"({'OR-union — Homer3 parity' if min_channels_flagged == 1 else 'vote threshold'})")
        print(f"  Good channels        : {n_good}  |  Bad (skipped): {len(bad_set)}")

    # ------------------------------------------------------------------
    # Step 1: Build initial OR-union motion mask (Discrepancy 4c fix)
    # ------------------------------------------------------------------
    initial_mask = _build_motion_mask(
        motion_results['masks'], good_names, n_times, min_channels_flagged
    )
    n_flagged_initial = int(initial_mask.sum())
    pct_flagged       = 100.0 * n_flagged_initial / n_times

    if verbose:
        mask_type = "OR-union" if min_channels_flagged == 1 else f"vote≥{min_channels_flagged}"
        print(f"\n  Initial motion mask ({mask_type}): "
              f"{n_flagged_initial}/{n_times} samples ({pct_flagged:.1f}%)")

    if n_flagged_initial == 0:
        if verbose:
            print("  No motion detected — no correction applied.")
        empty_summary = {
            'n_flagged_initial':    0,
            'n_flagged':            0,
            'pct_flagged_union':    0.0,
            'n_iterations':         0,
            'n_components_param':   n_components,
            'n_components_used':    0,
            'variance_removed_pct': 0.0,
            'n_good_channels':      n_good,
            'min_channels_flagged': min_channels_flagged,
            'max_iterations':       max_iterations,
            'svs_per_iter':         [],
            'k_per_iter':           [],
        }
        return raw_od_filt.copy(), empty_summary

    # ------------------------------------------------------------------
    # Iterative PCA loop (Discrepancy 4a + 4b fix)
    # Homer3: while any(tInc==0) AND ii < maxIter → apply PCA, re-detect
    # ------------------------------------------------------------------
    current_data = data.copy()
    current_mask = initial_mask.copy()

    svs_per_iter = []
    k_per_iter   = []
    iteration    = 0

    # loop runs while any channel in the motion mask == 0 (motion is present) and iteration count is below max_iterations
    while np.any(current_mask) and iteration < max_iterations:
        iteration += 1
        n_now = int(current_mask.sum())

        if verbose:
            print(f"\n  --- Iteration {iteration}/{max_iterations} ---")
            print(f"      Motion mask: {n_now} samples ({100.*n_now/n_times:.1f}%)")

        # Apply one PCA pass (Discrepancy 4b: full-signal projection removal)
        current_data, svs, k = _apply_pca_single_pass(
            current_data, good_idx, current_mask, n_components, verbose=verbose
        )

        #Save each iteration into array
        svs_per_iter.append(svs)
        k_per_iter.append(k)

        if verbose:
            var_rem = float(np.sum(svs[:k])) if len(svs) > 0 else 0.0
            print(f"      Removed {k} PC(s), "
                  f"{var_rem*100:.2f}% of variance in flagged segments")

        # Re-detect motion on corrected data (Homer3: re-call hmrR_MotionArtifact)
        temp_raw = raw_od_filt.copy()
        temp_raw._data = current_data.copy()

        new_motion = detect_motion_artifacts(
            temp_raw,
            amp_thresh   = amp_thresh,
            stdev_thresh = stdev_thresh,
            t_motion     = t_motion,
            t_mask       = t_mask,
            use_amp_only = use_amp_only,
            verbose      = False,
        )
        current_mask = _build_motion_mask(
            new_motion['masks'], good_names, n_times, min_channels_flagged
        )
        n_remaining = int(current_mask.sum())

        if verbose:
            print(f"      After correction: {n_remaining} samples still flagged "
                  f"({100.*n_remaining/n_times:.1f}%)")

    # ------------------------------------------------------------------
    # Build corrected Raw object
    # ------------------------------------------------------------------
    raw_od_corrected = raw_od_filt.copy()
    raw_od_corrected._data = current_data

    # Final-iteration metrics for summary (or zeros if no pass ran)
    k_final        = k_per_iter[-1]   if k_per_iter   else 0
    svs_final      = svs_per_iter[-1] if svs_per_iter else np.array([])
    var_removed    = float(np.sum(svs_final[:k_final])) if len(svs_final) > 0 else 0.0

    correction_summary = {
        'n_flagged_initial':    n_flagged_initial,
        'n_flagged':            n_flagged_initial,   # backward-compat alias
        'pct_flagged_union':    round(pct_flagged, 2),
        'n_iterations':         iteration,
        'n_components_param':   n_components,
        'n_components_used':    k_final,
        'variance_removed_pct': round(var_removed * 100, 2),
        'n_good_channels':      n_good,
        'min_channels_flagged': min_channels_flagged,
        'max_iterations':       max_iterations,
        'svs_per_iter':         [sv.tolist() for sv in svs_per_iter],
        'k_per_iter':           k_per_iter,
    }

    if verbose:
        if iteration == max_iterations and np.any(current_mask):
            print(f"\n  ⚠  Max iterations ({max_iterations}) reached; "
                  f"{int(current_mask.sum())} samples still flagged")
        else:
            print(f"\n  ✅ Motion fully corrected after {iteration} iteration(s)")
        print(f"\n  Summary (final pass):")
        print(f"    Iterations run        : {iteration}")
        print(f"    PCs removed (final)   : {k_final}")
        print(f"    Variance removed (final): {var_removed*100:.2f}%")
        print(f"    Initial mask coverage : {pct_flagged:.1f}%")
        print("=" * 70 + "\n")

    return raw_od_corrected, correction_summary


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
    For each selected channel, one row with three panels:
      [Before OD | After OD (corrected) | Removed signal (Before − After)]

      Before OD (blue):   filtered OD trace; red shading = union motion mask.
      After OD (green):   corrected OD trace; red shading = union motion mask.
      Removed (red):      difference (Before − After) = noise extracted by rPCA.

      Relationship: Before = After + Removed

    Channel selection: n_high channels with highest motion ratio (most artefact)
    + n_low channels with lowest motion ratio (least artefact / clean).

    Parameters
    ----------
    raw_od_filt      : mne.io.Raw   (Step B2 filtered OD — before correction)
    raw_od_corrected : mne.io.Raw   (Step B4 corrected OD — after correction)
    motion_results   : dict         (Step B3 masks + bad_channels)
    stats_before     : pd.DataFrame (per-channel motion stats, pre-correction)
    correction_summary : dict       (rPCA metrics from apply_rpca_correction)
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

    # OR-union mask for shading (first t_window seconds)
    n_times    = data_before.shape[1]
    good_names = [ch for ch in ch_names if ch not in bad_set]
    union_mask = np.zeros(n_times, dtype=bool)
    for ch in good_names:
        if ch in masks:
            union_mask |= masks[ch]
    union_seg = union_mask[:n_plot_samp]

    # Channel selection: highest and lowest motion ratios
    good_df     = stats_before[stats_before['motion_quality'] != 'bad_channel'].copy()
    sorted_asc  = good_df.sort_values('motion_ratio', ascending=True)
    sorted_desc = good_df.sort_values('motion_ratio', ascending=False)
    low_chs  = sorted_asc['channel'].tolist()[:n_low]
    high_chs = sorted_desc['channel'].tolist()[:n_high]
    plot_chs = high_chs + low_chs

    n_rows = len(plot_chs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3.2 * n_rows), sharex=True)
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
        tag       = f"{row_stats['motion_quality']}  {row_stats['pct_corrupted']:.1f}% flagged"
        ch_label  = f"{ch_name}  [{tag}]"

        var_before = float(np.var(before))
        var_after  = float(np.var(after))
        var_ratio  = (var_before - var_after) / var_before * 100 if var_before > 0 else 0.0

        for col, (sig, col_color) in enumerate(zip([before, after, removed], col_colors)):
            ax = axes[row_idx, col]
            if col < 2:
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

    from matplotlib.patches import Patch
    legend_patch = Patch(color='red', alpha=0.25, label='Motion-flagged (OR-union mask)')
    fig.legend(handles=[legend_patch], loc='upper right', fontsize=8,
               bbox_to_anchor=(1.0, 1.01))

    n_iters = correction_summary.get('n_iterations', 1)
    fig.suptitle(
        f'Subject {subject_id} — rPCA Motion Correction: Before vs After  '
        f'(Homer3-faithful)\n'
        f'nSV={correction_summary["n_components_param"]}  |  '
        f'Iterations={n_iters}  |  '
        f'PCs removed (final)={correction_summary["n_components_used"]}  |  '
        f'Var removed (final)={correction_summary["variance_removed_pct"]:.1f}%  |  '
        f'Mask={correction_summary["pct_flagged_union"]:.1f}% (OR-union)  |  '
        f'Window: first {t_window:.0f}s',
        fontsize=9, fontweight='bold', y=1.02,
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
    """Shade contiguous True blocks in a boolean mask along the time axis."""
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
    amp_thresh: float = _DEFAULT_AMP_THRESH,
    stdev_thresh: float = _DEFAULT_STDEV_THRESH,
    t_motion: float = _DEFAULT_T_MOTION,
    t_mask: float = _DEFAULT_T_MASK,
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
    raw_od_filt           : mne.io.Raw   (before — filtered OD)
    raw_od_corrected      : mne.io.Raw   (after  — rPCA corrected OD)
    motion_results_before : dict         (Step B3 masks on pre-correction OD)
    stats_before          : pd.DataFrame (pre-correction per-channel stats)
    amp_thresh            : float
    stdev_thresh          : float
    t_motion              : float
    t_mask                : float
    verbose               : bool

    Returns
    -------
    stats_after      : pd.DataFrame
        Per-channel motion stats on corrected OD (dual criterion).
    comparison_df    : pd.DataFrame
        channel, motion_ratio_before, motion_ratio_after, reduction_abs,
        reduction_pct, quality_before, quality_after.
    reduction_summary : dict
        Aggregate metrics: mean/median before & after, n_channels_improved.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.3 — POST-CORRECTION MOTION VERIFICATION")
        print("=" * 70)
        print("Re-running Homer3 dual criterion on rPCA-corrected OD...")

    # Re-detect on corrected data (dual criterion)
    motion_results_after = detect_motion_artifacts(
        raw_od_corrected,
        amp_thresh   = amp_thresh,
        stdev_thresh = stdev_thresh,
        t_motion     = t_motion,
        t_mask       = t_mask,
        use_amp_only = False,
        verbose      = False,
    )
    stats_after = compute_motion_stats(
        motion_results_after, raw_od_corrected, verbose=False
    )

    # Amplitude-only on corrected (checks if large spikes remain)
    motion_results_amp = detect_motion_artifacts(
        raw_od_corrected,
        amp_thresh   = amp_thresh,
        stdev_thresh = stdev_thresh,
        t_motion     = t_motion,
        t_mask       = t_mask,
        use_amp_only = True,
        verbose      = False,
    )
    stats_amp = compute_motion_stats(
        motion_results_amp, raw_od_corrected, verbose=False
    )

    # Per-channel comparison
    bad_set = set(motion_results_before['bad_channels'])
    records = []
    for ch in raw_od_filt.ch_names:
        if ch in bad_set:
            continue
        row_b   = stats_before[stats_before['channel'] == ch].iloc[0]
        row_a   = stats_after[stats_after['channel'] == ch].iloc[0]
        row_amp = stats_amp[stats_amp['channel'] == ch].iloc[0]

        mr_before = float(row_b['motion_ratio'])
        mr_after  = float(row_a['motion_ratio'])
        mr_amp    = float(row_amp['motion_ratio'])
        red_abs   = mr_before - mr_after
        red_pct   = (red_abs / mr_before * 100) if mr_before > 0 else 0.0

        records.append({
            'channel':               ch,
            'wavelength':            int(row_b['wavelength']),
            'motion_ratio_before':   round(mr_before, 4),
            'motion_ratio_after':    round(mr_after, 4),
            'motion_ratio_amp_only': round(mr_amp, 4),
            'reduction_abs':         round(red_abs, 4),
            'reduction_pct':         round(red_pct, 2),
            'quality_before':        row_b['motion_quality'],
            'quality_after':         row_a['motion_quality'],
        })

    comparison_df = pd.DataFrame(records)

    good_before = stats_before[stats_before['motion_quality'] != 'bad_channel']
    good_after  = stats_after[stats_after['motion_quality']  != 'bad_channel']
    good_amp    = stats_amp[stats_amp['motion_quality']      != 'bad_channel']

    mean_before = float(good_before['motion_ratio'].mean())
    mean_after  = float(good_after['motion_ratio'].mean())
    mean_amp    = float(good_amp['motion_ratio'].mean())
    n_improved  = int((comparison_df['reduction_abs'] > 0).sum())
    n_worse     = int((comparison_df['reduction_abs'] < 0).sum())

    reduction_summary = {
        'mean_motion_ratio_before':     round(mean_before, 4),
        'mean_motion_ratio_after_dual': round(mean_after,  4),
        'mean_motion_ratio_after_amp':  round(mean_amp,    4),
        'mean_reduction_abs':           round(mean_before - mean_after, 4),
        'mean_reduction_pct':           round((mean_before - mean_after) / mean_before * 100, 2)
                                        if mean_before > 0 else 0.0,
        'n_channels_improved':          n_improved,
        'n_channels_worse':             n_worse,
        'n_channels_unchanged':         len(comparison_df) - n_improved - n_worse,
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
      Row 2: Per-channel reduction (%) bar chart — green = improved, red = worse.

    Parameters
    ----------
    comparison_df     : pd.DataFrame  (output of verify_motion_reduction)
    reduction_summary : dict
    correction_summary : dict         (rPCA metrics from apply_rpca_correction)
    subject_id        : str
    output_dir        : Path
    verbose           : bool

    Returns
    -------
    fig_path : Path
    """
    df   = comparison_df.copy().sort_values('motion_ratio_before', ascending=False)
    n_ch = len(df)
    x    = np.arange(n_ch)
    w    = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_ch * 0.22), 9),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]
    ax1.bar(x - w/2, df['motion_ratio_before'] * 100,
            width=w, color='#ff7f0e', alpha=0.85, label='Before rPCA')
    ax1.bar(x + w/2, df['motion_ratio_after'] * 100,
            width=w, color='#17becf', alpha=0.85, label='After rPCA (dual)')
    ax1.scatter(x + w/2, df['motion_ratio_amp_only'] * 100,
                color='#2ca02c', s=18, zorder=5, label='After rPCA (amp-only)',
                marker='D')
    ax1.axhline(10, color='gray', lw=0.8, ls='--', alpha=0.6,
                label='Elevated threshold (10%)')
    ax1.set_ylabel('Motion Ratio (%)', fontsize=10)

    n_iters = correction_summary.get('n_iterations', '?')
    ax1.set_title(
        f'Subject {subject_id} — Per-Channel Motion Ratio: Before vs After rPCA  '
        f'(Homer3-faithful)\n'
        f'nSV={correction_summary["n_components_param"]}  |  '
        f'Iterations={n_iters}  |  '
        f'PCs (final)={correction_summary["n_components_used"]}  |  '
        f'Var removed (final)={correction_summary["variance_removed_pct"]:.1f}%',
        fontsize=9, fontweight='bold',
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['channel'], rotation=90, fontsize=5.5)
    ax1.legend(fontsize=8, loc='upper right')

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

    ax2 = axes[1]
    colors_bar = ['#2ca02c' if v >= 0 else '#d62728' for v in df['reduction_pct']]
    ax2.bar(x, df['reduction_pct'], color=colors_bar, alpha=0.85, width=0.7)
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
    """
    Save per-channel comparison CSV and reduction summary JSON.

    Parameters
    ----------
    comparison_df     : pd.DataFrame
    reduction_summary : dict
    subject_id        : str
    output_dir        : Path

    Returns
    -------
    csv_path  : Path
    json_path : Path
    """
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
    amp_thresh: float = _DEFAULT_AMP_THRESH,
    stdev_thresh: float = _DEFAULT_STDEV_THRESH,
    t_motion: float = _DEFAULT_T_MOTION,
    t_mask: float = _DEFAULT_T_MASK,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.4.5: Grid search over n_components × min_channels_flagged to find
    the configuration that maximises post-correction motion reduction.

    Note: max_iterations is fixed at the config default for all grid configs.
    Varying min_channels_flagged across the grid explores the trade-off between
    Homer3-faithful OR-union (=1) and more conservative vote thresholds (>1).

    Parameters
    ----------
    raw_od_filt        : mne.io.Raw
    motion_results     : dict   (Step B3 masks, use_amp_only=False)
    stats_before       : pd.DataFrame
    subject_id         : str
    output_dir         : Path
    n_components_grid  : list of float  (default [0.90, 0.95, 0.97, 0.99])
    min_channels_grid  : list of int    (default [1, 3, 5, 10, 15])
    amp_thresh etc.    : motion detection params (same as Step B3)
    verbose            : bool

    Returns
    -------
    results_df  : pd.DataFrame  (one row per configuration)
    best_params : dict          (optimal n_components, min_channels_flagged)
    """
    if n_components_grid is None:
        n_components_grid = [0.90, 0.95, 0.97, 0.99]
    if min_channels_grid is None:
        min_channels_grid = [1, 3, 5, 10, 15]

    good_df   = stats_before[stats_before['motion_quality'] != 'bad_channel']
    mr_before = float(good_df['motion_ratio'].mean())
    n_configs = len(n_components_grid) * len(min_channels_grid)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.4.5 — rPCA PARAMETER TUNING SWEEP")
        print("=" * 70)
        print(f"  Baseline mean MR (dual criterion): {mr_before*100:.2f}%")
        print(f"  n_components grid  : {n_components_grid}")
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
            raw_corr, corr_sum = apply_rpca_correction(
                raw_od_filt,
                motion_results,
                n_components         = n_comp,
                min_channels_flagged = min_ch,
                max_iterations       = _DEFAULT_MAX_ITER,
                amp_thresh           = amp_thresh,
                stdev_thresh         = stdev_thresh,
                t_motion             = t_motion,
                t_mask               = t_mask,
                verbose              = False,
            )

            mr_after_res   = detect_motion_artifacts(
                raw_corr, amp_thresh=amp_thresh, stdev_thresh=stdev_thresh,
                t_motion=t_motion, t_mask=t_mask, use_amp_only=False, verbose=False
            )
            mr_after_stats = compute_motion_stats(mr_after_res, raw_corr, verbose=False)

            mr_amp_res     = detect_motion_artifacts(
                raw_corr, amp_thresh=amp_thresh, stdev_thresh=stdev_thresh,
                t_motion=t_motion, t_mask=t_mask, use_amp_only=True, verbose=False
            )
            mr_amp_stats   = compute_motion_stats(mr_amp_res, raw_corr, verbose=False)

            good_a   = mr_after_stats[mr_after_stats['motion_quality'] != 'bad_channel']
            good_amp = mr_amp_stats[mr_amp_stats['motion_quality']     != 'bad_channel']
            mr_dual  = float(good_a['motion_ratio'].mean())
            mr_amp   = float(good_amp['motion_ratio'].mean())
            red      = (mr_before - mr_dual) / mr_before * 100 if mr_before > 0 else 0.0

            rec = {
                'n_components':         n_comp,
                'min_channels_flagged': min_ch,
                'pct_mask':             corr_sum['pct_flagged_union'],
                'n_pcs_removed':        corr_sum['n_components_used'],
                'variance_removed_pct': corr_sum['variance_removed_pct'],
                'n_iterations':         corr_sum['n_iterations'],
                'MR_before':            round(mr_before * 100, 2),
                'MR_after_dual':        round(mr_dual * 100, 2),
                'MR_after_amp':         round(mr_amp * 100, 2),
                'reduction_pct':        round(red, 2),
            }
            records.append(rec)

            if verbose:
                print(f"  {n_comp:>7.2f} {min_ch:>6d} {rec['pct_mask']:>7.1f} "
                      f"{rec['n_pcs_removed']:>6d} {rec['variance_removed_pct']:>7.1f}% "
                      f"{rec['MR_after_dual']:>8.2f}% {rec['MR_after_amp']:>7.2f}% "
                      f"{rec['reduction_pct']:>7.1f}%")

    results_df = pd.DataFrame(records)

    clean = results_df[results_df['MR_after_amp'] == 0.0]
    if clean.empty:
        clean = results_df
    best_row   = clean.sort_values('reduction_pct', ascending=False).iloc[0]
    best_params = {
        'n_components':         float(best_row['n_components']),
        'min_channels_flagged': int(best_row['min_channels_flagged']),
        'MR_after_dual':        float(best_row['MR_after_dual']),
        'MR_after_amp':         float(best_row['MR_after_amp']),
        'reduction_pct':        float(best_row['reduction_pct']),
        'pct_mask':             float(best_row['pct_mask']),
        'n_pcs_removed':        int(best_row['n_pcs_removed']),
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

    Parameters
    ----------
    results_df  : pd.DataFrame  (output of tune_rpca_parameters)
    best_params : dict
    subject_id  : str
    output_dir  : Path
    verbose     : bool

    Returns
    -------
    fig_path : Path
    """
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
        ax.set_xlabel('n_components (nSV)', fontsize=10)
        ax.set_ylabel('min_channels_flagged', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data.values[r, c]
                ax.text(c, r, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=8,
                        color='white' if val > data.values.mean() else 'black')
        best_r = list(min_ch_vals).index(best_params['min_channels_flagged'])
        best_c = list(n_comp_vals).index(best_params['n_components'])
        ax.plot(best_c, best_r, marker='*', color='gold', ms=18, zorder=5,
                markeredgecolor='k', markeredgewidth=0.8)

    _heatmap(axes[0], mr_grid,
             'Post-correction MR (dual, %)  ← lower is better', 'YlOrRd_r')
    _heatmap(axes[1], red_grid,
             'Reduction vs baseline (%)  ← higher is better', 'YlGn')

    fig.suptitle(
        f'Subject {subject_id} — rPCA Parameter Tuning  (Homer3-faithful)\n'
        f'Baseline mean MR = {results_df["MR_before"].iloc[0]:.1f}%  '
        f'★ = optimal: '
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
    Save the rPCA-corrected OD data and correction summary JSON.

    Parameters
    ----------
    raw_od_corrected  : mne.io.Raw
    correction_summary : dict
    subject_id        : str
    output_dir        : Path

    Returns
    -------
    fif_path     : Path
    summary_path : Path
    """
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    fif_path = output_dir / f"{subject_id}_od_corrected_rpca_raw.fif"
    raw_od_corrected.save(str(fif_path), overwrite=True, verbose=False)
    print(f"Saved corrected OD : {fif_path}")

    # Remove non-serialisable numpy arrays before JSON dump
    json_safe = {
        k: v for k, v in correction_summary.items()
        if k not in ('svs_per_iter',)
    }
    summary_path = output_dir / f"{subject_id}_rpca_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"Saved rPCA summary : {summary_path}")

    return fif_path, summary_path


# ===========================================================================
# End-to-end pipeline — run_rpca_correction
# ===========================================================================

def run_rpca_correction(
    dataset_name: str,
    subject_id: str,
    n_components: float = _DEFAULT_N_COMPONENTS,
    min_channels_flagged: int = _DEFAULT_MIN_CHANNELS,
    max_iterations: int = _DEFAULT_MAX_ITER,
    run_tuning: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Full Step B4 pipeline: Load filtered OD → motion detect → rPCA correct
    (iterative, full-signal, OR-union) → save → visualize → verify.

    All parameters default to values from config.yaml:
      n_components         ← stage_b.rpca.n_components        (Homer3 default: 0.97)
      min_channels_flagged ← stage_b.rpca.min_channels_flagged (Homer3 default: 1)
      max_iterations       ← stage_b.rpca.max_iterations       (Homer3 default: 5)

    Homer3 audit fixes applied (docs/reproduced_matlab_comparison.md §B4):
      4a — iterative PCA with re-detection between passes
      4b — full-signal projection removal
      4c — OR-union motion mask (min_channels_flagged=1)
      4d — nSV=0.97

    Parameters
    ----------
    dataset_name         : str
    subject_id           : str
    n_components         : float   nSV variance threshold (config: stage_b.rpca.n_components)
    min_channels_flagged : int     motion mask type (config: stage_b.rpca.min_channels_flagged)
    max_iterations       : int     max PCA passes (config: stage_b.rpca.max_iterations)
    run_tuning           : bool
        If True, run the Task 3.4.5 parameter grid-search sweep and save the
        heatmap/CSV.  Default False so the sweep is skipped during batch
        processing (07_preprocess_study.py).  The sweep runs 4×5=20 extra
        full rPCA passes per subject and has no effect on the correction output
        — it is purely diagnostic.  Set to True when running this script
        standalone (e.g. ``python 04_rPCA_motion_correct.py``) to explore
        parameter sensitivity on a single reference subject.
    verbose              : bool

    Returns
    -------
    results : dict
        {
          'raw_od_filt':        mne.io.Raw,
          'raw_od_corrected':   mne.io.Raw,
          'motion_results':     dict,
          'stats_before':       pd.DataFrame,
          'stats_after':        pd.DataFrame,
          'comparison_df':      pd.DataFrame,
          'reduction_summary':  dict,
          'correction_summary': dict,
          'tuning_df':          pd.DataFrame  or None  (None if run_tuning=False),
          'best_params':        dict           or None  (None if run_tuning=False),
          'output_dir':         Path,
          'fig_before_after':   Path,
          'fig_reduction':      Path,
          'fig_tuning':         Path           or None  (None if run_tuning=False),
        }
    """
    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B4: rPCA MOTION CORRECTION  |  Subject: {subject_id}")
    print(f"         Homer3-faithful (iterative, full-signal, OR-union)")
    print("=" * 70)

    # ---- Steps B1+B2: filtered OD ------------------------------------------
    print("\n[1/7] Loading filtered OD (Steps B1+B2)...")
    b2_results  = run_bandpass_filter(dataset_name, subject_id, verbose=False)
    raw_od_filt = b2_results['raw_od_filt']

    # ---- Step B3: motion detection (parameters from config) ----------------
    use_amp_only = bool(_B3_CFG.get("use_amp_only", False))
    print(f"[2/7] Detecting motion artifacts "
          f"({'amplitude-only' if use_amp_only else 'dual criterion'})...")
    motion_results = detect_motion_artifacts(
        raw_od_filt,
        amp_thresh   = _DEFAULT_AMP_THRESH,
        stdev_thresh = _DEFAULT_STDEV_THRESH,
        t_motion     = _DEFAULT_T_MOTION,
        t_mask       = _DEFAULT_T_MASK,
        use_amp_only = use_amp_only,
        verbose      = verbose,
    )
    stats_before = compute_motion_stats(motion_results, raw_od_filt, verbose=verbose)

    # ---- Step B4: iterative rPCA correction --------------------------------
    print("[3/7] Applying iterative rPCA correction (Homer3 parity)...")
    raw_od_corrected, correction_summary = apply_rpca_correction(
        raw_od_filt,
        motion_results,
        n_components         = n_components,
        min_channels_flagged = min_channels_flagged,
        max_iterations       = max_iterations,
        amp_thresh           = _DEFAULT_AMP_THRESH,
        stdev_thresh         = _DEFAULT_STDEV_THRESH,
        t_motion             = _DEFAULT_T_MOTION,
        t_mask               = _DEFAULT_T_MASK,
        use_amp_only         = use_amp_only,
        verbose              = verbose,
    )

    # ---- Save corrected OD -------------------------------------------------
    output_dir = get_output_path('stage_b', f'rPCA_correction/{subject_id}')
    fif_path, summary_path = save_corrected_od(
        raw_od_corrected, correction_summary, subject_id, output_dir
    )

    # ---- Before/after visualization ----------------------------------------
    print("[4/7] Generating before/after correction comparison plot...")
    fig_ba_path = plot_before_after_correction(
        raw_od_filt,
        raw_od_corrected,
        motion_results,
        stats_before,
        correction_summary,
        subject_id = subject_id,
        output_dir = output_dir,
        n_high     = 2,
        n_low      = 2,
        t_window   = 300.0,
        verbose    = verbose,
    )

    # ---- Post-correction motion verification -------------------------------
    print("[5/7] Verifying motion reduction post-rPCA...")
    stats_after, comparison_df, reduction_summary = verify_motion_reduction(
        raw_od_filt,
        raw_od_corrected,
        motion_results,
        stats_before,
        verbose = verbose,
    )

    # ---- Effectiveness documentation ---------------------------------------
    print("[6/7] Documenting rPCA effectiveness...")
    fig_red_path = plot_motion_reduction_summary(
        comparison_df, reduction_summary, correction_summary,
        subject_id=subject_id, output_dir=output_dir, verbose=verbose
    )
    csv_path, eff_path = save_motion_comparison(
        comparison_df, reduction_summary, subject_id, output_dir
    )

    # ---- Parameter tuning sweep (diagnostic only — skipped in batch) -------
    # The tuning runs 4×5=20 extra rPCA passes and produces CSV/heatmap
    # diagnostics.  It does NOT change the correction that was already applied
    # above.  It is disabled by default (run_tuning=False) to keep batch
    # processing fast.  Enable via run_tuning=True when running standalone.
    if run_tuning:
        print("[7/7] Running parameter tuning sweep (diagnostic only)...")
        tuning_df, best_params = tune_rpca_parameters(
            raw_od_filt, motion_results, stats_before,
            subject_id=subject_id, output_dir=output_dir, verbose=verbose
        )
        fig_tune_path = plot_tuning_results(
            tuning_df, best_params, subject_id, output_dir, verbose=verbose
        )
    else:
        print("[7/7] Skipping parameter tuning sweep (run_tuning=False; "
              "set run_tuning=True for standalone diagnostic use).")
        tuning_df     = None
        best_params   = None
        fig_tune_path = None

    # ---- Console summary ---------------------------------------------------
    rs = reduction_summary
    print("\n" + "=" * 70)
    print(f"✅ Step B4 complete for subject {subject_id}  (Homer3-faithful rPCA)")
    print(f"\n  rPCA Correction:")
    print(f"    Initial mask (OR-union) : {correction_summary['n_flagged_initial']} "
          f"({correction_summary['pct_flagged_union']:.1f}%)")
    print(f"    Iterations run          : {correction_summary['n_iterations']}")
    print(f"    PCs removed (final)     : {correction_summary['n_components_used']}")
    print(f"    Var removed (final)     : {correction_summary['variance_removed_pct']:.1f}%")
    print(f"\n  Motion Reduction Verification:")
    print(f"    Mean MR before      : {rs['mean_motion_ratio_before']*100:.1f}%")
    print(f"    Mean MR after (dual): {rs['mean_motion_ratio_after_dual']*100:.1f}%")
    print(f"    Mean MR after (amp) : {rs['mean_motion_ratio_after_amp']*100:.1f}%")
    print(f"    Mean reduction      : {rs['mean_reduction_pct']:.1f}%")
    print(f"    Channels improved   : {rs['n_channels_improved']}/{len(comparison_df)}")
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
    # When run standalone, enable the tuning sweep so parameter sensitivity
    # diagnostics are produced for the reference subject.
    run_rpca_correction(
        dataset_name = 'expert_novice',
        subject_id   = 'A1',
        run_tuning   = True,
        verbose      = True,
    )
