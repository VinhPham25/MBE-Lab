"""
Stage A - Step A3: Calculate Motion Ratio on Optical Density Data
Purpose: Calculate Motion Ratio on optical density (NOT raw intensity)

CRITICAL: Motion ratio MUST be calculated on OD, not raw intensity.
Reference: MR_logic.md, PROJECT.md Section 4.4, Paper Figure 4b,d,f,h

Motion Detection Parameters (from MATLAB hmrR_MotionArtifactByChannel):
- tMotion: 0.5 (time window for motion detection)
- tMask: 0.5 (time to mask around detected motion)
- STDEVthresh: 20 (standard deviation multiplier)
- AMPthresh: 5 (amplitude threshold in OD units)

Algorithm:
1. Convert raw intensity to optical density first
2. Calculate temporal derivative of OD signal
3. Flag Method 1: Where peak-to-peak change > 5 OD units (AMPthresh)
4. Flag Method 2: Where local std > 20 × global std (STDEVthresh)
5. Combine flagged timepoints (logical OR)
6. Motion Ratio = flagged_timepoints / total_timepoints
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import mne

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path

# Import load_data module
import importlib.util
spec = importlib.util.spec_from_file_location("load_data", Path(__file__).parent / "01_load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)
load_raw_snirf = load_data.load_raw_snirf


def _sanitize_intensity_data(raw, epsilon: float = 1e-6, verbose: bool = False):
    """
    Sanitize raw intensity data before OD conversion.

    Ensures data are finite and strictly positive to avoid warnings/errors in
    optical density conversion (division by zero, log of non-positive values).
    """
    raw_clean = raw.copy().load_data()
    data = raw_clean.get_data()

    # Replace non-finite values with per-channel median (or epsilon fallback).
    nonfinite_mask = ~np.isfinite(data)
    n_nonfinite = int(nonfinite_mask.sum())
    if n_nonfinite > 0:
        for ch_idx in range(data.shape[0]):
            ch = data[ch_idx]
            finite = ch[np.isfinite(ch)]
            fill = float(np.median(finite)) if finite.size > 0 else epsilon
            ch[~np.isfinite(ch)] = fill
            data[ch_idx] = ch

    # Convert negative values to absolute magnitude, then floor to epsilon.
    n_negative = int((data < 0).sum())
    if n_negative > 0:
        data = np.abs(data)
    n_nonpositive = int((data <= epsilon).sum())
    if n_nonpositive > 0:
        data[data <= epsilon] = epsilon

    raw_clean._data = data

    if verbose and (n_nonfinite > 0 or n_negative > 0 or n_nonpositive > 0):
        print(
            "   Sanitized intensity data before OD conversion: "
            f"nonfinite={n_nonfinite}, negative={n_negative}, "
            f"nonpositive_floored={n_nonpositive}"
        )

    return raw_clean


def convert_to_optical_density(raw, verbose: bool = True):
    """
    Convert raw intensity to optical density.
    
    Formula: OD = -log10(I / I₀)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS data object with intensity data
    verbose : bool
        Print conversion information
        
    Returns
    -------
    raw_od : mne.io.Raw
        Optical density data object
        
    Notes
    -----
    This is Step 2.3.1 - Convert raw intensity to OD first.
    OD makes motion detection more consistent across different baseline intensities.
    """
    if verbose:
        print("🔄 Converting raw intensity to optical density...")
        print("   Formula: OD = -log10(I / I₀)")
    
    # Sanitize intensity to avoid divide-by-zero and log issues in OD conversion.
    raw_clean = _sanitize_intensity_data(raw, epsilon=1e-6, verbose=verbose)

    # Convert to optical density using mne-nirs.
    raw_od = mne.preprocessing.nirs.optical_density(raw_clean)

    # Guard against non-finite values propagating into motion detection.
    od_data = raw_od.get_data()
    if not np.isfinite(od_data).all():
        n_bad = int((~np.isfinite(od_data)).sum())
        raw_od._data = np.nan_to_num(od_data, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(f"   Replaced {n_bad} non-finite OD samples with 0.0")
    
    if verbose:
        # Verify OD range is physiological (typically 0.1-2.0)
        od_data = raw_od.get_data()
        od_min = np.min(od_data)
        od_max = np.max(od_data)
        od_mean = np.mean(od_data)
        
        print(f"✓ Converted to OD")
        print(f"   OD range: [{od_min:.3f}, {od_max:.3f}]")
        print(f"   OD mean: {od_mean:.3f}")
        
        # Warn if outside typical physiological range
        if od_min < 0.05 or od_max > 3.0:
            print(f"   ⚠️  OD values outside typical range (0.1-2.0)")
    
    return raw_od


def detect_motion_homer3(od_signal, amp_thresh: float = 5.0,
                        stdev_thresh: float = 20.0,
                        sfreq: float = 20.0, 
                        t_motion: float = 0.5,
                        verbose: bool = False):
    """
    Detect motion artifacts using Homer3 hmrR_MotionArtifactByChannel algorithm.
    
    This implements the exact logic from Homer3:
    1. Calculate temporal derivative (diff of OD signal)
    2. Calculate global std of derivative (std_diff)
    3. For sliding window of tMotion samples, find max absolute difference
    4. Flag if max_diff > std_diff * stdev_thresh OR max_diff > amp_thresh
    
    Parameters
    ----------
    od_signal : np.ndarray
        Optical density signal (1D array)
    amp_thresh : float
        Amplitude threshold in OD units (default: 5.0)
    stdev_thresh : float
        Standard deviation multiplier (default: 20.0)
    sfreq : float
        Sampling frequency in Hz
    t_motion : float
        Time window for motion detection in seconds (default: 0.5)
    verbose : bool
        Print detection information
        
    Returns
    -------
    motion_mask : np.ndarray
        Boolean array, True where motion is detected (on derivative indices)
        Length is (n_samples - 1) because derivative reduces length by 1
        
    Notes
    -----
    Homer3 MATLAB code:
    ```matlab
    % calculate std_diff for each channel
    std_diff = std( d(2:end, :) - d(1:end-1, :), 0, 1);
    
    % calculate max_diff across channels for different time delays
    max_diff = zeros(size(d,1)-1, 1);
    for ii = 1:round(tMotion*fs)
        max_diff = max([abs(d((ii+1):end)-d(1:(end-ii))); zeros(ii-1,1)], max_diff);
    end
    
    % find indices with motion artifacts
    bad_inds = max( [ max_diff > std_diff*std_thresh, max_diff > amp_thresh ], [], 2);
    ```
    """
    od_signal = np.asarray(od_signal, dtype=float)

    # Replace non-finite values to prevent invalid arithmetic in derivatives.
    if not np.isfinite(od_signal).all():
        finite = od_signal[np.isfinite(od_signal)]
        fill = float(np.median(finite)) if finite.size > 0 else 0.0
        od_signal = np.nan_to_num(od_signal, nan=fill, posinf=fill, neginf=fill)

    n_samples = len(od_signal)
    if n_samples < 2:
        return np.zeros(max(n_samples - 1, 0), dtype=bool)
    
    # Step 1: Calculate temporal derivative
    # d(2:end) - d(1:end-1) in MATLAB = diff in Python
    derivative = np.diff(od_signal)  # Length: n_samples - 1
    n_derivative = len(derivative)
    
    # Step 2: Calculate global standard deviation of derivative
    std_diff = np.std(derivative)
    mc_thresh = std_diff * stdev_thresh
    
    # Step 3: Calculate max_diff for sliding window of tMotion samples
    # This finds the maximum absolute difference within tMotion seconds
    window_samples = int(np.round(t_motion * sfreq))
    max_diff = np.zeros(n_derivative)
    
    for ii in range(1, window_samples + 1):
        # abs(d((ii+1):end) - d(1:(end-ii))) in MATLAB
        # This compares signal at time t with signal at time t-ii
        if ii < n_samples:
            # Calculate absolute difference with time delay ii
            diff_delayed = np.abs(od_signal[ii:] - od_signal[:-ii])
            # Pad with zeros for the first (ii-1) samples
            padded = np.concatenate([np.zeros(ii-1), diff_delayed])
            # Take maximum with current max_diff
            max_diff = np.maximum(max_diff, padded)
    
    # Step 4: Flag samples exceeding either threshold
    # bad_inds = max_diff > mc_thresh OR max_diff > amp_thresh
    motion_mask_stdev = max_diff > mc_thresh
    motion_mask_amp = max_diff > amp_thresh
    motion_mask = np.logical_or(motion_mask_stdev, motion_mask_amp)
    
    if verbose:
        n_flagged = np.sum(motion_mask)
        ratio = n_flagged / n_derivative
        print(f"   Homer3 detection: {n_flagged}/{n_derivative} samples flagged ({ratio:.3f})")
        print(f"   std_diff: {std_diff:.4f}, mc_thresh: {mc_thresh:.4f}, amp_thresh: {amp_thresh:.4f}")
    
    return motion_mask


def apply_temporal_masking(motion_mask, sfreq: float, t_mask: float = 0.5):
    """
    Apply temporal masking around detected motion artifacts.
    
    Extends motion flags by ±t_mask seconds around each detection.
    This matches Homer3's art_buffer implementation.
    
    Parameters
    ----------
    motion_mask : np.ndarray
        Boolean array of detected motion (on derivative indices)
    sfreq : float
        Sampling frequency in Hz
    t_mask : float
        Time to mask around detected motion in seconds (default: 0.5)
        
    Returns
    -------
    masked : np.ndarray
        Boolean array with extended masking, same length as input
        
    Notes
    -----
    Homer3 MATLAB code:
    ```matlab
    art_buffer = round(tMask*fs);
    bad_inds = repmat(bad_inds, 1, 2*art_buffer+1) + ...
               repmat(-art_buffer:art_buffer, length(bad_inds), 1);
    bad_inds = bad_inds((bad_inds>0) & (bad_inds<=(size(d, 1)-1)));
    ```
    """
    art_buffer = int(np.round(t_mask * sfreq))
    
    # Find indices where motion is detected
    bad_inds = np.where(motion_mask)[0]
    
    if len(bad_inds) == 0:
        return motion_mask
    
    # Create extended indices: for each bad_ind, add [-art_buffer, ..., 0, ..., +art_buffer]
    # This replicates MATLAB's repmat logic
    offsets = np.arange(-art_buffer, art_buffer + 1)
    extended_inds = bad_inds[:, np.newaxis] + offsets[np.newaxis, :]
    extended_inds = extended_inds.flatten()
    
    # Keep only valid indices (>= 0 and < length)
    extended_inds = extended_inds[(extended_inds >= 0) & (extended_inds < len(motion_mask))]
    
    # Create masked array
    masked = np.zeros_like(motion_mask, dtype=bool)
    masked[extended_inds] = True
    
    return masked


def calculate_motion_ratio_od(raw_od, amp_thresh: float = 5.0,
                              stdev_thresh: float = 20.0,
                              t_motion: float = 0.5,
                              t_mask: float = 0.5,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Calculate Motion Ratio on optical density data using Homer3 algorithm.
    
    This is Steps 2.3.2-2.3.4: Apply motion detection and calculate ratio.
    
    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical density data object
    amp_thresh : float
        Amplitude threshold in OD units (default: 5.0)
    stdev_thresh : float
        Standard deviation multiplier (default: 20.0)
    t_motion : float
        Time window for motion detection (default: 0.5 s)
    t_mask : float
        Time to mask around detected motion (default: 0.5 s)
    verbose : bool
        Print calculation information
        
    Returns
    -------
    motion_df : pd.DataFrame
        DataFrame with columns:
        - channel: Channel name
        - wavelength: Wavelength (nm)
        - motion_ratio: Ratio of excluded samples to total samples
        - n_excluded: Number of excluded samples
        - n_total: Total number of samples
        - channel_type: Channel type
        
    Notes
    -----
    Algorithm from Homer3 hmrR_MotionArtifactByChannel:
    1. Calculate temporal derivative of OD signal: diff(OD)
    2. Calculate std of derivative: std_diff
    3. Find max absolute difference in sliding window of tMotion samples
    4. Flag if max_diff > std_diff*stdev_thresh OR max_diff > amp_thresh
    5. Apply temporal masking (extend flags by ±tMask samples)
    6. Motion Ratio = excluded_samples / total_samples
    
    Homer3 output is tIncCh where 1=included, 0=excluded.
    MATLAB: ChannelMR = 1 - sum(tIncCh{1}) / length(tIncCh{1})
    So: Motion Ratio = excluded_samples / total_samples
    """
    if verbose:
        print("🔍 Detecting motion artifacts on OD data (Homer3 algorithm)...")
        print(f"   Parameters:")
        print(f"     - AMPthresh: {amp_thresh} OD units")
        print(f"     - STDEVthresh: {stdev_thresh}x std of derivative")
        print(f"     - tMotion: {t_motion} s")
        print(f"     - tMask: {t_mask} s")
    
    # Get OD data
    od_data = raw_od.get_data()  # Shape: (n_channels, n_times)
    sfreq = raw_od.info['sfreq']
    n_times = od_data.shape[1]
    
    # Calculate motion ratio for each channel
    motion_results = []
    
    for i, ch_name in enumerate(raw_od.ch_names):
        # Get channel data
        ch_data = np.asarray(od_data[i, :], dtype=float)
        if not np.isfinite(ch_data).all():
            finite = ch_data[np.isfinite(ch_data)]
            fill = float(np.median(finite)) if finite.size > 0 else 0.0
            ch_data = np.nan_to_num(ch_data, nan=fill, posinf=fill, neginf=fill)
        
        # Detect motion using Homer3 algorithm
        # Returns motion_mask on derivative indices (length n_times - 1)
        motion_mask = detect_motion_homer3(
            ch_data, 
            amp_thresh=amp_thresh, 
            stdev_thresh=stdev_thresh,
            sfreq=sfreq, 
            t_motion=t_motion, 
            verbose=False
        )
        
        # Apply temporal masking (±tMask samples around each detection)
        motion_masked = apply_temporal_masking(
            motion_mask, sfreq=sfreq, t_mask=t_mask
        )
        
        # Homer3 adds 1 to bad_inds because they're on derivative
        # tInc{iBlk}(1+bad_inds) = 0
        # This means we need to create tInc array of length n_times
        tInc = np.ones(n_times, dtype=bool)  # 1 = included
        
        # bad_inds are on derivative (0-indexed), so map to original signal (1-indexed)
        bad_inds = np.where(motion_masked)[0]
        if len(bad_inds) > 0:
            # Add 1 to derivative indices to get original signal indices
            tInc[bad_inds + 1] = False  # 0 = excluded
        
        # Calculate motion ratio
        # MATLAB: ChannelMR = 1 - sum(tIncCh{1}) / length(tIncCh{1})
        n_included = np.sum(tInc)
        n_excluded = n_times - n_included
        motion_ratio = n_excluded / n_times if n_times > 0 else 0.0
        
        # Extract wavelength from channel name
        wavelength = extract_wavelength_from_channel(ch_name)
        
        # Get channel type
        ch_type = raw_od.get_channel_types()[i]
        
        motion_results.append({
            'channel': ch_name,
            'wavelength': wavelength,
            'motion_ratio': motion_ratio,
            'n_excluded': n_excluded,
            'n_total': n_times,
            'channel_type': ch_type
        })
    
    motion_df = pd.DataFrame(motion_results)
    
    if verbose:
        print(f"✓ Calculated motion ratio for {len(motion_df)} channels")
    
    return motion_df


def extract_wavelength_from_channel(ch_name: str) -> int:
    """
    Extract wavelength from channel name.
    
    Parameters
    ----------
    ch_name : str
        Channel name (e.g., 'S1_D1 690', 'S1_D2 830')
        
    Returns
    -------
    wavelength : int
        Wavelength in nm, or 0 if not found
    """
    # Skip hemoglobin channels
    if 'hbo' in ch_name.lower() or 'hbr' in ch_name.lower():
        return 0
    
    # Extract wavelength (last part of channel name)
    parts = ch_name.split()
    if len(parts) > 1 and parts[-1].isdigit():
        return int(parts[-1])
    
    return 0


def calculate_motion_statistics(motion_df: pd.DataFrame, by_wavelength: bool = True) -> dict:
    """
    Calculate summary statistics for motion ratio values.
    
    Parameters
    ----------
    motion_df : pd.DataFrame
        Motion ratio DataFrame from calculate_motion_ratio_od()
    by_wavelength : bool
        Calculate statistics separately for each wavelength
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - overall: Overall statistics (mean, median, std, min, max)
        - by_wavelength: Statistics per wavelength (if requested)
        - n_channels: Total number of channels
    """
    stats = {
        'n_channels': len(motion_df)
    }
    
    # Overall statistics
    stats['overall'] = {
        'mean': motion_df['motion_ratio'].mean(),
        'median': motion_df['motion_ratio'].median(),
        'std': motion_df['motion_ratio'].std(),
        'min': motion_df['motion_ratio'].min(),
        'max': motion_df['motion_ratio'].max(),
        'q25': motion_df['motion_ratio'].quantile(0.25),
        'q75': motion_df['motion_ratio'].quantile(0.75)
    }
    
    # By wavelength
    if by_wavelength:
        stats['by_wavelength'] = {}
        for wl in motion_df['wavelength'].unique():
            if wl > 0:  # Skip wavelength 0 (hemoglobin channels)
                wl_data = motion_df[motion_df['wavelength'] == wl]
                if len(wl_data) > 0:
                    stats['by_wavelength'][wl] = {
                        'mean': wl_data['motion_ratio'].mean(),
                        'median': wl_data['motion_ratio'].median(),
                        'std': wl_data['motion_ratio'].std(),
                        'min': wl_data['motion_ratio'].min(),
                        'max': wl_data['motion_ratio'].max(),
                        'n_channels': len(wl_data)
                    }
    
    return stats


def print_motion_summary(motion_df: pd.DataFrame, stats: dict):
    """
    Print formatted motion ratio summary.
    
    Parameters
    ----------
    motion_df : pd.DataFrame
        Motion ratio DataFrame
    stats : dict
        Statistics from calculate_motion_statistics()
    """
    print("\n" + "="*70)
    print("MOTION RATIO SUMMARY (OPTICAL DENSITY - HOMER3 ALGORITHM)")
    print("="*70)
    
    print(f"\n📊 CHANNEL STATISTICS:")
    print(f"  Total channels: {stats['n_channels']}")
    
    print(f"\n📈 OVERALL MOTION RATIO STATISTICS:")
    print(f"  Mean: {stats['overall']['mean']:.4f} ({stats['overall']['mean']*100:.2f}%)")
    print(f"  Median: {stats['overall']['median']:.4f} ({stats['overall']['median']*100:.2f}%)")
    print(f"  Std Dev: {stats['overall']['std']:.4f}")
    print(f"  Range: [{stats['overall']['min']:.4f}, {stats['overall']['max']:.4f}]")
    print(f"  IQR: [{stats['overall']['q25']:.4f}, {stats['overall']['q75']:.4f}]")
    
    if 'by_wavelength' in stats and stats['by_wavelength']:
        print(f"\n🌈 MOTION RATIO BY WAVELENGTH:")
        for wl, wl_stats in stats['by_wavelength'].items():
            print(f"  {wl} nm ({wl_stats['n_channels']} channels):")
            print(f"    Mean: {wl_stats['mean']:.4f} ({wl_stats['mean']*100:.2f}%), "
                  f"Median: {wl_stats['median']:.4f}, "
                  f"Range: [{wl_stats['min']:.4f}, {wl_stats['max']:.4f}]")
    
    # Show channels with highest motion ratio
    motion_sorted = motion_df.sort_values('motion_ratio', ascending=False)
    
    if len(motion_sorted) > 0:
        print(f"\n🔼 HIGHEST MOTION RATIO CHANNELS (top 5):")
        for i, row in motion_sorted.head(5).iterrows():
            n_excluded = row.get('n_excluded', row.get('n_flagged', 0))
            print(f"  {row['channel']:30s} {row['motion_ratio']:6.4f} "
                  f"({row['motion_ratio']*100:.2f}%) - {n_excluded}/{row['n_total']} samples")
        
        print(f"\n🔽 LOWEST MOTION RATIO CHANNELS (top 5):")
        for i, row in motion_sorted.tail(5).iterrows():
            n_excluded = row.get('n_excluded', row.get('n_flagged', 0))
            print(f"  {row['channel']:30s} {row['motion_ratio']:6.4f} "
                  f"({row['motion_ratio']*100:.2f}%) - {n_excluded}/{row['n_total']} samples")
    
    print("\n" + "="*70 + "\n")


def save_motion_results(motion_df: pd.DataFrame, subject_id: str, output_dir: Path):
    """
    Save motion ratio results to CSV file.
    
    Parameters
    ----------
    motion_df : pd.DataFrame
        Motion ratio DataFrame
    subject_id : str
        Subject ID
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    output_path = output_dir / f"{subject_id}_motion_od.csv"
    motion_df.to_csv(output_path, index=False)
    
    print(f"💾 Motion ratio results saved: {output_path}")
    
    return output_path


# Test function
def test_calculate_motion_ratio():
    """
    Test motion ratio calculation on subject A1.
    This is Step 2.3.5: Verify results.
    """
    print("="*70)
    print("TESTING MOTION RATIO CALCULATION - Subject A1")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading data...")
    raw, metadata = load_raw_snirf('expert_novice', 'A1', preload=True, verbose=True)
    print()
    
    # Step 2.3.1: Convert to optical density
    print("Step 2.3.1: Converting to optical density...")
    raw_od = convert_to_optical_density(raw, verbose=True)
    print()
    
    # Steps 2.3.2-2.3.4: Calculate motion ratio
    print("Steps 2.3.2-2.3.4: Detecting motion and calculating ratio...")
    motion_df = calculate_motion_ratio_od(
        raw_od,
        amp_thresh=5.0,      # AMPthresh in OD units
        stdev_thresh=20.0,   # STDEVthresh multiplier
        t_motion=0.5,        # Time window for detection
        t_mask=0.5,          # Time to mask around motion
        verbose=True
    )
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats = calculate_motion_statistics(motion_df, by_wavelength=True)
    
    # Print summary
    print_motion_summary(motion_df, stats)
    
    # Save results
    output_dir = get_output_path('stage_a', 'qa_reports')
    save_motion_results(motion_df, 'A1', output_dir)
    
    # Step 2.3.5: Verify results
    print("🔍 VERIFICATION:")
    median_ratio = stats['overall']['median']
    mean_ratio = stats['overall']['mean']
    print(f"  Median motion ratio: {median_ratio:.4f} ({median_ratio*100:.2f}%)")
    print(f"  Mean motion ratio: {mean_ratio:.4f} ({mean_ratio*100:.2f}%)")
    
    # Expected: ~0 median motion ratio (from paper Figure 4b,d,f,h)
    if median_ratio < 0.05:  # Less than 5% motion
        print(f"  ✅ Median motion ratio is very low (~0, as expected from Figure 4)")
    elif median_ratio < 0.20:  # Less than 20% motion
        print(f"  ✅ Median motion ratio is low (reasonable for good quality data)")
    else:
        print(f"  ⚠️  Median motion ratio is higher than expected (should be ~0 for A1)")
    
    # Check distribution
    low_motion = motion_df[motion_df['motion_ratio'] < 0.10]
    pct_low_motion = 100 * len(low_motion) / len(motion_df)
    print(f"  Channels with <10% motion: {len(low_motion)}/{len(motion_df)} ({pct_low_motion:.1f}%)")
    
    print("\n✅ Motion ratio calculation test complete!")
    print("📋 Completed all 5 tasks in Section 2.3!")
    
    return motion_df, stats


if __name__ == '__main__':
    # Run test
    motion_df, stats = test_calculate_motion_ratio()
