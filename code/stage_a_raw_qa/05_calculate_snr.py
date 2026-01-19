"""
Stage A - Step A2: Calculate SNR on Raw Intensity Data
Purpose: Calculate Signal-to-Noise Ratio on unprocessed raw intensity data

Formula: SNR = 20 * log10(mean / std)
Reference: PROJECT.md Section 4.3, Paper Figure 4a,c,e,g

IMPORTANT: Calculate on RAW intensity data WITHOUT any preprocessing or denoising.
This is for transparency and quality documentation (as stated in paper).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path

# Import load_data module
import importlib.util
spec = importlib.util.spec_from_file_location("load_data", Path(__file__).parent / "01_load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)
load_raw_snirf = load_data.load_raw_snirf


def calculate_snr_raw(raw, verbose: bool = True) -> pd.DataFrame:
    """
    Calculate Signal-to-Noise Ratio on raw intensity data.
    
    Formula: SNR = 20 * log10(mean / std)
    
    where:
    - mean = mean of raw intensity signal
    - std = standard deviation of raw intensity signal
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS data object (unprocessed)
    verbose : bool
        Print calculation information
        
    Returns
    -------
    snr_df : pd.DataFrame
        DataFrame with columns:
        - channel: Channel name
        - wavelength: Wavelength (nm)
        - mean: Mean intensity
        - std: Standard deviation
        - snr_db: SNR in decibels
        - channel_type: Channel type
        
    Notes
    -----
    This calculation is performed on RAW intensity data without any
    preprocessing or denoising, as stated in the paper:
    "To give an accurate assessment of the quality of the data, no channels 
    have been excluded from the analysis provided here and no denoising 
    methods have been used prior to analysis."
    """
    if verbose:
        print("📊 Calculating SNR on raw intensity data...")
        print("   Formula: SNR = 20 * log10(mean / std)")
    
    # Get raw data (this is raw intensity, NO preprocessing)
    data = raw.get_data()  # Shape: (n_channels, n_times)
    
    # Calculate SNR for each channel
    snr_results = []
    
    for i, ch_name in enumerate(raw.ch_names):
        # Get channel data
        ch_data = data[i, :]
        
        # Calculate mean and std
        mean_intensity = np.mean(ch_data)
        std_intensity = np.std(ch_data)
        
        # Calculate SNR in dB
        if std_intensity > 0:
            snr_db = 20 * np.log10(mean_intensity / std_intensity)
        else:
            snr_db = np.nan  # Avoid division by zero
        
        # Extract wavelength from channel name
        wavelength = extract_wavelength_from_channel(ch_name)
        
        # Get channel type
        ch_type = raw.get_channel_types()[i]
        
        snr_results.append({
            'channel': ch_name,
            'wavelength': wavelength,
            'mean': mean_intensity,
            'std': std_intensity,
            'snr_db': snr_db,
            'channel_type': ch_type
        })
    
    snr_df = pd.DataFrame(snr_results)
    
    if verbose:
        print(f"✓ Calculated SNR for {len(snr_df)} channels")
    
    return snr_df


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


def calculate_snr_statistics(snr_df: pd.DataFrame, by_wavelength: bool = True) -> dict:
    """
    Calculate summary statistics for SNR values.
    
    Parameters
    ----------
    snr_df : pd.DataFrame
        SNR DataFrame from calculate_snr_raw()
    by_wavelength : bool
        Calculate statistics separately for each wavelength
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - overall: Overall statistics (mean, median, std, min, max)
        - by_wavelength: Statistics per wavelength (if requested)
        - n_channels: Total number of channels
        - n_valid: Number of valid SNR values (not NaN)
    """
    # Remove NaN values
    valid_snr = snr_df.dropna(subset=['snr_db'])
    
    stats = {
        'n_channels': len(snr_df),
        'n_valid': len(valid_snr),
        'n_invalid': len(snr_df) - len(valid_snr)
    }
    
    # Overall statistics
    if len(valid_snr) > 0:
        stats['overall'] = {
            'mean': valid_snr['snr_db'].mean(),
            'median': valid_snr['snr_db'].median(),
            'std': valid_snr['snr_db'].std(),
            'min': valid_snr['snr_db'].min(),
            'max': valid_snr['snr_db'].max(),
            'q25': valid_snr['snr_db'].quantile(0.25),
            'q75': valid_snr['snr_db'].quantile(0.75)
        }
    else:
        stats['overall'] = None
    
    # By wavelength
    if by_wavelength:
        stats['by_wavelength'] = {}
        for wl in snr_df['wavelength'].unique():
            if wl > 0:  # Skip wavelength 0 (hemoglobin channels)
                wl_data = valid_snr[valid_snr['wavelength'] == wl]
                if len(wl_data) > 0:
                    stats['by_wavelength'][wl] = {
                        'mean': wl_data['snr_db'].mean(),
                        'median': wl_data['snr_db'].median(),
                        'std': wl_data['snr_db'].std(),
                        'min': wl_data['snr_db'].min(),
                        'max': wl_data['snr_db'].max(),
                        'n_channels': len(wl_data)
                    }
    
    return stats


def print_snr_summary(snr_df: pd.DataFrame, stats: dict):
    """
    Print formatted SNR summary.
    
    Parameters
    ----------
    snr_df : pd.DataFrame
        SNR DataFrame
    stats : dict
        Statistics from calculate_snr_statistics()
    """
    print("\n" + "="*70)
    print("SNR SUMMARY (RAW INTENSITY)")
    print("="*70)
    
    print(f"\n📊 CHANNEL STATISTICS:")
    print(f"  Total channels: {stats['n_channels']}")
    print(f"  Valid SNR values: {stats['n_valid']}")
    if stats['n_invalid'] > 0:
        print(f"  Invalid SNR values: {stats['n_invalid']} (likely zero std)")
    
    if stats['overall']:
        print(f"\n📈 OVERALL SNR STATISTICS:")
        print(f"  Mean: {stats['overall']['mean']:.2f} dB")
        print(f"  Median: {stats['overall']['median']:.2f} dB")
        print(f"  Std Dev: {stats['overall']['std']:.2f} dB")
        print(f"  Range: [{stats['overall']['min']:.2f}, {stats['overall']['max']:.2f}] dB")
        print(f"  IQR: [{stats['overall']['q25']:.2f}, {stats['overall']['q75']:.2f}] dB")
    
    if 'by_wavelength' in stats and stats['by_wavelength']:
        print(f"\n🌈 SNR BY WAVELENGTH:")
        for wl, wl_stats in stats['by_wavelength'].items():
            print(f"  {wl} nm ({wl_stats['n_channels']} channels):")
            print(f"    Mean: {wl_stats['mean']:.2f} dB, "
                  f"Median: {wl_stats['median']:.2f} dB, "
                  f"Range: [{wl_stats['min']:.2f}, {wl_stats['max']:.2f}]")
    
    # Show sample of channels with lowest and highest SNR
    valid_snr = snr_df.dropna(subset=['snr_db']).sort_values('snr_db')
    
    if len(valid_snr) > 0:
        print(f"\n🔽 LOWEST SNR CHANNELS (top 5):")
        for i, row in valid_snr.head(5).iterrows():
            print(f"  {row['channel']:30s} {row['snr_db']:6.2f} dB")
        
        print(f"\n🔼 HIGHEST SNR CHANNELS (top 5):")
        for i, row in valid_snr.tail(5).iterrows():
            print(f"  {row['channel']:30s} {row['snr_db']:6.2f} dB")
    
    print("\n" + "="*70 + "\n")


def save_snr_results(snr_df: pd.DataFrame, subject_id: str, output_dir: Path):
    """
    Save SNR results to CSV file.
    
    Parameters
    ----------
    snr_df : pd.DataFrame
        SNR DataFrame
    subject_id : str
        Subject ID
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    output_path = output_dir / f"{subject_id}_snr_raw.csv"
    snr_df.to_csv(output_path, index=False)
    
    print(f"💾 SNR results saved: {output_path}")
    
    return output_path


# Test function
def test_calculate_snr():
    """
    Test SNR calculation on subject A1.
    """
    print("="*70)
    print("TESTING SNR CALCULATION - Subject A1")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading data...")
    raw, metadata = load_raw_snirf('expert_novice', 'A1', preload=True, verbose=True)
    print()
    
    # Calculate SNR
    snr_df = calculate_snr_raw(raw, verbose=True)
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats = calculate_snr_statistics(snr_df, by_wavelength=True)
    
    # Print summary
    print_snr_summary(snr_df, stats)
    
    # Save results
    output_dir = get_output_path('stage_a', 'qa_reports')
    save_snr_results(snr_df, 'A1', output_dir)
    
    # Verify expected range
    print("🔍 VERIFICATION:")
    if stats['overall']:
        mean_snr = stats['overall']['mean']
        print(f"  Mean SNR: {mean_snr:.2f} dB")
        
        # Expected range from paper: ~15-30 dB
        if 10 < mean_snr < 40:
            print(f"  ✅ Mean SNR is in reasonable range (10-40 dB)")
        else:
            print(f"  ⚠️  Mean SNR outside expected range (should be ~15-30 dB)")
        
        # Check if most channels are in reasonable range
        valid_snr = snr_df.dropna(subset=['snr_db'])
        in_range = valid_snr[(valid_snr['snr_db'] >= 10) & (valid_snr['snr_db'] <= 40)]
        pct_in_range = 100 * len(in_range) / len(valid_snr)
        print(f"  Channels in range (10-40 dB): {len(in_range)}/{len(valid_snr)} ({pct_in_range:.1f}%)")
    
    print("\n✅ SNR calculation test complete!")
    
    return snr_df, stats


if __name__ == '__main__':
    # Run test
    snr_df, stats = test_calculate_snr()
