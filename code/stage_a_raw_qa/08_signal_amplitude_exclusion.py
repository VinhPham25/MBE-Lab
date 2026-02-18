"""
Stage A - Step A5: Signal Amplitude Assessment (for Stage B exclusion)
Purpose: Calculate raw signal amplitude in dB for all channels.

Task coverage in this file:
  2.5.1 - Calculate signal amplitude in dB for all channels (implemented)
  2.5.2 - Identify channels outside 80-140 dB range (implemented)
  2.5.3 - Create exclusion-list JSON structure (implemented)
  2.5.4 - Save final exclusion JSON to excluded_channels/ (next subtask)

Reference: PROJECT.md Section 4, Stage A/B handoff notes

IMPORTANT:
- This is separate from SNR.
- No channels are excluded in Stage A technical validation.
- Stage A only documents channels; exclusion is applied later in Stage B.
"""

import numpy as np
import pandas as pd
import sys
import json
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
    if 'hbo' in ch_name.lower() or 'hbr' in ch_name.lower():
        return 0

    parts = ch_name.split()
    if len(parts) > 1 and parts[-1].isdigit():
        return int(parts[-1])

    return 0


def calculate_signal_amplitude_db(raw, verbose: bool = True) -> pd.DataFrame:
    """
    Calculate channel-level signal amplitude in dB from raw intensity.

    Amplitude metric used for Stage B exclusion documentation:
      amplitude_db = 20 * log10(RMS(raw_intensity))

    where RMS (root mean square) is calculated per channel over time.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS data object (unprocessed intensity channels)
    verbose : bool
        Print progress information

    Returns
    -------
    amplitude_df : pd.DataFrame
        DataFrame with columns:
        - channel
        - wavelength
        - channel_type
        - mean_intensity
        - std_intensity
        - rms_intensity
        - amplitude_db
    """
    if verbose:
        print("📊 Calculating signal amplitude (dB) on raw intensity...")
        print("   Formula: amplitude_db = 20 * log10(RMS(raw_intensity))")

    data = raw.get_data()  # Shape: (n_channels, n_times)
    ch_types = raw.get_channel_types()

    results = []
    for i, ch_name in enumerate(raw.ch_names):
        ch_data = data[i, :]

        mean_intensity = float(np.mean(ch_data))
        std_intensity = float(np.std(ch_data))
        rms_intensity = float(np.sqrt(np.mean(np.square(ch_data))))

        if rms_intensity > 0:
            amplitude_db = float(20 * np.log10(rms_intensity))
        else:
            amplitude_db = np.nan

        results.append({
            'channel': ch_name,
            'wavelength': extract_wavelength_from_channel(ch_name),
            'channel_type': ch_types[i],
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'rms_intensity': rms_intensity,
            'amplitude_db': amplitude_db,
        })

    amplitude_df = pd.DataFrame(results)

    if verbose:
        print(f"✓ Calculated amplitude for {len(amplitude_df)} channels")

    return amplitude_df


def identify_amplitude_out_of_range(
    amplitude_df: pd.DataFrame,
    low_db: float = 80.0,
    high_db: float = 140.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Task 2.5.2: Identify channels outside the 80-140 dB amplitude range.

    IMPORTANT:
    - This only tags candidate channels for Stage B exclusion.
    - No channels are excluded in Stage A.

    Parameters
    ----------
    amplitude_df : pd.DataFrame
        Output of calculate_signal_amplitude_db()
    low_db : float
        Lower threshold (inclusive pass boundary)
    high_db : float
        Upper threshold (inclusive pass boundary)
    verbose : bool
        Print summary to console

    Returns
    -------
    flagged_df : pd.DataFrame
        Original DataFrame with added columns:
        - within_80_140_db (bool)
        - exclusion_candidate (bool)
        - exclusion_reason (str)
    """
    flagged_df = amplitude_df.copy()

    low_mask = flagged_df['amplitude_db'] < low_db
    high_mask = flagged_df['amplitude_db'] > high_db
    invalid_mask = flagged_df['amplitude_db'].isna()

    flagged_df['within_80_140_db'] = ~(low_mask | high_mask | invalid_mask)
    flagged_df['exclusion_candidate'] = ~flagged_df['within_80_140_db']

    flagged_df['exclusion_reason'] = 'within_range'
    flagged_df.loc[low_mask, 'exclusion_reason'] = f'below_{int(low_db)}db'
    flagged_df.loc[high_mask, 'exclusion_reason'] = f'above_{int(high_db)}db'
    flagged_df.loc[invalid_mask, 'exclusion_reason'] = 'invalid_amplitude'

    if verbose:
        n_total = len(flagged_df)
        n_bad = int(flagged_df['exclusion_candidate'].sum())
        n_good = n_total - n_bad
        n_low = int(low_mask.sum())
        n_high = int(high_mask.sum())
        n_invalid = int(invalid_mask.sum())

        print("\n" + "=" * 70)
        print("AMPLITUDE RANGE ASSESSMENT (TASK 2.5.2)")
        print("=" * 70)
        print(f"Threshold range: [{low_db:.0f}, {high_db:.0f}] dB")
        print(f"Total channels: {n_total}")
        print(f"Within range (good): {n_good}")
        print(f"Outside range (candidate bad): {n_bad}")
        print(f"  - Below {low_db:.0f} dB: {n_low}")
        print(f"  - Above {high_db:.0f} dB: {n_high}")
        if n_invalid > 0:
            print(f"  - Invalid amplitude: {n_invalid}")
        print("NOTE: Stage A documents candidates only; no exclusion applied yet.")
        print("=" * 70 + "\n")

    return flagged_df


def calculate_amplitude_statistics(amplitude_df: pd.DataFrame, by_wavelength: bool = True) -> dict:
    """
    Calculate summary statistics for amplitude dB values.
    """
    valid = amplitude_df.dropna(subset=['amplitude_db'])

    stats = {
        'n_channels': len(amplitude_df),
        'n_valid': len(valid),
        'n_invalid': len(amplitude_df) - len(valid),
    }

    if len(valid) > 0:
        stats['overall'] = {
            'mean': float(valid['amplitude_db'].mean()),
            'median': float(valid['amplitude_db'].median()),
            'std': float(valid['amplitude_db'].std()),
            'min': float(valid['amplitude_db'].min()),
            'max': float(valid['amplitude_db'].max()),
            'q25': float(valid['amplitude_db'].quantile(0.25)),
            'q75': float(valid['amplitude_db'].quantile(0.75)),
        }
    else:
        stats['overall'] = None

    if by_wavelength:
        stats['by_wavelength'] = {}
        for wl in sorted(amplitude_df['wavelength'].unique()):
            if wl <= 0:
                continue
            wl_df = valid[valid['wavelength'] == wl]
            if len(wl_df) == 0:
                continue
            stats['by_wavelength'][int(wl)] = {
                'mean': float(wl_df['amplitude_db'].mean()),
                'median': float(wl_df['amplitude_db'].median()),
                'std': float(wl_df['amplitude_db'].std()),
                'min': float(wl_df['amplitude_db'].min()),
                'max': float(wl_df['amplitude_db'].max()),
                'n_channels': int(len(wl_df)),
            }

    return stats


def print_amplitude_summary(amplitude_df: pd.DataFrame, stats: dict):
    """
    Print formatted amplitude summary.
    """
    print("\n" + "=" * 70)
    print("SIGNAL AMPLITUDE SUMMARY (RAW INTENSITY)")
    print("=" * 70)

    print(f"\n📊 CHANNEL STATISTICS:")
    print(f"  Total channels: {stats['n_channels']}")
    print(f"  Valid amplitude values: {stats['n_valid']}")
    if stats['n_invalid'] > 0:
        print(f"  Invalid amplitude values: {stats['n_invalid']}")

    if stats['overall']:
        print(f"\n📈 OVERALL AMPLITUDE STATISTICS:")
        print(f"  Mean: {stats['overall']['mean']:.2f} dB")
        print(f"  Median: {stats['overall']['median']:.2f} dB")
        print(f"  Std Dev: {stats['overall']['std']:.2f} dB")
        print(f"  Range: [{stats['overall']['min']:.2f}, {stats['overall']['max']:.2f}] dB")
        print(f"  IQR: [{stats['overall']['q25']:.2f}, {stats['overall']['q75']:.2f}] dB")

    if 'by_wavelength' in stats and stats['by_wavelength']:
        print(f"\n🌈 AMPLITUDE BY WAVELENGTH:")
        for wl, wl_stats in stats['by_wavelength'].items():
            print(f"  {wl} nm ({wl_stats['n_channels']} channels):")
            print(f"    Mean: {wl_stats['mean']:.2f} dB, "
                  f"Median: {wl_stats['median']:.2f} dB, "
                  f"Range: [{wl_stats['min']:.2f}, {wl_stats['max']:.2f}]")

    sorted_df = amplitude_df.dropna(subset=['amplitude_db']).sort_values('amplitude_db')
    if len(sorted_df) > 0:
        print(f"\n🔽 LOWEST AMPLITUDE CHANNELS (top 5):")
        for _, row in sorted_df.head(5).iterrows():
            print(f"  {row['channel']:30s} {row['amplitude_db']:7.2f} dB")

        print(f"\n🔼 HIGHEST AMPLITUDE CHANNELS (top 5):")
        for _, row in sorted_df.tail(5).iterrows():
            print(f"  {row['channel']:30s} {row['amplitude_db']:7.2f} dB")

    print("\n" + "=" * 70 + "\n")


def save_signal_amplitude_results(amplitude_df: pd.DataFrame, subject_id: str, output_dir: Path) -> Path:
    """
    Save signal amplitude results to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_signal_amplitude_raw.csv"
    amplitude_df.to_csv(output_path, index=False)
    print(f"💾 Signal amplitude results saved: {output_path}")
    return output_path


def save_flagged_amplitude_results(flagged_df: pd.DataFrame, subject_id: str, output_dir: Path) -> Path:
    """
    Save Task 2.5.2 flagged amplitude results to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_signal_amplitude_flagged.csv"
    flagged_df.to_csv(output_path, index=False)
    print(f"💾 Flagged amplitude results saved: {output_path}")
    return output_path


def create_exclusion_list_json(
    flagged_df: pd.DataFrame,
    subject_id: str,
    low_db: float = 80.0,
    high_db: float = 140.0,
    verbose: bool = True,
) -> dict:
    """
    Task 2.5.3: Build exclusion-list JSON payload for Stage B handoff.

    Required structure:
      {
        "subject_id": ...,
        "bad_channels": [...],
        "good_channels": [...],
        "exclusion_reasons": {...}
      }
    """
    required_cols = {'channel', 'exclusion_candidate', 'exclusion_reason'}
    missing = required_cols - set(flagged_df.columns)
    if missing:
        raise ValueError(f"flagged_df missing required columns: {sorted(missing)}")

    bad_df = flagged_df[flagged_df['exclusion_candidate']].copy()
    good_df = flagged_df[~flagged_df['exclusion_candidate']].copy()

    bad_channels = sorted(bad_df['channel'].tolist())
    good_channels = sorted(good_df['channel'].tolist())
    exclusion_reasons = {
        row['channel']: row['exclusion_reason']
        for _, row in bad_df[['channel', 'exclusion_reason']].iterrows()
    }

    exclusion_json = {
        'subject_id': subject_id,
        'bad_channels': bad_channels,
        'good_channels': good_channels,
        'exclusion_reasons': exclusion_reasons,
        'thresholds_db': {
            'low': low_db,
            'high': high_db,
        },
        'stage_a_note': (
            "Stage A documentation only: channels listed as exclusion candidates "
            "for Stage B. No channels excluded from Stage A QA analysis."
        ),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("EXCLUSION LIST JSON (TASK 2.5.3)")
        print("=" * 70)
        print(f"subject_id: {subject_id}")
        print(f"bad_channels: {len(bad_channels)}")
        print(f"good_channels: {len(good_channels)}")
        if bad_channels:
            print("candidate bad channels:")
            for ch in bad_channels:
                print(f"  - {ch} ({exclusion_reasons[ch]})")
        print("=" * 70 + "\n")

    return exclusion_json


def save_exclusion_list_preview(exclusion_json: dict, subject_id: str, output_dir: Path) -> Path:
    """
    Save Task 2.5.3 JSON preview to signal_quality_exclusion directory.

    Final Stage B handoff save location is handled in Task 2.5.4.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_exclusion_list_preview.json"
    with open(output_path, 'w') as f:
        json.dump(exclusion_json, f, indent=2)
    print(f"💾 Exclusion-list preview saved: {output_path}")
    return output_path


def test_calculate_signal_amplitude():
    """
    Test tasks 2.5.1-2.5.3 on subject A1.
    """
    print("=" * 70)
    print("TESTING SIGNAL AMPLITUDE CALCULATION - Subject A1")
    print("=" * 70 + "\n")

    print("📂 Loading data...")
    raw, _ = load_raw_snirf('expert_novice', 'A1', preload=True, verbose=True)
    print()

    amplitude_df = calculate_signal_amplitude_db(raw, verbose=True)
    stats = calculate_amplitude_statistics(amplitude_df, by_wavelength=True)
    print_amplitude_summary(amplitude_df, stats)

    flagged_df = identify_amplitude_out_of_range(
        amplitude_df,
        low_db=80.0,
        high_db=140.0,
        verbose=True,
    )

    output_dir = get_output_path('stage_a', 'signal_quality_exclusion')
    save_signal_amplitude_results(amplitude_df, 'A1', output_dir)
    save_flagged_amplitude_results(flagged_df, 'A1', output_dir)
    exclusion_json = create_exclusion_list_json(
        flagged_df,
        subject_id='A1',
        low_db=80.0,
        high_db=140.0,
        verbose=True,
    )
    save_exclusion_list_preview(exclusion_json, 'A1', output_dir)

    print("✅ Tasks 2.5.1-2.5.3 test complete!")
    return exclusion_json, stats


if __name__ == '__main__':
    test_calculate_signal_amplitude()
