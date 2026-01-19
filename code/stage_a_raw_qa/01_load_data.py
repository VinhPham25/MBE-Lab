"""
Stage A - Step A1: Load Raw Data and Extract Metadata
Purpose: Load .snirf files without preprocessing and verify structure

Reference: PROJECT.md Section 4.2
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_dataset_config, get_snirf_files, get_subject_path


def load_raw_snirf(dataset_name: str, subject_id: str, 
                   preload: bool = False, verbose: bool = True):
    """
    Load .snirf file without any preprocessing.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., 'expert_novice')
    subject_id : str
        Subject ID (e.g., 'A1')
    preload : bool
        Whether to preload data into memory
    verbose : bool
        Print loading information
        
    Returns
    -------
    raw : mne.io.Raw
        Raw fNIRS data object
    metadata : dict
        Extracted metadata dictionary
        
    Examples
    --------
    >>> raw, meta = load_raw_snirf('expert_novice', 'A1')
    >>> print(f"Sampling rate: {meta['sfreq']} Hz")
    """
    # Suppress MNE verbose output if requested
    if not verbose:
        mne.set_log_level('WARNING')
    
    # Get .snirf file path
    snirf_files = get_snirf_files(dataset_name, subject_id)
    
    if len(snirf_files) > 1:
        print(f"⚠️  Multiple .snirf files found for {subject_id}. Using first: {snirf_files[0].name}")
    
    snirf_path = snirf_files[0]
    
    if verbose:
        print(f"📂 Loading: {snirf_path.name}")
        print(f"   Size: {snirf_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load .snirf file - NO preprocessing
    raw = mne.io.read_raw_snirf(snirf_path, preload=preload, verbose=verbose)
    
    # Extract metadata
    metadata = extract_metadata(raw, dataset_name, subject_id, snirf_path)
    
    if verbose:
        print(f"✓ Loaded: {metadata['n_channels']} channels, "
              f"{metadata['duration']:.1f}s, {metadata['sfreq']}Hz")
    
    return raw, metadata


def extract_metadata(raw, dataset_name: str, subject_id: str, 
                    snirf_path: Path) -> dict:
    """
    Extract metadata from raw fNIRS data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS data object
    dataset_name : str
        Dataset name
    subject_id : str
        Subject ID
    snirf_path : Path
        Path to .snirf file
        
    Returns
    -------
    metadata : dict
        Dictionary containing:
        - subject_id, dataset_name, file_path
        - sfreq (sampling rate)
        - n_channels, n_times
        - duration
        - ch_names (list of channel names)
        - ch_types (list of channel types)
        - wavelengths (unique wavelengths detected)
        - n_annotations (number of event markers)
        - annotation_descriptions (unique event types)
    """
    metadata = {
        'subject_id': subject_id,
        'dataset_name': dataset_name,
        'file_path': str(snirf_path),
        'file_size_mb': snirf_path.stat().st_size / (1024*1024),
        
        # Basic info
        'sfreq': raw.info['sfreq'],
        'n_channels': len(raw.ch_names),
        'n_times': len(raw.times),
        'duration': raw.times[-1],
        
        # Channel information
        'ch_names': raw.ch_names,
        'ch_types': raw.get_channel_types(),
        
        # Wavelength information
        'wavelengths': extract_wavelengths(raw.ch_names),
        
        # Event markers
        'n_annotations': len(raw.annotations) if raw.annotations else 0,
        'annotation_descriptions': list(set(raw.annotations.description)) if raw.annotations else [],
    }
    
    # Count channels by type
    ch_types = raw.get_channel_types()
    metadata['ch_type_counts'] = {
        ch_type: ch_types.count(ch_type) 
        for ch_type in set(ch_types)
    }
    
    return metadata


def extract_wavelengths(ch_names: list) -> list:
    """
    Extract wavelengths from channel names.
    
    Parameters
    ----------
    ch_names : list
        List of channel names (e.g., ['S1_D1 690', 'S1_D1 830'])
        
    Returns
    -------
    wavelengths : list
        Sorted list of unique wavelengths (e.g., [690, 830])
    """
    wavelengths = set()
    
    for ch_name in ch_names:
        # Skip hemoglobin channels (already converted)
        if 'hbo' in ch_name.lower() or 'hbr' in ch_name.lower():
            continue
        
        # Extract wavelength from channel name (last part)
        parts = ch_name.split()
        if len(parts) > 1 and parts[-1].isdigit():
            wavelengths.add(int(parts[-1]))
    
    return sorted(list(wavelengths))


def print_metadata_summary(metadata: dict):
    """
    Print formatted metadata summary.
    
    Parameters
    ----------
    metadata : dict
        Metadata dictionary from extract_metadata()
    """
    print("\n" + "="*70)
    print("METADATA SUMMARY")
    print("="*70)
    
    print(f"\n📊 SUBJECT INFORMATION:")
    print(f"  Subject ID: {metadata['subject_id']}")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  File: {Path(metadata['file_path']).name}")
    print(f"  Size: {metadata['file_size_mb']:.2f} MB")
    
    print(f"\n⏱️  RECORDING INFORMATION:")
    print(f"  Duration: {metadata['duration']:.2f} seconds ({metadata['duration']/60:.2f} minutes)")
    print(f"  Sampling rate: {metadata['sfreq']} Hz")
    print(f"  Number of samples: {metadata['n_times']}")
    
    print(f"\n📡 CHANNEL INFORMATION:")
    print(f"  Total channels: {metadata['n_channels']}")
    for ch_type, count in metadata['ch_type_counts'].items():
        print(f"    • {ch_type}: {count}")
    
    print(f"\n🌈 WAVELENGTH INFORMATION:")
    print(f"  Wavelengths: {metadata['wavelengths']} nm")
    
    print(f"\n📍 EVENT MARKERS:")
    print(f"  Total annotations: {metadata['n_annotations']}")
    if metadata['annotation_descriptions']:
        print(f"  Event types: {metadata['annotation_descriptions']}")
    
    print("\n" + "="*70 + "\n")


def save_metadata(metadata: dict, output_dir: Path):
    """
    Save metadata to CSV file.
    
    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create flattened version for CSV (exclude lists)
    metadata_flat = {
        k: v for k, v in metadata.items() 
        if not isinstance(v, (list, dict))
    }
    
    # Add counts
    metadata_flat['wavelengths_str'] = str(metadata['wavelengths'])
    metadata_flat['n_event_types'] = len(metadata['annotation_descriptions'])
    
    # Save as CSV
    df = pd.DataFrame([metadata_flat])
    output_path = output_dir / f"{metadata['subject_id']}_metadata.csv"
    df.to_csv(output_path, index=False)
    
    print(f"💾 Metadata saved: {output_path}")


# Test function
def test_load_data():
    """
    Test data loading on subject A1 from Expert-Novice dataset.
    """
    print("="*70)
    print("TESTING DATA LOADING - Subject A1")
    print("="*70 + "\n")
    
    # Load data
    raw, metadata = load_raw_snirf('expert_novice', 'A1', preload=False, verbose=True)
    
    # Print summary
    print_metadata_summary(metadata)
    
    # Additional checks
    print("🔍 ADDITIONAL CHECKS:")
    
    # Check data shape
    print(f"  Data shape: {raw.get_data().shape}")
    
    # Check first few channel names
    print(f"\n  First 5 channels:")
    for i, ch_name in enumerate(raw.ch_names[:5]):
        print(f"    {i+1}. {ch_name}")
    
    # Check annotations
    if raw.annotations:
        print(f"\n  First 3 annotations:")
        for i in range(min(3, len(raw.annotations))):
            print(f"    {i+1}. Onset: {raw.annotations.onset[i]:.2f}s, "
                  f"Duration: {raw.annotations.duration[i]:.2f}s, "
                  f"Description: {raw.annotations.description[i]}")
    
    print("\n✅ Data loading test complete!")
    
    return raw, metadata


if __name__ == '__main__':
    # Run test
    raw, metadata = test_load_data()
