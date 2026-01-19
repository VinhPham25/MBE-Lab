"""
Stage A - Step A1: Verify Metadata Against Expected Hardware Specs
Purpose: Validate that loaded data matches expected hardware specifications

Reference: PROJECT.md Section 4.2, Section 1.2 (Hardware Specifications)
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_dataset_config

# Import from local module (will be renamed)
import importlib.util
spec = importlib.util.spec_from_file_location("load_data", Path(__file__).parent / "01_load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)
load_raw_snirf = load_data.load_raw_snirf
extract_metadata = load_data.extract_metadata


def verify_metadata(metadata: dict, dataset_name: str, 
                    tolerance: float = 0.1) -> dict:
    """
    Verify metadata against expected hardware specifications.
    
    Parameters
    ----------
    metadata : dict
        Metadata extracted from raw data
    dataset_name : str
        Dataset name for looking up expected specs
    tolerance : float
        Tolerance for numerical comparisons (e.g., 0.1 Hz for sampling rate)
        
    Returns
    -------
    verification : dict
        Dictionary containing:
        - checks: dict of individual check results
        - all_passed: bool indicating if all checks passed
        - warnings: list of warning messages
        - errors: list of error messages
    """
    config = load_config()
    expected = get_dataset_config(dataset_name, config)
    
    verification = {
        'checks': {},
        'warnings': [],
        'errors': [],
        'all_passed': True
    }
    
    # Check 1: Sampling rate
    check_sampling_rate(metadata, expected, verification, tolerance)
    
    # Check 2: Wavelengths
    check_wavelengths(metadata, expected, verification)
    
    # Check 3: Channel count
    check_channel_count(metadata, expected, verification)
    
    # Check 4: Event markers
    check_event_markers(metadata, verification)
    
    # Update all_passed flag
    verification['all_passed'] = len(verification['errors']) == 0
    
    return verification


def check_sampling_rate(metadata: dict, expected: dict, 
                       verification: dict, tolerance: float):
    """Check if sampling rate matches expected value."""
    actual_sfreq = metadata['sfreq']
    expected_sfreq = expected['sfreq']
    
    diff = abs(actual_sfreq - expected_sfreq)
    
    if diff <= tolerance:
        verification['checks']['sampling_rate'] = {
            'status': 'PASS',
            'actual': actual_sfreq,
            'expected': expected_sfreq,
            'message': f"✓ Sampling rate: {actual_sfreq} Hz (expected: {expected_sfreq} Hz)"
        }
    elif diff <= 5.0:  # Allow up to 5 Hz difference with warning
        verification['checks']['sampling_rate'] = {
            'status': 'WARN',
            'actual': actual_sfreq,
            'expected': expected_sfreq,
            'message': f"⚠️  Sampling rate: {actual_sfreq} Hz (expected: {expected_sfreq} Hz, diff: {diff} Hz)"
        }
        verification['warnings'].append(
            f"Sampling rate differs from expected: {actual_sfreq} vs {expected_sfreq} Hz"
        )
    else:
        verification['checks']['sampling_rate'] = {
            'status': 'FAIL',
            'actual': actual_sfreq,
            'expected': expected_sfreq,
            'message': f"✗ Sampling rate: {actual_sfreq} Hz (expected: {expected_sfreq} Hz, diff: {diff} Hz)"
        }
        verification['errors'].append(
            f"Sampling rate mismatch: {actual_sfreq} vs {expected_sfreq} Hz"
        )
        verification['all_passed'] = False


def check_wavelengths(metadata: dict, expected: dict, verification: dict):
    """Check if wavelengths match expected values."""
    actual_wl = set(metadata['wavelengths'])
    expected_wl = set(expected['wavelengths'])
    
    if actual_wl == expected_wl:
        verification['checks']['wavelengths'] = {
            'status': 'PASS',
            'actual': sorted(list(actual_wl)),
            'expected': sorted(list(expected_wl)),
            'message': f"✓ Wavelengths: {sorted(list(actual_wl))} nm"
        }
    else:
        verification['checks']['wavelengths'] = {
            'status': 'FAIL',
            'actual': sorted(list(actual_wl)),
            'expected': sorted(list(expected_wl)),
            'message': f"✗ Wavelengths: {sorted(list(actual_wl))} nm (expected: {sorted(list(expected_wl))} nm)"
        }
        verification['errors'].append(
            f"Wavelength mismatch: {sorted(list(actual_wl))} vs {sorted(list(expected_wl))} nm"
        )
        verification['all_passed'] = False


def check_channel_count(metadata: dict, expected: dict, verification: dict):
    """Check if channel count matches expected value."""
    # Calculate actual channels per wavelength
    # Total channels = (n_long + n_short) * n_wavelengths
    n_wavelengths = len(metadata['wavelengths'])
    actual_ch_per_wl = metadata['n_channels'] // n_wavelengths if n_wavelengths > 0 else 0
    
    expected_total = expected['n_long_channels'] + expected['n_short_channels']
    
    if actual_ch_per_wl == expected_total:
        verification['checks']['channel_count'] = {
            'status': 'PASS',
            'actual': actual_ch_per_wl,
            'expected': expected_total,
            'message': f"✓ Channels: {actual_ch_per_wl} per wavelength "
                      f"({expected['n_long_channels']} long + {expected['n_short_channels']} short)"
        }
    elif abs(actual_ch_per_wl - expected_total) <= 1:
        # Allow 1 channel difference with warning (e.g., Learning Curve has 7 short instead of 8)
        verification['checks']['channel_count'] = {
            'status': 'WARN',
            'actual': actual_ch_per_wl,
            'expected': expected_total,
            'message': f"⚠️  Channels: {actual_ch_per_wl} per wavelength "
                      f"(expected: {expected_total}, diff: {abs(actual_ch_per_wl - expected_total)})"
        }
        verification['warnings'].append(
            f"Channel count slightly differs: {actual_ch_per_wl} vs {expected_total} (may be hardware issue)"
        )
    else:
        verification['checks']['channel_count'] = {
            'status': 'FAIL',
            'actual': actual_ch_per_wl,
            'expected': expected_total,
            'message': f"✗ Channels: {actual_ch_per_wl} per wavelength (expected: {expected_total})"
        }
        verification['errors'].append(
            f"Channel count mismatch: {actual_ch_per_wl} vs {expected_total}"
        )
        verification['all_passed'] = False


def check_event_markers(metadata: dict, verification: dict):
    """Check if event markers are present and valid."""
    n_annotations = metadata['n_annotations']
    event_types = metadata['annotation_descriptions']
    
    if n_annotations == 0:
        verification['checks']['event_markers'] = {
            'status': 'FAIL',
            'actual': n_annotations,
            'message': "✗ No event markers found"
        }
        verification['errors'].append("No event markers found in data")
        verification['all_passed'] = False
    elif n_annotations > 0:
        # Check if we have expected event types (1=start, 2=end)
        expected_events = {'1', '2', 'np.str_(\'1\')', 'np.str_(\'2\')'}  # Allow numpy string format
        event_types_str = set(str(e) for e in event_types)
        
        has_start = any('1' in str(e) for e in event_types)
        has_end = any('2' in str(e) for e in event_types)
        
        if has_start and has_end:
            verification['checks']['event_markers'] = {
                'status': 'PASS',
                'actual': n_annotations,
                'event_types': event_types,
                'message': f"✓ Event markers: {n_annotations} annotations, types: {event_types}"
            }
        else:
            verification['checks']['event_markers'] = {
                'status': 'WARN',
                'actual': n_annotations,
                'event_types': event_types,
                'message': f"⚠️  Event markers: {n_annotations} annotations, but unexpected types: {event_types}"
            }
            verification['warnings'].append(
                f"Unexpected event types: {event_types} (expected: condition 1=start, 2=end)"
            )


def print_verification_report(verification: dict, metadata: dict):
    """
    Print formatted verification report.
    
    Parameters
    ----------
    verification : dict
        Verification results
    metadata : dict
        Original metadata
    """
    print("\n" + "="*70)
    print("METADATA VERIFICATION REPORT")
    print("="*70)
    
    print(f"\nSubject: {metadata['subject_id']}")
    print(f"Dataset: {metadata['dataset_name']}")
    
    print("\n📋 VERIFICATION CHECKS:")
    for check_name, check_result in verification['checks'].items():
        print(f"  {check_result['message']}")
    
    if verification['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in verification['warnings']:
            print(f"  • {warning}")
    
    if verification['errors']:
        print("\n❌ ERRORS:")
        for error in verification['errors']:
            print(f"  • {error}")
    
    print("\n" + "="*70)
    
    if verification['all_passed']:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    
    print("="*70 + "\n")


def save_verification_report(verification: dict, metadata: dict, output_dir: Path):
    """
    Save verification report to CSV.
    
    Parameters
    ----------
    verification : dict
        Verification results
    metadata : dict
        Original metadata
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create flattened report
    report = {
        'subject_id': metadata['subject_id'],
        'dataset_name': metadata['dataset_name'],
        'all_passed': verification['all_passed'],
        'n_warnings': len(verification['warnings']),
        'n_errors': len(verification['errors']),
    }
    
    # Add individual check results
    for check_name, check_result in verification['checks'].items():
        report[f'{check_name}_status'] = check_result['status']
        if 'actual' in check_result:
            report[f'{check_name}_actual'] = check_result['actual']
        if 'expected' in check_result:
            report[f'{check_name}_expected'] = check_result['expected']
    
    # Save to CSV
    df = pd.DataFrame([report])
    output_path = output_dir / f"{metadata['subject_id']}_verification.csv"
    df.to_csv(output_path, index=False)
    
    print(f"💾 Verification report saved: {output_path}")


# Test function
def test_verify_metadata():
    """
    Test metadata verification on subject A1.
    """
    print("="*70)
    print("TESTING METADATA VERIFICATION - Subject A1")
    print("="*70 + "\n")
    
    # Load data using the imported function
    raw, metadata = load_raw_snirf('expert_novice', 'A1', preload=False, verbose=True)
    
    # Verify metadata
    verification = verify_metadata(metadata, 'expert_novice')
    
    # Print report
    print_verification_report(verification, metadata)
    
    return verification


if __name__ == '__main__':
    # Run test
    verification = test_verify_metadata()
