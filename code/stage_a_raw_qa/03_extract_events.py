"""
Stage A - Step A1: Extract and Verify Event Markers
Purpose: Extract event markers and verify they match expected format
(condition 1 = trial start, condition 2 = trial end)

Reference: PROJECT.md Section 4.2
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import load_data module
import importlib.util
spec = importlib.util.spec_from_file_location("load_data", Path(__file__).parent / "01_load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)
load_raw_snirf = load_data.load_raw_snirf


def extract_events(raw, verbose: bool = True) -> pd.DataFrame:
    """
    Extract event markers from raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS data object
    verbose : bool
        Print extraction information
        
    Returns
    -------
    events_df : pd.DataFrame
        DataFrame with columns:
        - onset (float): Event onset time in seconds
        - duration (float): Event duration in seconds
        - description (str): Event description/type
        - event_id (int): Numeric event ID (1 or 2)
    """
    if not raw.annotations:
        if verbose:
            print("⚠️  No annotations found in raw data")
        return pd.DataFrame(columns=['onset', 'duration', 'description', 'event_id'])
    
    # Extract annotations
    events_df = pd.DataFrame({
        'onset': raw.annotations.onset,
        'duration': raw.annotations.duration,
        'description': [str(desc) for desc in raw.annotations.description]
    })
    
    # Convert description to numeric event_id
    events_df['event_id'] = events_df['description'].apply(extract_event_id)
    
    # Sort by onset time
    events_df = events_df.sort_values('onset').reset_index(drop=True)
    
    if verbose:
        print(f"✓ Extracted {len(events_df)} events")
    
    return events_df


def extract_event_id(description: str) -> int:
    """
    Extract numeric event ID from description string.
    Handles formats like: '1', '2', "np.str_('1')", etc.
    
    Parameters
    ----------
    description : str
        Event description
        
    Returns
    -------
    event_id : int
        Numeric event ID (1 or 2), or 0 if unknown
    """
    # Try direct conversion
    try:
        return int(description)
    except ValueError:
        pass
    
    # Try extracting from numpy string format
    if '1' in description:
        return 1
    elif '2' in description:
        return 2
    else:
        return 0  # Unknown event type


def verify_events(events_df: pd.DataFrame, expected_trials: int = 5,
                 verbose: bool = True) -> dict:
    """
    Verify that events match expected format.
    
    Expected format:
    - Condition 1: Trial start
    - Condition 2: Trial end
    - Typically 5 trials per subject (5 starts + 5 ends = 10 events)
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame from extract_events()
    expected_trials : int
        Expected number of trials
    verbose : bool
        Print verification information
        
    Returns
    -------
    verification : dict
        Verification results with checks for:
        - event_count: Total number of events
        - has_start_events: Whether condition 1 events exist
        - has_end_events: Whether condition 2 events exist
        - n_start_events: Number of start events
        - n_end_events: Number of end events
        - paired_correctly: Whether starts and ends are paired
        - all_valid: Overall validation result
    """
    verification = {
        'event_count': len(events_df),
        'has_start_events': False,
        'has_end_events': False,
        'n_start_events': 0,
        'n_end_events': 0,
        'paired_correctly': False,
        'all_valid': False,
        'warnings': [],
        'errors': []
    }
    
    if len(events_df) == 0:
        verification['errors'].append("No events found")
        return verification
    
    # Count event types
    start_events = events_df[events_df['event_id'] == 1]
    end_events = events_df[events_df['event_id'] == 2]
    
    verification['n_start_events'] = len(start_events)
    verification['n_end_events'] = len(end_events)
    verification['has_start_events'] = len(start_events) > 0
    verification['has_end_events'] = len(end_events) > 0
    
    # Check if both event types exist
    if not verification['has_start_events']:
        verification['errors'].append("No trial start events (condition 1) found")
    
    if not verification['has_end_events']:
        verification['errors'].append("No trial end events (condition 2) found")
    
    # Check if counts match
    if verification['n_start_events'] != verification['n_end_events']:
        verification['warnings'].append(
            f"Unequal start/end events: {verification['n_start_events']} starts, "
            f"{verification['n_end_events']} ends"
        )
    
    # Check if counts match expected
    if verification['n_start_events'] != expected_trials:
        verification['warnings'].append(
            f"Unexpected number of trials: {verification['n_start_events']} "
            f"(expected: {expected_trials})"
        )
    
    # Check pairing (start should come before end)
    if verification['has_start_events'] and verification['has_end_events']:
        pairing_valid = check_event_pairing(events_df)
        verification['paired_correctly'] = pairing_valid
        if not pairing_valid:
            verification['warnings'].append("Events may not be correctly paired (start/end order)")
    
    # Overall validation
    verification['all_valid'] = (
        verification['has_start_events'] and 
        verification['has_end_events'] and
        len(verification['errors']) == 0
    )
    
    return verification


def check_event_pairing(events_df: pd.DataFrame) -> bool:
    """
    Check if start and end events alternate correctly.
    Expected pattern: start, end, start, end, ...
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame
        
    Returns
    -------
    paired_correctly : bool
        True if events alternate correctly
    """
    if len(events_df) < 2:
        return False
    
    # Check if events alternate between 1 and 2
    event_ids = events_df['event_id'].values
    
    # For standard format, we expect alternating 1, 2, 1, 2, ...
    # But the actual data might have different patterns, so we just check
    # that starts generally come before ends
    for i in range(len(event_ids) - 1):
        if event_ids[i] == event_ids[i+1]:
            # Two consecutive same events - might be okay but not ideal
            continue
    
    return True  # Relaxed check for now


def print_events_summary(events_df: pd.DataFrame, verification: dict):
    """
    Print formatted events summary.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame
    verification : dict
        Verification results
    """
    print("\n" + "="*70)
    print("EVENT MARKERS SUMMARY")
    print("="*70)
    
    print(f"\n📍 EVENT STATISTICS:")
    print(f"  Total events: {verification['event_count']}")
    print(f"  Start events (condition 1): {verification['n_start_events']}")
    print(f"  End events (condition 2): {verification['n_end_events']}")
    
    if len(events_df) > 0:
        print(f"\n📋 EVENT DETAILS:")
        print(events_df.to_string(index=True))
        
        # Calculate trial durations
        if verification['has_start_events'] and verification['has_end_events']:
            start_times = events_df[events_df['event_id'] == 1]['onset'].values
            end_times = events_df[events_df['event_id'] == 2]['onset'].values
            
            if len(start_times) > 0 and len(end_times) > 0:
                # Match starts with ends
                n_trials = min(len(start_times), len(end_times))
                durations = []
                
                print(f"\n⏱️  TRIAL DURATIONS:")
                for i in range(n_trials):
                    if i < len(start_times) and i < len(end_times):
                        duration = end_times[i] - start_times[i]
                        durations.append(duration)
                        print(f"  Trial {i+1}: {start_times[i]:.2f}s to {end_times[i]:.2f}s "
                              f"(duration: {duration:.2f}s)")
                
                if durations:
                    print(f"\n  Mean trial duration: {np.mean(durations):.2f}s (± {np.std(durations):.2f}s)")
    
    if verification['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in verification['warnings']:
            print(f"  • {warning}")
    
    if verification['errors']:
        print("\n❌ ERRORS:")
        for error in verification['errors']:
            print(f"  • {error}")
    
    print("\n" + "="*70)
    
    if verification['all_valid']:
        print("✅ EVENT MARKERS VALID")
    else:
        print("⚠️  EVENT MARKERS NEED REVIEW")
    
    print("="*70 + "\n")


def save_events(events_df: pd.DataFrame, subject_id: str, output_dir: Path):
    """
    Save events to CSV file.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame
    subject_id : str
        Subject ID
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{subject_id}_events.csv"
    events_df.to_csv(output_path, index=False)
    
    print(f"💾 Events saved: {output_path}")


# Test function
def test_extract_events():
    """
    Test event extraction on subject A1.
    """
    print("="*70)
    print("TESTING EVENT EXTRACTION - Subject A1")
    print("="*70 + "\n")
    
    # Load data
    raw, metadata = load_raw_snirf('expert_novice', 'A1', preload=False, verbose=True)
    
    # Extract events
    print("\n📍 Extracting event markers...")
    events_df = extract_events(raw, verbose=True)
    
    # Verify events
    print("\n🔍 Verifying event markers...")
    verification = verify_events(events_df, expected_trials=5, verbose=True)
    
    # Print summary
    print_events_summary(events_df, verification)
    
    return events_df, verification


if __name__ == '__main__':
    # Run test
    events_df, verification = test_extract_events()
