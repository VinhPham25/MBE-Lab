"""
Stage A - Step A1: Test Complete Data Loading Pipeline
Purpose: Test entire data loading pipeline on subject A1 and document any issues

Reference: PROJECT.md Section 4.2
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, get_output_path

# Import local modules
import importlib.util

def import_module(module_name, file_path):
    """Helper to import local modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import Stage A modules
stage_a_dir = Path(__file__).parent
load_data_module = import_module("load_data", stage_a_dir / "01_load_data.py")
verify_metadata_module = import_module("verify_metadata", stage_a_dir / "02_verify_metadata.py")
extract_events_module = import_module("extract_events", stage_a_dir / "03_extract_events.py")


def test_complete_pipeline(dataset_name: str, subject_id: str,
                          save_outputs: bool = True) -> dict:
    """
    Test complete data loading pipeline for a subject.
    
    Pipeline steps:
    1. Load .snirf file and extract metadata
    2. Verify metadata against expected specifications
    3. Extract and verify event markers
    4. Document any issues or warnings
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., 'expert_novice')
    subject_id : str
        Subject ID (e.g., 'A1')
    save_outputs : bool
        Whether to save outputs to files
        
    Returns
    -------
    pipeline_results : dict
        Complete pipeline results including:
        - metadata, verification, events
        - success status
        - issues and warnings
        - timestamps
    """
    print("\n" + "="*70)
    print(f"DATA LOADING PIPELINE TEST")
    print(f"Subject: {subject_id} | Dataset: {dataset_name}")
    print("="*70 + "\n")
    
    pipeline_results = {
        'subject_id': subject_id,
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'issues': [],
        'warnings': [],
        'metadata': None,
        'verification': None,
        'events': None
    }
    
    # Step 1: Load data and extract metadata
    print("STEP 1: Loading .snirf file and extracting metadata")
    print("-" * 70)
    try:
        raw, metadata = load_data_module.load_raw_snirf(
            dataset_name, subject_id, preload=False, verbose=True
        )
        load_data_module.print_metadata_summary(metadata)
        pipeline_results['metadata'] = metadata
        print("✅ Step 1 Complete\n")
    except Exception as e:
        print(f"❌ Step 1 Failed: {e}\n")
        pipeline_results['issues'].append(f"Data loading failed: {str(e)}")
        return pipeline_results
    
    # Step 2: Verify metadata
    print("STEP 2: Verifying metadata against expected specifications")
    print("-" * 70)
    try:
        verification = verify_metadata_module.verify_metadata(metadata, dataset_name)
        verify_metadata_module.print_verification_report(verification, metadata)
        pipeline_results['verification'] = verification
        
        # Collect warnings and errors
        if verification['warnings']:
            pipeline_results['warnings'].extend(verification['warnings'])
        if verification['errors']:
            pipeline_results['issues'].extend(verification['errors'])
        
        print("✅ Step 2 Complete\n")
    except Exception as e:
        print(f"❌ Step 2 Failed: {e}\n")
        pipeline_results['issues'].append(f"Metadata verification failed: {str(e)}")
        return pipeline_results
    
    # Step 3: Extract and verify events
    print("STEP 3: Extracting and verifying event markers")
    print("-" * 70)
    try:
        events_df = extract_events_module.extract_events(raw, verbose=True)
        events_verification = extract_events_module.verify_events(
            events_df, expected_trials=5, verbose=True
        )
        extract_events_module.print_events_summary(events_df, events_verification)
        
        pipeline_results['events'] = {
            'n_events': len(events_df),
            'n_start_events': events_verification['n_start_events'],
            'n_end_events': events_verification['n_end_events'],
            'all_valid': events_verification['all_valid']
        }
        
        # Collect warnings and errors
        if events_verification['warnings']:
            pipeline_results['warnings'].extend(events_verification['warnings'])
        if events_verification['errors']:
            pipeline_results['issues'].extend(events_verification['errors'])
        
        print("✅ Step 3 Complete\n")
    except Exception as e:
        print(f"❌ Step 3 Failed: {e}\n")
        pipeline_results['issues'].append(f"Event extraction failed: {str(e)}")
        return pipeline_results
    
    # Determine overall success
    pipeline_results['success'] = (
        len(pipeline_results['issues']) == 0 and
        verification['all_passed'] and
        events_verification['all_valid']
    )
    
    # Save outputs if requested
    if save_outputs:
        save_pipeline_outputs(pipeline_results, metadata, events_df)
    
    # Print final summary
    print_pipeline_summary(pipeline_results)
    
    return pipeline_results


def save_pipeline_outputs(pipeline_results: dict, metadata: dict, events_df):
    """
    Save pipeline outputs to files.
    
    Parameters
    ----------
    pipeline_results : dict
        Pipeline results
    metadata : dict
        Metadata dictionary
    events_df : pd.DataFrame
        Events DataFrame
    """
    output_dir = get_output_path('stage_a', 'qa_reports')
    subject_id = pipeline_results['subject_id']
    
    # Save metadata
    load_data_module.save_metadata(metadata, output_dir)
    
    # Save events
    extract_events_module.save_events(events_df, subject_id, output_dir)
    
    # Save pipeline summary as JSON
    summary = {
        'subject_id': pipeline_results['subject_id'],
        'dataset_name': pipeline_results['dataset_name'],
        'timestamp': pipeline_results['timestamp'],
        'success': pipeline_results['success'],
        'n_issues': len(pipeline_results['issues']),
        'n_warnings': len(pipeline_results['warnings']),
        'issues': pipeline_results['issues'],
        'warnings': pipeline_results['warnings'],
        'metadata_summary': {
            'sfreq': metadata['sfreq'],
            'n_channels': metadata['n_channels'],
            'duration': metadata['duration'],
            'wavelengths': metadata['wavelengths'],
            'n_annotations': metadata['n_annotations']
        },
        'events_summary': pipeline_results['events']
    }
    
    summary_path = output_dir / f"{subject_id}_pipeline_test.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Pipeline summary saved: {summary_path}")


def print_pipeline_summary(pipeline_results: dict):
    """
    Print formatted pipeline summary.
    
    Parameters
    ----------
    pipeline_results : dict
        Pipeline results
    """
    print("\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    
    print(f"\n📊 SUBJECT: {pipeline_results['subject_id']}")
    print(f"📁 DATASET: {pipeline_results['dataset_name']}")
    print(f"🕐 TIMESTAMP: {pipeline_results['timestamp']}")
    
    print(f"\n✅ SUCCESS: {pipeline_results['success']}")
    print(f"⚠️  WARNINGS: {len(pipeline_results['warnings'])}")
    print(f"❌ ISSUES: {len(pipeline_results['issues'])}")
    
    if pipeline_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in pipeline_results['warnings']:
            print(f"  • {warning}")
    
    if pipeline_results['issues']:
        print("\n❌ ISSUES:")
        for issue in pipeline_results['issues']:
            print(f"  • {issue}")
    
    if pipeline_results['success']:
        print("\n" + "="*70)
        print("✅ ALL PIPELINE TESTS PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  PIPELINE TESTS COMPLETED WITH ISSUES")
        print("="*70)
    
    print()


def test_subject_A1():
    """
    Test complete pipeline on subject A1 from Expert-Novice dataset.
    """
    results = test_complete_pipeline('expert_novice', 'A1', save_outputs=True)
    return results


if __name__ == '__main__':
    # Run test on subject A1
    results = test_subject_A1()
