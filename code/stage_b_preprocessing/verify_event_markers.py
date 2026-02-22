"""
Stage B — 3.6+ : Manually verify Stage B output format
Purpose: Ensures the output from Stage B maintains the conditional event markers (1 = start, 2 = end) and 
        rest periods and task periods are still mixed (not segmented)

Task coverage:
- 3.6.9 - Verify event markers are preserved in processed data; stored in .fif file under annotations
- 3.6.10 - Check that rest periods and task periods are still mixed (not segmented)
"""

import sys
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

output_dir = get_output_path('stage_b', f'ols_regression/{"A1"}')

raw_processed = mne.io.read_raw_fif(output_dir / 'A1_ols_haemo_raw.fif')

events_df = raw_processed.annotations.to_data_frame()
print(events_df.head())