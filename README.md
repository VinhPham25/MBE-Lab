# ASTaUND Phase 1: fNIRS Preprocessing Pipeline

**Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND)**  
Reproduction of Kamat et al. (2023) preprocessing pipeline in Python

---

## Overview

This project implements the exact preprocessing and technical validation pipeline from the ASTaUND paper using MNE-Python and mne-nirs. The pipeline processes fNIRS data from four surgical training studies to extract hemodynamic responses.

**Reference:** Kamat, A., Eastmond, C., Gao, Y. et al. (2023). Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND). *Sci Data* 10, 699. https://doi.org/10.1038/s41597-023-02603-3

---

## Quick Start

### 1. Installation

```bash
# Clone repository
cd /path/to/MBE-Lab

# Install dependencies (Python 3.9+)
pip install -r requirements.txt
```

### 2. Verify Setup

```python
# Test imports
import mne
import mne_nirs
import numpy as np
import pandas as pd

print(f"MNE version: {mne.__version__}")
print(f"MNE-NIRS version: {mne_nirs.__version__}")
```

### 3. Configuration

All parameters are defined in `config.yaml`:
- Dataset specifications (sampling rates, wavelengths, PPF values)
- Stage A: Quality assessment thresholds (SNR 80-140 dB, motion detection)
- Stage B: Preprocessing parameters (filter 0.01-0.5 Hz, rPCA, Beer-Lambert)
- Stage C: Validation parameters (epochs, HRF windows, CNR calculations)

---

## Pipeline Structure

### Three-Stage Workflow

```
Stage A: Raw Quality Assessment → Stage B: Preprocessing → Stage C: Validation
```

#### Stage A - Raw Quality Assessment
- Calculate SNR on raw intensity (80-140 dB threshold)
- Detect motion artifacts (tMotion=0.5, tStdev=20)
- Generate exclusion lists for bad channels
- Output: QA reports, channel exclusions

#### Stage B - Preprocessing (6 Steps)
1. Intensity → Optical Density conversion
2. Bandpass filtering (0.01-0.5 Hz)
3. Motion artifact detection
4. rPCA motion correction
5. Beer-Lambert Law (OD → HbO₂/HbR)
6. Short-channel regression (OLS)

#### Stage C - Post-Processing Validation
- Extract hemodynamic response functions (HRF)
- Calculate contrast-to-noise ratio (CNR)
- Statistical validation (t-tests, Cohen's d)
- Compare to paper's Figure 3 and 4

---

## Project Structure

```
MBE-Lab/
├── data/
│   └── BIDS/                    # BIDS-formatted fNIRS data
│       ├── Expert-Novice/       # 14 subjects, TechEn CW6, 25 Hz
│       ├── Learning_Curve/      # 13 subjects, TechEn CW6, 25 Hz
│       ├── Short-Term_tES/      # 12 subjects, NIRx NIRScout, 7.8125 Hz
│       └── Long-Term_tES/       # 21 subjects, NIRx NIRScout, 7.8125 Hz
│
├── code/
│   ├── stage_a_raw_qa/          # Quality assessment scripts
│   ├── stage_b_preprocessing/   # Preprocessing pipeline scripts
│   └── stage_c_validation/      # Validation and analysis scripts
│
├── outputs/
│   ├── stage_a_qa/              # QA reports, figures, exclusion lists
│   ├── stage_b_processed/       # Processed .fif files
│   └── stage_c_validation/      # HRF plots, CNR metrics, statistics
│
├── config.yaml                  # Pipeline parameters
├── requirements.txt             # Python dependencies
├── PROJECT.md                   # Detailed PRD
└── PROGRESS.md                  # Implementation tracking
```

---

## Datasets

### Expert-Novice Study
- **Subjects:** 14 (8 experts, 6 novices)
- **System:** TechEn CW6
- **Sampling:** 25.0 Hz
- **Channels:** 25 long + 8 short
- **Wavelengths:** 690 nm, 830 nm
- **PPF:** 6.4 (690nm), 5.8 (830nm)

### Learning Curve Study
- **Subjects:** 13 (8 trainees, 5 controls)
- **System:** TechEn CW6
- **Sampling:** 25.0 Hz
- **Channels:** 25 long + 7 short (hardware issue)
- **Sessions:** 12 training days + retention

### Short-Term tES Study
- **Subjects:** 12
- **System:** NIRx NIRScout
- **Sampling:** 7.8125 Hz
- **Channels:** 28 long + 8 short
- **Wavelengths:** 760 nm, 850 nm
- **Design:** tRNS vs Sham (counterbalanced)

### Long-Term tES Study
- **Subjects:** 21 (7 tDCS, 7 tRNS, 7 Sham)
- **System:** NIRx NIRScout
- **Sampling:** 7.8125 Hz
- **Duration:** 12 training days + 4-week follow-up

---

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| SNR Threshold | 80-140 dB | Channel quality (Stage A) |
| Motion tMotion | 0.5 | Amplitude threshold |
| Motion tStdev | 20 | Standard deviation multiplier |
| Bandpass Filter | 0.01-0.5 Hz | Remove drift and noise |
| rPCA Variance | 0.97 | Motion correction | Note, adjusted from 0.97 to 0.99 for rPCA after hyperparameter tuning
| Epoch Window | -25 to +100 s | HRF extraction |
| HRF Peak Window | 5-15 s | CNR calculation |

---

## Development Status

- [x] **Section 1.1:** Development environment setup (3/3 tasks)
- [x] **Section 1.2:** Data access verification (3/3 tasks)
- [ ] **Section 1.3:** Utility functions (1 task)
- [ ] **Section 2:** Stage A implementation (22 tasks)
- [ ] **Section 3:** Stage B implementation (36 tasks)
- [ ] **Section 4:** Stage C implementation (20 tasks)
- [ ] **Section 5:** Multi-dataset extension (7 tasks)

**Total Progress:** 6/92 tasks (7%)

See `PROGRESS.md` for detailed task tracking.

---

## Dependencies

### Core Libraries
- **MNE:** 1.11.0 - fNIRS data processing
- **MNE-NIRS:** 0.7.3 - fNIRS-specific preprocessing
- **NumPy:** 2.4.1 - Numerical computing
- **Pandas:** 2.2.2 - Data manipulation
- **SciPy:** Latest - Signal processing
- **Scikit-learn:** Latest - PCA for motion correction

### Visualization
- **Matplotlib:** Latest - Plotting
- **Seaborn:** Latest - Statistical visualization

### Utilities
- **PyYAML:** Latest - Config management
- **h5py:** Latest - HDF5 support for .snirf files

---

## Usage Examples

### Load and Inspect Data
```python
import mne
import yaml

# Load config
config = yaml.safe_load(open('config.yaml'))

# Load .snirf file
raw = mne.io.read_raw_snirf('data/BIDS/Expert-Novice/sub-A1/nirs/sub-A1_task-ExpNovPatternCutting_nirs.snirf')

print(f"Duration: {raw.times[-1]:.1f} seconds")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Channels: {len(raw.ch_names)}")
```

### Run Pipeline (Coming Soon)
```bash
# Stage A: Quality assessment
python code/stage_a_raw_qa/run_qa.py --subject A1 --dataset Expert_Novice

# Stage B: Preprocessing
python code/stage_b_preprocessing/run_pipeline.py --subject A1

# Stage C: Validation
python code/stage_c_validation/run_validation.py --dataset Expert_Novice
```

---

## Important Notes

### System Differences
- **TechEn CW6:** Custom PPF values (6.4, 5.8), 25.0 Hz sampling
- **NIRx NIRScout:** Default PPF values, 7.8125 Hz sampling
- Code must handle both systems appropriately

### Known Issues
- **Learning Curve:** Only 7 short channels (hardware failure)
- **Subject R9:** Split recordings with missing Trial 3
- **Sampling Rate:** Actual data is 25 Hz, not 20 Hz as in paper

### Validation Targets
- SNR plots match Figure 4a,c,e,g from paper
- Motion ratio matches Figure 4b,d,f,h
- HRF morphology matches Figure 3c,3d
- CNR values match Figure 4i

---

## Contact

**Project:** MBE Lab - ASTaUND Phase 1  
**Author:** Vinh Pham  
**Started:** January 18, 2026

For questions or issues, see `PROJECT.md` for detailed specifications.
