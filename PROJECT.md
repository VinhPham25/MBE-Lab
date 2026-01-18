# ASTaUND Phase 1: Data Processing Pipeline PRD

**Project:** Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND)  
**Objective:** Reproduce the exact preprocessing and technical validation pipeline from Kamat et al. (2023) using Python and mne-nirs  
**Target:** BIDS-compliant .snirf format fNIRS data from Figshare  
**Timeline:** Phase 1 (Reproduction and Validation)

---

## 1. PROJECT OVERVIEW

### 1.1 Background
The ASTaUND dataset contains fNIRS recordings from four studies:
- **Expert/Novice Study** (n=14): Single session, 5 trials per subject
- **Learning Curve Study** (n=13): 12 consecutive training days + retention test
- **Short-Term tES Study** (n=12): Single day, tRNS vs sham stimulation
- **Long-Term tES Study** (n=21): 12 training days + 4-week follow-up

### 1.2 Hardware Specifications

#### TechEn CW6 System (Expert/Novice & Learning Curve)
- **Long-Separation Channels:** 25
- **Short-Separation Channels:** 8 (7 in Learning Curve due to hardware issues)
- **Sampling Rate:** 20 Hz
- **Wavelengths:** 690 nm, 830 nm
- **Coverage:** Prefrontal Cortex (PFC), Primary Motor Cortex (M1), Supplementary Motor Area (SMA)

#### NIRx NIRScout System (Short/Long-Term tES)
- **Long-Separation Channels:** 28
- **Short-Separation Channels:** 8
- **Sampling Rate:** 7.8125 Hz
- **Wavelengths:** 760 nm, 850 nm
- **Coverage:** PFC, M1, SMA

### 1.3 Key Concepts
- **Long-Separation Channels (30-40mm):** Capture cortical brain activity
- **Short-Separation Channels (~8mm):** Capture superficial scalp noise for regression
- **Modified Beer-Lambert Law:** Converts light intensity to hemoglobin concentrations
- **Partial Path-length Factors (PPF):** 6.4 (690nm), 5.8 (830nm) for TechEn system

---

## 2. ENVIRONMENT SETUP

### 2.1 Required Libraries
```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# fNIRS-specific
import mne
import mne_nirs
from mne_nirs.channels import get_long_channels, get_short_channels

# File I/O
from pathlib import Path
import json

# Statistical/Signal Processing
from scipy import signal, stats
from sklearn.decomposition import PCA
```

### 2.2 Directory Structure
```
project/
├── data/
│   ├── BIDS/
│   │   ├── Expert_Novice/
│   │   ├── Learning_Curve/
│   │   ├── Short_Term_tES/
│   │   └── Long_Term_tES/
│   └── processed/
├── code/
│   ├── stage_a_raw_qa/
│   │   ├── 01_load_data.py
│   │   ├── 02_calculate_snr_raw.py
│   │   ├── 03_calculate_motion_ratio_raw.py
│   │   └── 04_generate_qa_report.py
│   ├── stage_b_preprocessing/
│   │   ├── 01_intensity_to_od.py
│   │   ├── 02_bandpass_filter.py
│   │   ├── 03_motion_detection.py
│   │   ├── 04_rpca_correction.py
│   │   ├── 05_beer_lambert.py
│   │   └── 06_short_channel_regression.py
│   └── stage_c_validation/
│       ├── 01_extract_hrf.py
│       ├── 02_calculate_cnr.py
│       └── 03_final_validation.py
├── outputs/
│   ├── stage_a_qa/
│   │   ├── figures/
│   │   ├── excluded_channels/
│   │   └── qa_reports/
│   ├── stage_b_processed/
│   │   └── data/
│   └── stage_c_validation/
│       ├── figures/
│       └── metrics/
└── README.md
```

---

## 3. THREE-STAGE PROCESSING WORKFLOW

### 3.1 Workflow Overview

```
┌─────────────────────────────────────────────────────┐
│ STAGE A: RAW QUALITY ASSESSMENT (Gating)           │
│ Purpose: Verify data collection quality            │
│ Input: Raw .snirf intensity data                   │
│ Output: Channel exclusion lists, QA reports        │
└─────────────────────────────────────────────────────┘
                         ↓
          ┌──────────────────────────┐
          │ DECISION POINT:          │
          │ Proceed with good        │
          │ channels only            │
          └──────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│ STAGE B: PREPROCESSING PIPELINE (6 Steps)          │
│ Purpose: Clean and process neural signals          │
│ Input: Raw data (good channels only)               │
│ Output: HbO₂ and HbR concentration time series     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│ STAGE C: POST-PROCESSING VALIDATION (Success)      │
│ Purpose: Validate preprocessing effectiveness      │
│ Input: Processed hemoglobin data                   │
│ Output: CNR, HRF plots, statistical comparisons    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Key Insight from Paper

**Important Quote from Technical Validation:**
> "To give an accurate assessment of the quality of the data, no channels have been excluded from the analysis provided here and no denoising methods have been used prior to analysis."

**Interpretation:**
- The paper calculated SNR and Motion Ratio on **raw intensity data** for Figure 4
- This was for **transparency and documentation** purposes
- For actual scientific analysis, they applied the 80-140 dB threshold to exclude bad channels
- **We replicate this approach:** Calculate metrics on raw data, but exclude bad channels before preprocessing

---

## 4. STAGE A: RAW QUALITY ASSESSMENT

### 4.1 Stage A Overview

**Objectives:**
1. Verify hardware functionality - Did the fNIRS device collect data properly?
2. Identify dead/saturated channels - Which channels should be excluded?
3. Document data quality - Create transparency reports like paper's Figure 4
4. Create exclusion lists - Gate the preprocessing pipeline

**Critical Requirement:** All metrics calculated on **unprocessed raw intensity data** before any filtering or denoising.

---

### 4.2 Step A1: Load Raw Data and Extract Metadata

**Objective:** Load .snirf files without any preprocessing and verify structure

**Key Functions:**
- Load .snirf using `mne.io.read_raw_snirf()`
- Extract and verify metadata (sampling rate, wavelengths, channel count)
- Verify event markers exist (condition 1 = trial start, condition 2 = trial end)

**Quality Checks:**
- [ ] Sampling rate matches hardware specs (20 Hz or 7.8125 Hz)
- [ ] Wavelengths match (690/830 or 760/850 nm)
- [ ] Channel count matches expected (25+8 or 28+8)
- [ ] Event markers are present and correctly labeled

**Deliverables:**
- [ ] Data loading function
- [ ] Metadata extraction and verification script
- [ ] Initial data structure report

---

### 4.3 Step A2: Calculate SNR on Raw Intensity

**Objective:** Calculate Signal-to-Noise Ratio on unprocessed intensity data

**Formula (from paper):**
$$SNR = 20 \times \log_{10}\left(\frac{\mu}{\sigma}\right)$$

where μ = mean, σ = standard deviation of raw intensity signal

**Apply 80-140 dB Threshold:**
- Channels with SNR < 80 dB → Exclude (too noisy)
- Channels with SNR > 140 dB → Exclude (likely saturated)
- Channels within 80-140 dB → Keep for preprocessing

**Deliverables:**
- [ ] SNR calculation function for raw intensity
- [ ] Channel exclusion identification (80-140 dB threshold)
- [ ] CSV file: `{subject_id}_channel_exclusions.csv`
- [ ] Bar plot replicating Figure 4a, c, e, g from paper

**Example Output:**
```
Subject: A1, Dataset: Expert_Novice
Total Channels: 33 (25 long + 8 short)
Good Channels: 31
Excluded Channels: 2 (ch_12: SNR=75.3 dB, ch_24: SNR=145.2 dB)
```

---

### 4.4 Step A3: Calculate Motion Ratio on Raw Intensity

**Objective:** Calculate Motion Ratio on raw intensity using paper's MATLAB parameters

**Formula:**
$$Motion\ Ratio = \frac{\text{Duration of motion artifacts}}{\text{Total acquisition time}}$$

**Motion Detection Parameters (from MATLAB code):**
- `tMotion`: 0.5 (amplitude threshold)
- `tStdev`: 20 (standard deviation multiplier)
- `AMPThresh`: 0.5
- `STDEVthresh`: 5

**Algorithm:**
1. Calculate derivative (rate of change) of raw intensity
2. Flag samples where |derivative| > 0.5 (amplitude threshold)
3. Flag samples where |derivative - mean| > 20×std (outlier detection)
4. Combine both masks (logical OR)
5. Calculate ratio: flagged_samples / total_samples

**Deliverables:**
- [ ] Motion detection function on raw intensity
- [ ] Motion ratio calculation per channel
- [ ] CSV file: `{subject_id}_motion_ratios.csv`
- [ ] Bar plot replicating Figure 4b, d, f, h from paper

---

### 4.5 Step A4: Generate Stage A QA Report

**Objective:** Create comprehensive quality assessment report matching paper's Figure 4

**Report Components:**

1. **Individual Subject Reports:**
   - SNR bar plot with 80-140 dB threshold lines
   - Motion ratio bar plot
   - Bad channels highlighted in red
   - Summary statistics table

2. **Dataset-Level Aggregation:**
   - Mean ± SD across all subjects
   - Channel-wise statistics (which channels fail most often)
   - Subject-wise box plots

3. **Exclusion Decision Log:**
   - JSON/CSV with bad channels per subject
   - Reason for exclusion (SNR too low/high)
   - Pass to Stage B as exclusion list

**Deliverables:**
- [ ] QA report generation function
- [ ] Individual subject plots (SNR + Motion Ratio)
- [ ] Dataset-level aggregated plots (replicate Fig 4)
- [ ] Summary CSV files
- [ ] Channel exclusion lists for Stage B (JSON format)

**Example Exclusion List:**
```json
{
  "subject_id": "A1",
  "dataset": "Expert_Novice",
  "bad_channels": ["S1_D1 690", "S5_D3 830"],
  "good_channels": ["S1_D2 690", "S1_D2 830", ...],
  "total_excluded": 2,
  "exclusion_reasons": {
    "S1_D1 690": "SNR too low: 75.3 dB",
    "S5_D3 830": "SNR too high: 145.2 dB"
  }
}
```

---

### 4.6 Stage A Decision Point

**Gate to Stage B:** Only proceed with preprocessing if:
- [ ] At least 70% of channels pass SNR threshold
- [ ] Motion ratio < 0.5 (less than 50% motion artifacts)
- [ ] Event markers are valid

If subject fails these criteria, flag for review but still document in Stage A report (for transparency like paper).

---

## 5. STAGE B: PREPROCESSING PIPELINE (6 Steps)

### 5.1 Stage B Overview

**Input:** Raw .snirf files + Channel Exclusion Lists from Stage A  
**Output:** Clean HbO₂ and HbR concentration time series  
**Apply to:** Good channels only (bad channels excluded before Step B1)

**Critical Note:** All preprocessing operates only on channels that passed Stage A quality checks.

---

### 5.2 Step B1: Intensity to Optical Density Conversion

**Objective:** Convert raw light intensity to optical density

**Formula:**
$$OD = -\log_{10}\left(\frac{I}{I_0}\right)$$

**Implementation Notes:**
- Use `mne.preprocessing.nirs.optical_density()`
- Mark bad channels (from Stage A) in `raw.info['bads']` before conversion
- Verify OD values are in physiological range (typically 0.1 to 2.0)

**Deliverables:**
- [ ] Intensity to OD conversion function
- [ ] OD range verification (flag if outside 0.1-2.0)
- [ ] Visualization of OD time series for sample channels

---

### 5.3 Step B2: Bandpass Filtering

**Objective:** Apply bandpass filter to remove noise

**Parameters (from paper):**
- **Filter range:** 0-0.50 Hz (this is effectively a low-pass filter)
- **Low cutoff:** 0.01 Hz (remove very slow drift)
- **High cutoff:** 0.5 Hz (remove high-frequency noise)
- **Method:** IIR filter

**Implementation Notes:**
- Use `raw.filter(l_freq=0.01, h_freq=0.5, method='iir')`
- Apply to all good channels

**Deliverables:**
- [ ] Bandpass filtering function
- [ ] Before/after filtering comparison plots
- [ ] Power spectral density plots to verify filtering

---

### 5.4 Step B3: Motion Artifact Detection

**Objective:** Detect motion artifacts on filtered OD data

**Same algorithm as Stage A3, but applied to OD instead of intensity:**
- Use same thresholds: tMotion=0.5, tStdev=20
- Detect on derivative of OD signal
- Output: Boolean mask of motion-corrupted timepoints

**Purpose:** This mask is used for rPCA correction in next step

**Deliverables:**
- [ ] Motion detection function for OD data
- [ ] Motion artifact visualization (marked on OD traces)

---

### 5.5 Step B4: Recursive PCA (rPCA) Motion Correction

**Objective:** Correct motion artifacts using recursive Principal Component Analysis

**Paper Quote:**
> "Motion artifacts and systemic physiology interference were corrected using recursive principal component analysis"

**Algorithm Overview:**
1. Identify motion artifact segments (from B3)
2. For each corrupted segment:
   - Apply PCA to decompose signal
   - Remove high-variance components (likely motion)
   - Reconstruct signal without motion artifacts
3. Iterate until motion is sufficiently reduced

**Implementation Notes:**
- Use `sklearn.decomposition.PCA`
- Typical n_components: 0.97 (retain 97% variance)
- Process in sliding windows around motion artifacts

**Alternative Methods (DO NOT USE):**
Although mne-nirs provides Temporal Derivative Distribution Repair (TDDR), **do not use this method**. Your task is to replicate the paper's rPCA approach exactly.

**Deliverables:**
- [ ] rPCA correction implementation
- [ ] Before/after motion correction comparison plots
- [ ] Validation that motion ratio decreased after correction

---

### 5.6 Step B5: Modified Beer-Lambert Law Conversion

**Objective:** Convert optical density to hemoglobin concentrations

**Formula:**
$$\Delta C = \frac{\Delta OD}{\epsilon \cdot L \cdot PPF}$$

**Parameters (from paper - TechEn system):**
- **PPF at 690 nm:** 6.4
- **PPF at 830 nm:** 5.8

**For NIRx system (760nm, 850nm):**
- Use mne-nirs default PPF values

**Implementation Notes:**
- Use `mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf={690: 6.4, 830: 5.8})`
- Verify hemoglobin values are physiological (typically ±50 μM)
- Separate HbO₂ (oxygenated) and HbR (deoxygenated) channels

**Deliverables:**
- [ ] Beer-Lambert conversion function (TechEn system with custom PPF)
- [ ] Beer-Lambert conversion function (NIRx system with defaults)
- [ ] Hemoglobin concentration time series visualization
- [ ] Physiological range verification (flag if > ±100 μM)

---

### 5.7 Step B6: Short-Channel Regression

**Objective:** Remove superficial physiological interference using short-separation channels

**Paper Quote:**
> "The most correlated short-distance channels are regressed from the long-distance channels to remove any interference from superficial layers using a consecutive sequence of Gaussian basis functions via ordinary least squares"

**Implementation Notes:**
- Use `mne_nirs.signal_enhancement.short_channel_regression(method='OLS')`
- Short channels (~8mm) capture scalp blood flow, not brain activity
- OLS regression removes this systemic noise from long channels (30-40mm)

**What This Does:**
- Identifies most correlated short channel for each long channel
- Regresses out the shared variance (superficial noise)
- Leaves only deep cortical signals in long channels

**Deliverables:**
- [ ] Short-channel regression function
- [ ] Before/after regression comparison plots
- [ ] Correlation analysis (should decrease after regression)
- [ ] Final cleaned HbO₂ and HbR time series
- [ ] Save processed data as .fif files

---

### 5.8 Stage B Output Format

**Save Processed Data:**
- Format: `.fif` (MNE-Python native format)
- Location: `outputs/stage_b_processed/data/`
- Naming: `{subject_id}_{dataset}_{session}_processed_raw.fif`

**Include Metadata:**
- Processing parameters used
- Excluded channels list
- Timestamp of processing
- Software versions (mne, mne-nirs)

---

## 6. STAGE C: POST-PROCESSING VALIDATION

### 6.1 Stage C Overview

**Input:** Processed HbO₂ and HbR time series from Stage B  
**Output:** Validation metrics proving preprocessing was successful  
**Purpose:** Demonstrate that neural signals are detectable and group differences exist

**This stage replicates paper's Figure 3c, 3d (HRF plots) and validates CNR.**

---

### 6.2 Step C1: Extract Hemodynamic Response Function (HRF)

**Objective:** Extract task-evoked hemodynamic responses using block averaging

**Algorithm:**
1. Load processed hemoglobin data
2. Extract event markers (condition 1 = trial start)
3. Create epochs: -5 to +30 seconds around each trial
4. Apply baseline correction (-5 to 0 seconds)
5. Average across trials within subject
6. Grand average across subjects within group

**Implementation Notes:**
- Use `mne.Epochs(tmin=-5, tmax=30, baseline=(-5, 0))`
- Separate by brain region (PFC, M1, SMA)
- Calculate for both HbO₂ and HbR

**Deliverables:**
- [ ] HRF extraction function (epoching + averaging)
- [ ] Grand average HRF calculation
- [ ] HRF plots by region (PFC, M1, SMA) - replicate Figure 3c, 3d
- [ ] Separate plots for each dataset

**Expected HRF Shape:**
- HbO₂: Increases 5-10 seconds post-stimulus, peaks ~8-12s, returns to baseline by 20-25s
- HbR: Decreases (inverse relationship), smaller amplitude than HbO₂

---

### 6.3 Step C2: Calculate Contrast-to-Noise Ratio (CNR)

**Objective:** Quantify how well preprocessing separated group differences

**Formula (from von Lühmann et al., 2020):**
$$CNR = \frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}$$

where μ₁, μ₂ are mean HRF peak amplitudes for two groups

**Group Comparisons Required:**
1. Expert vs. Novice (Expert/Novice study)
2. Trainee vs. Control (Learning Curve study)
3. Sham vs. tDCS (Long-Term tES study)
4. Sham vs. tRNS (Long-Term tES study)

**Implementation Notes:**
- Extract HRF peak amplitude (5-15 seconds post-stimulus window)
- Calculate mean and std for each group
- Compute CNR per channel
- Higher CNR = better separation between groups

**Deliverables:**
- [ ] CNR calculation function
- [ ] Bar plot showing CNR for each comparison (replicate Figure 4i)
- [ ] CSV file with CNR values per channel
- [ ] Statistical significance tests

**Expected Results:**
- CNR > 1.0 indicates detectable difference
- CNR > 2.0 indicates strong separation
- Should see higher CNR in motor regions (M1, SMA) during motor task

---

### 6.4 Step C3: Statistical Validation

**Objective:** Perform statistical tests to confirm group differences

**Tests to Perform:**
1. **Independent t-tests** between groups (Expert vs Novice, etc.)
2. **Effect size calculation** (Cohen's d)
3. **ANOVA** for multi-group comparisons (tDCS vs tRNS vs Sham)

**Implementation Notes:**
- Use `scipy.stats.ttest_ind()` for two-group comparisons
- Calculate Cohen's d: (mean₁ - mean₂) / pooled_std
- Report p-values and effect sizes

**Deliverables:**
- [ ] Statistical testing function
- [ ] Results table with t-statistics, p-values, effect sizes
- [ ] Summary report comparing all groups
- [ ] Validation that results align with paper's conclusions

---

### 6.5 Stage C Final Report

**Comprehensive Validation Report Should Include:**

1. **HRF Validation:**
   - [ ] HRF plots match expected morphology (Figure 3c, 3d)
   - [ ] Peak times are physiologically reasonable (8-12s)
   - [ ] HbO₂ and HbR show inverse relationship

2. **CNR Validation:**
   - [ ] CNR values indicate detectable group differences
   - [ ] CNR plot replicates Figure 4i
   - [ ] Motor regions show higher CNR than prefrontal

3. **Statistical Validation:**
   - [ ] Significant differences between expert/novice groups
   - [ ] Effect sizes are meaningful (Cohen's d > 0.5)
   - [ ] Results align with paper's reported findings

4. **Quality Assurance:**
   - [ ] Compare Stage A raw metrics to paper's Figure 4
   - [ ] Verify preprocessing improved signal quality
   - [ ] Document any deviations from expected results

---

## 7. DATASET-SPECIFIC CONSIDERATIONS

### 7.1 Expert/Novice Dataset
- **Special handling:** Subject reordering may be needed (see MATLAB code)
- **Known issues:** R9 has split recordings with missing Trial 3
- **Channel count:** 25 long + 8 short
- **Expected result:** Experts should show more efficient cortical activation (lower HbO₂ amplitude)

### 7.2 Learning Curve Dataset
- **Special handling:** Separate Control vs Training groups
- **Known issues:** Only 7 short-separation channels (hardware issue)
- **Missing data:** FLS5 Retention file has no stimulus marks
- **Session structure:** 12 training days + retention day
- **Expected result:** Skill improvement over days (decreasing HbO₂, faster task completion)

### 7.3 Short-Term tES Dataset
- **Complex file nomenclature:**
  - `0-PT`: Pre-trials
  - `1-BS1`: Before Stimulation 1
  - `2/4-T`: After tRNS (counterbalanced)
  - `2/4-S`: After Sham (counterbalanced)
  - `3-BS2`: Before Stimulation 2
- **Sampling rate:** 7.8125 Hz (NIRx system)
- **Missing data:** S9 Pretrial 1 has Trial 1 missing
- **Expected result:** Acute effects of tRNS vs sham

### 7.4 Long-Term tES Dataset
- **Groups:** tDCS (n=7), tRNS (n=7), Sham (n=7)
- **Duration:** 12 consecutive days + 4-week follow-up
- **Missing data:** Multiple documented issues in README
- **Sampling rate:** 7.8125 Hz (NIRx system)
- **Expected result:** Long-term skill acquisition differences between stimulation types

---

## 8. VALIDATION CHECKLIST

### 8.1 Stage A Validation
- [ ] SNR values fall within ranges shown in paper's Figure 4a, c, e, g
- [ ] Motion ratios comparable to Figure 4b, d, f, h
- [ ] Number of excluded channels is reasonable (< 30%)
- [ ] Exclusion lists created successfully

### 8.2 Stage B Validation
- [ ] All preprocessing steps complete without errors
- [ ] Hemoglobin values are physiologically plausible (±50 μM)
- [ ] Motion artifacts reduced after rPCA
- [ ] Short-channel regression decreased scalp signal correlation

### 8.3 Stage C Validation
- [ ] HRF morphology matches Figure 3c, 3d
- [ ] Peak times are physiologically reasonable (8-12s post-stimulus)
- [ ] HbO₂ and HbR show expected inverse relationship
- [ ] CNR plot replicates Figure 4i
- [ ] Group differences are statistically significant (where expected)
- [ ] Results align with paper's conclusions

### 8.4 Reproducibility Validation
- [ ] Processing parameters exactly match paper
- [ ] All deviations documented with justification
- [ ] Comparison table: Paper values vs reproduced values
- [ ] Code is well-documented and reusable

---

## 9. DELIVERABLES SUMMARY

### Stage A Outputs
```
outputs/stage_a_qa/
├── figures/
│   ├── {dataset}_snr_aggregated.png
│   ├── {dataset}_motion_ratio_aggregated.png
│   └── {subject}_individual_qa_plots.png
├── excluded_channels/
│   └── {subject}_exclusions.json
└── qa_reports/
    ├── {subject}_stage_a_metrics.csv
    └── {dataset}_summary_statistics.csv
```

### Stage B Outputs
```
outputs/stage_b_processed/
└── data/
    └── {subject}_{dataset}_{session}_processed_raw.fif
```

### Stage C Outputs
```
outputs/stage_c_validation/
├── figures/
│   ├── {dataset}_hrf_by_region.png
│   ├── {dataset}_cnr_comparison.png
│   └── {dataset}_group_statistics.png
└── metrics/
    ├── {dataset}_cnr_values.csv
    ├── {dataset}_statistical_tests.csv
    └── final_validation_report.pdf
```

---

## 10. TIMELINE AND MILESTONES

### Week 1-2: Stage A - Raw Quality Assessment
- [ ] Implement data loading and metadata extraction
- [ ] Calculate SNR on raw intensity
- [ ] Calculate Motion Ratio on raw intensity
- [ ] Generate QA reports for all datasets
- [ ] Create channel exclusion lists

**Milestone:** Stage A QA reports match paper's Figure 4

### Week 3-4: Stage B - Core Preprocessing (Steps 1-3)
- [ ] Implement OD conversion
- [ ] Apply bandpass filtering
- [ ] Detect motion artifacts on OD
- [ ] Test on single subject from each dataset

**Milestone:** Successfully process 1 subject per dataset

### Week 5-6: Stage B - Advanced Processing (Steps 4-6)
- [ ] Implement rPCA motion correction
- [ ] Apply Beer-Lambert Law conversion
- [ ] Perform short-channel regression
- [ ] Process all subjects across all datasets

**Milestone:** All subjects processed, saved as .fif files

### Week 7-8: Stage C - Validation
- [ ] Extract HRFs and create visualizations
- [ ] Calculate CNR for all group comparisons
- [ ] Perform statistical tests
- [ ] Generate final validation report

**Milestone:** Validation metrics match paper's results

### Week 9-10: Documentation and Final Review
- [ ] Document all deviations from paper
- [ ] Create comprehensive comparison table
- [ ] Write final report
- [ ] Prepare presentation for Dr. Nezafati

**Milestone:** Phase 1 complete, ready for Phase 2 (ML pipeline)

---

## 11. PARAMETER REFERENCE TABLE

| Processing Step | Parameter | TechEn CW6 | NIRx NIRScout | Source |
|----------------|-----------|------------|---------------|---------|
| **Stage A Quality** |
| SNR Range | threshold_low | 80 dB | 80 dB | Paper Methods |
| SNR Range | threshold_high | 140 dB | 140 dB | Paper Methods |
| Motion Detection | tMotion | 0.5 | 0.5 | MATLAB Code |
| Motion Detection | tStdev | 20 | 20 | MATLAB Code |
| Motion Detection | AMPThresh | 0.5 | 0.5 | MATLAB Code |
| Motion Detection | STDEVthresh | 5 | 5 | MATLAB Code |
| **Stage B Preprocessing** |
| Bandpass Filter | l_freq | 0.01 Hz | 0.01 Hz | Paper Methods |
| Bandpass Filter | h_freq | 0.5 Hz | 0.5 Hz | Paper Methods |
| rPCA | n_components | 0.97 | 0.97 | Standard |
| Beer-Lambert | PPF (λ₁) | 6.4 (690nm) | Default | Paper Methods |
| Beer-Lambert | PPF (λ₂) | 5.8 (830nm) | Default | Paper Methods |
| Short-Channel | method | OLS | OLS | Paper Methods |
| Short-Channel | separation | ~8mm | ~8mm | Hardware Spec |
| **Stage C Validation** |
| Epochs | tmin | -5 s | -5 s | Standard |
| Epochs | tmax | 30 s | 30 s | Standard |
| Epochs | baseline | (-5, 0) s | (-5, 0) s | Standard |
| HRF Peak Window | t_start | 5 s | 5 s | Physiological |
| HRF Peak Window | t_end | 15 s | 15 s | Physiological |

---

## 12. TROUBLESHOOTING GUIDE

### Common Issues and Solutions

**Issue:** SNR values too low across many channels
- **Check:** Raw data quality, hardware connection issues
- **Solution:** Document in Stage A report, may indicate bad recording session

**Issue:** Motion artifacts remain after rPCA
- **Check:** rPCA parameters (n_components, window size)
- **Solution:** Adjust parameters or increase number of iterations

**Issue:** HbO₂/HbR not showing expecte