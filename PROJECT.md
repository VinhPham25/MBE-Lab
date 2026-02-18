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
│   │   ├── 02_calculate_snr.py
│   │   ├── 03_calculate_motion_ratio.py
│   │   ├── 04_calculate_signal_amplitude.py
│   │   └── 05_generate_qa_report.py
│   ├── stage_b_preprocessing/
│   │   ├── 01_exclude_and_convert_to_od.py
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
│   │   ├── exclusion_lists/
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
│ STAGE A: RAW QUALITY ASSESSMENT (Documentation)     │
│ Purpose: Document data collection quality           │
│ Input: Raw .snirf intensity data                    │
│ Output: Quality metrics, exclusion lists            │
│ Note: NO channels excluded in this stage            │
└─────────────────────────────────────────────────────┘
                         ↓
          ┌──────────────────────────┐
          │ DECISION POINT:          │
          │ Identify which channels  │
          │ to exclude in Stage B    │
          └──────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│ STAGE B: PREPROCESSING PIPELINE (6 Steps)           │
│ Purpose: Clean and process neural signals           │
│ Step 1: Exclude bad channels, then convert to OD    │
│ Output: HbO₂ and HbR concentration time series      │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│ STAGE C: POST-PROCESSING VALIDATION (Success)       │
│ Purpose: Validate preprocessing effectiveness       │
│ Input: Processed hemoglobin data                    │
│ Output: CNR, HRF plots, statistical comparisons     │
└─────────────────────────────────────────────────────┘
```

### 3.2 Key Insight from Paper

**Important Quote from Technical Validation:**
> "To give an accurate assessment of the quality of the data, no channels have been excluded from the analysis provided here and no denoising methods have been used prior to analysis."

**Interpretation:**
- The paper calculated SNR and Motion Ratio on **all channels** (no exclusions) for Figure 4
- This was for **transparency and documentation** purposes
- For actual preprocessing, they excluded channels based on **signal amplitude** (80-140 dB)
- **We replicate this approach:** 
  - Stage A: Calculate metrics on ALL channels, document quality
  - Stage B: Exclude bad channels FIRST, then preprocess

---

## 4. STAGE A: RAW QUALITY ASSESSMENT

### 4.1 Stage A Overview

**Objectives:**
1. Verify hardware functionality - Did the fNIRS device collect data properly?
2. Calculate quality metrics on ALL channels (for transparency)
3. Document data quality - Create transparency reports like paper's Figure 4
4. Identify which channels to exclude in Stage B

**Critical Requirement:** All metrics calculated on **unprocessed raw data** with **NO channel exclusions**.

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

**Objective:** Calculate Signal-to-Noise Ratio on unprocessed intensity data for quality documentation

**Formula (from paper):**
$$SNR = 20 \times \log_{10}\left(\frac{\mu}{\sigma}\right)$$

where μ = mean, σ = standard deviation of raw intensity signal

**Important Note:** SNR is for **documentation only** (Figure 4a,c,e,g). Channel exclusion uses **signal amplitude** (calculated in A4), not SNR.

**Expected SNR Values:**
- Good quality: 15-30 dB
- Excellent quality: > 25 dB  
- Poor quality: < 10 dB

**Deliverables:**
- [ ] SNR calculation function for raw intensity
- [ ] CSV file: `{subject_id}_snr_metrics.csv`
- [ ] Bar plot replicating Figure 4a, c, e, g from paper (show SNR values ~10-30 dB range)

**Example Output:**
```
Subject: A1, Dataset: Expert_Novice
Total Channels: 33 (25 long + 8 short)
Median SNR: 17.2 dB
SNR Range: 2.5 - 37.3 dB
NOTE: All channels included (no exclusions in Stage A)
```

---

### 4.4 Step A3: Calculate Motion Ratio on Optical Density

**Objective:** Calculate Motion Ratio on OD data using paper's MATLAB parameters

**Important:** Motion Ratio is calculated on **Optical Density**, not raw intensity

**Formula:**
$$Motion\ Ratio = \frac{\text{Number of motion-corrupted samples}}{\text{Total samples}}$$

**Motion Detection Parameters (from MATLAB code):**
- `tMotion`: 0.5 (amplitude change threshold)
- `tStdev`: 20 (standard deviation multiplier)
- `AMPThresh`: 0.5 (absolute amplitude threshold)
- `STDEVthresh`: 5 (statistical outlier threshold)

**Algorithm:**
1. Convert raw intensity to optical density (for ALL channels)
2. Calculate derivative (rate of change) of OD signal
3. Flag samples where |derivative| > 0.5 (amplitude threshold)
4. Flag samples where |derivative - mean| > 20×std (outlier detection)
5. Combine both masks (logical OR) - a sample is flagged if EITHER method detects it
6. Calculate ratio: flagged_samples / total_samples

**Expected Motion Ratio Values:**
- Excellent: < 0.05 (< 5%)
- Good: 0.05-0.10 (5-10%)
- Marginal: 0.10-0.20 (10-20%)
- Poor: > 0.20 (> 20%)

**Deliverables:**
- [ ] OD conversion function (for motion detection only)
- [ ] Motion detection function using BOTH amplitude AND statistical thresholds
- [ ] Motion ratio calculation per channel
- [ ] CSV file: `{subject_id}_motion_ratios.csv`
- [ ] Bar plot replicating Figure 4b, d, f, h from paper

---

### 4.5 Step A4: Calculate Signal Amplitude and Create Exclusion List

**Objective:** Identify channels outside 80-140 dB signal amplitude range for Stage B exclusion

**Important Distinction:**
- **Signal Amplitude:** Detector reading level (80-140 dB threshold) - **USED FOR EXCLUSION**
- **SNR:** Signal quality metric (10-30 dB range) - **DOCUMENTATION ONLY**

**Exclusion Criteria (applied in Stage B, not here):**
- Signal amplitude < 80 dB → Will exclude (too weak, poor contact)
- Signal amplitude > 140 dB → Will exclude (saturated detector)
- Signal amplitude 80-140 dB → Will keep for preprocessing

**Deliverables:**
- [ ] Signal amplitude calculation function
- [ ] Exclusion list creation (channels to exclude in Stage B)
- [ ] JSON file: `{subject_id}_exclusions.json`

**Example Exclusion List:**
```json
{
  "subject_id": "A1",
  "dataset": "Expert_Novice",
  "total_channels": 33,
  "good_channels": ["S1_D2 690", "S1_D2 830", ...],
  "bad_channels": ["S1_D1 690", "S5_D3 830"],
  "exclusion_reasons": {
    "S1_D1 690": "Signal amplitude too low: 75.3 dB",
    "S5_D3 830": "Signal amplitude too high: 145.2 dB"
  },
  "note": "Channels NOT excluded yet - exclusion happens in Stage B"
}
```

---

### 4.6 Step A5: Generate Stage A QA Report

**Objective:** Create comprehensive quality assessment report matching paper's Figure 4

**Report Components:**

1. **Individual Subject Reports:**
   - SNR bar plot (threshold lines at ~10 dB low, ~25 dB high)
   - Motion ratio bar plot (threshold lines at 0.10 marginal, 0.20 poor)
   - Channels to be excluded highlighted in red
   - Summary statistics table

2. **Dataset-Level Aggregation:**
   - Mean ± SD across all subjects
   - Channel-wise statistics (which channels fail most often)
   - Subject-wise box plots

3. **Exclusion Decision Log:**
   - JSON/CSV with bad channels per subject
   - Reason for exclusion (signal amplitude outside 80-140 dB)
   - Pass to Stage B as exclusion list

**Deliverables:**
- [ ] QA report generation function
- [ ] Individual subject plots (SNR + Motion Ratio)
- [ ] Dataset-level aggregated plots (replicate Fig 4)
- [ ] Summary CSV files
- [ ] Channel exclusion lists for Stage B (JSON format)

---

### 4.7 Stage A Critical Reminder

**NO CHANNELS ARE EXCLUDED IN STAGE A**

Stage A only:
- ✓ Calculates quality metrics
- ✓ Documents which channels will be excluded
- ✓ Creates exclusion lists for Stage B
- ✗ Does NOT actually exclude any channels

All exclusion happens at the start of Stage B.

---

## 5. STAGE B: PREPROCESSING PIPELINE (6 Steps)

### 5.1 Stage B Overview

**Input:** Raw .snirf files + Channel Exclusion Lists from Stage A  
**Output:** Clean HbO₂ and HbR concentration time series  

**Critical Note from Paper:**
> "First, channels with signal quality outside 80 dB to 140 dB were excluded. Next, the remaining raw optical signals (wavelengths at 690 nm and 830 nm) were converted into optical density."

**Correct Order:** Exclude channels FIRST, then convert to OD

---

### 5.2 Step B1: Channel Exclusion and Optical Density Conversion

**Objective:** Exclude bad channels, then convert remaining channels to optical density

**Formula:**
$$OD = -\log_{10}\left(\frac{I}{I_0}\right)$$

**Implementation Order (from paper):**
```python
# Step 1: Load raw intensity data (ALL channels)
raw = mne.io.read_raw_snirf(filepath, preload=True)

# Step 2: Load exclusion list from Stage A
exclusion_list = load_exclusion_list(subject_id)

# Step 3: Mark bad channels BEFORE any processing
raw.info['bads'] = exclusion_list['bad_channels']

# Step 4: NOW convert to OD (MNE will automatically skip bad channels)
raw_od = mne.preprocessing.nirs.optical_density(raw)

# Step 5: Verify OD values are in physiological range
verify_od_range(raw_od, expected_range=(0.1, 2.0))
```

**Deliverables:**
- [ ] Channel exclusion application function
- [ ] OD conversion function
- [ ] OD range verification (flag if outside 0.1-2.0)
- [ ] Visualization of OD time series for sample channels

---

### 5.3 Step B2: Bandpass Filtering

**Objective:** Apply bandpass filter to remove drift and high-frequency noise

**Parameters (from paper):**
- **Filter range:** "0-0.50 Hz" 
- **Actual implementation:** 0.01-0.5 Hz
  - Low cutoff: 0.01 Hz (removes DC offset and ultra-slow drift)
  - High cutoff: 0.5 Hz (removes cardiac ~1 Hz, respiratory ~0.3 Hz)
- **Method:** IIR filter

**Why 0.01 Hz instead of 0 Hz?**
- 0 Hz is technically impossible to implement (would be infinite DC)
- 0.01 Hz is the practical equivalent that removes baseline drift
- Preserves all hemodynamic frequencies (HRF peaks at ~0.08-0.12 Hz)

**Implementation Notes:**
- Use `raw_od.filter(l_freq=0.01, h_freq=0.5, method='iir')`
- Apply to good channels only (bad channels already marked in info['bads'])

**Deliverables:**
- [ ] Bandpass filtering function
- [ ] Before/after filtering comparison plots
- [ ] Power spectral density plots to verify filtering

---

### 5.4 Step B3: Motion Artifact Detection on Filtered OD

**Objective:** Detect motion artifacts on filtered OD data

**Same algorithm as Stage A3, but applied to filtered OD:**
- Use same thresholds: tMotion=0.5, tStdev=20, AMPthresh=0.5
- Detect using derivative of OD signal
- Apply BOTH amplitude AND statistical methods (logical OR)
- Output: Boolean mask of motion-corrupted timepoints

**Purpose:** This mask is used for rPCA correction in next step

**Deliverables:**
- [ ] Motion detection function for filtered OD data
- [ ] Motion artifact visualization (marked on OD traces)
- [ ] Motion mask for rPCA input

---

### 5.5 Step B4: Recursive PCA (rPCA) Motion Correction

**Objective:** Correct motion artifacts using recursive Principal Component Analysis

**Paper Quote:**
> "Motion artifacts and systemic physiology interference were corrected using recursive principal component analysis"

**Algorithm Overview:**
1. Identify motion artifact segments (from B3 motion mask)
2. For each corrupted segment:
   - Apply PCA to decompose signal into components
   - High-variance components = motion artifacts (large spikes)
   - Low-variance components = neural signal (small changes)
   - Remove high-variance components
   - Reconstruct signal from low-variance components only
3. Iterate until motion is sufficiently reduced

**Implementation Notes:**
- Use `sklearn.decomposition.PCA`
- Typical n_components: 0.97 (retain 97% variance)
- Process segments containing motion artifacts (not entire signal)
- Motion removal targets HIGH variance (spikes), preserves LOW variance (neural)

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

**Output Structure:**
```
raw_haemo.ch_names = [
    'S1_D1 hbo',  # Oxygenated hemoglobin
    'S1_D1 hbr',  # Deoxygenated hemoglobin
    'S1_D2 hbo',
    'S1_D2 hbr',
    ...
]
```

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

**Mechanism:**
- Long channels (30-40mm): Brain signal + Scalp noise
- Short channels (~8mm): Only scalp noise (too shallow for brain)
- **OLS Regression:** For each long channel, find most correlated short channel
- Remove shared variance (scalp noise) using regression coefficient
- Result: Long channels contain only deep brain signal

**Implementation Notes:**
- Use `mne_nirs.signal_enhancement.short_channel_regression(method='OLS')`
- Automatically pairs each long channel with most correlated short channel
- Regression removes systemic noise (heart rate, blood pressure, respiration)

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

**Output Characteristics:**
- **Still continuous** (not epoched - that happens in Stage C)
- Contains HbO₂ and HbR for both long and short channels
- Spans entire recording session (rest + task periods mixed)
- Event markers preserved
- Motion artifacts reduced
- Scalp noise regressed out

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
**Purpose:** Extract HRF and demonstrate that neural signals are detectable

**This stage replicates paper's Figure 3c, 3d (HRF plots) and Figure 4i (CNR).**

---

### 6.2 Step C1: Extract Hemodynamic Response Function (HRF)

**Objective:** Extract task-evoked hemodynamic responses using block averaging

**Why This Step?**
- Stage B output is continuous time series (rest + task mixed together)
- Need to extract task-related responses (HRF)
- Averaging improves signal-to-noise ratio

**Algorithm:**
1. Load processed continuous hemoglobin data
2. Extract event markers (condition 1 = trial start)
3. Create epochs: -5 to +30 seconds around each trial start
4. Apply baseline correction (-5 to 0 seconds as baseline reference)
   - Subtract mean of baseline period from entire trial
   - Centers each trial at 0 (removes baseline shifts)
5. Average across trials within subject (subject-level HRF)
   - Improves SNR by √n_trials
6. Grand average across subjects within group (group-level HRF)
   - Further improves SNR by √n_subjects

**Implementation Notes:**
- Use `mne.Epochs(tmin=-5, tmax=30, baseline=(-5, 0))`
- Separate by brain region (PFC, M1, SMA)
- Calculate for both HbO₂ and HbR

**Deliverables:**
- [ ] HRF extraction function (epoching + averaging)
- [ ] Subject-level HRF calculation
- [ ] Grand average HRF calculation
- [ ] HRF plots by region (PFC, M1, SMA) - replicate Figure 3c, 3d
- [ ] Separate plots for each dataset

**Expected HRF Shape:**
- HbO₂: Increases 2-3s post-stimulus, peaks ~8-12s, returns to baseline ~20-25s
- HbR: Decreases (inverse relationship), smaller amplitude than HbO₂
- Peak amplitude: 5-15 μM for HbO₂, 2-8 μM for HbR

---

### 6.3 Step C2: Calculate Contrast-to-Noise Ratio (CNR)

**Objective:** Quantify how well preprocessing separated group differences

**Formula (from von Lühmann et al., 2020):**
$$CNR = \frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}$$

where μ₁, μ₂ are mean HRF peak amplitudes for two groups

**What CNR Measures:**
- **Group separation:** How different are Expert vs Novice brain patterns?
- **Effect size:** Magnitude of the difference
- **Preprocessing quality:** Higher CNR suggests better signal extraction

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
- [ ] SNR values in 10-30 dB range (matches paper's Figure 4a, c, e, g)
- [ ] Motion ratios 0.02-0.10 range (matches Figure 4b, d, f, h)
- [ ] Number of excluded channels is reasonable (< 30%)
- [ ] Exclusion lists created successfully
- [ ] NO channels actually excluded in Stage A (only documented)

### 8.2 Stage B Validation
- [ ] All preprocessing steps complete without errors
- [ ] Hemoglobin values are physiologically plausible (±50 μM)
- [ ] Motion artifacts reduced after rPCA
- [ ] Short-channel regression decreased scalp signal correlation
- [ ] Output is continuous time series (not epoched)

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
├── exclusion_lists/
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
- [ ] Calculate SNR on raw intensity (10-30 dB range)
- [ ] Calculate Motion Ratio on optical density (0.02-0.10 range)
- [ ] Calculate signal amplitude and create exclusion lists (80-140 dB)
- [ ] Generate QA reports for all datasets

**Milestone:** Stage A QA reports match paper's Figure 4

### Week 3-4: Stage B - Core Preprocessing (Steps 1-3)
- [ ] Exclude bad channels and convert to OD
- [ ] Apply bandpass filtering (0.01-0.5 Hz)
- [ ] Detect motion artifacts on filtered OD
- [ ] Test on single subject from each dataset

**Milestone:** Successfully process 1 subject per dataset

### Week 5-6: Stage B - Advanced Processing (Steps 4-6)
- [ ] Implement rPCA motion correction
- [ ] Apply Beer-Lambert Law conversion
- [ ] Perform short-channel regression (OLS)
- [ ] Process all subjects across all datasets

**Milestone:** All subjects processed, saved as .fif files

### Week 7-8: Stage C - Validation
- [ ] Extract HRFs (epoch, baseline correct, average)
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
| **Stage A Quality (Documentation)** |
| SNR | Formula | 20×log₁₀(μ/σ) | 20×log₁₀(μ/σ) | Paper Methods |
| SNR | Expected Range | 10-30 dB | 10-30 dB | Paper Figure 4 |
| Signal Amplitude | Exclusion Low | 80 dB | 80 dB | Paper Methods |
| Signal Amplitude | Exclusion High | 140 dB | 140 dB | Paper Methods |
| Motion Detection | tMotion | 0.5 | 0.5 | MATLAB Code |
| Motion Detection | tStdev | 20 | 20 | MATLAB Code |
| Motion Detection | AMPThresh | 0.5 | 0.5 | MATLAB Code |
| Motion Detection | STDEVthresh | 5 | 5 | MATLAB Code |
| Motion Ratio | Expected Range | 0.02-0.10 | 0.02-0.10 | Paper Figure 4 |
| **Stage B Preprocessing** |
| Bandpass Filter | l_freq | 0.01 Hz | 0.01 Hz | Paper Methods |
| Bandpass Filter | h_freq | 0.5 Hz | 0.5 Hz | Paper Methods |
| Bandpass Filter | method | IIR | IIR | Paper Methods |
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

**Issue:** SNR values appear too high (80-140 dB)
- **Check:** You may be calculating signal amplitude, not SNR
- **Solution:** SNR = 20×log₁₀(μ/σ) should give ~10-30 dB, not 80-140 dB

**Issue:** Motion artifacts remain after rPCA
- **Check:** rPCA parameters (n_components, window size)
- **Solution:** Adjust parameters or increase number of iterations

**Issue:** HbO₂/HbR not showing expected inverse relationship
- **Check:** PPF values in Beer-Lambert conversion
- **Solution:** Verify 6.4 (690nm) and 5.8 (830nm) for TechEn; use defaults for NIRx

**Issue:** Short-channel regression not effective
- **Check:** Short channels correctly identified (~8mm separation)
- **Solution:** Verify channel labeling in .snirf metadata

**Issue:** HRF peaks at wrong time
- **Check:** Event markers (condition 1 should be trial start)
- **Solution:** Verify event timing in raw data annotations

**Issue:** CNR values much lower than expected
- **Check:** Group assignment (Expert vs Novice, etc.)
- **Solution:** Verify subject metadata, check if preprocessing removed too much signal

**Issue:** .snirf file won't load
- **Check:** File corruption, BIDS compliance
- **Solution:** Re-download from Figshare, check file integrity

---

## 13. REFERENCES

### Primary Paper
Kamat, A., Eastmond, C., Gao, Y. et al. Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND). *Sci Data* 10, 699 (2023). https://doi.org/10.1038/s41597-023-02603-3

### Key Methodological References
- **fNIRS Best Practices:** Yücel, M. A. et al. Best practices for fNIRS publications. *Neurophotonics* 8 (2021).
- **CNR Calculation:** von Lühmann, A., Li, X., Gilmore, N., Boas, D. A. & Yücel, M. A. Open Access Multimodal fNIRS Resting State Dataset. *Front Neurosci* 14 (2020).
- **Beer-Lambert Law:** Cui, X., Bray, S. & Reiss, A. L. Functional near infrared spectroscopy (NIRS) signal improvement. *Neuroimage* 49 (2010).

### Software Documentation
- **MNE-Python:** https://mne.tools/stable/index.html
- **MNE-NIRS:** https://mne.tools/mne-nirs/stable/index.html
- **HOMER2/3 (Reference Only):** https://homer-fnirs.org/

### Dataset Access
- **Figshare Repository:** https://doi.org/10.6084/m9.figshare.c.6429713
- **BIDS Specification:** https://bids.neuroimaging.io/
- **SNIRF Format:** https://github.com/fNIRS/snirf

---

## 14. SUCCESS CRITERIA

### Phase 1 is considered successful when:

1. **Stage A Replication:**
   - [ ] SNR values in 10-30 dB range (matches Figure 4a,c,e,g)
   - [ ] Motion Ratio values in 0.02-0.10 range (matches Figure 4b,d,f,h)
   - [ ] Channel exclusion rates are comparable to paper
   - [ ] QA metrics calculated for all 4 datasets
   - [ ] All channels documented (no exclusions in Stage A)

2. **Stage B Processing:**
   - [ ] All subjects successfully processed without errors
   - [ ] Bad channels excluded BEFORE OD conversion (correct order)
   - [ ] Hemoglobin values are physiologically plausible (±50 μM)
   - [ ] Processing pipeline uses exact parameters from paper

3. **Stage C Validation:**
   - [ ] HRF plots match paper's Figure 3c, 3d morphology
   - [ ] CNR values show expected group differences (Figure 4i)
   - [ ] Statistical tests confirm significance where expected
   - [ ] Expert vs Novice comparison shows expertise effects

4. **Documentation:**
   - [ ] All code is well-documented and reproducible
   - [ ] Deviations from paper are clearly explained
   - [ ] Comparison table shows reproduced vs paper values
   - [ ] Final report ready for Dr. Nezafati's review

5. **Readiness for Phase 2:**
   - [ ] Processed data ready for ML pipeline
   - [ ] Understanding of fNIRS preprocessing complete
   - [ ] Baseline results established for comparison

---

## 15. NOTES FOR IMPLEMENTATION

### For the Implementation Agent:

1. **Start with One Subject:**
   - Pick one subject from Expert/Novice dataset
   - Run through entire pipeline (Stages A, B, C)
   - Verify outputs before scaling to all subjects

2. **Modular Code Structure:**
   - Each stage should be independent script/module
   - Use consistent data structures (pandas DataFrames for metrics)
   - Save intermediate outputs at each stage

3. **Error Handling:**
   - Implement try/except blocks for file loading
   - Log all warnings and errors
   - Continue processing even if one subject fails

4. **Visualization Standards:**
   - Use same plotting style as paper (white bars, black edges)
   - Include error bars where appropriate
   - Save high-resolution figures (300 dpi)
   - Use consistent color schemes

5. **Data Validation:**
   - Add assert statements for expected data ranges
   - Print summary statistics at each stage
   - Compare intermediate results to expected values

6. **Configuration File:**
   - Create `config.yaml` with all parameters
   - Avoid hardcoding values in scripts
   - Make it easy to adjust parameters for testing

### Example Config Structure:
```yaml
# config.yaml
datasets:
  expert_novice:
    system: "TechEn_CW6"
    sfreq: 20
    wavelengths: [690, 830]
    ppf: {690: 6.4, 830: 5.8}
  
stage_a:
  snr_expected_range: [10, 30]  # dB
  motion_ratio_expected_range: [0.02, 0.10]
  signal_amplitude_low: 80  # dB
  signal_amplitude_high: 140  # dB
  motion_tMotion: 0.5
  motion_tStdev: 20
  motion_AMPthresh: 0.5
  motion_STDEVthresh: 5

stage_b:
  filter_low: 0.01
  filter_high: 0.5
  filter_method: 'iir'
  rpca_n_components: 0.97
  
stage_c:
  epoch_tmin: -5
  epoch_tmax: 30
  baseline: [-5, 0]
  hrf_peak_window: [5, 15]
```

---

## 16. EXPECTED OUTCOMES

### Stage A Expected Results:

**SNR Values (from Figure 4):**
- Expert/Novice: Median ~15-25 dB
- Learning Curve: Median ~10-20 dB
- Short-term tES: Median ~15-25 dB
- Long-term tES: Median ~10-20 dB

**Motion Ratio Values (from Figure 4):**
- Expert/Novice: Median ~0.03 (range 0.01-0.08)
- Learning Curve: Median ~0.04 (range 0.02-0.10)
- Short-term tES: Median ~0.03 (range 0.01-0.06)
- Long-term tES: Median ~0.05 (range 0.02-0.12)

**Channel Exclusions:**
- Typically 0-5 channels per subject excluded (based on signal amplitude 80-140 dB)
- Short-separation channels may have higher exclusion rates

### Stage B Expected Results:

**Optical Density:**
- Range: 0.1 to 2.0 (typical physiological)
- Should be smooth after filtering
- Motion spikes should be reduced after rPCA

**Hemoglobin Concentrations:**
- HbO₂: ±10 to ±50 μM during task
- HbR: ±5 to ±30 μM during task (smaller magnitude than HbO₂)
- Baseline-corrected values should center around 0

### Stage C Expected Results:

**HRF Morphology:**
- **HbO₂:** Increases starting ~2-3s, peaks ~8-12s, returns to baseline ~20-25s
- **HbR:** Decreases (inverse), smaller amplitude, similar timing
- **Peak amplitude:** 5-15 μM for HbO₂, 2-8 μM for HbR

**CNR Values (from Figure 4i):**
- Expert vs Novice: CNR ~1.5-3.0 (should show clear separation)
- Trainee vs Control: CNR ~1.0-2.5
- Sham vs tDCS: CNR ~1.0-2.0
- Sham vs tRNS: CNR ~1.0-2.0

**Statistical Significance:**
- Expert vs Novice: p < 0.05 (experts show lower activation)
- Learning effects: p < 0.05 (activation decreases over training)
- Stimulation effects: May or may not reach significance (depends on individual differences)

---

## 17. FINAL NOTES

### Remember:

1. **Exact Replication is the Goal:**
   - Use exact parameters from paper
   - Don't "improve" the pipeline
   - Document any necessary deviations

2. **Stage A is Critical:**
   - Documents quality for ALL channels (no exclusions)
   - Creates exclusion lists for Stage B
   - Transparency in reporting (like paper's Figure 4)

3. **Validate at Each Stage:**
   - Don't proceed if outputs look wrong
   - Compare intermediate results to expected ranges
   - Ask for help if stuck

4. **Document Everything:**
   - Code comments explaining why, not just what
   - Save all intermediate outputs
   - Keep a lab notebook of issues encountered

5. **This is Phase 1:**
   - Goal is reproduction, not innovation
   - Establishes baseline for Phase 2 (ML pipeline)
   - Demonstrates understanding of fNIRS processing

---

**END OF PRD**

*Version: 3.0: Corrected SNR vs Signal Amplitude, Motion Detection, Stage B Ordering*  
*Last Updated: February 17, 2026*  
*Authored By: Vinh Pham*  
*Corrections Based On: Detailed technical review with Claude*
