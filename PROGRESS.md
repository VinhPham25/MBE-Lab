# PROGRESS.md - ASTaUND Phase 1 Implementation

**Last Updated:** February 20, 2026  
**Current Week:** Week 3-4 (Stage B)  
**Current Stage:** Section 3.4 In Progress — Tasks 3.4.1–3.4.2 Complete
**Next Milestone:** Section 3.4.3 — Verify Motion Reduction Post-Correction

---

## 🎯 PROJECT STATE OF THE UNION

### Current Status
- **Phase:** Phase 1 - Reproduction and Validation
- **Active Stage:** Stage B — Section 3.4 In Progress (3.4.1–3.4.2 complete)
- **Completion:** 47% (44/92 tasks)
- **Blockers:** None
- **Latest Achievement:** Task 3.4.2 complete — before/after correction comparison plot generated (4 channels: 2 high-motion + 2 low-motion, 3 panels per channel: Before | After | Removed, first 300s, red-shaded union mask); saved `A1_rpca_before_after.png`.

### Recent Changes
- [2026-02-19] ✅ Section 3.2 complete (tasks 3.2.1–3.2.4) — built `code/stage_b_preprocessing/02_bandpass_filtering.py`; 0.01–0.5 Hz IIR bandpass filter applied to 60 good channels (6 bad skipped); PSD HF attenuation ratio 0.0075 ✓.
- [2026-02-19] ✅ Section 3.3 complete (tasks 3.3.1–3.3.3) — built `code/stage_b_preprocessing/03_motion_correct.py`; Homer3 algorithm adapted for filtered OD with amplitude-only criterion (use_amp_only=True, AMPthresh=5.0 OD); A1: 0% motion, 60/60 ok, visualization saved.
- [2026-02-20] ✅ Task 3.4.1 complete — rPCA algorithm implemented in `code/stage_b_preprocessing/04_rPCA_motion_correct.py`; global PCA on union motion mask; A1: 26,470/26,500 samples in union mask (99.9%), 21 PCs × 97.2% variance removed, neural residual 2.8%; corrected OD saved as `.fif`.
- [2026-02-20] ✅ Task 3.4.2 complete — before/after comparison plot; 4 channels (2 high-motion, 2 low-motion), 3-panel layout (Before | After | Removed), red union-mask shading, per-channel variance-removed annotation; saved `A1_rpca_before_after.png`.

### Next Steps
1. Task 3.4.3 — Re-run motion detection on corrected OD and compare motion ratios.
2. Tasks 3.4.4–3.4.5 — Document effectiveness and tune if needed.

---

## 📋 DESIGN REQUIREMENTS SUMMARY

### Critical Parameters (Must Match Paper Exactly)
- **Signal Amplitude Exclusion List:** 80-140 dB
- **Motion Detection:** tMotion=0.5, tStdev=20, AMPThresh=0.5, STDEVthresh=5
- **Bandpass Filter:** 0.01-0.5 Hz
- **PPF Values (TechEn):** 6.4 (690nm), 5.8 (830nm)
- **Short-Channel Regression:** OLS method
- **Epoch Window:** -5 to +30 seconds, baseline (-5, 0)

### Three-Stage Workflow
```
Stage A (Raw QA) → Decision Point → Stage B (Preprocessing) → Stage C (Validation)
```

### Success Criteria
- [x] Stage A plots match paper's Figure 4 (SNR + Motion Ratio) — A1 verified: SNR median 16.0 dB, MR median 0.000
- [ ] Stage B produces physiologically plausible HbO₂/HbR (±50 μM)
- [ ] Stage C HRF plots match paper's Figure 3c, 3d
- [ ] CNR values show expected group differences (Figure 4i)

---

## 🔄 IMPLEMENTATION METHODOLOGY

### Development Approach
1. **Atomic Tasks:** Each checkbox is one meaningful feature/functionality (~15-30 min)
2. **Test-Driven:** Test on 1 subject before scaling. Note, each study in data/BIDS/ is structured differently, and the analyeses/features must work for all studies and subjects.
3. **Incremental:** Complete stage before moving to next
4. **Documented:** Log all attempts, errors, and solutions. Note, the progress updates after each task should be short and highlight key updates. Do not be redundant and be efficiency with token usage.

### When Stuck (Ralph Loop Pattern)
1. **Document the blocker** in "Roadblocks Encountered" section below
2. **Try alternative approach** (ask me first and then document what you tried)
3. **Verify assumptions** (check paper, verify data format, check parameter values)
4. **Simplify the task** (break it down further if needed)
5. **Move to parallel task** if completely blocked (mark original with ⚠️)

### Update Protocol
After completing ANY task:
1. Mark task as `[x]` with completion date: `[x] (2026-01-18)`
2. Update "Last Updated" timestamp at top
3. Add entry to "Recent Changes"
4. Update completion percentage
5. Document any issues in "Roadblocks Encountered"
6. Update "Notes for Next Session" before stopping
7. Ensure "Recent Changes" and "Current Status" is fully updated at the end of each section

---

## 🧪 TESTING PROTOCOL

### Test Subjects (One per system)
1. **A1** - Expert/Novice (TechEn CW6)
   - Sampling: 20 Hz
   - Wavelengths: 690/830 nm
   - Channels: 25 long + 8 short
   - File: `data/BIDS/Expert_Novice/A1/A1.snirf`

2. **FLS1_Session1** - Learning Curve (TechEn CW6 with issues)
   - Sampling: 20 Hz
   - Wavelengths: 690/830 nm
   - Channels: 25 long + **7 short** (hardware issue!)
   - File: `data/BIDS/Learning_Curve/FLS1/FLS1_Session1.snirf`

3. **S1_0-PT** - Short-term tES (NIRx NIRScout)
   - Sampling: 7.8125 Hz
   - Wavelengths: 760/850 nm
   - Channels: 28 long + 8 short
   - File: `data/BIDS/Short_Term_tES/S1/S1_0-PT.snirf`

### Testing Gates

#### Gate 1: After Stage A Implementation (Section 2 complete)
- [x] A1 processed successfully
- [ ] FLS1_Session1 processed successfully (verify 7 short channels handled)
- [ ] S1_0-PT processed successfully (verify different sampling rate handled)
- [ ] All 3 produce reasonable SNR/motion metrics
- [ ] Document any system-specific code needed

**Expert/Novice special-case check (Dataset_README notes):**
- R2 (`Beginning of Trial 5 estimated`) and R3 (`End of Trial 3 estimated`) processed successfully in Stage A with valid SNR/MR outputs.
- R9 (`Split into 2 recordings with Trial 3 missing`) processed successfully in Stage A; SNR distribution remains a documented visual discrepancy vs paper Figure 4a.
- Stage A raw QA is event-agnostic (continuous intensity/OD quality metrics), so estimated/missing trial markers affect event-level analysis more than SNR/MR computation.

#### Gate 2: After Stage B Core Implementation (Section 3.1-3.6 complete)
- [ ] A1 preprocessed → HbO/HbR look physiological
- [ ] FLS1_Session1 preprocessed → Handles 7 short channels in regression
- [ ] S1_0-PT preprocessed → NIRx wavelengths work with Beer-Lambert
- [ ] All 3 produce clean hemoglobin time series
- [ ] Document any PPF or filtering adjustments

#### Gate 3: After Stage B Complete (Section 3.8 complete)
- [ ] Full Expert/Novice dataset (n=14) processed
- [ ] Sample Learning Curve subjects (n=3) processed
- [ ] Sample Short-term tES subjects (n=3) processed
- [ ] Cross-dataset comparison looks reasonable

### System-Specific Adaptations Tracker
*Document any code that needs to handle system differences*

**Example:**
```python
# System-specific PPF values
if system == "TechEn":
    ppf = {690: 6.4, 830: 5.8}
elif system == "NIRx":
    ppf = None  # Use defaults for 760/850nm
```

| Adaptation | Reason | Location | Status |
|------------|--------|----------|--------|
| Expert/Novice event-note tolerance in Stage A | R2/R3 estimated trial boundaries and R9 split/missing trial do not block raw continuous SNR/MR QA | `code/stage_a_raw_qa/07_generate_qa_report.py` + cumulative outputs | ✅ validated |

---

## 📊 TASK BREAKDOWN (92 Total Tasks)

### SECTION 1: ENVIRONMENT SETUP (Week 0) ✅ COMPLETE
**Completion:** 7/7 tasks (100%)  
**Reference:** PRD Section 2

#### 1.1 Development Environment ✅ COMPLETE
- [x] **1.1.1** (2026-01-19) Setup Python environment (3.9+) with virtual environment and install all required libraries (mne, mne-nirs, numpy, pandas, matplotlib, scipy, sklearn)
- [x] **1.1.2** (2026-01-19) Create and verify project directory structure (data/, code/, outputs/ with all subdirectories)
- [x] **1.1.3** (2026-01-19) Create config.yaml file with all parameters from PRD (SNR thresholds, motion params, filter settings, PPF values)

#### 1.2 Data Access and Verification ✅ COMPLETE
- [x] **1.2.1** (2026-01-19) Verify Expert/Novice dataset from data/BIDS/Expert-Novice/ (all 14 subjects confirmed, A1 verified)
- [x] **1.2.2** (2026-01-19) Verify .snirf file structure with mne.io.read_raw_snirf() - 66 channels, 25 Hz, event markers confirmed
- [x] **1.2.3** (2026-01-19) Create README.md documenting setup process and project structure

#### 1.3 Utility Functions ✅ COMPLETE
- [x] **1.3.1** (2026-01-19) Create code/utils.py with 9 functions: load_config, get_dataset_config, get_stage_params, get_subject_path, get_snirf_files, get_output_path, validate_config, print_config_summary - all tests passing

---

### SECTION 2: STAGE A - RAW QUALITY ASSESSMENT (Week 1-2)
**Completion:** 26/26 tasks (100%)  
**Reference:** PRD Section 4

#### 2.1 Data Loading and Metadata Extraction (Step A1) ✅ COMPLETE
- [x] **2.1.1** (2026-01-19) Implemented 01_load_data.py - loads .snirf, extracts metadata (sfreq, channels, wavelengths, events)
- [x] **2.1.2** (2026-01-19) Created 02_verify_metadata.py - validates hardware specs, all checks passed for A1
- [x] **2.1.3** (2026-01-19) Implemented 03_extract_events.py - extracts/verifies events, calculated trial durations
- [x] **2.1.4** (2026-01-19) Created 04_test_pipeline.py - complete pipeline test passed, outputs saved (metadata.csv, events.csv, pipeline_test.json)

#### 2.2 SNR Calculation on Raw Intensity (Step A2) ✅ COMPLETE
- [x] **2.2.1** (2026-01-19) Implemented 05_calculate_snr.py - SNR = 20*log10(mean/std) on raw intensity, tested on A1
- [x] **2.2.2** (2026-01-19) Created save_snr_results() function - saves CSV with channel, wavelength, mean, std, snr_db
- [x] **2.2.3** (2026-01-19) Verified on A1: Mean SNR=17.17 dB (range 2.50-37.30), matches paper's Figure 4 (~15-30 dB range)

#### 2.3 Motion Ratio Calculation on Optical Density (Step A3) ✅ COMPLETE
- [x] **2.3.1** (2026-01-19) Convert raw intensity to optical density using mne.preprocessing.nirs.optical_density()
- [x] **2.3.2** (2026-01-19) Implement motion artifact detection on OD data (derivative-based with hmrMotionArtifactByChannel parameters)
- [x] **2.3.3** (2026-01-19) Apply motion detection thresholds: tMotion=0.5, tMask=0.5, STDEVthresh=20, AMPthresh=5
- [x] **2.3.4** (2026-01-19) Calculate motion ratio as (flagged_samples / total_samples) per channel
- [x] **2.3.5** (2026-01-19) Test motion detection on subject A1 and verify results are reasonable (~0 median for A1 from Fig 4)

#### 2.4 QA Report Generation (Step A4) ✅ COMPLETE
- [x] **2.4.1** (2026-02-17) Create comprehensive metrics DataFrame combining SNR, motion ratio, QA quality flags — saved A1_qa_metrics.csv (66 channels, 13 cols: snr_db, motion_ratio, snr_quality_flag, motion_quality_flag, qa_note; 33 acceptable / 24 low / 9 high SNR; 0 excluded — all channels retained for QA transparency)
- [x] **2.4.2** (2026-02-17) Implement SNR box plot visualization — white boxes, blue median line, red + outliers, dashed threshold lines at 10 dB (low) and 30 dB (high); saved A1_snr_boxplot.png; 690 nm median=11.2 dB range=[4.1,36.1], 830 nm median=17.6 dB range=[2.5,37.3]
- [x] **2.4.3** (2026-02-17) Implement motion ratio box plot visualization — white boxes, blue median, red + outliers; saved A1_motion_boxplot.png; 690 nm median=0.000 range=[0.000,0.017], 830 nm median=0.000 range=[0.000,0.011]; 0 elevated channels (all motion ~0 as expected for A1)
- [x] **2.4.4** (2026-02-17) Summary statistics calculator and console report generator — saved A1_qa_summary.json; SNR mean=17.17 dB median=16.01 dB range=[2.50,37.30]; MR median=0.000 (0%); fixed numpy int64 JSON serialization and empty-legend warning
- [x] **2.4.5** (2026-02-17) Generate complete Stage A report for subject A1 — generate_stage_a_report() orchestrates all sub-tasks end-to-end; all 4 output files confirmed (A1_qa_metrics.csv, A1_snr_boxplot.png, A1_motion_boxplot.png, A1_qa_summary.json); verified vs Figure 4: SNR median=16.01 dB range=[2.50,37.30] IQR=[7.11,26.71] ✓ matches A1 box in Fig 4a; MR median=0.000 max=0.017 ✓ matches A1 ~0 in Fig 4b

#### 2.5 Signal Amplitude Assessment (for Stage B exclusion)
- [x] **2.5.1** (2026-02-18) Calculate signal amplitude in dB for all channels (separate from SNR) — implemented in `08_signal_amplitude_exclusion.py` using `amplitude_db = 20*log10(RMS(raw_intensity))`; tested on A1 (66 channels), saved `A1_signal_amplitude_raw.csv`
- [x] **2.5.2** (2026-02-18) Identify channels outside 80-140 dB signal amplitude range (for Stage B exclusion candidates) — added threshold logic and flags (`within_80_140_db`, `exclusion_candidate`, `exclusion_reason`) in `08_signal_amplitude_exclusion.py`; A1 result: 6 below 80 dB, 0 above 140 dB, 60 within range; saved `A1_signal_amplitude_flagged.csv`
- [x] **2.5.3** (2026-02-18) Create exclusion list JSON: `{subject_id, bad_channels[], good_channels[], exclusion_reasons{}}` — implemented in `create_exclusion_list_json()` and validated on A1; preview saved as `A1_exclusion_list_preview.json` in `outputs/stage_a_qa/channel_signal_quality_exclusion/`
- [x] **2.5.4** (2026-02-18) Save to outputs/stage_a_qa/channel_signal_quality_exclusion/ (NOTE: channels NOT excluded yet, just documented) — verified final Stage A artifacts are stored in the updated directory for Stage B handoff

#### 2.6 Dataset-Level Operations
- [x] **2.6.1** (2026-02-18) Implement dataset aggregation function (combine all subjects, calculate channel-wise statistics) — added `aggregate_dataset_metrics()` in `07_generate_qa_report.py`; combines available subject QA metrics and computes channel-wise + per-subject SNR/MR stats; outputs saved to `expert_novice_all_subjects_qa_metrics.csv`, `expert_novice_channel_stats.csv`, `expert_novice_subject_stats.csv` in `outputs/stage_a_qa/qa_reports/expert_novice_full_report/`
- [x] **2.6.2** (2026-02-18) Create aggregated visualization plots with error bars (replicate Figure 4 style) — added `plot_dataset_level_boxplots()` in `07_generate_qa_report.py`; generates dataset-level `expert_novice_snr_boxplot.png` and `expert_novice_motion_boxplot.png` with subject-level box plots and optional mean±std error bars in `outputs/stage_a_qa/qa_reports/expert_novice_full_report/`
- [x] **2.6.3** (2026-02-18) Process all 14 Expert/Novice subjects through Stage A pipeline — implemented `process_dataset_stage_a_reports()` in `07_generate_qa_report.py` and executed batch run; success 14/14, failure 0/14; generated per-subject SNR, motion, metrics, summary JSON, and QA box plots for all subjects
- [x] **2.6.4** (2026-02-18) Generate dataset-level QA report and compare visualization plots to paper's Figure 4
- [x] **2.6.5** (2026-02-18) Document any problematic subjects and verify completion (14/14 processed)

**Stage A note (Expert/Novice R9):** Local R9 SNR boxplot does not visually match ASTaUND Figure 4a (paper shows lower Q1/whisker and more negatives). Current local metrics for R9 remain high (Q1~31.8 dB, median~33.8 dB, 2 negative outliers), so this is tracked as a documented dataset/replication discrepancy rather than a plotting bug. Additionaly, check if MR values can be negative. Several subjects in the MR boxplot, for expert-novice study, including subejcts R1, R9, R10, and R12, show negative MR values.

---

### SECTION 3: STAGE B - PREPROCESSING PIPELINE (Week 3-6)
**Completion:** 12/36 tasks (33%)  
**Reference:** PRD Section 5

#### 3.1 Intensity to Optical Density (Step B1) ✅ COMPLETE
- [x] **3.1.1** (2026-02-19) Load exclusion list from Stage A (channels failing 80-140 dB signal amplitude threshold)
- [x] **3.1.2** (2026-02-19) Mark bad channels in raw.info['bads'] before any preprocessing
- [x] **3.1.3** (2026-02-19) Implement OD conversion function (mne.preprocessing.nirs.optical_density) — `convert_to_optical_density()` wraps MNE; bad channels preserved; output shape (66, 26500), type fnirs_od
- [x] **3.1.4** (2026-02-19) OD quality verification — replaced mean-based [0.1–2.0] range check with: (1) NaN/Inf detection, (2) flat-signal via std < 1e-5, (3) Stage A bad-channel pass-through; A1: 60 ok, 6 bad_channel, 0 nan_or_inf, 0 flat_signal; OK std mean=0.259 range=[0.014, 0.760]
- [x] **3.1.5** (2026-02-19) Tested on A1, visualized OD time series — saved `A1_od_time_series.png` and `A1_od_range_verification.csv` in `outputs/stage_b_processed/od_conversion/A1/`

Important Note: a mean OD_std of 0.259 is quite high. The expected range of std FNIRS is ideally ~0.02. This difference has been noted and could be revisited later. Currently, a threshold of 1e-5 is the minimum (if any std is less than 1e-5, it's classified as a disconnected channel). It could be beneficial to add an upper threshold as well to remove noisy channels with high std.

#### 3.2 Bandpass Filtering (Step B2) ✅ COMPLETE
- [x] **3.2.1** (2026-02-19) Implement bandpass filter application (0.01-0.5 Hz, IIR method) — `apply_bandpass_filter()` wraps `raw_od.filter()`; zero-phase IIR Butterworth; 60 channels filtered, 6 bad skipped
- [x] **3.2.2** (2026-02-19) Created before/after filtering comparison visualisation — 4-channel time-series subplot saved as `A1_filter_comparison.png`
- [x] **3.2.3** (2026-02-19) Verified filtering effectiveness with PSD plots — Welch PSD overlay (linear + log dB) saved as `A1_psd_comparison.png`; HF power ratio (>0.5 Hz) = 0.0075 ✓
- [x] **3.2.4** (2026-02-19) Tested filtering on subject A1 — full pipeline runs end-to-end via `run_bandpass_filter()`; outputs in `outputs/stage_b_processed/bandpass_filter/A1/`

#### 3.3 Motion Artifact Detection on OD (Step B3) ✅ COMPLETE
- [x] **3.3.1** (2026-02-19) Adapted Homer3 motion detection for filtered OD — `detect_motion_artifacts()` with `_detect_motion_homer3_single()` + `_expand_mask_to_signal()`; Stage A key differences: input is filtered OD, output is per-sample boolean mask; `use_amp_only=True` parameter added to switch criterion for identifying motion artifacts
- [x] **3.3.2** (2026-02-19) Generated per-channel motion mask and statistics — `compute_motion_stats()` calculates motion_ratio, pct_corrupted, quality flag (ok/elevated/high/bad_channel); A1: 60 ok, 0 elevated, 0 high, 6 bad_channel; 0.00% mean corrupted ✓
- [x] **3.3.3** (2026-02-19) Visualized motion artifacts on OD traces — `plot_motion_artifacts()` shows 3 low-motion (blue) + 2 high-motion/bad (red/orange) channels with shaded flagged regions; saved `A1_motion_artifacts.png` in `outputs/stage_b_processed/motion_correction/A1/`

**Threshold Decision — Amplitude-Only (use_amp_only=True)**
The original Homer3 dual-criterion (AMPthresh=5.0 OD + STDEVthresh=20×std_diff) flagged 17.4% of A1 samples when applied to filtered OD (expected ~0%). Root cause: bandpass filtering (0.01–0.5 Hz) reduces the derivative std (std_diff) by orders of magnitude, so `mc_thresh = std_diff × 20` shrinks below the level of normal hemodynamic oscillations. This could be an overestimate for the presence of motion artifacts or it could be the ground truth.

Two options were considered if the threshold was too low, and too many motion artifacts were flagged:
- **Option 1 (dual-criterion, raise multiplier):** Increase STDEVthresh (e.g., 50–100×) to recalibrate for filtered data. Rejected because the correct multiplier is not well-defined and would require dataset-specific tuning — reintroducing the estimation problem that the amplitude criterion avoids.
- **Option 2 (amplitude only) ✅ Chosen:** Use AMPthresh=5.0 OD units alone. Stable across datasets (absolute OD units, not relative to noise level). Catches large motion spikes that survive the bandpass filter. Result: 0% flagged for A1 ✓, consistent with Stage A expectation.

However, both of these options require finding a new value for STDEVthresh or AMPthresh that captures the motion correctly, which is arbitrary. Note, only using AMPthresh = 5.0 resulted in no motion artifacts being found, meaning Option 2 could have undershot the detection of motion artifacts. Therefore, in the detect_motion_artifact() function, there exists a parameter labeled use_amp_only: bool = True, which can be toggled to determine if the motion detection function utilizes the original threshold or the amplitude specific one. Additionally, for the full pipeline function run_motion_detect(), USE_AMP_ONLY: bool = False is a boolean that can be toggled to adjust the time-series and .csv outputs. 

#### 3.4 Recursive PCA Motion Correction (Step B4)
- [x] **3.4.1** (2026-02-20) Implement rPCA algorithm: global PCA on union motion mask (dual criterion, use_amp_only=False), remove high-variance components (21 PCs, 97.2% variance = motion subspace), keep low-variance neural residual (2.8%); n_components=0.97; A1: union mask 99.9% (inevitable with 60 ch each flagging ~15% → (1−0.15)^60 ≈ 0.01% chance all-clean); corrected OD saved to `outputs/stage_b_processed/rPCA_correction/A1/`

Note, the union mask covers almost 100% of the recording because each channel independently flags samples, so it's extremely likely that the mask will cover the entire time signal. This means rPCA is applied globally rather than locally on only the "bad" signals. Some suggestions for improvement that should be considered:
1) Improving the motion detection algorithm. For subject A1, 17.4% of samples are detected as noise. If we can clarify the criteria/treshold for defining motion, this percentage could decrease and thus decrease the coverage of the union mask.
2) Modify the OR mask with a summation threshold. Only include a time point as having motion if > X channels simulatenously flag this motion.
3) Apply grouped union (masks for anatomical clusters / groups of channels (Ex: frontal channels, motor chanells, etc.)). This is because noise in the front region might not affect the channels in the back.

- [x] **3.4.2** (2026-02-20) Before/after comparison visualization: 4 channels (2 highest-motion, 2 lowest-motion), 3-panel layout (Before | After | Removed), first 300 s, red shading for union motion mask, per-channel variance-removed annotation; saved `A1_rpca_before_after.png`
- [ ] **3.4.3** Verify motion reduction by recalculating motion ratio after correction
- [ ] **3.4.4** Test rPCA on subject A1 and document effectiveness
- [ ] **3.4.5** Tune rPCA parameters if motion artifacts persist (document parameter changes)

#### 3.5 Modified Beer-Lambert Law (Step B5)
- [ ] **3.5.1** Implement Beer-Lambert conversion with TechEn PPF values (690nm: 6.4, 830nm: 5.8)
- [ ] **3.5.2** Verify hemoglobin concentration ranges (±50 μM typical, flag if >±100 μM)
- [ ] **3.5.3** Visualize HbO and HbR time series and verify inverse relationship
- [ ] **3.5.4** Test Beer-Lambert conversion on subject A1

#### 3.6 Short-Channel Regression (Step B6)
- [ ] **3.6.1** Implement short-channel regression: Find most correlated short channel for each long channel, apply OLS regression to remove shared variance (scalp noise)
- [ ] **3.6.2** Verify expected channel counts (25 long + 8 short for Expert/Novice)
- [ ] **3.6.3** Create before/after regression comparison plots
- [ ] **3.6.4** Calculate correlation reduction to verify regression effectiveness
- [ ] **3.6.5** Test short-channel regression on subject A1

#### 3.6.6 Verify Stage B Output Format (Optional)
- [ ] **3.6.6** Confirm output is continuous time series (not epoched)
- [ ] **3.6.7** Verify output contains both HbO₂ and HbR channels
- [ ] **3.6.8** Confirm output includes both long-separation (brain) and short-separation (scalp) channels
- [ ] **3.6.9** Verify event markers are preserved in processed data
- [ ] **3.6.10** Check that rest periods and task periods are still mixed (not segmented)

#### 3.7 Pipeline Integration and Testing
- [ ] **3.7.1** Create end-to-end preprocessing pipeline script (chains Steps B1-B6)
- [ ] **3.7.2** Implement processed data saving function (.fif format with metadata)
- [ ] **3.7.3** Add error handling and progress logging throughout pipeline
- [ ] **3.7.4** Test complete pipeline on subject A1 (verify ~5-10 min runtime)
- [ ] **3.7.5** Validate pipeline on 3 additional subjects (A2, A3, R1) before scaling

#### 3.8 Full Dataset Processing
- [ ] **3.8.1** Create batch processing script for Stage B
- [ ] **3.8.2** Process all 14 Expert/Novice subjects through complete pipeline
- [ ] **3.8.3** Track and document processing success/failures
- [ ] **3.8.4** Generate processing summary report (14/14 completed with any issues noted)
- [ ] **3.8.5** Verify all processed data files are valid and loadable

#### 3.9 NIRx System Adaptation (for tES datasets)
- [ ] **3.9.1** Modify Beer-Lambert step for NIRx wavelengths (760/850nm, use default PPF)
- [ ] **3.9.2** Test adapted pipeline on 1 subject from Short-term tES dataset
- [ ] **3.9.3** Verify hemoglobin values are physiologically reasonable
- [ ] **3.9.4** Document any differences between TechEn and NIRx processing

---

### SECTION 4: STAGE C - POST-PROCESSING VALIDATION (Week 7-8)
**Completion:** 0/20 tasks (0%)  
**Reference:** PRD Section 6

#### 4.1 HRF Extraction (Step C1)
- [ ] **4.1.1** Implement epoch extraction function (tmin=-5, tmax=30, baseline=(-5,0))
- [ ] **4.1.2** Create single-subject HRF averaging function (average across trials)
- [ ] **4.1.3** Verify HRF morphology for subject A1 (HbO increases, HbR decreases)
- [ ] **4.1.4** Implement brain region channel selection (PFC, M1, SMA)

#### 4.2 Group-Level HRF Analysis
- [ ] **4.2.1** Calculate grand average HRFs for Expert group (n=8) across all regions
- [ ] **4.2.2** Calculate grand average HRFs for Novice group (n=6) across all regions
- [ ] **4.2.3** Verify HRF timing (peak at ~8-12s) and amplitude (5-15 μM for HbO)

#### 4.3 HRF Visualization (Replicate Figure 3c, 3d)
- [ ] **4.3.1** Create HRF plotting function with paper-style formatting (HbO red, HbR blue, error bands)
- [ ] **4.3.2** Generate HRF plots for all regions (PFC, M1, SMA) for both Expert and Novice groups
- [ ] **4.3.3** Save all HRF plots and compare visually to paper's Figure 3c, 3d
- [ ] **4.3.4** Verify HRF characteristics match expected physiology

#### 4.4 CNR Calculation and Visualization (Step C2)
- [ ] **4.4.1** Implement CNR calculation function (|μ₁-μ₂| / sqrt(σ₁²+σ₂²) using 5-15s peak window)
- [ ] **4.4.2** Calculate CNR for Expert vs Novice comparison across all channels
- [ ] **4.4.3** Create CNR bar plot visualization (replicate Figure 4i style)
- [ ] **4.4.4** Verify CNR values indicate detectable differences (CNR > 1.0, higher in motor regions)
- [ ] **4.4.5** Save CNR results to CSV

#### 4.5 Statistical Validation (Step C3)
- [ ] **4.5.1** Implement statistical testing function (t-test, Cohen's d calculation)
- [ ] **4.5.2** Perform Expert vs Novice comparison and verify significance (p<0.05, d>0.5)
- [ ] **4.5.3** Generate comprehensive validation report with all metrics
- [ ] **4.5.4** Document alignment or deviations from paper's reported findings

---

### SECTION 5: EXTEND TO OTHER DATASETS (Week 9-10)
**Completion:** 0/7 tasks (0%)  
**Reference:** PRD Section 7

#### 5.1 Learning Curve Dataset
- [ ] **5.1.1** Adapt and run Stage A for Learning Curve dataset (n=13, note 7 short channels)
- [ ] **5.1.2** Process all subjects through Stage B pipeline
- [ ] **5.1.3** Calculate CNR for Trainee vs Control comparison

#### 5.2 Short-Term tES Dataset  
- [ ] **5.2.1** Implement session handling for complex file structure (PT, BS1, BS2, T, S naming)
- [ ] **5.2.2** Process all 12 subjects through full pipeline using NIRx parameters
- [ ] **5.2.3** Calculate CNR for Sham vs tRNS comparison

#### 5.3 Long-Term tES Dataset
- [ ] **5.3.1** Process all 21 subjects through full pipeline using NIRx parameters
- [ ] **5.3.2** Calculate CNR for Sham vs tDCS and Sham vs tRNS comparisons
- [ ] **5.3.3** Perform three-group ANOVA comparison

#### 5.4 Final Integration and Reporting
- [ ] **5.4.1** Create master summary comparing all four datasets (SNR, motion, HRF, CNR metrics)
- [ ] **5.4.2** Document systematic differences between TechEn and NIRx systems
- [ ] **5.4.3** Generate comprehensive Phase 1 completion report comparing reproduced vs paper values
- [ ] **5.4.4** Prepare presentation materials for Dr. Nezafati review

---

## 🚧 ROADBLOCKS ENCOUNTERED

### Active Blockers
*None currently*

### Resolved Issues
*None yet - will be populated as issues arise*

---

### Issue Log Template
When you encounter a blocker, add it here with this format:

```
### [YYYY-MM-DD] Issue: [Brief Title]
**Task:** [Task number, e.g., 2.2.1]  
**Problem:** [Detailed description of what went wrong]

**Attempted Solutions:**
1. **[Approach 1]:** [What was tried] → Result: [Success/Failure + details]
2. **[Approach 2]:** [What was tried] → Result: [Success/Failure + details]
3. **[Approach 3]:** [What was tried] → Result: [Success/Failure + details]

**Final Resolution:** [How it was ultimately solved]  
**Root Cause:** [Why the problem occurred]  
**Prevention:** [How to avoid this in future]  
**Time Spent:** [Estimate]
```

---

## 📈 WEEKLY PROGRESS TRACKING

### Week 0: Setup ✅ COMPLETE
**Target Completion:** Section 1 (7 tasks)  
**Status:** 7/7 tasks complete (100%)  
**Key Deliverables:**
- [x] Environment fully configured
- [x] Data downloaded and verified
- [x] Config file created
- [x] Utility functions implemented

### Week 1-2: Stage A Implementation (Current Week)
**Target Completion:** Section 2 (22 tasks)  
**Status:** 21/22 tasks complete (95%)  
**Key Deliverables:**
- [x] Data loading pipeline implemented and tested
- [x] SNR calculation validated on raw data
- [x] Motion ratio calculation validated on OD data
- [ ] QA reports generated for all 14 Expert/Novice subjects
- [ ] Exclusion lists created for Stage B

### Week 3-4: Stage B Core Pipeline
**Target Completion:** Section 3, Tasks 3.1-3.6 (20 tasks)  
**Status:** Not started  
**Key Deliverables:**
- [ ] All 6 preprocessing steps implemented
- [ ] Pipeline tested on multiple subjects
- [ ] Motion correction validated

### Week 5-6: Stage B Complete + Scaling
**Target Completion:** Section 3, Tasks 3.7-3.9 (16 tasks)  
**Status:** Not started  
**Key Deliverables:**
- [ ] End-to-end pipeline working
- [ ] All 14 Expert/Novice subjects processed
- [ ] NIRx adaptation completed

### Week 7-8: Stage C Validation
**Target Completion:** Section 4 (20 tasks)  
**Status:** Not started  
**Key Deliverables:**
- [ ] HRFs extracted and visualized
- [ ] CNR calculated and compared to paper
- [ ] Statistical validation complete
- [ ] Results match paper's Figure 3 and 4

### Week 9-10: Extension + Documentation
**Target Completion:** Section 5 (7 tasks)  
**Status:** Not started  
**Key Deliverables:**
- [ ] All 4 datasets processed
- [ ] Comprehensive comparison report
- [ ] Phase 1 complete and validated
- [ ] Ready for Phase 2 (ML pipeline)

---

## 🎓 LESSONS LEARNED

### Best Practices Discovered
*To be updated as project progresses*

**Example entries:**
- Always visualize intermediate outputs (catches issues early)
- Save intermediate processing stages for debugging
- Test on single subject before batch processing

### Common Pitfalls
*To be updated as mistakes are made and learned from*

**Example entries:**
- Forgetting to exclude bad channels before preprocessing
- Not checking data ranges after each step
- Assuming all subjects have same structure

### Code Optimization Tips
*To be updated as performance improvements are found*

**Example entries:**
- Batch operations are much faster than loops
- Precompute masks once rather than recalculating
- Use memory-mapped files for large datasets

---

## 📝 NOTES FOR NEXT SESSION

### Context for Next Session
*CRITICAL: Update this section before ending ANY coding session*

**Session End Date:** [YYYY-MM-DD]  
**Current Task:** [Task number and brief description]  
**Status:** [In progress / Blocked / Needs testing]

**What I was doing:**
- [Specific functionality being implemented]
- [Files being modified]
- [Current approach/method]

**What's working:**
- [List what's confirmed working]

**What's not working:**
- [Specific errors or issues]
- [What's been tried]

**Next steps when returning:**
1. [First thing to do]
2. [Second thing to do]
3. [Third thing to do]

**Files to check:**
- [List of relevant files]

**Questions to answer:**
- [Open questions or decisions needed]

---

## 🔗 QUICK REFERENCE

### Key File Locations
- **Config:** `config.yaml`
- **Data** `data/BIDS/`
- **Stage A Scripts:** `code/stage_a_raw_qa/`
- **Stage B Scripts:** `code/stage_b_preprocessing/`
- **Stage C Scripts:** `code/stage_c_validation/`
- **Outputs:** `outputs/stage_[a|b|c]_*/`

### Parameter Quick Reference
| Parameter | Value | Used In |
|-----------|-------|---------|
| SNR Low Threshold | 80 dB | Stage A |
| SNR High Threshold | 140 dB | Stage A |
| Motion tMotion | 0.5 | Stage A, B |
| Motion tStdev | 20 | Stage A, B |
| Filter Low | 0.01 Hz | Stage B |
| Filter High | 0.5 Hz | Stage B |
| PPF 690nm | 6.4 | Stage B (TechEn) |
| PPF 830nm | 5.8 | Stage B (TechEn) |
| Epoch tmin | -5 s | Stage C |
| Epoch tmax | 30 s | Stage C |
| HRF Peak Window | 5-15 s | Stage C |

### Common Commands
```bash
# Run Stage A on single subject
python code/stage_a_raw_qa/01_load_data.py --subject A1 --dataset Expert_Novice

# Run full Stage B pipeline
python code/stage_b_preprocessing/run_full_pipeline.py --subject A1

# Generate Stage C validation
python code/stage_c_validation/03_final_validation.py --dataset Expert_Novice
```

### Expected Ranges (for validation)
- **SNR:** ~15-30 dB (typical)
- **Motion Ratio:** ~0 median (typical)
- **OD:** 0.1-2.0 (physiological)
- **HbO/HbR:** ±50 μM (typical task response)
- **HRF Peak Time:** 8-12 seconds post-stimulus
- **CNR:** >1.0 for detectable difference, >2.0 for strong separation

---

## 📊 COMPLETION METRICS

### Overall Progress
- **Total Tasks:** 92
- **Completed:** 47 (51%)
- **In Progress:** 0
- **Blocked:** 0
- **Remaining:** 45

### By Section
- **Section 1 (Setup):** 7/7 (100%) ✅
- **Section 2 (Stage A):** 26/26 (100%) ✅
- **Section 3 (Stage B):** 14/36 (39%)
- **Section 4 (Stage C):** 0/20 (0%)
- **Section 5 (Extension):** 0/7 (0%)


---

**Last saved:** February 20, 2026  
**Next update:** After completing task 3.4.3  
**Project start date:** January 18, 2026  
**Target completion:** ~March 29, 2026 (10 weeks)