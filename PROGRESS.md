# PROGRESS.md - ASTaUND Phase 1 Implementation

**Last Updated:** January 19, 2026  
**Current Week:** Week 1-2 (Stage A)  
**Current Stage:** Section 2.3 Complete - Motion Ratio Calculation Working  
**Next Milestone:** Section 2.4 - QA Report Generation (5 tasks)

---

## 🎯 PROJECT STATE OF THE UNION

### Current Status
- **Phase:** Phase 1 - Reproduction and Validation
- **Active Stage:** Section 2.3 Complete - Motion Ratio Calculation Working
- **Completion:** 21% (19/92 tasks)
- **Blockers:** None
- **Latest Achievement:** Motion ratio calculation on OD validated (median: 0.00%, mean: 0.11%, matches paper's ~0 median from Figure 4)

### Recent Changes
- [2026-01-19] ✅ Completed Section 2.3 - Motion Ratio Calculation on OD (5/5 tasks)
- [2026-01-19] Implemented 06_calculate_motion_ratio.py - calculates motion ratio on OD (NOT raw intensity)
- [2026-01-19] Tested on A1: Median 0.00%, Mean 0.11% - matches paper's Figure 4 (~0 median for good data)
- [2026-01-19] Uses Homer3 parameters: AMPthresh=5 OD, STDEVthresh=20x, tMotion=0.5s, tMask=0.5s
- [2026-01-19] All 66 channels have <10% motion (100% pass rate) - confirms A1 is excellent quality
- [2026-01-19] ✅ Completed Section 2.2 - SNR Calculation (3/3 tasks)
- [2026-01-19] Implemented 05_calculate_snr.py - calculates SNR on raw intensity (20*log10(mean/std))
- [2026-01-19] Tested on A1: Mean SNR 17.17 dB (range: 2.50-37.30 dB) - matches paper's Figure 4 (~15-30 dB)

### Next Steps
1. Begin Section 2.4 - QA Report Generation (5 tasks)
2. Then Section 2.5 - Exclusion List Management (4 tasks)
3. Then Section 2.6 - Dataset-Level Operations (5 tasks)

---

## 📋 DESIGN REQUIREMENTS SUMMARY

### Critical Parameters (Must Match Paper Exactly)
- **SNR Threshold:** 80-140 dB
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
- [ ] Stage A plots match paper's Figure 4 (SNR + Motion Ratio)
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
- [ ] A1 processed successfully
- [ ] FLS1_Session1 processed successfully (verify 7 short channels handled)
- [ ] S1_0-PT processed successfully (verify different sampling rate handled)
- [ ] All 3 produce reasonable SNR/motion metrics
- [ ] Document any system-specific code needed

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
| *None yet* | | | |

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
**Completion:** 12/22 tasks (55%)  
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

#### 2.4 QA Report Generation (Step A4)
- [x] **2.4.1** (2026-02-17) Create comprehensive metrics DataFrame combining SNR, motion ratio, QA quality flags — saved A1_qa_metrics.csv (66 channels, 13 cols: snr_db, motion_ratio, snr_quality_flag, motion_quality_flag, qa_note; 33 acceptable / 24 low / 9 high SNR; 0 excluded — all channels retained for QA transparency)
- [ ] **2.4.2** Implement SNR box plot visualization (white bars, red for bad channels, lower expected threshold lines at ~10 dB low quality, higher expected threshold lines at ~30 dB high quality)
- [ ] **2.4.3** Implement motion ratio box plot visualization
- [ ] **2.4.4** Create summary statistics calculator and console report generator
- [ ] **2.4.5** Generate complete Stage A report for subject A1 and verify plots look similar to Figure 4

#### 2.5 Signal Amplitude Assessment (for Stage B exclusion)
- [ ] **2.5.1** Calculate signal amplitude in dB for all channels (this is separate from SNR)
- [ ] **2.5.2** Identify channels outside 80-140 dB signal amplitude range (for Stage B exclusion)
- [ ] **2.5.3** Create exclusion list JSON: {subject_id, bad_channels[], good_channels[], exclusion_reasons{}}
- [ ] **2.5.4** Save to outputs/stage_a_qa/excluded_channels/ (NOTE: channels NOT excluded yet, just documented)

#### 2.6 Dataset-Level Operations
- [ ] **2.6.1** Implement dataset aggregation function (combine all subjects, calculate channel-wise statistics)
- [ ] **2.6.2** Create aggregated visualization plots with error bars (replicate Figure 4 style)
- [ ] **2.6.3** Process all 14 Expert/Novice subjects through Stage A pipeline
- [ ] **2.6.4** Generate dataset-level QA report and compare to paper's Figure 4
- [ ] **2.6.5** Document any problematic subjects and verify completion (14/14 processed)

#### 2.7 Stage A Validation Checkpoint (Optional)
- [ ] **2.7.1** Verify NO channels have been excluded (all channels should be in output files)
- [ ] **2.7.2** Confirm SNR values are in 10-30 dB range (NOT 80-140 dB)
- [ ] **2.7.3** Confirm Motion Ratios are in 0.01-0.10 range (1-10%, NOT 10-25%)
- [ ] **2.7.4** Compare Stage A outputs to paper's Figure 4 and verify visual similarity

---

### SECTION 3: STAGE B - PREPROCESSING PIPELINE (Week 3-6)
**Completion:** 0/36 tasks (0%)  
**Reference:** PRD Section 5

#### 3.1 Intensity to Optical Density (Step B1)
- [ ] **3.1.1** Load exclusion list from Stage A (channels failing 80-140 dB signal amplitude threshold)
- [ ] **3.1.2** Mark bad channels in raw.info['bads'] before any preprocessing
- [ ] **3.1.3** Implement OD conversion function (mne.preprocessing.nirs.optical_density)
- [ ] **3.1.4** Add OD range verification (0.1-2.0) and flag anomalies
- [ ] **3.1.5** Test OD conversion on subject A1 and visualize time series for sample channels

#### 3.2 Bandpass Filtering (Step B2)
- [ ] **3.2.1** Implement bandpass filter application (0.01-0.5 Hz, IIR method)
- [ ] **3.2.2** Create before/after filtering comparison visualization
- [ ] **3.2.3** Verify filtering effectiveness with power spectral density plots
- [ ] **3.2.4** Test filtering on subject A1

#### 3.3 Motion Artifact Detection on OD (Step B3)
- [ ] **3.3.1** Adapt motion detection algorithm for OD data (reuse Stage A logic)
- [ ] **3.3.2** Generate motion mask and calculate percentage of motion-corrupted data
- [ ] **3.3.3** Visualize detected motion artifacts on OD traces for subject A1

#### 3.4 Recursive PCA Motion Correction (Step B4)
- [ ] **3.4.1** Implement rPCA algorithm: Apply PCA to motion-corrupted segments only, remove high-variance components (motion), keep low-variance (neural signal), n_components=0.97
- [ ] **3.4.2** Create before/after motion correction comparison visualization
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
**Status:** 12/22 tasks complete (55%)  
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
- **Completed:** 19 (21%)
- **In Progress:** 0
- **Blocked:** 0
- **Remaining:** 73

### By Section
- **Section 1 (Setup):** 7/7 (100%) ✅
- **Section 2 (Stage A):** 12/22 (55%)
- **Section 3 (Stage B):** 0/36 (0%)
- **Section 4 (Stage C):** 0/20 (0%)
- **Section 5 (Extension):** 0/7 (0%)

### By Week
- **Week 0:** 0% complete
- **Week 1-2:** 0% complete
- **Week 3-4:** 0% complete
- **Week 5-6:** 0% complete
- **Week 7-8:** 0% complete
- **Week 9-10:** 0% complete

---

**Last saved:** January 19, 2026  
**Next update:** After completing task 2.3.1  
**Project start date:** January 18, 2026  
**Target completion:** ~March 29, 2026 (10 weeks)