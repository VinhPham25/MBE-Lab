# GC_LINEAR_PRD.md — Linear Granger Causality Pipeline

**Reference MATLAB Script:** `GC_freq_main_TRGC_NovExp.m`  
**Reference Toolbox:** GCCA (`cca_pwcausal`), bsmart, MVGC v1.0  
**Input:** `outputs/stage_b_processed/processed_data/expert-novice/sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif`  
**Output Root:** `outputs/stage_e_granger/linear_gc/expert-novice/`  
**Language:** Python 3.9+  
**Status:** Not started

---

## 🎯 OBJECTIVE

Implement the frequency-domain pairwise conditional Granger Causality (GC) pipeline as an extension of the existing ASTaUND preprocessing pipeline, operating on the preprocessed `.fif` files produced by Stage B. The GC computation methodology is adapted from `GC_freq_main_TRGC_NovExp.m` (MATLAB reference), but the ROI definitions, channel mappings, and data loading are carried forward directly from the validated Stage C pipeline (`build_region_map_5()` in `code/stage_c_validation/01_hrf_extraction.py`). The output is a per-subject, per-trial, per-window connectivity matrix (subjects × 20 directed connections × windows) that serves as the validated linear baseline before implementing the nonlinear LSTM-based GC in Phase 2.

> **ROI Note:** This implementation uses the 5 anatomical regions validated in Stage C, which differ from the MATLAB reference script's ROIs. The MATLAB script used `[LPFC, RPFC, LPMC, RPMC, SMA]` (bilateral motor cortex). This pipeline uses `[Left Lat PFC, Central PFC, Right Lat PFC, Left Medial M1, SMA]` — reflecting the actual optode coverage of the ASTaUND TechEn CW6 cap and the already-validated Stage C channel assignments. This is an intentional anatomical choice, not a deviation. Any comparison of GC results against the MATLAB paper must account for this difference.

---

## 📐 DESIGN REQUIREMENTS

### ROI Definitions (from Stage C `build_region_map_5()`)
```python
# Carried forward directly from code/stage_c_validation/01_hrf_extraction.py
ROI_NAMES = ['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']
# Full names:
#   LPFC = Left Lateral PFC
#   CPFC = Central PFC
#   RPFC = Right Lateral PFC
#   LM1  = Left Medial M1
#   SMA  = Supplementary Motor Area
```
5 ROIs → 5×5 directed GC matrix → remove diagonal → **20 directed connections**

> **Deviation from MATLAB reference:** The MATLAB script used `[LPFC, RPFC, LPMC, RPMC, SMA]`. This pipeline replaces `LPMC` and `RPMC` (bilateral motor cortex) with `CPFC` and `LM1` to match the validated Stage C ROI map. The GC computation itself is unchanged — only the region labels and channel assignments differ.

### Directed Connections (20 off-diagonal, row→col convention)
```
LPFC→CPFC, LPFC→RPFC, LPFC→LM1,  LPFC→SMA,
CPFC→LPFC, CPFC→RPFC, CPFC→LM1,  CPFC→SMA,
RPFC→LPFC, RPFC→CPFC, RPFC→LM1,  RPFC→SMA,
LM1→LPFC,  LM1→CPFC,  LM1→RPFC,  LM1→SMA,
SMA→LPFC,  SMA→CPFC,  SMA→RPFC,  SMA→LM1
```

### Critical Parameters (must match MATLAB exactly)
| Parameter | Value | Source |
|---|---|---|
| Sampling frequency | 25 Hz | ASTaUND TechEn CW6 |
| Sliding window size | 54 s × 25 Hz = **1350 samples** | MATLAB: `window_size = 54*25` |
| Window stride | 1350 samples (**no overlap**) | MATLAB: `sT = window_size` |
| VAR model order | **20** | MATLAB: `cca_pwcausal(...,20,20,...)` |
| GC frequency range | 0–0.1 Hz, step 0.01 Hz | MATLAB: `freq = [0:0.01:0.1]` |
| Neurophysio band | **0.01–0.07 Hz** | MATLAB: `idx1=0.01, idx2=0.07` |
| GC averaging | Mean over 0.01–0.07 Hz | MATLAB: `mean(GW(:,:,idx1:idx2),3)` |
| Subject groups | Expert: A1,A2,R7–R12 (n=7); Novice: R1–R6 (n=6) | MATLAB subject lists |

### Python Equivalents of MATLAB Functions
| MATLAB | Python |
|---|---|
| `cca_pwcausal()` | Custom `compute_pairwise_conditional_gc()` using `statsmodels.tsa.VAR` |
| `swtest()` | `scipy.stats.shapiro()` |
| `anova1()` | `scipy.stats.f_oneway()` |
| `mafdr()` | `statsmodels.stats.multitest.fdrcorrection()` |
| `nanmean()` | `np.nanmean()` |

---

## 📁 OUTPUT STRUCTURE

```
outputs/stage_e_granger/linear_gc/expert-novice/
├── roi_signals/
│   └── sub-{id}_roi_signals.npz          # 5×T array per subject
├── gc_raw/
│   └── sub-{id}_gc_raw.npz               # GW: 5×5×freq per trial/window
├── gc_connectivity/
│   └── sub-{id}_gc_connectivity.npz      # 20 connections × windows per trial
├── group/
│   ├── B_Exp.npz                          # subjects × 20 conn × windows (Expert)
│   ├── B_Nov.npz                          # subjects × 20 conn × windows (Novice)
│   └── gc_summary.json
├── statistics/
│   ├── normality_test.csv
│   ├── H1_expert_vs_novice.csv
│   ├── H0_within_trial_stability.csv
│   └── HIV_across_trial_stability.csv
└── visualizations/
    ├── sub-{id}_connectivity_matrix.png
    └── group_gc_comparison.png
```

---

## 📋 TASK BREAKDOWN

---

### SECTION 1: ENVIRONMENT AND CONFIGURATION

#### 1.1 Setup
- [ ] **1.1.1** Add GC dependencies to environment: `statsmodels`, `scipy`, `nitime` (for spectral validation); verify VAR import works with `from statsmodels.tsa.vector_ar.var_model import VAR`
- [ ] **1.1.2** Create `code/stage_e_granger/` directory and `__init__.py`; create `config` block in `config.yaml` under `stage_e_granger.linear_gc` with all parameters from the Design Requirements table above
- [ ] **1.1.3** Create `code/stage_e_granger/utils_gc.py` — shared utility functions: `load_preprocessed_fif(subject_id, study)`, `get_event_onsets(raw)`, `get_subject_group(subject_id)` returning `'Expert'` or `'Novice'`, and `get_connection_labels()` returning the 20 directed connection strings in row→col order for `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']`

---

### SECTION 2: ROI SIGNAL EXTRACTION

This mirrors the MATLAB section that builds `Data_Reg` (5×T) from averaged channels per ROI. Because the Stage C pipeline has already validated the channel-to-ROI mapping via `build_region_map_5()` in `code/stage_c_validation/01_hrf_extraction.py`, there is no new channel mapping work required — that function is imported directly.

#### 2.1 Import Stage C ROI Mapping
- [ ] **2.1.1** Confirm that `build_region_map_5()` in `code/stage_c_validation/01_hrf_extraction.py` returns a dict of the form `{'LPFC': [ch1, ch2, ...], 'CPFC': [...], 'RPFC': [...], 'LM1': [...], 'SMA': [...]}` containing HbO channel name strings; run it on A1 and print the resulting channel lists to verify the 5 regions are fully populated with no empty lists
- [ ] **2.1.2** Add `get_roi_channel_map(raw, study='expert-novice') -> dict` to `utils_gc.py` as a thin wrapper around `build_region_map_5()` that also filters out any channels present in `raw.info['bads']` before returning; this is the single entry point for ROI channel lookup throughout the GC pipeline

#### 2.2 ROI Extraction Function
- [ ] **2.2.1** Implement `extract_roi_signals(raw) -> np.ndarray` in `01_extract_roi_signals.py`: calls `get_roi_channel_map(raw)` to get channel lists per ROI, selects HbO channels only, averages channels within each ROI, returns `Data_Reg` as shape `(5, T)` — mirrors MATLAB `mean(LPFC,1)` per ROI; signal is in μM (MNE haemo units from Stage B output)
- [ ] **2.2.2** Add NaN-row removal logic to `extract_roi_signals()`: if all samples in a ROI are NaN (all contributing channels were bad), remove that ROI from `Data_Reg` and record which ROIs were dropped in a returned `valid_roi_names` list — mirrors MATLAB `Data_Reg(any(isnan(Data_Reg),2),:) = []` and `valid_Ch_names`; log a warning if any ROI is dropped
- [ ] **2.2.3** Validate on A1: confirm `Data_Reg.shape == (5, ~26500)` with no all-NaN rows; plot all 5 ROI time series labelled with `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']` and visually confirm they are smooth HbO traces with plausible hemodynamic fluctuations; save plot to `outputs/stage_e_granger/linear_gc/expert-novice/roi_signals/sub-A1_roi_signals.png`
- [ ] **2.2.4** Implement `run_roi_extraction(study, subjects)` batch function; process all 13 Expert/Novice subjects; save each `Data_Reg` as `sub-{id}_roi_signals.npz` with keys `data` (5×T), `roi_names` (list of valid ROI names), `sfreq`

---

### SECTION 3: TRIAL SEGMENTATION

This mirrors the MATLAB `st = find(da.s)` trigger detection and `Y = Data_Reg(:,st(stt):st(stt+1))` trial extraction.

#### 3.1 Stimulus Trigger Extraction
- [ ] **3.1.1** Implement `get_trial_segments(raw) -> list[tuple[int,int]]` in `02_trial_segmentation.py`: reads MNE annotations/events from the `.fif`, finds trial onset and offset sample indices — mirrors MATLAB `st = find(da.s)` and `for stt = 1:2:(size_st-1)` loop; returns list of `(onset_sample, offset_sample)` tuples
- [ ] **3.1.2** Validate trial segments on A1: print number of trials found, duration of each in seconds; confirm trial durations are physiologically reasonable (1–3 min as per BMES poster); compare to `duration_task` logic in MATLAB (`(st(stt+1)-st(stt))/25`)
- [ ] **3.1.3** Add trial duration logging: compute `duration_sec = (offset - onset) / sfreq` per trial and save to `sub-{id}_trial_durations.csv`; flag any trial shorter than 54 s (too short for even one window) with a warning

---

### SECTION 4: CORE GC COMPUTATION — `cca_pwcausal` EQUIVALENT

This is the central translation task. MATLAB's `cca_pwcausal(Y, 1, NL, 20, 20, freq, 1)` computes pairwise-conditional, frequency-domain Granger Causality using a VAR(20) model. The function returns `GW` (5×5×length(freq)) — the GC spectrum matrix.

#### 4.1 VAR Model Fitting
- [ ] **4.1.1** Implement `fit_var_model(Y: np.ndarray, order: int = 20) -> VARResults` in `03_linear_gc.py`: fits a VAR(order) model on `Y` (n_rois × T) using `statsmodels.tsa.vector_ar.var_model.VAR`; transpose to (T × n_rois) as required by statsmodels; return the fitted `VARResults` object
- [ ] **4.1.2** Add stationarity pre-check to `fit_var_model()`: run ADF test on each ROI signal via `statsmodels.tsa.stattools.adfuller`; if any series is non-stationary (p > 0.05), apply first-differencing and log the occurrence — matches the implicit stationarity assumption of linear VAR/GC
- [ ] **4.1.3** Validate VAR fit on a single A1 window: confirm `model.coefs` shape is `(20, 5, 5)` (lag × rois × rois); print AIC/BIC; confirm no fitting errors on the 1350-sample window

#### 4.2 Frequency-Domain GC Spectrum
- [ ] **4.2.1** Implement `compute_gc_spectrum(var_result, freq: np.ndarray, sfreq: float) -> np.ndarray` in `03_linear_gc.py`: converts VAR coefficients to spectral domain using the transfer function `H(f) = (I - A(f))^{-1}`, computes the spectral GC measure `GW[i,j,f] = log(Σ_jj(f) / (Σ_jj(f) - (H_ij(f)² * Σ_ii.noise(f))))` per the Geweke (1982) / Barnett & Seth (2014) formulation; returns array of shape `(n_rois, n_rois, len(freq))`
- [ ] **4.2.2** Implement the VAR coefficient-to-transfer-function conversion: `A(f) = Σ_k A_k * exp(-2πi*f*k/sfreq)` summed over lags k=1..order; `H(f) = inv(I - A(f))`; compute the noise covariance matrix `Σ` from VAR residuals — this is the spectral factorization step at the core of `cca_pwcausal`
- [ ] **4.2.3** Validate GC spectrum on A1 trial 1 window 1: confirm `GW.shape == (5, 5, 11)` (11 frequency points from 0 to 0.1 in steps of 0.01); confirm diagonal is zero (no self-causality); confirm values are non-negative real numbers; plot the 20 off-diagonal spectra and visually confirm they peak in the 0.01–0.07 Hz range

#### 4.3 Pairwise Conditional GC
- [ ] **4.3.1** Implement `compute_pairwise_conditional_gc(Y_window, order=20, freq=None, sfreq=25) -> np.ndarray` in `03_linear_gc.py` as the single entry-point function mirroring `cca_pwcausal(Y_windowed, 1, NL, 20, 20, freq, 1)`. It calls `fit_var_model()` then `compute_gc_spectrum()` and returns `GW` of shape `(n_rois, n_rois, len(freq))`. Set default `freq = np.arange(0, 0.11, 0.01)`
- [ ] **4.3.2** Add conditioning logic: the "pairwise conditional" aspect means GC from i→j is conditioned on all other ROIs k≠i,j. Confirm your VAR-based implementation is inherently multivariate (all ROIs modeled jointly), which provides the conditioning automatically — document this confirmation in a code comment referencing Barnett & Seth (2014) §2.3

#### 4.4 Sliding Window Loop
- [ ] **4.4.1** Implement `compute_windowed_gc(Y_trial, window_size=1350, stride=1350, order=20, freq=None, sfreq=25) -> dict` in `03_linear_gc.py`: slides over `Y_trial` (5×T) from `w = window_size` to `T` in steps of `stride`; calls `compute_pairwise_conditional_gc()` per window; returns dict `{'Win_1': GW_array, 'Win_2': GW_array, ...}` — mirrors MATLAB `for w = window_size+1:sT:LY` loop
- [ ] **4.4.2** Add minimum-length guard: skip a window if `Y_windowed` contains NaN values in any ROI channel (can occur for subjects with bad ROI channels); log skipped windows with reason

#### 4.5 Frequency Band Averaging
- [ ] **4.5.1** Implement `average_gc_over_band(GW, freq, f_low=0.01, f_high=0.07) -> np.ndarray` in `03_linear_gc.py`: finds indices of `f_low` and `f_high` in `freq` array; returns `np.mean(GW[:,:,idx1:idx2+1], axis=2)` — exactly mirrors MATLAB `mean(GW(:,:,idx1:idx2),3)`; output shape `(n_rois, n_rois)`
- [ ] **4.5.2** Validate band averaging on A1: confirm output shape `(5,5)`; confirm diagonal is zero; print all 20 off-diagonal values and confirm they are in a physiologically reasonable range (typically 0–5 for log-GC)

---

### SECTION 5: SUBJECT-LEVEL PIPELINE AND STORAGE

This mirrors the full MATLAB outer loops: `for m in [Exp, Nov]`, `for s in 1:N`, building `B_Exp` and `B_Nov`.

#### 5.1 Per-Subject GC Pipeline
- [ ] **5.1.1** Implement `process_subject_gc(subject_id, study='expert-novice') -> dict` in `04_subject_gc_pipeline.py`: orchestrates ROI extraction → trial segmentation → windowed GC → band averaging for a single subject; returns nested dict `{trial_id: {window_id: connectivity_matrix_20x1}}` — mirrors MATLAB per-subject logic building `GC_fqmean`
- [ ] **5.1.2** Add diagonal removal inside `process_subject_gc()`: reshape 5×5 matrix to 25 elements, remove indices [0,6,12,18,24] (diagonal), leaving 20 elements in the exact MATLAB connection order — mirrors MATLAB `B_Exp.(trial)(:,[1 7 13 19 25],:) = []`
- [ ] **5.1.3** Save per-subject output as `sub-{id}_gc_connectivity.npz` with keys: `connectivity` (shape: n_trials × 20_connections × n_windows), `trial_ids`, `window_counts`, `roi_names_used`, `params` dict

#### 5.2 Group Array Construction
- [ ] **5.2.1** Implement `build_group_arrays(study='expert-novice') -> tuple[np.ndarray, np.ndarray]` in `04_subject_gc_pipeline.py`: loads all per-subject `.npz` files; stacks into `B_Exp` (n_exp_subjects × 20_conn × n_windows) and `B_Nov` (n_nov_subjects × 20_conn × n_windows) — mirrors MATLAB `B_Exp.(trial).Exp(h,:,Wn+1)` accumulation; pads with NaN for subjects with fewer windows than the max
- [ ] **5.2.2** Save `B_Exp.npz` and `B_Nov.npz` to `outputs/stage_e_granger/linear_gc/expert-novice/group/`; also save `gc_summary.json` with: n_subjects per group, n_trials, n_windows per trial, connection labels, processing parameters

#### 5.3 Single-Subject Validation
- [ ] **5.3.1** Run `process_subject_gc('A1')` end-to-end; print shape of output connectivity array; confirm it has the expected number of trials (should match trial count from `02_trial_segmentation.py`); print mean GC per connection across all windows
- [ ] **5.3.2** Run on all 13 Expert/Novice subjects; confirm 13/13 succeed; log any subjects with reduced window counts (due to short trials or NaN ROIs); save batch processing log to `gc_batch_log.json`

---

### SECTION 6: STATISTICAL TESTING

Directly mirrors the MATLAB statistical sections (normality, H0, H1, H4, HIV).

#### 6.1 Normality Testing
- [ ] **6.1.1** Implement `run_normality_tests(B_Exp, B_Nov) -> pd.DataFrame` in `05_gc_statistics.py`: applies `scipy.stats.shapiro()` to each connection × window cell across subjects; mirrors MATLAB `swtest` loops; returns DataFrame with columns `[group, trial, window, connection, W_stat, p_value, is_normal]`; save to `statistics/normality_test.csv`

#### 6.2 H1 — Expert vs. Novice (Primary Hypothesis)
- [ ] **6.2.1** Implement `test_expert_vs_novice(B_Exp, B_Nov, trial='Tr_1') -> pd.DataFrame` in `05_gc_statistics.py`: for each of the 20 connections × each window, run `scipy.stats.f_oneway(B_Exp[:,conn,win], B_Nov[:,conn,win])`; compute mean and std for each group; mirrors MATLAB H1 section; return DataFrame with F-stat, raw p-value, group means and SDs
- [ ] **6.2.2** Apply FDR correction on the 20 p-values per window using `statsmodels.stats.multitest.fdrcorrection()`; mirrors MATLAB `mafdr()`; add `p_fdr` and `q_value` columns to the DataFrame; save to `statistics/H1_expert_vs_novice.csv`

#### 6.3 H0 — Within-Trial Stability
- [ ] **6.3.1** Implement `test_within_trial_stability(B_Exp, B_Nov) -> pd.DataFrame` in `05_gc_statistics.py`: for each group × trial × connection, run one-way ANOVA across windows (`f_oneway(*[data[:,conn,win] for win in range(n_windows)])`); mirrors MATLAB H0 section testing whether connectivity changes within a single trial; apply FDR correction per trial; save to `statistics/H0_within_trial_stability.csv`

#### 6.4 HIV — Across-Trial Stability
- [ ] **6.4.1** Implement `test_across_trial_stability(B_Exp, B_Nov, window_idx=0) -> pd.DataFrame` in `05_gc_statistics.py`: for each group × connection, collect the first window's GC value across all trials; run ANOVA across trials; mirrors MATLAB HIV section testing whether connectivity stays stable across task repetitions; apply FDR; save to `statistics/HIV_across_trial_stability.csv`

---

### SECTION 7: VISUALIZATION

#### 7.1 Per-Subject Connectivity Matrix
- [ ] **7.1.1** Implement `plot_connectivity_matrix(gc_matrix_5x5, subject_id, trial, window, group)` in `06_gc_visualization.py`: renders a 5×5 heatmap with ROI labels `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']`, masked diagonal, color scale 0–max; save to `visualizations/sub-{id}_Tr{n}_Win{n}_connectivity.png`
- [ ] **7.1.2** Plot connectivity matrices for A1 across all windows of trial 1 in a single multi-panel figure to visualize how connectivity evolves across windows; save as `sub-A1_Tr1_connectivity_evolution.png`

#### 7.2 Group Comparison
- [ ] **7.2.1** Implement `plot_group_gc_comparison(B_Exp, B_Nov, connection_labels)` in `06_gc_visualization.py`: for each of the 20 connections, plot Expert mean ± SEM and Novice mean ± SEM as a bar chart; mark FDR-significant connections (from H1) with an asterisk; save as `group_gc_comparison_Tr1_Win1.png`
- [ ] **7.2.2** Generate a brain-topology connection diagram: render the 5 ROI nodes in their approximate anatomical positions (LPFC/CPFC/RPFC anterior, LM1 lateral, SMA medial) and draw directed arrows scaled by mean GC magnitude, separately for Expert and Novice; save as `group_gc_topology_Expert.png` and `group_gc_topology_Novice.png`

---

### SECTION 8: INTEGRATION AND BATCH RUNNER

#### 8.1 Full Pipeline Script
- [ ] **8.1.1** Create `07_run_linear_gc_pipeline.py`: a CLI batch script that accepts `--study`, `--subjects`, `--trials` flags and runs the full pipeline (Sections 2–6) end-to-end; mirrors the role of `07_preprocess_study.py` in Stage B; writes a `gc_provenance.json` with all parameters, run timestamp, and per-subject success/failure status
- [ ] **8.1.2** Add a `--validate` flag to `07_run_linear_gc_pipeline.py` that runs a quick sanity check: confirms `B_Exp` and `B_Nov` shapes are correct, all GC values are non-negative, no all-zero connectivity windows exist, and FDR-corrected p-values are in [0,1]; prints PASS/FAIL per check

---

## 🔗 KEY DEPENDENCIES

| Library | Version | Use |
|---|---|---|
| `statsmodels` | ≥0.14 | VAR model fitting, FDR correction |
| `scipy` | ≥1.10 | Shapiro-Wilk, f_oneway, signal processing |
| `numpy` | ≥1.24 | All array operations |
| `mne` / `mne-nirs` | existing | `.fif` loading |
| `matplotlib` | existing | Visualization |
| `pandas` | existing | Statistics output DataFrames |

---

## ✅ SUCCESS CRITERIA

| Check | Target |
|---|---|
| ROI extraction | 5 ROI signals extracted for all 13/13 subjects |
| Trial segmentation | Trial count matches BIDS event file per subject |
| GC spectrum shape | `GW.shape == (5, 5, 11)` per window |
| Band-averaged GC | Shape `(5, 5)`, diagonal = 0, values ≥ 0 |
| `B_Exp` shape | `(7, 20, n_windows)` |
| `B_Nov` shape | `(6, 20, n_windows)` |
| H1 Expert vs Novice | At least some connections significant (FDR p < 0.05), particularly in motor/PFC-motor pathways |
| Directionality sanity | SMA→LM1 and LM1→SMA connections show meaningful GC in both groups (bidirectional motor-SMA coupling expected during laparoscopic task); PFC→SMA connections expected to be higher in Experts (top-down motor control) |

---

## 📊 COMPLETION METRICS

- **Total Tasks:** 34
- **Completed:** 0
- **In Progress:** 0
- **Remaining:** 34

### By Section
- **Section 1 (Setup):** 0/3
- **Section 2 (ROI Extraction):** 0/6
- **Section 3 (Trial Segmentation):** 0/3
- **Section 4 (Core GC):** 0/12
- **Section 5 (Subject Pipeline):** 0/5
- **Section 6 (Statistics):** 0/4
- **Section 7 (Visualization):** 0/4
- **Section 8 (Batch Runner):** 0/2

---

## 🔗 REFERENCE NOTES

### ROI Divergence from MATLAB Reference
This pipeline intentionally uses different ROIs from `GC_freq_main_TRGC_NovExp.m`. The MATLAB script's `[LPFC, RPFC, LPMC, RPMC, SMA]` mapping was hand-tuned to the Homer3 `.nirs` file channel ordering of a specific recording session. This pipeline uses `[LPFC, CPFC, RPFC, LM1, SMA]` from `build_region_map_5()`, which was derived from the actual 3D optode coordinates in the ASTaUND BIDS files and validated against physiology in Stage C (M1 HbO peak +0.506 μM, SMA HbO peak +0.446 μM). This is the more principled choice. The bilateral motor cortex coverage (LPMC/RPMC) in the MATLAB script is not reproducible from the ASTaUND optode layout without making assumptions about which channels physically cover right motor cortex.

### MATLAB `cca_pwcausal` Behavior (for validation)
The MATLAB function `cca_pwcausal(Y, 1, NL, 20, 20, freq, 1)` from the GCCA toolbox computes pairwise conditional spectral GC. The key arguments are:
- `Y` — multivariate time series (channels × time)
- `1` — regression mode (1 = VAR)
- `NL` — number of samples
- `20, 20` — VAR model order for forward and backward models
- `freq` — frequency vector
- `1` — include conditioning on all other channels

Your Python `compute_pairwise_conditional_gc()` must produce a `GW` array of the same shape `(n_rois, n_rois, len(freq))` with the same sign convention (positive = causal influence from column → row).

### Connection Between Linear and Nonlinear GC
The Section 4 `compute_pairwise_conditional_gc()` function is the **only component replaced** when moving to the nonlinear LSTM-based GC. All of Sections 2, 3, 5, 6, and 7 are reused identically. Design Section 4 so the function signature is cleanly swappable:
```python
# Linear
gc_matrix = compute_pairwise_conditional_gc(Y_window, order=20, ...)

# Nonlinear (Phase 2 — same interface)
gc_matrix = compute_lstm_gc(Y_window, model=trained_lstm, ...)
```