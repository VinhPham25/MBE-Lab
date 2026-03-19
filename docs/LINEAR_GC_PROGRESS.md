# GC_LINEAR_PRD.md — Linear Granger Causality Pipeline

**Reference MATLAB Script:** `GC_freq_main_TRGC_NovExp.m`  
**Reference Toolbox:** GCCA (`cca_pwcausal`), bsmart, MVGC v1.0  
**Input:** `outputs/stage_b_processed/processed_data/expert-novice/sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif`  
**Output Root:** `outputs/stage_d_granger/linear_gc/expert-novice/`  
**Language:** Python 3.9+  
**Status:** Sections 1–3 complete (2026-03-17); Section 4.1–4.2 complete (2026-03-18); bivariate bsmart implementation added (2026-03-18)

---

## 🎯 OBJECTIVE

Implement the frequency-domain pairwise conditional Granger Causality (GC) pipeline in Python as an extension of the existing ASTaUND preprocessing pipeline, operating on the preprocessed `.fif` files produced by Stage B. The GC computation methodology is adapted from `GC_freq_main_TRGC_NovExp.m` (MATLAB reference), but the ROI definitions, channel mappings, and data loading are carried forward directly from the validated Stage C pipeline (`build_region_map_5()` in `code/stage_c_validation/01_hrf_extraction.py`, which now returns abbreviated keys `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']`). The output is a per-subject, per-trial, per-window connectivity matrix (subjects × 20 directed connections × windows) that serves as the validated linear baseline before implementing the nonlinear LSTM-based GC in Phase 2.

> **ROI Note:** This implementation uses the 5 anatomical regions validated in Stage C, which differ from the MATLAB reference script's ROIs. The MATLAB script used `[LPFC, RPFC, LPMC, RPMC, SMA]` (bilateral motor cortex). This pipeline uses `[LPFC, CPFC, RPFC, LM1, SMA]` — reflecting the actual optode coverage of the ASTaUND TechEn CW6 cap and the already-validated Stage C channel assignments. These are the abbreviated key names returned by `build_region_map_5()` as of the Stage D update. This is an intentional anatomical choice, not a deviation. Any comparison of GC results against the MATLAB paper must account for this difference.

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

> **Deviation from MATLAB reference:** The MATLAB script used `[LPFC, RPFC, LPMC, RPMC, SMA]`. This pipeline replaces `LPMC` and `RPMC` (bilateral motor cortex) with `CPFC` and `LM1` to match the validated Stage C ROI map. The GC computation itself is unchanged — only the region labels and channel assignments differ. `build_region_map_5()` now returns abbreviated keys directly; `03_hrf_visualization.py` maintains its own local copy with full-name keys for plot labels.

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
| Sampling frequency | 25 Hz | ASTaUND TechEn CW6. All Stage B `.fif` files except R9 are natively at 25 Hz. **R9's `.fif` is at 50 Hz** (hardware clock of split-recording session not corrected in Stage B). `load_preprocessed_fif()` detects `sfreq != 25 Hz` and resamples R9 in-memory to 25 Hz — file on disk is unchanged. Same approach used by Stage C `_harmonise_evoked()`. |
| Sliding window size | 54 s × 25 Hz = 1350 samples; MATLAB inclusive slice gives **1351 samples** (`Y(:,w-window_size:w)`) | MATLAB: `window_size = 54*25` |
| Window stride | 1350 samples (**no overlap**) | MATLAB: `sT = window_size` |
| VAR model order | **20** | MATLAB: `cca_pwcausal(...,20,20,...)` |
| GC frequency range | 0–0.1 Hz, step 0.01 Hz | MATLAB: `freq = [0:0.01:0.1]` |
| Neurophysio band | **0.01–0.07 Hz** | MATLAB: `idx1=0.01, idx2=0.07` |
| GC averaging | Mean over 0.01–0.07 Hz | MATLAB: `mean(GW(:,:,idx1:idx2),3)` |
| Subject groups | Expert: A1, A2, R7, R8, R9, R11, R12 (n=7); Novice: R1–R6 (n=6) | MATLAB subject lists; **R10 explicitly excluded** — trailing zeros in 690nm channels cause NaN after bandpass filter (Stage B quality flag; anomalously high rPCA var_removed: HbO 87%, HbR 99%) |

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
outputs/stage_d_granger/linear_gc/expert-novice/
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
- [x] **1.1.1** (2026-03-17) Add GC dependencies to environment: `statsmodels`, `scipy`; verify VAR import works with `from statsmodels.tsa.vector_ar.var_model import VAR`; verify `scipy.stats.shapiro` and `statsmodels.stats.multitest.fdrcorrection` are importable (`nitime` is **not** required — spectral factorization uses `numpy` directly)
- [x] **1.1.2** (2026-03-17) Create `code/stage_d_granger/` directory and `__init__.py`; create `config` block in `config.yaml` under `stage_d_granger.linear_gc` with all parameters from the Design Requirements table above; also add `stage_d_outputs: "outputs/stage_d_granger"` to the `paths` section of `config.yaml`
- [x] **1.1.3** (2026-03-17) Create `code/stage_d_granger/utils_gc.py` — shared utility functions: `load_preprocessed_fif(subject_id, study)`, `get_event_onsets(raw)`, `get_subject_group(subject_id)` returning `'Expert'` or `'Novice'`, and `get_connection_labels()` returning the 20 directed connection strings in row→col order for `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']`

---

### SECTION 2: ROI SIGNAL EXTRACTION

This mirrors the MATLAB section that builds `Data_Reg` (5×T) from averaged channels per ROI. Because the Stage C pipeline has already validated the channel-to-ROI mapping via `build_region_map_5()` in `code/stage_c_validation/01_hrf_extraction.py`, there is no new channel mapping work required — that function is imported directly.

#### 2.1 Import Stage C ROI Mapping
- [x] **2.1.1** (2026-03-17) Confirmed `build_region_map_5()` returns abbreviated keys `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']` with base channel names (no ` hbo` suffix). A1 result: LPFC(2), CPFC(2), RPFC(2), LM1(16), SMA(2) = 24 total channels, all 5 ROIs populated, no empty lists. A1 bad channels (`S5_D21`, `S6_D22`, `S8_D24`) are all short-sep and already excluded by `_is_short_sep()` before the region map is built.
- [x] **2.1.2** (2026-03-17) Added `get_roi_channel_map(raw) -> dict` to `utils_gc.py`: wraps `build_region_map_5()` via importlib, filters `raw.info['bads']`, appends ` hbo` suffix. Returns `{'LPFC': ['S1_D1 hbo', ...], ...}` with full MNE channel names. Logs warnings per-ROI for any bad channels dropped; logs ERROR if a full ROI is emptied. A1 verified: 5/5 ROIs populated, 24 HbO channels total, 0 empty ROIs.

#### 2.2 ROI Extraction Function
- [x] **2.2.1** (2026-03-17) Implemented `extract_roi_signals(raw)` in `01_extract_roi_signals.py`: calls `get_roi_channel_map(raw)`, averages HbO channels per ROI via `np.nanmean`, returns `(Data_Reg, valid_roi_names)`.
- [x] **2.2.2** (2026-03-17) NaN-row removal: any ROI with all channels bad gets an all-NaN row inserted and then removed with warning; `valid_roi_names` tracks which ROIs survive.
- [x] **2.2.3** (2026-03-17) A1 validated: `Data_Reg.shape == (5, 26500)` ✓, 0 NaN rows ✓, data range `[-4.6, +5.7] μM` (physiologically plausible). Plot saved to `roi_signals/sub-A1_roi_signals.png`.
- [x] **2.2.4** (2026-03-17) Batch complete — 13/13 subjects, 0 failed. Key findings: **R9** recorded at 50 Hz (confirmed — 477 s × 50 Hz = 23,850 samples; GC pipeline uses `.fif` sfreq at load time as documented). **R11** lost LM1+SMA (16+2 channels all bad → 3-ROI output). **R2** lost SMA (2 channels all bad → 4-ROI output). All other subjects 5/5 ROIs retained.

---

### SECTION 3: TRIAL SEGMENTATION

This mirrors the MATLAB `st = find(da.s)` trigger detection and `Y = Data_Reg(:,st(stt):st(stt+1))` trial extraction.

#### 3.1 Stimulus Trigger Extraction
- [x] **3.1.1** (2026-03-17) Implemented `get_trial_segments(raw) -> list[tuple[int,int]]` in `02_trial_segmentation.py`: reads all MNE events via `get_event_onsets(raw)`, sorts by sample, pairs them 2-at-a-time `(id=1 start, id=2 end)` — exact mirror of MATLAB `st = find(da.s)` + `for stt = 1:2:(size_st-1)`; validates each pair's id order and raises `RuntimeError` on malformed annotations; includes window-feasibility check (warns if trial < 54 s). Window size auto-scales to `window_size_s × sfreq` so R9 (50 Hz) correctly uses 2700-sample windows = 54 s.
- [x] **3.1.2** (2026-03-17) A1 validated: 5 trials, durations `[88, 104, 116, 124, 138]` s — all within the 1–3 min physiological range. `dur_s == dur_MATLAB` exactly (`(offset-onset)/sfreq == (offset-onset)/25` for A1 at 25 Hz). Total 8 non-overlapping windows. No SHORT flags.
- [x] **3.1.3** (2026-03-17) Trial duration logging implemented in `save_trial_durations()`; saves `sub-{id}_trial_durations.csv` with columns `[trial, onset_sample, offset_sample, duration_s, duration_samples, n_windows, short_flag]`; flags trials < `window_size_s` with "SHORT". Batch complete — 13/13 subjects OK. Key per-subject findings:
  - **R9** (50 Hz .fif → resampled to 25 Hz in-memory by `load_preprocessed_fif()`): 2 trials only (vs 5 for all others); durations 99 s and 105 s; 2 total windows of 1350 samples (54 s × 25 Hz) — uniform with all other subjects after resampling.
  - **R4**: 9 triggers found (5 trial_start, 4 trial_end — missing final trial-end annotation); code warns "odd number of triggers" and uses the 4 complete pairs → 4 trials, 7 total windows.
  - **R3** / **R6**: notably long trials (195–277 s / up to 343 s) → 20 and 16 total windows respectively.
  - All subjects: zero SHORT flags (every trial ≥ 88 s ≥ 54 s window minimum).

---

### SECTION 4: CORE GC COMPUTATION — `cca_pwcausal` EQUIVALENT

This is the central translation task. MATLAB's `cca_pwcausal(Y, 1, NL, 20, 20, freq, 1)` computes pairwise-conditional, frequency-domain Granger Causality using a VAR(20) model. The function returns `GW` (5×5×length(freq)) — the GC spectrum matrix.

#### 4.1 VAR Model Fitting
- [x] **4.1.1** (2026-03-18) Implemented `fit_var_model(Y, order=20, stationarity_check, adf_pvalue_threshold, trend)` in `03_linear_gc.py`. Takes `Y` of shape `(n_rois, T)`, transposes to `(T, n_rois)` for statsmodels, fits `VAR(order)` using `model.fit(maxlags=order, ic=None, trend='c')` — `ic=None` forces exact order, `trend='c'` includes a constant matching GCCA toolbox convention. Returns `(VARResults, differenced_rois_list)`. Key attributes used downstream: `coefs` (order × n_rois × n_rois), `sigma_u` (n_rois × n_rois), `resid` ((T−order) × n_rois), `.aic`, `.bic`.
- [x] **4.1.2** (2026-03-18) ADF stationarity pre-check implemented; controlled by `config.yaml: stage_d_granger.linear_gc.var.stationarity_check`. **Deviation from plan (confirmed intentional):** `stationarity_check` is set to `false` in config. Testing on A1 showed 4/5 ROIs were non-stationary under ADF — a systematic finding because short 54-second fNIRS windows look like slow random walks to the ADF test. First-differencing would change what quantity the GC measures (differenced dynamics ≠ raw hemodynamic dynamics), so the check is disabled to match the MATLAB `cca_pwcausal` behavior (no pre-check). The ADF + differencing code is fully implemented and re-activatable by toggling the config flag.
- [x] **4.1.3** (2026-03-18) Validated on A1 Trial 1, Window 1 (1351 samples). All assertions pass: `coefs.shape == (20, 5, 5)` ✓, `sigma_u.shape == (5, 5)` ✓, `resid.shape == (1331, 5)` ✓. AIC = −262.22, BIC = −260.25. Window shape confirmed `(5, 1351)` — matching the MATLAB-inclusive 1351-sample slice.

  > **Window size note:** MATLAB's `Y(:,w-window_size:w)` with `window_size=1350` produces **1351** samples (1-indexed inclusive). Python replicates this as `Y[:, w-window_size : w+1]`. Config lists `size_samples: 1350` as the base step size, but each window is 1351 samples. The 1-sample overlap per step is a faithful translation of the MATLAB behavior, not an error.

#### 4.2 Frequency-Domain GC Spectrum
- [x] **4.2.1** (2026-03-18) Implemented `compute_gc_spectrum(var_result, freq, sfreq)` in `03_linear_gc.py`. Returns `GW` of shape `(n_rois, n_rois, n_freqs)`. **Critical deviation discovered and resolved — diagonal-Σ approximation required:**

  The textbook formula `S_jj(f) = [H(f) Σ H*(f)]_jj` (full cross-spectral density) suffers **catastrophic numerical cancellation** for this dataset. The ASTaUND fNIRS ROI innovations are near-rank-1 (the 5×5 `sigma_u` matrix is effectively rank-1 with row sums ≈ 0 — all ROIs share a dominant common hemodynamic mode after Stage B preprocessing). Combined with the near-unit-root transfer function (`|H| ~ 7.46×10⁶` at 0.05 Hz — expected for slow hemodynamic signals in 54-second windows), the off-diagonal cross-terms in `H Σ H*` nearly completely cancel, producing `S_jj = 8.55×10⁻¹²` while `σ_src · |H_ji|² = 0.0139` — a factor of **1.6×10⁹** larger. This makes the denominator negative and GC = 0 for all connections.

  **Fix — diagonal-Σ approximation (implemented):**
  ```python
  S_jj(f) = Σ_k  σ_kk · |H(f)[j,k]|²     # only diagonal of Σ
  ```
  This is a vectorized dot product `H_abs2[j,:] @ sigma_diag` with no cancellation. It is exact when innovations are uncorrelated (standard assumption). For correlated innovations, it measures each ROI's *unique* noise contribution to the target spectrum. This is consistent with the progress doc formula notation `Σ_ii.noise(f)` = scalar noise variance, and matches the intent of the Geweke (1982) diagonal-Σ formulation. A1 Trial 1 Window 1 result after fix: `GW range [0.000, 1.015]`, off-diag mean = 0.266 — physiologically meaningful and in expected log-GC range (0–5).

  > **Comparison to MATLAB cca_pwcausal:** The MATLAB reference data uses Homer3 units (μM·mm, not mol/L). While unit-scaling does not eliminate the near-rank-1 structure of Σ, the MATLAB data likely has lower inter-ROI innovation correlation due to different channel layouts and preprocessing. The diagonal-Σ approximation is the most practical choice that avoids catastrophic cancellation while preserving directional GC interpretability. Future work could explore Wilson spectral factorization (Barnett & Seth MVGC toolbox approach) for an exact formula.

- [x] **4.2.2** (2026-03-18) Implemented `_compute_transfer_function(coefs, freq, sfreq)` as a private helper. Uses `np.einsum('kij,kf->fij', coefs, phase)` to vectorise the lag-polynomial sum over all frequencies simultaneously, then `np.linalg.inv` (broadcast over the frequency axis). Returns `H` of shape `(n_freqs, n_rois, n_rois)`. `sigma_u` is taken directly from `VARResults.sigma_u` (statsmodels OLS residual covariance estimator) — no separate computation from residuals needed.
- [x] **4.2.3** (2026-03-18) Validated on A1 Trial 1, Window 1. All assertions pass: `GW.shape == (5, 5, 11)` ✓, diagonal all-zero ✓, all values ≥ 0 ✓. Physiological snapshot (0.01–0.07 Hz band): LPFC→ other regions ~0.47–0.82 (PFC as causal driver); RPFC→ other regions ~0.38–0.92; CPFC/LM1/SMA→ others <0.04 (receivers). LPFC→CPFC is ~35× stronger than CPFC→LPFC — meaningful directional asymmetry confirmed.

- [x] **4.2.4** (2026-03-18) Implemented `compute_gc_spectrum_bivariate(Y, order, freq, sfreq)` — a faithful translation of bsmart's `pwcausal.m` that underlies `cca_pwcausal` (see **D5** below). For each of the 10 unique pairs (src, tgt), a separate **bivariate** VAR(order) is fitted on `Y[[src,tgt], :]`. The spectral GC formula uses the **Schur complement** of the 2×2 noise covariance as the source variance:

  ```python
  e_src = Σ2[0,0] − Σ2[0,1]² / Σ2[1,1]   # ≡ L[0,0]² from lower Cholesky of Σ2
  S2_f  = H2_f @ Σ2 @ H2_f.conj().T        # 2×2 only (4 terms, stable)
  GW[src, tgt, f] = log( Re(S2_f[1,1]) / (Re(S2_f[1,1]) − |H2_f[1,0]|² · e_src) )
  ```

  A1 Trial 1 Window 1 bivariate results: `GW range [0.000, 5.358]`, off-diag mean = 0.936. Side-by-side comparison plot saved to `visualizations/sub-A1_Tr1_Win1_gc_spectrum_comparison.png`. The bivariate values are systematically higher because the 2×2 Σ2 per pair is much better conditioned than the full 5×5 Σ, yielding a larger ratio of total to intrinsic power. Directional asymmetries are preserved (LPFC→LM1 ~2.25 vs LPFC→RPFC ~0.29; CPFC→SMA ~3.4 vs SMA→CPFC ~0.65).

#### 4.3 Pairwise Conditional GC Entry Point
- [ ] **4.3.1** Implement `compute_pairwise_conditional_gc(Y_window, order=20, freq=None, sfreq=25) -> np.ndarray` in `03_linear_gc.py` as the single entry-point function mirroring `cca_pwcausal(Y_windowed, 1, NL, 20, 20, freq, 1)`. It calls `compute_gc_spectrum_bivariate(Y_window, order, freq, sfreq)` directly and returns `GW` of shape `(n_rois, n_rois, len(freq))`. Set default `freq = np.arange(0, 0.11, 0.01)`. Note: the multivariate `fit_var_model` + `compute_gc_spectrum` path is no longer called from this function; `compute_gc_spectrum_bivariate` does its own internal bivariate fitting per pair.
- [ ] **4.3.2** Document in a docstring that this implements bivariate unconditional GC (not full pairwise-conditional), matching the actual `cca_pwcausal`/bsmart behavior. Include a note that the Schur complement handles instantaneous innovation cross-correlation, and that the function does not condition on the remaining n−2 channels.

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
- [ ] **5.2.2** Save `B_Exp.npz` and `B_Nov.npz` to `outputs/stage_d_granger/linear_gc/expert-novice/group/`; also save `gc_summary.json` with: n_subjects per group, n_trials, n_windows per trial, connection labels, processing parameters

#### 5.3 Single-Subject Validation
- [ ] **5.3.1** Run `process_subject_gc('A1')` end-to-end; print shape of output connectivity array; confirm it has the expected number of trials (should match trial count from `02_trial_segmentation.py`); print mean GC per connection across all windows
- [ ] **5.3.2** Run on all 13 GC-eligible subjects (R10 excluded; see Stage B quality flag); confirm 13/13 succeed; log any subjects with reduced window counts (due to short trials or NaN ROIs); save batch processing log to `gc_batch_log.json`

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

> **Note:** `nitime` is **not** a dependency. It was listed in an early draft but is unused — the spectral factorization uses `numpy` and the VAR transfer function formula directly.

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

- **Total Tasks:** 35
- **Completed:** 19
- **In Progress:** 0
- **Remaining:** 16

### By Section
- **Section 1 (Setup):** 3/3 ✅
- **Section 2 (ROI Extraction):** 6/6 ✅
- **Section 3 (Trial Segmentation):** 3/3 ✅
- **Section 4 (Core GC):** 7/13 (4.1 + 4.2 complete including bivariate implementation; 4.3–4.5 pending)
- **Section 5 (Subject Pipeline):** 0/5
- **Section 6 (Statistics):** 0/4
- **Section 7 (Visualization):** 0/4
- **Section 8 (Batch Runner):** 0/2

---

## 🔗 REFERENCE NOTES

### ROI Divergence from MATLAB Reference
This pipeline intentionally uses different ROIs from `GC_freq_main_TRGC_NovExp.m`. The MATLAB script's `[LPFC, RPFC, LPMC, RPMC, SMA]` mapping was hand-tuned to the Homer3 `.nirs` file channel ordering of a specific recording session. This pipeline uses `[LPFC, CPFC, RPFC, LM1, SMA]` from `build_region_map_5()`, which was derived from the actual 3D optode coordinates in the ASTaUND BIDS files and validated against physiology in Stage C (M1 HbO peak +0.506 μM, SMA HbO peak +0.446 μM). This is the more principled choice. The bilateral motor cortex coverage (LPMC/RPMC) in the MATLAB script is not reproducible from the ASTaUND optode layout without making assumptions about which channels physically cover right motor cortex.

### MATLAB `cca_pwcausal` Behavior (verified from source, 2026-03-18)

`cca_pwcausal` from the GCCA toolbox (Seth 2010) is a **thin wrapper** around the bsmart toolbox function `pwcausal.m` (Chen, Bressler & Ding 2006). The actual GC computation is entirely inside `pwcausal`. Key facts verified from source code:

1. **Bivariate models, not multivariate.** For each pair (i, j), `pwcausal` extracts `y = [X[i,:], X[j,:]]` and fits an independent bivariate VAR(p) using the Morf-LWR algorithm (`armorf`). There is no shared multivariate VAR — each pair gets its own 2×2 model, coefficients, and noise covariance.

2. **Schur complement, not diagonal Σ.** Source contribution to spectral GC uses the Schur complement of the 2×2 noise covariance: `e_src = Z2[0,0] − Z2[0,1]² / Z2[1,1]`. This equals `L[0,0]²` from the lower Cholesky factor of Z2 — so the "Cholesky of Σ" description is algebraically correct for the 2×2 case.

3. **Numerical stability from 2×2 products.** S_jj(f) = [H2 Σ2 H2*]_jj has only 4 product terms (2×2 matrix). Near-rank-1 Σ in the bivariate case is much rarer than in the full 5×5 case; each pair's bivariate model captures their shared variance in the AR coefficients, leaving less correlated residuals.

4. **"Pairwise conditional" is a misnomer.** The name refers to conditioning on instantaneous innovation cross-correlation (via Schur complement), not on the other n−2 channels. This is **unconditional bivariate GC**.

5. **MATLAB GW convention is target-major.** `GW_MATLAB(i,j)` = GC from j→i (column j causes row i). The Python convention is source-major: `GW_Python[src, tgt]` = GC from src→tgt. Therefore `GW_Python = GW_MATLAB.T` (transposed in the first two dims).

Python implementation: `compute_gc_spectrum_bivariate(Y, order, freq, sfreq)` in `03_linear_gc.py`.

### Section 4 Implementation Deviations (recorded 2026-03-18)

#### D1 — Stationarity check disabled (4.1.2)
The ADF stationarity pre-check is implemented but disabled via `config.yaml: stationarity_check: false`. On A1, 4/5 ROIs failed ADF in a 54-second window (p ~ 0.2–0.88) because short fNIRS windows behave like slow random walks to the test. First-differencing is mathematically valid but changes the GC interpretation from raw hemodynamic dynamics to differenced dynamics, diverging from the MATLAB reference which applies no pre-check. The code is fully functional and can be re-enabled by toggling the config flag for methodological exploration.

#### D2 — VAR returns tuple not VARResults (4.1.1)
`fit_var_model()` returns `(VARResults, differenced_rois_list)` rather than bare `VARResults`. This accommodates the stationarity-check bookkeeping without altering the VARResults object itself. All downstream callers unpack with `var_result, _ = fit_var_model(...)`.

#### D3 — Window is 1351 samples, not 1350 (4.1.3)
MATLAB's `Y(:,w-window_size:w)` produces 1351 samples (1-indexed inclusive). Python replicates as `Y[:, w-window_size:w+1]`. The config value `size_samples: 1350` is the stride, not the window length. This is a faithful translation of the MATLAB inclusive-index behavior, confirmed by the `(5, 1351)` window shape in the validation output.

#### D4 — Diagonal-Σ approximation for spectral density (4.2.1)
The standard Geweke formula `S_jj = [H Σ H*]_jj` undergoes catastrophic numerical cancellation for this dataset. The ASTaUND ROI innovations are near-rank-1 (the row-sum of `sigma_u` ≈ 0 — a dominant common mode is present in all 5 ROI residuals), and the VAR transfer function is near-unit-root (`|H| ~ 7.46×10⁶` at 0.05 Hz). Their combined effect: each term in the 25-element double sum `H_jk Σ_kl H*_jl` is ~0.0056, but they nearly completely cancel, producing `S_jj = 8.55×10⁻¹²` while `σ_src|H_ji|² = 0.014` — a 10⁹× violation of the theoretical constraint `S_jj ≥ σ_src|H_ji|²`.

**Resolution:** Replace `S_jj = [H Σ H*]_jj` with the diagonal-Σ approximation:
```
S_jj(f) = Σ_k  σ_kk · |H(f)[j,k]|²
```
This is a stable dot product, exact for uncorrelated innovations. For the highly correlated innovations in fNIRS data, it measures each ROI's unique noise contribution to the target's spectrum. GC values with fix: range [0, 1.015], off-diag mean 0.266, with clear directional asymmetry (LPFC→CPFC ~35× stronger than reverse). Documented in code with a detailed comment.

**Status:** Retained in `compute_gc_spectrum(var_result, ...)` for comparison. The bivariate approach in D5 is now the default for `compute_pairwise_conditional_gc()`.

#### D5 — cca_pwcausal uses bivariate VAR per pair, not multivariate (4.2.4) — CRITICAL CORRECTION

After verifying the `cca_pwcausal`/bsmart source code (2026-03-18), the original assumption that MATLAB uses one multivariate VAR was **incorrect**. The actual implementation fits a separate bivariate VAR(20) for each pair. This is the correct approach for two reasons:

1. **Numerical stability**: the bivariate 2×2 Σ2 is much better conditioned than the full 5×5 Σ. The catastrophic cancellation in D4 is a consequence of using a multivariate model — it does not occur in the bivariate approach.

2. **Faithfulness to MATLAB**: `compute_gc_spectrum_bivariate()` replicates the exact bsmart formula including the Schur complement source variance. The bivariate A1 results (max ~5.36, mean ~0.94) are higher than the diagonal-Σ results (max ~1.02, mean ~0.27) because the 2×2 model yields a larger ratio of total to intrinsic spectral power per pair. Directional asymmetries are preserved in both methods.

**Implementation:** `compute_gc_spectrum_bivariate(Y, order, freq, sfreq)` — takes raw `Y` (n_rois × T), fits bivariate VAR for each unique pair (n_rois*(n_rois-1)/2 = 10 models for 5 ROIs), returns `GW` (n_rois × n_rois × n_freqs). NaN channels (R11/R2) are automatically skipped.

**Impact on 4.3:** `compute_pairwise_conditional_gc()` will call `compute_gc_spectrum_bivariate()` directly instead of `compute_gc_spectrum()`.

### Connection Between Linear and Nonlinear GC
The Section 4 `compute_pairwise_conditional_gc()` function is the **only component replaced** when moving to the nonlinear LSTM-based GC. All of Sections 2, 3, 5, 6, and 7 are reused identically. Design Section 4 so the function signature is cleanly swappable:
```python
# Linear
gc_matrix = compute_pairwise_conditional_gc(Y_window, order=20, ...)

# Nonlinear (Phase 2 — same interface)
gc_matrix = compute_lstm_gc(Y_window, model=trained_lstm, ...)
```