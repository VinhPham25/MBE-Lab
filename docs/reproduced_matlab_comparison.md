# Preprocessing Pipeline Audit: Python vs. MATLAB Ground Truth

**Date:** March 3, 2026  
**Purpose:** Systematic comparison of Python implementation against Homer3 MATLAB ground truth  
**Status:** Critical discrepancies identified in B4 (rPCA) and B6 (GLM/OLS)

---

## B1 — Intensity → OD (`01_intensity_to_od.py` vs. `hmrR_Intensity2OD.m`)

### MATLAB Formula:
```matlab
dm = mean(abs(d), 1);                          % mean of |intensity|
dod = -log(abs(d) ./ (ones(nTpts,1)*dm));      % -ln(|I| / mean|I|)
```

### Python:
Uses `mne.preprocessing.nirs.optical_density()`, which computes `-ln(I / mean(I))`.

### Verdict: ✅ **Match**
For positive-valued intensity data (fNIRS measurements are always positive), `|I| = I` and both are natural-log formulations. This is numerically equivalent.

---

## B2 — Bandpass Filter (`02_bandpass_filtering.py` vs. `hmrR_BandpassFilt.m`)

This is a **two-part discrepancy**.

### Discrepancy 2a — High-pass cutoff (the paper's "0 Hz"):

The Homer3 function header shows the default as:
```matlab
% PARAMETERS:
% hpf: [0.000]
% lpf: [0.500]
```

And the implementation enforces:
```matlab
if hpf_norm > 0    % <-- high-pass is SKIPPED if hpf = 0
    [z, p, k] = butter(FilterOrder, hpf_norm, 'high');
    ...
end
```

**When `hpf = 0.000` (the default), the high-pass filter is never applied.** The pipeline becomes a pure 3rd-order Butterworth low-pass at 0.5 Hz only. The paper's "0–0.50 Hz" is therefore literally correct — DC to 0.5 Hz, with no high-pass removal.

Your `DataPreprocessing.m` infers `hpf = 0.001`, and the Python config also uses `l_freq: 0.001 Hz` (per `ASTaUND_parameters.md`). This deviates from the Homer3 default behavior. **The paper's 0 for the lower bound is not a rounding of 0.01 Hz — it means no high-pass filtering at all.**

### Discrepancy 2b — Filter order and method:

| Property | MATLAB Homer3 | Python MNE |
|----------|---------------|------------|
| Low-pass order | 3rd-order Butterworth (`filtfilt`) | Auto-determined IIR (MNE default ~5th order) |
| High-pass order | 5th-order Butterworth (`filtfilt`) | Auto-determined IIR |
| Both applied via | `filtfilt` (zero-phase) | MNE `filter()` with `method='iir'` (also zero-phase internally) |

The different filter orders mean the roll-off characteristics are different. A 3rd-order LPF has a gentler roll-off than a 5th-order one. For strict paper fidelity, you'd want a 3rd-order Butterworth applied as a low-pass only.

### Verdict: ⚠️ **Moderate discrepancy**
The Python applies a high-pass component that Homer3 does not, and uses a different filter order. This affects slow-drift behavior and baseline shape.

---

## B3 — Motion Detection (`03_motion_correct.py` vs. `hmrR_MotionArtifactByChannel.m`)

### Discrepancy 3a — `max_diff` padding orientation:

**In MATLAB**, for each lag `ii`:
```matlab
max_diff = max([abs(d((ii+1):end) - d(1:(end-ii))); zeros(ii-1,...) ], max_diff);
%              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^
%              n-ii elements (differences at positions 0..n-ii-1)    ii-1 zeros at END
```

**In Python**:
```python
abs_change = np.abs(od_signal[lag:] - od_signal[:-lag])   # n-lag elements
padded = np.concatenate([np.zeros(lag - 1), abs_change])   # zeros at START
```

MATLAB places the differences at the beginning of the array (positions 0 through n-lag-1), with zeros padding at the end. Python places zeros at the beginning and differences at positions lag-1 through n-2. **This means motion events are time-shifted by ~(lag-1) samples in Python relative to MATLAB.**

In practice this is partially absorbed by the `tMask` buffer expansion (±0.5 s ≈ ±10 samples at 20 Hz), so the practical impact is small but non-zero.

### Discrepancy 3b — Default parameter values:

Looking at the two Homer3 function parameter headers:
- `hmrR_MotionArtifactByChannel` PARAMETERS: `STDEVthresh: 50.0`, `tMask: 1.0`
- `hmrR_MotionCorrectPCArecurse` PARAMETERS: `STDEVthresh: 20.0`, `tMask: 1.0`

Both show `tMask: 1.0` as default, but your `DataPreprocessing.m` and Python both use `tMask = 0.5` (from Ref #36). This is correct per the cited paper, but the Homer3 function defaults are `tMask=1.0`. **If the ASTaUND study used default Homer3 parameters, their `tMask` may have been 1.0 s rather than 0.5 s.**

### Verdict: ⚠️ **Minor-to-moderate discrepancy**
The `max_diff` timing is slightly off, and `tMask` may differ from what was actually used.

---

## B4 — rPCA Motion Correction (`04_rPCA_motion_correct.py` vs. `hmrR_MotionCorrectPCArecurse.m`)

This has **three independent structural discrepancies**, all significant.

### Discrepancy 4a — Iterative vs. single-pass:

**MATLAB Homer3** applies PCA recursively up to `maxIter=5` times:
```matlab
while length(find(tInc{iBlk}==0)) > 0 && ii < maxIter
    [data_dN, svs_ii, nSV] = hmrR_MotionCorrectPCA(data_d, ...);
    tInc = hmrR_MotionArtifact(data_dN, ...);   % re-detect on corrected data
    data_d.Copy(data_dN);                         % corrected data becomes next input
    ii = ii + 1;
end
```

After each PCA step, motion detection runs again on the corrected signal. If motion is still present, another PCA step is applied. **Python applies PCA exactly once.**

### Discrepancy 4b — Scope of correction (flagged segments only vs. full signal):

Homer3's inner function `hmrR_MotionCorrectPCA` (called inside the recursion) performs PCA on the motion-flagged segments, identifies the motion subspace, then **subtracts the motion projection from the entire continuous signal** — both flagged and unflagged time points get the correction applied.

The Python implementation does the opposite:
```python
data[np.ix_(good_idx, motion_idx)] = X_corrected.T   # ONLY flagged time points are modified
```

**Only the samples inside the union motion mask are replaced with their PCA-residuals.** Unflagged samples are never touched. This means any global correlated noise that exists everywhere (not just in flagged windows) is never removed by the Python rPCA.

### Discrepancy 4c — Motion mask construction:

Homer3's `hmrR_MotionCorrectPCArecurse` calls `hmrR_MotionArtifact` (not `hmrR_MotionArtifactByChannel`). `hmrR_MotionArtifact` returns a **single global `tInc` vector** — if any channel is flagged at time t, `tInc[t]=0` for all channels. This is a strict OR-union.

Python builds per-channel masks from `hmrR_MotionArtifactByChannel`, then applies a vote threshold (`min_channels_flagged=10`): **a time point must be simultaneously flagged by at least 10 channels to enter the mask.** This is considerably more conservative than the Homer3 OR-union behavior.

### Discrepancy 4d — `nSV` parameter:

The MATLAB defaults and `DataPreprocessing.m` specify `nSV = 0.97`. The Python config (`ASTaUND_parameters.md`) specifies `n_components: 0.99`. The `ASTaUND_parameters.md` itself flags this as a needed correction: `n_components: 0.99 → 0.97`.

### Verdict: 🔴 **Major discrepancy**
The iterative nature, the scope of correction (full signal vs. flagged-only), and the mask construction strategy are all fundamentally different from Homer3.

---

## B5 — MBLL (`05_mbll.py` vs. `hmrR_OD2Conc.m`)

### Formula comparison:

**MATLAB:**
```matlab
e = GetExtinctions(Lambda);
e = e(:,1:2) / 10;          % /cm → /mm
einv = inv(e'*e) * e';
y2(:,k:k+1) = (einv * (y(:,[idx1 idx2'])./(ones(nTpts,1)*rho*ppf))' )';
```

**Formula:** `ΔC = (ε^T ε)^{-1} ε^T · (ΔOD / (ρ · ppf))`

**Python (MNE):**
```python
mne.preprocessing.nirs.beer_lambert_law(raw_od_corrected, ppf=(6.4, 5.8))
```

**MNE formula:** `ΔC = ε^{-1} · (ΔOD / (ppf · d))`

where `d` is the source-detector separation (distance in meters, converted internally to mm).

**The formulas are structurally equivalent.** Both divide OD by (ppf × distance) before multiplying by the inverted extinction matrix.

### Potential discrepancy — Extinction coefficients:

Homer3's `GetExtinctions(Lambda)` uses the Prahl et al. extinction coefficient database, returning coefficients in units of /cm/(mol/L). MNE has its own built-in extinction coefficient table (from a slightly different source). **If these tables differ at 690 nm and 830 nm, there will be a constant scaling difference in the final concentration values.**

### Potential discrepancy — Distance units:

Homer3 uses `SrcPos/DetPos` in whatever units the SNIRF probe stores them (typically mm for TechEn). MNE reads SNIRF probe positions in meters. If the SNIRF file stores positions in mm but MNE interprets them as meters, the distance ρ would be off by a factor of 1000. However, MNE's `beer_lambert_law` documentation specifies it expects positions in meters and handles the conversion, so this should be handled correctly as long as the SNIRF file stores positions in meters (standard SNIRF convention).

### Special PPF behavior in Homer3:
```matlab
if ~isempty(find(ppf==1))
    ppf = ones(size(ppf));
end
```

If ANY ppf value equals 1, all are set to 1. With `ppf=[6.4, 5.8]` (neither equals 1), this branch is NOT triggered, so it is not relevant here. ✅

### Verdict: ✅/⚠️ **Mostly equivalent, with a minor caveat**
The formula is the same. The main risk is differing extinction coefficients between Homer3 (`GetExtinctions`) and MNE's internal table. This should be empirically verified by comparing the E matrix values at 690 nm and 830 nm.

---

## B6 — GLM/OLS Regression (`06_ols_regression.py` vs. `hmrS_GLM.m`)

This is the **most critical structural discrepancy** in the entire pipeline.

### What Homer3 actually does (from `hmrS_GLM.m`):

The Gaussian basis functions in Homer3 are used to **model the shape of the HRF**, NOT to model the short-channel signal:

```matlab
% idxBasis=1 → Gaussian HRF model:
gms = paramsBasis(1);    % step between Gaussians (spacing)
gstd = paramsBasis(2);   % standard deviation (width)
nB = floor((trange(2)-trange(1))/gms) - 1;   % number of basis functions
tbasis(:,b) = exp(-(tHRF - (trange(1)+b*gms)).^2 / (2*gstd.^2));
% tbasis is defined over the HRF time axis tHRF

% HRF regressor column = convolve stimulus onset with Gaussian basis:
clmn = conv(onset(:,iCond), tbasis(:,b));   % <-- convolved with STIMULUS ONSET
dA(:,iC,iConc) = clmn(1:nT);
```

The short-channel is then added as a **single raw nuisance regressor** — one column, no Gaussian convolution:

```matlab
% flagNuisanceRMethod=1 (most correlated SS):
Ass = y(:,conc,mlSSlst(iSS));   % raw short-channel signal — ONE column
At = [A(:,:,conc)  Ass];         % HRF model | drift polynomial | short-channel
```

The final design matrix is:
```
At = [HRF_Gaussian_basis_columns (nB × nConditions) | drift_polynomial | SS_raw_column]
```

Everything is solved in one unified OLS (`pinvA = ATA \ At(lstInc,:)'`), and the output `dcAvg` is directly the estimated HRF (reconstructed from the Gaussian beta weights: `yavg = tbasis * beta`).

### What the Python does:

Python convolves the **short-channel signal** with Gaussian kernels:

```python
kernel = np.exp(-0.5 * (t_kernel - center_samp)**2 / std_samp**2)
X[:, k] = np.convolve(s_sig, kernel, mode='same')   # s_sig = short-channel
```

The design matrix is:
```
X = [SS_conv_Gaussian_1 | SS_conv_Gaussian_2 | SS_conv_Gaussian_3]
```

And it solves:
```python
L_corrected = L - X @ beta   # strips scalp contribution from long channel
```

This produces a "cleaned" continuous signal. **The HRF is then extracted in Stage C by epoch averaging.**

### Summary of the fundamental difference:

| Aspect | Homer3 (`hmrS_GLM`) | Python (`06_ols_regression.py`) |
|--------|---------------------|----------------------------------|
| **Role of Gaussian basis** | Model the HRF shape | Model the short-channel's time-varying coupling |
| **Gaussian is convolved with** | Stimulus onset vector | Short-channel signal |
| **Number of Gaussians** | `floor(epoch_duration / step) - 1` e.g., ~34 for [-5,30]s window | Fixed at 3 |
| **Short-channel role** | Single raw nuisance column in OLS (no convolution) | Convolved into the Gaussian design matrix |
| **Drift correction** | 3rd-order polynomial in the design matrix | None |
| **HRF output** | Directly from GLM (`dcAvg = tbasis * beta`) | Not produced here; extracted in Stage C by epoching |
| **Statistics** | GLM β, t-stats, R² produced per channel | Not produced |

### Gaussian parameter discrepancy:

Homer3 default `paramsBasis = [1.0, 1.0, 0, 0, 0, 0]` means `step=1.0 s`, `stdev=1.0 s`. Python uses `gaussian_spacing_s = 1.5 s` and `gaussian_std_s = 1.0 s`. **The spacing is 50% wider in Python.** Additionally, the number of Gaussians differs dramatically (3 vs. ~34).

### Verdict: 🔴 **Major architectural discrepancy**
This is not a parameter mismatch — it is a fundamentally different algorithm. Homer3 performs **simultaneous GLM deconvolution** to extract HRFs. Python performs **sequential scalp regression** (as preprocessing) followed by trial-averaging. The paper's statement that Gaussian basis functions are used "to create the hemodynamic response function" refers to Homer3's GLM-HRF estimation, not to a preprocessing regression on the short-channel signal.

---

## Summary Table

| Step | File | MATLAB Equivalent | Severity | Type |
|------|------|-------------------|----------|------|
| **B1** Intensity→OD | `01_intensity_to_od.py` | `hmrR_Intensity2OD.m` | ✅ Match | — |
| **B2** Bandpass filter | `02_bandpass_filtering.py` | `hmrR_BandpassFilt.m` | ⚠️ Moderate | HPF applied in Python (not in Homer3 default); different filter order |
| **B3** Motion detection | `03_motion_correct.py` | `hmrR_MotionArtifactByChannel.m` | ⚠️ Minor | `max_diff` padding orientation offset; `tMask` may differ from Homer3 defaults |
| **B4** rPCA | `04_rPCA_motion_correct.py` | `hmrR_MotionCorrectPCArecurse.m` | 🔴 Major | Not iterative; corrects only flagged samples (not full signal); vote-threshold mask vs. OR-union; `n_components=0.99` vs. `0.97` |
| **B5** MBLL | `05_mbll.py` | `hmrR_OD2Conc.m` | ✅/⚠️ Minor | Formulas equivalent; potential extinction coefficient difference (MNE vs. Prahl) |
| **B6** GLM/OLS | `06_ols_regression.py` | `hmrS_GLM.m` | 🔴 Critical | Different algorithm: Python does preprocessing scalp regression; Homer3 does unified HRF+short-channel GLM deconvolution |

---

## Impact on CNR and HRF Mismatch

### For the CNR technical validation branch (OD → BP → MBLL only):

The relevant discrepancy is **B2** — the Python applies a 0.01 Hz high-pass that Homer3 does not. This changes the baseline of slow hemodynamic signals and can affect the amplitude of the grand-average HRF that feeds into CNR calculation.

### For the full preprocessing pipeline (HRF morphology mismatch):

- **B4 (rPCA):** Python over-corrects some subjects (HOLD flags for R9, A2, R10, R11) because it applies a vote-threshold mask and single-pass correction — Homer3's iterative refinement would handle these more gracefully.

- **B6 (GLM):** This is the **root cause of the HRF morphology mismatch.** Homer3's output `dcAvg` is a deconvolved GLM estimate of the HRF; Python's output from epoch-averaging produces a noisier estimate that doesn't benefit from the simultaneous drift and scalp regression. The Expert vs. Novice contrast in M1 that's improving after successive audit fixes is converging toward a correct GLM-equivalent result from the wrong direction — the sequential approach is working but is mathematically non-equivalent.

---

## Prioritized Fix Recommendations

### P0 — Critical, implement before anything else:

1. **B6 architecture:** The correct paper-faithful approach requires reproducing the Homer3 `hmrS_GLM` logic: build a Gaussian basis over the epoch window convolved with stimulus onsets, add the raw short-channel as a single nuisance column, add drift polynomial, and solve one OLS per channel. **The output IS the HRF estimate.** This replaces both `06_ols_regression.py` and Stage C's epoch-averaging approach.

2. **B4 rPCA scope:** Apply PCA correction to the **entire signal** (all time points, not just flagged ones). The motion-flagged segments are used to identify the motion subspace (fit the PCA), but the projection/subtraction is applied globally. Also consider adding recursion (up to 5 iterations, re-detect between each) to match Homer3.

### P1 — High, implement after P0:

3. **B2 high-pass:** Test running with `l_freq=None` (no high-pass, pure 0.5 Hz LPF) to match Homer3's default of `hpf=0`. Use 3rd-order Butterworth for the LPF specifically.

4. **B4 n_components:** Change from `0.99` to `0.97` to match both `hmrR_MotionCorrectPCArecurse` parameters and the `DataPreprocessing.m` specification.

### P2 — Medium, investigate after P1:

5. **B5 extinction coefficients:** Extract and compare the extinction coefficient values that Homer3's `GetExtinctions(690, 830)` returns (from the Prahl database) against MNE's built-in values. If they differ, apply a correction factor or override MNE's internal table.

6. **B3 tMask:** Confirm whether Homer3 ASTaUND study used `tMask=0.5` (from Ref #36) or `tMask=1.0` (Homer3 function default). The function-header annotation shows 1.0, the cited reference says 0.5.

---

**Last Updated:** March 3, 2026  
**Next Review:** After implementing P0 critical fixes