# Preprocessing Pipeline Audit: Python vs. MATLAB Ground Truth

**Date:** March 3, 2026 (updated March 5, 2026)
**Purpose:** Systematic comparison of Python implementation against Homer3 MATLAB ground truth  
**Status:** B2, B3, B6 fixed; B4 partially fixed — Discrepancy 4b (full-signal projection) was incorrectly implemented and must be reverted to flagged-segment-only + DC continuity to match actual MATLAB behavior

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
while length(find(tInc{iBlk}==0)) > 0 & ii < maxIter
    ii = ii + 1;
    [data_dN, svs_ii, nSV] = hmrR_MotionCorrectPCA(data_d, ...);
    tInc = hmrR_MotionArtifact(data_dN, ...);   % re-detect on corrected data
    data_d.Copy(data_dN);                         % corrected data becomes next input
    svs{iBlk}(:,ii) = svs_ii{1};
end
```

After each PCA step, motion detection runs again on the corrected signal. If motion is still present, another PCA step is applied. **Python applies PCA exactly once.**

### Discrepancy 4b — Scope of correction: ⚠️ AUDIT CORRECTION (2026-03-05)

> **Original audit claim (incorrect):** MATLAB subtracts the motion projection from the entire continuous signal — both flagged and unflagged time points.
>
> **Corrected finding:** This was a misreading of `hmrR_MotionCorrectPCA.m`. MATLAB applies PCA correction **only to the flagged segments**, then applies **DC continuity offsets** to non-flagged segments to remove boundary discontinuities. Non-flagged samples are never PCA-projected.

**What MATLAB actually does (from `hmrR_MotionCorrectPCA.m`):**
```matlab
y = d(lstNoInc, lstAct);   % Extract ONLY flagged segments (n_flagged × n_active)
yo = y;
c = y.' * y;               % Cross-product of FLAGGED data only
[V, St, foo] = svd(c);
% ... select top k components ...
yc = yo - y*V*ev*V';       % PCA residuals — applied to y (FLAGGED) only

dN = d;  % Start with full copy of original data
% Flagged segments  → replaced with yc (PCA-corrected + DC offset at boundary)
% Non-flagged gaps  → dN(lst) = d(lst) - d(lst(1)) + dN(lst(1))
%                    (DC shift for continuity only — NOT PCA-projected)
```

The non-flagged segments receive a scalar baseline shift so the signal is continuous at every flagged/non-flagged boundary, but their shape and variance are otherwise unchanged.

**What the original Python did (before B4 rewrite):**
```python
data[np.ix_(good_idx, motion_idx)] = X_corrected.T   # ONLY flagged time points replaced
```
This was structurally correct (flagged-only correction) but lacked the DC continuity stitching for non-flagged segments.

**What the B4 Homer3-faithful rewrite implemented (incorrect):**
```python
# Applied to ALL n_times columns — NOT what MATLAB does:
motion_projection = V_k @ (V_k.T @ data_good)
data_out[good_idx, :] = data_good - motion_projection   # full-signal projection
```
This was introduced under the incorrect belief that MATLAB applies full-signal projection. It is not what `hmrR_MotionCorrectPCA.m` does and is the proximate cause of signal destruction when the motion mask contains false-positives (see "Impact" section below).

**Correct Python implementation should:**
1. Fit PCA on flagged segments: `y = data[good_idx, :][:, motion_idx].T`
2. Compute residuals: `yc = y - y @ V_k @ V_k.T`
3. Splice corrected segments back at flagged indices with DC continuity stitching at boundaries

### Discrepancy 4c — Motion mask construction:

Homer3's `hmrR_MotionCorrectPCArecurse` calls `hmrR_MotionArtifact` (not `hmrR_MotionArtifactByChannel`). `hmrR_MotionArtifact` returns a **single global `tInc` vector** — if any channel is flagged at time t, `tInc[t]=0` for all channels. This is a strict OR-union.

Python builds per-channel masks from `hmrR_MotionArtifactByChannel`, then applies a vote threshold (`min_channels_flagged=10`): **a time point must be simultaneously flagged by at least 10 channels to enter the mask.** This is considerably more conservative than the Homer3 OR-union behavior.

**Note:** OR-union (min_channels_flagged=1) is correct when the detection thresholds are properly calibrated. When STDEVthresh misfires on filtered OD (see B3 and "Impact" below), a higher min_channels_flagged acts as a practical safeguard.

### Discrepancy 4d — `nSV` parameter:

The MATLAB defaults and `DataPreprocessing.m` specify `nSV = 0.97`. The Python config (`ASTaUND_parameters.md`) specifies `n_components: 0.99`. The `ASTaUND_parameters.md` itself flags this as a needed correction: `n_components: 0.99 → 0.97`.

### Verdict: 🔴 **Major discrepancy (revised)**
The iterative nature (4a) and mask construction (4c) are real discrepancies from the original Python. The scope-of-correction discrepancy (4b) was mis-stated in the original audit — MATLAB is flagged-segment-only (same as old Python), not full-signal. The B4 Homer3-faithful rewrite incorrectly introduced full-signal projection, which must be reverted to flagged-segment correction + DC continuity stitching.

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
| **B2** Bandpass filter | `02_bandpass_filtering.py` | `hmrR_BandpassFilt.m` | ✅ Fixed (2026-03-03) | Pure 3rd-order LPF; HPF skipped when hpf=0 |
| **B3** Motion detection | `03_motion_correct.py` | `hmrR_MotionArtifactByChannel.m` | ✅ Fixed (2026-03-05) | `max_diff` padding corrected to MATLAB parity (zeros at END); `tStdev` renamed `STDEVthresh` |
| **B4** rPCA | `04_rPCA_motion_correct.py` | `hmrR_MotionCorrectPCArecurse.m` | 🔴 Partially fixed, partially broken | 4a iterative ✅; 4b full-signal projection ❌ WRONG (MATLAB is flagged-segment + DC continuity); 4c OR-union ✅; 4d nSV=0.97 ✅ |
| **B5** MBLL | `05_mbll.py` | `hmrR_OD2Conc.m` | ✅/⚠️ Minor | Formulas equivalent; potential extinction coefficient difference (MNE vs. Prahl) |
| **B6** GLM/OLS | `06_ols_regression.py` | `hmrS_GLM.m` | ✅ Fixed (2026-03-05) | Homer3-style GLM/OLS with Gaussian basis on stimulus onsets, raw short-channel nuisance, drift polynomial |

---

## Impact on CNR and HRF Mismatch

### For the CNR technical validation branch (OD → BP → MBLL only):

The relevant discrepancy was **B2** — the Python applied a 0.01 Hz high-pass that Homer3 does not. Fixed 2026-03-03 (now pure 3rd-order 0.5 Hz LPF).

### For the full preprocessing pipeline (HRF morphology mismatch):

- **B4 (rPCA) — current active problem:** The B4 rewrite introduced full-signal projection removal based on a misreading of the MATLAB code (Discrepancy 4b). MATLAB's `hmrR_MotionCorrectPCA` only corrects flagged segments and applies DC continuity shifts to non-flagged segments. The Python full-signal projection, combined with the iterative loop (up to 5 passes) and the STDEVthresh-on-filtered-OD over-detection problem, catastrophically destroys the neural signal for subjects where the motion mask captures hemodynamic peaks (A2: 100% variance removed; R6, R9, R12: machine-epsilon GLM output). This must be reverted.

- **B4 × B3 interaction:** Even with the correct flagged-segment-only correction, the STDEVthresh arm of the dual criterion misfires on filtered OD because `std(diff(filtered_OD))` is very small (~1×10⁻⁴ OD at 25 Hz), making `mc_thresh = STDEVthresh × std_diff` smaller than normal hemodynamic oscillations. This flags task-response peaks as motion. Setting `use_amp_only: true` or raising `min_channels_flagged` guards against this when the flagged-only correction is in place.

- **B6 (GLM):** Fixed 2026-03-05. The Python now implements Homer3-style GLM/OLS with Gaussian basis functions convolved with stimulus onsets and raw short-channel as nuisance regressor.

---

## Prioritized Fix Recommendations

### P0 — Critical, must fix before pipeline is valid:

1. **B4 Discrepancy 4b — revert full-signal projection:** In `_apply_pca_single_pass()`, replace the full-signal `data -= V_k @ V_k.T @ data` with flagged-segment-only correction + DC continuity stitching matching `hmrR_MotionCorrectPCA.m`:
   - Corrected flagged segments: `yc = y - y @ V_k @ V_k.T`
   - Non-flagged gaps: DC-shifted at segment boundaries for continuity
   - This bounds any correction damage to the flagged samples only

### P1 — High, implement after P0:

2. **B4 × B3 STDEVthresh calibration:** After fixing 4b, evaluate whether dual criterion with STDEVthresh=20 is appropriate for filtered OD. Options:
   - `use_amp_only: true` (amplitude-only, most conservative)
   - `min_channels_flagged: 5–10` (require consensus, reduces false-positives)
   - STDEVthresh raised to ~500+ (equivalent to amplitude-only for filtered data)

3. **B4 n_components:** Already fixed to 0.97 (Homer3/Yucel et al. default). ✅

### P2 — Medium, investigate after P1:

4. **B5 extinction coefficients:** Extract and compare the extinction coefficient values that Homer3's `GetExtinctions(690, 830)` returns (from the Prahl database) against MNE's built-in values. If they differ, apply a correction factor or override MNE's internal table.

5. **B3 tMask:** Confirm whether Homer3 ASTaUND study used `tMask=0.5` (from Ref #36) or `tMask=1.0` (Homer3 function default). The function-header annotation shows 1.0, the cited reference says 0.5.

---

**Last Updated:** March 5, 2026  
**Next Review:** After implementing B4 Discrepancy 4b revert (flagged-segment-only + DC continuity)