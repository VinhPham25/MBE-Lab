# GC_TE_STAGE_E_PRD.md — Transfer Entropy Effective Connectivity Pipeline

**Stage:** E (Transfer Entropy — parallel to Stage D Linear GC)  
**Reference Papers:**
- Wang, Y., & Chen, W. (2020). *Effective brain connectivity for fNIRS data analysis based on multi-delays symbolic phase transfer entropy.* J. Neural Eng., 17(5), 056024. https://doi.org/10.1088/1741-2552/abb4a4 **(dSPTE — Phase 1 foundation)**
- Wang, Y., Chen, C., & Chen, W. (2022). *Nonlinear directed information flow estimation for fNIRS brain network analysis based on the modified multivariate transfer entropy.* Biomed. Signal Process. Control, 74, 103422. https://doi.org/10.1016/j.bspc.2021.103422 **(MTENUE — Phase 2 multivariate extension)**

**Input:** `outputs/stage_b_processed/processed_data/expert-novice/sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif`  
**Output Root:** `outputs/stage_e_transfer_entropy/expert-novice/`  
**Language:** Python 3.9+  
**Status:** Sections 1–2 complete; Section 3.1–3.4.3 complete; Section 4.1 complete (2026-03-24)

---

## 🎯 OBJECTIVE

Implement a three-phase Transfer Entropy (TE) effective connectivity pipeline that produces directed connectivity matrices in the **same format as Stage D GC** — enabling direct methodological comparison between linear (GC) and nonlinear (TE) approaches. The pipeline advances sequentially through three algorithmic phases:

- **Phase 1 — dSPTE:** Faithful implementation of Wang (2020)'s delay Symbolic Phase Transfer Entropy, producing a bivariate N×N directed connectivity matrix per window.
- **Phase 2 — Hybrid-dSPTE:** Extension of Phase 1 with multivariate conditioning (inspired by MTENUE from Wang 2022) to suppress spurious indirect connections — addressing dSPTE's core limitation.
- **Phase 3 — Bimodal dSPTE:** Further extension incorporating amplitude-based TE alongside phase-based TE, with weighted fusion, to recover amplitude-coupled connections discarded by phase-only analysis.

Each phase builds directly on the previous one and shares ROI extraction, trial segmentation, statistical testing, and output formatting infrastructure with Stage D. The single new component at each phase is the core connectivity function — the exact same swappable interface design used between Stage D's linear and nonlinear GC.

> **Comparison to Stage D:** Stage D computed bivariate unconditional GC using `compute_pairwise_gc(Y_window)`. Stage E replaces that single function with `compute_dspte(Y_window)` (Phase 1), `compute_hybrid_dspte(Y_window)` (Phase 2), and `compute_bimodal_dspte(Y_window)` (Phase 3), while all upstream (ROI extraction, segmentation) and downstream (statistics, visualization) code is reused directly.

---

## ⚗️ ALGORITHMIC BACKGROUND

### Phase 1: dSPTE (Wang 2020) — Full Mathematical Specification

The following stages are implemented in sequence for every pair of channels (X → Y). All equation numbers refer to Wang (2020).

#### Step 1 — Phase Extraction via Hilbert Transform (Eq. 8)
For each raw HbO ROI-averaged signal `X(t)`, compute the analytic signal and extract instantaneous phase:

```
X(t) = A(t) · exp(i·θ(t))
→ θ_x(t) = angle(hilbert(X(t)))
→ θ_y(t) = angle(hilbert(Y(t)))
```

Implementation note: `scipy.signal.hilbert` returns the analytic signal; `np.angle()` extracts phase in `[-π, π]`. The phase sequences replace the raw amplitude sequences for all downstream steps.

> ⚠️ **Critical methodological concern — Hilbert phase on broadband fNIRS:** The Hilbert transform produces a meaningful instantaneous phase *only* for narrowband, approximately monocomponent signals — the classic use case being an EEG oscillation within a defined frequency band (alpha, gamma, etc.), where the signal locally resembles `A(t)·cos(ω₀t + φ(t))` with slowly varying A and φ. fNIRS HbO signals after HRF extraction are **not** that. They are slow, broadband, non-stationary hemodynamic envelopes. Applied to such signals, θ(t) will spin at irregular rates largely driven by noise crossings and signal morphology rather than genuine neural oscillation timing. Wang (2020) validated dSPTE on finger-tapping fNIRS and obtained significant results, so the method does not break entirely — but the physical interpretation of "phase" in this context is the relative timing of hemodynamic peaks between channels, not oscillatory phase in the EEG sense. This is a limitation worth stating explicitly in any report. **Practical implication:** the amplitude branch in Phase 3 may be the scientifically more appropriate primary branch for broadband fNIRS — consider raising its default weight relative to phase (see Phase 3 discussion). Raise this with your supervisor before interpreting Phase 1 results.

#### Step 2 — Uniform Time-Delay Embedding (Eq. 10)
Reconstruct the state space of each phase time series using uniform embedding:

```
θ_t^{xd} = [ θ_x(t),  θ_x(t-τ),  ...,  θ_x(t-(d-1)τ) ]    → shape: T × d
θ_t^{yd} = [ θ_y(t),  θ_y(t-τ),  ...,  θ_y(t-(d-1)τ) ]    → shape: T × d
```

**Parameters:**
- Embedding delay `τ`: set to 1 autocorrelation time of the signal (computed as the lag at which the autocorrelation first crosses zero, or first drops below `1/e` — use `1/e` threshold as default, configurable)
- Embedding dimension `d`: set to 3 as default (Wang 2020 does not specify a value; d=3 is standard for phase-space TE in fNIRS; configurable via `config.yaml`)

> ⚠️ **Uniform embedding limitation:** The uniform embedding scheme (constant τ across all lags) is a simplifying assumption known to be suboptimal for non-stationary signals. For a 25 Hz fNIRS signal, the 1/e autocorrelation time is typically 5–20 samples (200–800 ms), so with d=3 your embedding vectors span ~30 samples (~1.2 seconds of history) — physiologically reasonable for hemodynamic dynamics. However, uniform embedding can over-represent correlated lags and under-represent informative distant lags. This is the core motivation for Wang (2022)'s Non-Uniform Embedding (NUE) in MTENUE, which selects embedding lags greedily based on conditional mutual information. Consider incorporating NUE into Phase 2 (not just multivariate conditioning) as a more complete adoption of Wang (2022)'s methodology. Document as limitation E7 if kept uniform.

#### Step 3 — Neural-Gas Symbolization (Eqs. 11–14) ← Most complex stage
This is the novel contribution of Wang (2020). Steps in order:

**3a. Joint state matrix construction:**
```
θ_t^{xyd} = [θ_t^{xd} ; θ_t^{yd}]    → shape: 2T × d
```
Both trajectories are brought to a common reconstructed state space.

**3b. Neural-Gas (NG) codebook learning:**
Apply the Neural-Gas algorithm (Martinetz et al. 1993) to the joint matrix to learn `k` prototype vectors. The NG algorithm is a soft competitive learning vector quantizer — it adapts prototypes via stochastic gradient descent with a soft-max neighbourhood function, converging to `k` codebook vectors `{M_i}_{i=1:k}` with minimised average distortion.

**3c. Distortion-based codebook size selection:**
The optimal codebook size `k₀` is the **smallest k** satisfying:
```
nDis = Σ‖θ_t^{xyd} - θ_t^{xyd,rec}‖² / Σ‖θ_t^{xyd} - θ_t^{xyd,mean}‖²  <  0.05
```
Procedure: start at `k=2`, increment by 1, refit NG, evaluate nDis. Stop when nDis < 5%. `k₀` defines the final codebook.

**3d. Vector quantisation to symbolic sequences:**
Using the nearest-prototype rule, each embedded vector in `θ_t^{xd}` and `θ_t^{yd}` is independently assigned to the most similar codebook vector from `{M_i}_{i=1:k₀}`:
```
θ_t^{xd}  →  S_t^{θx} = j₁(t),    j₁ ∈ {1, ..., k₀}
θ_t^{yd}  →  S_t^{θy} = j₂(t),    j₂ ∈ {1, ..., k₀}
```
Result: two integer symbolic sequences of length `T`.

**Neural-Gas implementation strategy (ordered by preference):**
1. **First attempt:** `neuralgas` or `neural-gas` on PyPI (`pip install neuralgas`). If it exposes a `fit(X)` / `predict(X)` interface with configurable `n_neurons`, distortion metric, and iteration count, use it directly.
2. **Fallback — scratch implementation:** If no suitable PyPI package exists or the package doesn't support the nDis stopping criterion, implement NG directly. The algorithm is ~40 lines: initialise prototypes randomly from data, iterate stochastic gradient updates with soft-ranking of distances, compute nDis, stop. Reference: Martinetz, T.M., Berkovich, S.G., & Schulten, K.J. (1993). IEEE Trans. Neural Netw., 4(4), 558–569.
3. **Validation strategy:** Either implementation is validated on the Wang (2020) simulation data (AR model, Eq. 26, δ=1, a=0.9, N=20,000 samples) before applying to ASTaUND data. Expected: nDis < 0.05 with k₀ ≤ 20 for typical fNIRS window lengths.

> 📌 **Dataset note — Try to obtain Wang (2020)'s real fNIRS data:** Wang (2020) used an open-access finger-tapping fNIRS dataset (Bak et al. 2019, *Electronics* 8(12):1486) available at https://figshare.com/articles/Open_access_fNIRS_dataset_for_classification_of_the_unilateral_finger-_and_foot-tapping/9783755/1. This dataset (30 subjects, 20 channels, 13.3 Hz, left/right/foot finger-tapping) is explicitly the dataset Wang (2020) applied dSPTE to and reported significant EC results for (p=0.0055 right finger-tapping, p=0.0041 left finger-tapping). **Running your pipeline on this dataset before ASTaUND is strongly recommended** — it gives you a ground-truth check (can you replicate Wang's Figure 9 results?) before trusting Phase 1 on your own data. Download the dataset early; it should take an afternoon to adapt the loader.

#### Step 4 — dSPTE Computation for a Single Lag u (Eq. 17)
With symbolic sequences in hand, compute directed TE at lag `u`:

```
dSPTE_{X→Y,u} = I(S_t^{θy} ; S_{t-u}^{θx} | S_{t-1}^{θy})
              = Σ p(S_t^{θy}, S_{t-1}^{θy}, S_{t-u}^{θx})
                  · log[ p(S_t^{θy} | S_{t-1}^{θy}, S_{t-u}^{θx}) /
                         p(S_t^{θy} | S_{t-1}^{θy}) ]
```

**Entropy estimation:** Since the symbolic sequences are discrete integers in `{1,...,k₀}`, probabilities are estimated by **frequency counting** (histogram method) — not the KSG estimator. The KSG estimator is for continuous data; once symbolisation is complete, the TE formula reduces to a sum over discrete joint distributions estimated from counts. This is more efficient and equally valid for discretised data. The TRENTOOL/KSG reference in Wang (2020) applies to the SPTE step before symbolisation; after symbolisation, counting is correct.

Implementation: build 3-way contingency table `count[s_ty, s_ty_lag1, s_tx_lagu]` with shape `(k₀, k₀, k₀)`. Normalise to joint probability. Derive conditional probabilities by marginalisation.

> 📦 **Library note — consider PyInform or infomeasure for entropy computation:** Rather than building the contingency table manually, two well-maintained libraries handle discrete TE cleanly. **PyInform** (`pip install pyinform`) supports `transfer_entropy(source, target, k)` on integer sequences with a `condition` parameter — directly usable on symbolic sequences post-NG. **infomeasure** (`pip install infomeasure`, 2025) supports conditional variants and multiple estimators including ordinal and discrete counting. Either can serve as the entropy backend, making the NG symbolisation the sole custom component. Manual implementation (~20 lines) is also fine and avoids a dependency. Whichever path is taken, cross-validate one window's TE value against the other approach to confirm numerical agreement.

#### Step 5 — Multi-Delay Search and Optimal Delay Selection (Eqs. 17, 24–25)
Per Wang (2020) Theorem: `dSPTE_{X→Y,u}` is **maximised at u = δ**, where δ is the true causal delay. Therefore:

```
dSPTE_{X→Y}^{opt} = max_u  dSPTE_{X→Y,u}    for u ∈ {1, 2, ..., u_max}
δ̂_{X→Y}          = argmax_u  dSPTE_{X→Y,u}   (estimated optimal delay)
```

**Parameter — `u_max`:** The maximum lag to search. Wang (2020) tested `u=3` to `u=13` on simulated data sampled at a high rate. For ASTaUND at 25 Hz, hemodynamic delays are on the order of 0.5–3 seconds → lags 12–75 samples. Default: `u_max = 50` (2 seconds at 25 Hz). Configurable.

**Output per pair:** scalar `dSPTE_{X→Y}^{opt}` (the maximum across lags) and `δ̂_{X→Y}` (the lag at which max occurred). The scalar is used to populate the connectivity matrix; the optimal delay matrix is saved as auxiliary output.

#### Step 6 — Directed Connectivity Matrix Assembly
For each ordered pair (src, tgt) with `src ≠ tgt`:
```
EC_matrix[src, tgt] = dSPTE_{src→tgt}^{opt}
```
Result: `(n_rois × n_rois)` matrix with zeros on diagonal. Remove diagonal → 20-element vector (identical structure to Stage D `gc_vec_20`).

The directionality accuracy metric from Eq. 25 is stored as auxiliary output:
```
AC_{X→Y} = (dSPTE_{X→Y}^{opt} - dSPTE_{Y→X}^{opt}) × 100%
```

---

### Phase 2: Hybrid-dSPTE — Multivariate Conditioning

**Motivation (from report):** dSPTE is bivariate — it cannot distinguish direct from indirect connections. If X→Y and X→Z, dSPTE may falsely detect Y→Z. The fix is to condition on a third channel Z when computing TE from X to Y, analogous to conditional GC.

**Core formula change (Eq. from report):**
```
dSPTE_{X→Y|Z,u} = I(S_t^{θy} ; S_{t-u}^{θx} | S_{t-1}^{θy}, S_{t-u}^{θz})
                = Σ p(S_t^{θy}, S_{t-1}^{θy}, S_{t-u}^{θx}, S_{t-u}^{θz})
                    · log[ p(S_t^{θy} | S_{t-1}^{θy}, S_{t-u}^{θx}, S_{t-u}^{θz}) /
                           p(S_t^{θy} | S_{t-1}^{θy}, S_{t-u}^{θz}) ]
```

**Conditioning strategy:** For a given pair (X→Y), condition on all other ROIs `Z ∈ {all ROIs} \ {X, Y}`. With 5 ROIs, conditioning set has 3 channels. This is the full multivariate conditioning scheme.

**Symbolisation extension:** The NG codebook is now fit on the joint matrix of ALL channels (not just the pair), so the prototype vectors encode the full multivariate state space. Each channel's embedded vectors are then separately quantised using this shared codebook.

**Entropy estimation change:** The conditioning on Z expands the contingency table to 4 dimensions: `count[s_ty, s_ty_lag1, s_tx_lagu, s_tz_lagu]` with shape `(k₀, k₀, k₀, k₀)`. For 5 ROIs, `k₀` must be kept small (≤ 8–10) to avoid the curse of dimensionality. Monitor table sparsity; if >40% of cells are zero, log a WARNING and fall back to bivariate dSPTE for that pair.

**Sample complexity guard:** With 1351 samples per window and `k₀^4` cells, there is a real sample scarcity risk. Per config: set `max_k0_multivariate = 8` (overrides the nDis-based k₀ if NG selects a larger codebook). Document this as a known limitation.

---

### Phase 3: Bimodal dSPTE — Amplitude + Phase Fusion

**Motivation (from report):** Phase-only analysis discards power-modulated coupling. The bimodal pipeline computes TE independently on phase and amplitude branches, then fuses via weighted sum.

**Branch A — Phase (identical to Phase 1/2):**
- Extract instantaneous phase `θ(t)` via Hilbert transform
- Embed, symbolise (shared NG codebook across both branches for a given pair), compute dSPTE_phase

**Branch B — Amplitude:**
- Extract instantaneous amplitude `A(t) = |hilbert(X(t))|`
- Apply same embedding, symbolisation (using the same joint NG codebook), compute dSPTE_amplitude

**Shared NG codebook rationale:** Both phase and amplitude are extracted from the same analytic signal and reflect the same underlying neural process. A single joint codebook over the combined [phase, amplitude] embedding captures the bimodal structure of the signal. If this proves to degrade nDis (phase and amplitude live on very different scales), implement separate codebooks per branch and document the deviation.

**Fusion (from report, Eq.):**
```
TE_final = w_phase · TE_phase + w_amplitude · TE_amplitude
```

**Weight optimisation:** Phase 3 implements two modes, configurable:
- `fusion_mode: fixed` — default weights `w_phase=0.7, w_amplitude=0.3` (phase carries more information for fNIRS per Wang 2020 Section 1; configurable)
- `fusion_mode: optimise` — use PSO (Particle Swarm Optimisation) on a held-out window to maximise a proxy objective (e.g., directionality accuracy AC). Uses `pyswarms` library. Computationally expensive; disabled by default.

---

## 📐 DESIGN REQUIREMENTS

### ROI Definitions (carried forward from Stage C / Stage D)
```python
ROI_NAMES = ['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']
```
5 ROIs → 20 directed off-diagonal connections. Identical to Stage D.

### Critical Parameters
| Parameter | Value | Source / Justification |
|---|---|---|
| Sampling frequency | 25 Hz (R9 resampled from 50 Hz in-memory) | Identical to Stage D |
| Window size | 1351 samples (54 s × 25 Hz + 1, MATLAB inclusive) | Identical to Stage D |
| Window stride | 1350 samples (no overlap) | Identical to Stage D |
| Embedding delay τ | 1 autocorrelation time (lag at which ACF drops to 1/e) | Wang (2020) Section 3.1; computed per-signal per-window |
| Embedding dimension d | 3 | Standard for fNIRS phase-TE; configurable |
| nDis threshold | 0.05 (5%) | Wang (2020) Section 3.2 |
| k₀ search start | 2 | Wang (2020) |
| k₀ max (bivariate) | 20 | Guard against codebook explosion |
| k₀ max (multivariate, Phase 2) | 8 | Dimensionality / sample-scarcity guard |
| Multi-delay range | u ∈ {1, ..., 50} | Covers ~2 s at 25 Hz; hemodynamic delay range |
| ROI-level EC aggregation | Mean of pairwise channel dSPTE across ROI (Eq. 30) | Wang (2020); same as Stage D ROI averaging |
| TE entropy estimator | Frequency counting (histogram) on discrete symbols | Valid post-symbolisation; more efficient than KSG |
| Fusion weights (Phase 3) | w_phase=0.7, w_amplitude=0.3 (fixed mode default) | Configurable; based on phase-primacy argument |
| Subject exclusion | R10 excluded (same reason as Stage D) | Trailing zeros in 690nm channels |

### Python Dependencies (new relative to Stage D)
| Library | Version | Use |
|---|---|---|
| `scipy` | existing | `hilbert()`, `autocorrelation`, `stats` |
| `numpy` | existing | All array ops |
| `neuralgas` or `neural-gas` | latest PyPI | Neural-Gas NG algorithm (Phase 1–3) |
| `pyswarms` | ≥1.3 | PSO weight optimisation (Phase 3, optional) |
| `idtxl` | optional | Alternative KSG-based TE estimator for validation |
| All Stage D dependencies | existing | VAR, MNE, statsmodels, matplotlib |

Strategic recommendation based on libraries
Rather than building everything from scratch, the most defensible implementation path is:

1. Phase extraction + NG symbolization — build custom (this is the unique methodological contribution; no library does it)
2. Phase 1 dSPTE entropy computation — use PyInform.transfer_entropy() on the symbolic sequences with a custom lag loop, or build the contingency table manually (it's ~20 lines)
3. Phase 2 multivariate conditioning — use IDTxl.MultivariateTE with your pre-symbolized sequences as discrete data, using IDTxl's built-in NUE rather than uniform embedding. This gives you Wang 2022's NUE essentially for free and with proper statistical testing
4. Phase 3 amplitude — pipe amplitude sequences through the same NG → PyInform path as Phase 1

> **NG library evaluation checklist** (task 1.1.2): Before committing to a PyPI library, verify it: (a) accepts a 2D numpy array as input, (b) exposes `n_neurons`/`k` as a hyperparameter, (c) allows iterative fitting or exposes distortion error, (d) returns prototype vectors and assignments. If any criterion fails → implement from scratch per fallback spec above.

---

## 📁 OUTPUT STRUCTURE

```
outputs/stage_e_transfer_entropy/expert-novice/
├── roi_signals/                             # Reused from Stage D if cached; otherwise re-extracted
│   └── sub-{id}_roi_signals.npz
│
├── phase_1_dspte/
│   ├── te_raw/
│   │   └── sub-{id}_dspte_raw.npz          # EC_matrix (5×5) per trial/window + optimal delays (5×5)
│   ├── te_connectivity/
│   │   └── sub-{id}_dspte_connectivity.npz # 20 connections × windows per trial (IDENTICAL to GC .npz schema)
│   ├── codebook/
│   │   └── sub-{id}_ng_codebook.npz        # k₀ per pair, codebook vectors, nDis values (diagnostic)
│   ├── group/
│   │   ├── B_Exp_dspte.npz                 # (7 subjects × 20 conn × n_windows) per trial
│   │   ├── B_Nov_dspte.npz
│   │   └── dspte_summary.json
│   ├── statistics/
│   │   ├── normality_test.csv
│   │   ├── H1_expert_vs_novice.csv
│   │   ├── H0_within_trial_stability.csv
│   │   └── HIV_across_trial_stability.csv
│   └── visualizations/
│       ├── sub-{id}_connectivity_matrix.png
│       ├── sub-{id}_optimal_delay_matrix.png  # New: heatmap of δ̂ per pair
│       ├── group_te_comparison.png
│       ├── group_te_topology.png
│       └── gc_vs_te_comparison.png            # New: side-by-side GC vs dSPTE for same subject/window
│
├── phase_2_hybrid_dspte/
│   ├── te_connectivity/
│   │   └── sub-{id}_hybrid_dspte_connectivity.npz
│   ├── group/
│   │   ├── B_Exp_hybrid.npz
│   │   ├── B_Nov_hybrid.npz
│   │   └── hybrid_summary.json
│   ├── statistics/
│   │   └── [same 4 CSV files as Phase 1]
│   └── visualizations/
│       ├── group_hybrid_te_topology.png
│       └── phase1_vs_phase2_comparison.png    # New: bivariate vs multivariate connectivity diff
│
├── phase_3_bimodal_dspte/
│   ├── te_connectivity/
│   │   ├── sub-{id}_bimodal_dspte_connectivity.npz     # fused TE_final
│   │   ├── sub-{id}_phase_branch_connectivity.npz      # TE_phase branch alone
│   │   └── sub-{id}_amplitude_branch_connectivity.npz  # TE_amplitude branch alone
│   ├── group/
│   │   ├── B_Exp_bimodal.npz
│   │   ├── B_Nov_bimodal.npz
│   │   └── bimodal_summary.json
│   ├── statistics/
│   │   └── [same 4 CSV files as Phase 1]
│   └── visualizations/
│       ├── phase_vs_amplitude_contribution.png  # Per-connection breakdown of w·TE for each branch
│       └── group_bimodal_topology.png
│
└── cross_phase_comparison/
    ├── all_methods_connectivity_summary.csv    # GC, dSPTE, hybrid-dSPTE, bimodal for all subjects
    ├── method_correlation_matrix.png           # Pearson correlation of 20-conn vectors across methods
    └── expert_novice_delta_comparison.png      # (Expert−Novice) for each method side-by-side
```

---

## 📋 TASK BREAKDOWN

---

### SECTION 1: ENVIRONMENT AND CONFIGURATION

#### 1.1 Setup
- [x] **1.1.1** Add new dependencies to environment: `pip install neuralgas` (or equivalent — see NG library evaluation checklist above); `pip install pyswarms` (optional, for Phase 3 PSO); verify `scipy.signal.hilbert`, `np.angle`, and `np.correlate` are available; confirm `idtxl` is importable if present (optional validation only)
  > **Result (2026-03-23):** `neuralgas` and `neural-gas` do not exist on PyPI. `pyswarms` is installed to `.pip_deps/` (macOS SIP prevents system-site-package writes; `.pip_deps/` added to `sys.path` in `utils_te.py`). `scipy.signal.hilbert` and all existing deps confirmed OK. Neural-Gas is implemented from scratch — see 1.1.2.

- [x] **1.1.2** Evaluate PyPI Neural-Gas library against the checklist: (a) numpy array input, (b) configurable `k`, (c) distortion access, (d) prototype retrieval. Log findings to `code/stage_e_transfer_entropy/NG_LIBRARY_EVAL.md`. Decision: use PyPI or scratch. If scratch, implement `NeuralGas` class in `utils_te.py` before proceeding. **This is a gate task — all Phase 1 symbolisation tasks block on this.**
  > **Result (2026-03-23):** Both `neuralgas` and `neural-gas` confirmed absent from PyPI. K-means and SOM alternatives evaluated and rejected (not methodologically equivalent). **Decision: scratch implementation.** `NeuralGas` class implemented in `utils_te.py` (~80 lines). Evaluation record in `code/stage_e_transfer_entropy/NG_LIBRARY_EVAL.md`. Smoke tested: 3-cluster 2D test (nDis=0.058, correct labels), codebook search test (k₀=8, nDis=0.050 for 3-cluster 3D data), quantise_to_symbols test all PASS.

- [x] **1.1.3** Add `stage_e_transfer_entropy` block to `config.yaml`:
  > **Result (2026-03-23):** Block added to `config.yaml`. Contains: `paths`, `subjects` (Expert/Novice, capitalised to match Stage D convention), `signal` (sfreq, embedding_dimension=3, embedding_delay_method, u_max=50), `window` (size_s=54, stride=1350), `symbolisation` (ndis_threshold=0.05, k0_min=2, k0_max_bivariate=20, k0_max_multivariate=8, NG hyperparameters), `phase_3` (fusion_mode, w_phase, w_amplitude), `output_subdirs` (all 15 subdirectory keys). **Discrepancy noted:** PRD spec used lowercase `expert`/`novice` keys; implementation uses `Expert`/`Novice` (capitalized) to match Stage D convention.

- [x] **1.1.4** Create directory structure: `code/stage_e_transfer_entropy/` with `__init__.py`. Create `utils_te.py` for shared utilities. Reuse `utils_gc.py` functions (`load_preprocessed_fif`, `get_event_onsets`, `get_subject_group`, `get_connection_labels`, `build_region_map_5`) via importlib — do not copy.
  > **Result (2026-03-23):** Directory created. `__init__.py` written. `utils_te.py` implemented with: `NeuralGas` class, `fit_neural_gas_codebook()`, `quantise_to_symbols()`, `get_te_output_dir()`, `load_or_extract_roi_signals()`, importlib wrappers for all Stage D utilities (`load_preprocessed_fif`, `get_event_onsets`, `get_subject_group`, `get_connection_labels`). All Stage D function calls go via `_import_utils_gc()` — no code copying. All 6 smoke tests PASS including A1 ROI shape validation: `(5, 26500)` @ 25 Hz.

---

### SECTION 2: ROI SIGNAL EXTRACTION

ROI extraction is identical to Stage D. If Stage D output `.npz` files exist in `outputs/stage_d_granger/linear_gc/expert-novice/roi_signals/`, load them directly. If not, re-run extraction.

#### 2.1 ROI Signal Loading / Extraction
- [x] **2.1.1** Implement `load_or_extract_roi_signals(subject_id) -> np.ndarray` in `utils_te.py`. Check for Stage D `.npz` first; if present, load and return. If absent, call Stage D's `extract_roi_signals()` via importlib and save to Stage E `roi_signals/` directory. Log which path was taken. This avoids redundant computation and ensures identical ROI signals across both stages.
  > **Result (2026-03-23):** Implemented in `utils_te.py`. Three-path strategy: (1) Stage D cache, (2) Stage E local cache, (3) re-extraction via importlib. All 13 subject `.npz` files confirmed present in Stage D cache — no re-extraction needed.

- [x] **2.1.2** Validate for A1: `Data_Reg.shape == (5, 26500)`, same values as Stage D. No reprocessing should occur if Stage D completed successfully.
  > **Result (2026-03-23):** A1 validated — `Data_Reg.shape == (5, 26500)`, sfreq=25.0 Hz, ROIs=['LPFC','CPFC','RPFC','LM1','SMA'], NaN rows=0. Loaded from Stage D cache (no re-extraction). All 13 subject `.npz` files present in Stage D roi_signals/ directory.

---

### SECTION 3: PHASE 1 — dSPTE CORE FUNCTIONS

All functions live in `code/stage_e_transfer_entropy/01_dspte_core.py`.

#### 3.1 Phase Extraction
- [x] **3.1.1** Implement `extract_phase_signal(x: np.ndarray) -> np.ndarray` using `scipy.signal.hilbert` + `np.angle`. Input: 1D time series. Output: instantaneous phase in `[-π, π]`. Add unit test: pure sine wave `sin(2πft)` should yield constant phase derivative ≈ 2πf/sfreq.
  > **Result (2026-03-23):** Implemented in `01_dspte_core.py`. Unit test: 1 Hz sine at 25 Hz → phase range [-3.08, 2.95] rad ✓ ∈ [-π,π]; mean |dθ/dt| = 0.2513 rad/sample vs expected 0.2513 (0.0% error). Shape, range, 2D-input-rejection all PASS. Docstring includes the critical Hilbert-on-broadband-fNIRS methodological limitation from TRANSFER_ENTROPY_PROGRESS.md §Step 1.

- [x] **3.1.2** Implement `extract_amplitude_signal(x: np.ndarray) -> np.ndarray` using `np.abs(scipy.signal.hilbert(x))`. Needed for Phase 3 but implemented now. Unit test: pure sine wave amplitude should be ≈ constant.
  > **Result (2026-03-23):** Implemented in `01_dspte_core.py`. Unit test: 3·sin(2πt) → amplitude range [3.0000, 3.0000], interior max deviation 0.000% < 2% tolerance. All PASS.

- [x] **3.1.3** Implement `compute_embedding_delay(x: np.ndarray, method: str = 'autocorr_1_over_e') -> int`. Computes the autocorrelation lag τ at which the normalised ACF first drops below `1/e ≈ 0.368`. Uses `np.correlate(x - x.mean(), x - x.mean(), mode='full')[n-1:]` (one-sided). Returns τ as integer sample lag (minimum value: 1). Log τ per channel per window — expected range for 25 Hz fNIRS: 3–15 samples.
  > **Result (2026-03-23):** Implemented in `01_dspte_core.py`. Unit tests: (a) 1 Hz sine → τ=5 ✓ ∈ [3,6]; (b) white noise → τ=1 ✓; (c) slow ramp → τ=544 (WARNING logged) ✓; (d) constant → τ=1 ✓; (e) 0.1 Hz → τ=48 (WARNING logged) ✓. All PASS.
  >
  > ⚠️ **PRD estimate corrected — observed τ is 20–300 samples, not 3–15.** Running on real A1 data (phase signal, first 1351-sample window): LPFC τ=39, CPFC τ=118, RPFC τ=164, LM1 τ=22, SMA τ=169 (mean 102.4 samples = 4.1 s). The ACF at lag 1 is 0.987–0.999 for all ROIs — the signals are extremely slowly decorrelating because Stage B's 0.5 Hz LP filter + GLM HRF deconvolution produces broadband hemodynamic envelopes, not narrowband oscillations. This is physically correct and expected. **Implication for embedding:** with d=3 and τ≈100 samples, each embedding vector spans ~200 samples (~8 s of signal history) — this covers most of the hemodynamic response peak, which is physiologically reasonable. The warning threshold in `compute_embedding_delay` has been updated from [3, 15] to [5, 500] to reflect the actual data characteristics. **Note for report:** this large τ further reinforces that Hilbert phase on these signals tracks the relative timing of hemodynamic peaks, not oscillatory phase — consistent with the methodological note in Step 1.

#### 3.2 Uniform Embedding
- [x] **3.2.1** Implement `uniform_embed(theta: np.ndarray, d: int, tau: int) -> np.ndarray`. Input: 1D phase sequence of length T. Output: 2D matrix of shape `(T - (d-1)*tau) × d`. Row i is `[θ(i), θ(i-τ), ..., θ(i-(d-1)τ)]`. Valid rows only (no NaN padding). Validate: embedding of length-1000 signal with d=3, τ=2 should yield shape `(996, 3)`.
  > **Result (2026-03-23):** Implemented in `01_dspte_core.py` via vectorised index array — no Python loop over rows. Tests: (a) spec shape (996,3) ✓; (b) row content [t, t-τ, t-2τ] verified at rows 0,10,100,995 ✓; (c) d=1 identity embedding ✓; (d) d=3,tau=1 ✓; (e) shape formula verified for 4 (d,τ) combinations ✓; (f) all 4 error cases raise ValueError ✓; (g) real A1 LPFC pipeline run: τ=39, embedded shape (1273,3), history span 3.1 s, phase values ∈ [-3.14, 3.14] rad ✓. All PASS. No obstacles.

#### 3.3 Neural-Gas Symbolisation
- [x] **3.3.1** Implement `fit_neural_gas_codebook(theta_xyd: np.ndarray, k0_max: int, ndis_threshold: float) -> tuple[np.ndarray, int, float]`. Input: joint state matrix of shape `(2T, d)`. Procedure: iterate k from 2 to k0_max; at each k, fit NG (PyPI or scratch), compute nDis (Eq. 11), stop when nDis < threshold. Returns `(codebook_vectors, k0, final_nDis)`. Logs k₀ and nDis per call. If nDis never reaches threshold by k0_max, log WARNING and use k=k0_max.
  > **Result (2026-03-24):** Implemented in `utils_te.py` as part of Section 1.1.2 (scratch NeuralGas). Iterates k from 2 to k0_max, fits NeuralGas at each k, evaluates nDis (Eq. 11), stops when nDis < threshold. WARNING logged if threshold not met. Returns `(codebook, k0, final_nDis)`. Smoke tested: k₀=8, nDis≈0.050 for 3-cluster 3D data. PASS.

  > **nDis formula (Eq. 11):**
  > ```
  > nDis = Σ‖θ_t^{xyd} - θ_t^{xyd,rec}‖² / Σ‖θ_t^{xyd} - mean(θ_t^{xyd})‖²
  > ```
  > `θ_t^{xyd,rec}` = reconstruction using nearest prototype. Numerator = total quantisation error. Denominator = total variance. A smaller nDis = better encoding.

- [x] **3.3.2** Implement `quantise_to_symbols(embedded: np.ndarray, codebook: np.ndarray) -> np.ndarray`. Input: embedded matrix `(T', d)` and codebook `(k₀, d)`. For each row, find nearest prototype via Euclidean distance. Returns 1D integer array of length T' with values in `{0, ..., k₀-1}` (0-indexed internally; display as 1-indexed in logs). Uses `scipy.spatial.distance.cdist` for efficiency.
  > **Result (2026-03-24):** Implemented in `utils_te.py`. Uses `scipy.spatial.distance.cdist(embedded, codebook, metric='euclidean')` → `np.argmin(axis=1)`. Returns 0-indexed integer array. PASS.

- [x] **3.3.3** Implement `symbolise_pair(theta_x: np.ndarray, theta_y: np.ndarray, d: int, tau: int, k0_max: int, ndis_threshold: float) -> tuple[np.ndarray, np.ndarray, dict]`. Orchestrates 3.3.1 and 3.3.2 for a single (X, Y) pair:
  1. Embed both series → `θ_xd`, `θ_yd`
  2. Build joint matrix `θ_xyd = [θ_xd; θ_yd]`
  3. Fit NG → codebook
  4. Quantise each series independently → `S_x`, `S_y`
  5. Return `(S_x, S_y, {'k0': k0, 'nDis': nDis, 'tau': tau})`
  > **Result (2026-03-24):** Implemented in `01_dspte_core.py`. Imports `fit_neural_gas_codebook` and `quantise_to_symbols` from `utils_te`. Meta dict includes `k0`, `nDis`, `tau`, `d`, `T_prime`, and `codebook` (retained for Phase 2 reuse). Validates 1D inputs, equal-length constraint, and raises ValueError on mismatches. Unit tests (5 cases): (a) synthetic sine pair (T=500, d=3, tau=4) → T'=492, k₀=9, nDis=0.0467 < 0.05 ✓; (b) length mismatch raises ValueError ✓; (c) 2D input raises ValueError ✓; (d) nDis ∈ [0,1] ✓; (e) real A1 LPFC→CPFC pair (first 1351-sample window): tau=39, k₀=13, nDis=0.0424 < 0.05, S shape=(1273,), symbols ∈ [0,12] ✓. All PASS.

#### 3.4 dSPTE Computation
- [x] **3.4.1** Implement `compute_dspte_at_lag(S_x: np.ndarray, S_y: np.ndarray, u: int, k0: int) -> float`. For a single lag `u`, compute dSPTE_{X→Y,u} (Eq. 17) using frequency counting:
  1. Build joint count array: `counts[s_ty, s_ty_lag1, s_tx_lagu]` with shape `(k₀, k₀, k₀)`. Index as: for `t` from `max(1,u)` to `T-1`: increment `counts[S_y[t], S_y[t-1], S_x[t-u]]`.
  2. Normalise to joint probability `p(s_ty, s_ty_lag1, s_tx_lagu)`.
  3. Compute marginals by summation.
  4. Compute TE sum: add `p_joint * log(p_cond_joint / p_cond_marg)` for all non-zero cells. Use `np.where(p_joint > 0, ...)` to avoid log(0).
  5. Return scalar TE value (≥ 0 by definition; clip negative numerical noise to 0).
  > **Result (2026-03-24):** Implemented in `01_dspte_core.py`. Vectorised contingency table construction via `np.add.at`. All marginal/conditional probabilities computed by broadcasting over the (k₀, k₀, k₀) array. Zero-cell guard via `np.where`. Returns non-negative float, clipped against floating-point noise. Unit tests: (a) AR model (X→Y, δ=3) → TE(X→Y, u=3)=0.333 > TE(Y→X, u=3)=0.127 ✓ directionality confirmed; (b) return type float, non-negative ✓; (c) u ≥ T' returns 0.0 ✓. All PASS.

- [x] **3.4.2** Implement `compute_dspte_multidelay(S_x: np.ndarray, S_y: np.ndarray, k0: int, u_max: int) -> tuple[float, int, np.ndarray]`. Loops over `u ∈ {1, ..., u_max}`, calls `compute_dspte_at_lag` at each. Returns `(max_te, optimal_delay, te_vs_lag_array)`. The `te_vs_lag_array` is saved as diagnostic output for visualising the lag-TE profile.
  > **Result (2026-03-24):** Implemented in `01_dspte_core.py`. Returns `(max_te, optimal_delay, te_vs_lag)` where `te_vs_lag[i] = TE at lag i+1`. Unit tests: AR model (X→Y, δ=3) → max_TE=0.333, **opt_delay=3** (exact match to Wang 2020's expected causal lag) ✓; TE(X→Y) > TE(Y→X) ✓; te_vs_lag shape=(u_max,) ✓. All PASS.

- [x] **3.4.3** Implement `compute_dspte_matrix(Y_window: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]`. This is the **drop-in replacement for `compute_pairwise_gc()`**. Input: ROI signal window of shape `(n_rois, T)`. For each ordered pair (src, tgt) with src ≠ tgt:
  1. Extract phase of each ROI signal
  2. Compute τ per signal
  3. Symbolise pair
  4. Run multi-delay dSPTE
  5. Fill EC_matrix[src, tgt] = max_dSPTE, delay_matrix[src, tgt] = optimal_delay
  Returns `(EC_matrix (n_rois × n_rois), delay_matrix (n_rois × n_rois), metadata_dict)`. Diagonal = 0. Values ≥ 0. NaN handling: if either ROI in a pair is all-NaN, set EC=0 for that pair (identical to Stage D's NaN behavior).

  > **Function signature (for Stage F swappability):**
  > ```python
  > def compute_dspte_matrix(Y_window: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
  > ```
  > This is the one function replaced in Phase 2 and Phase 3.
  > **Result (2026-03-24):** Implemented in `01_dspte_core.py`. Pre-computes per-ROI phase and τ to avoid redundant Hilbert transforms. Shared τ per pair = `min(τ_src, τ_tgt)`. Metadata dict includes `k0_matrix`, `ndis_matrix`, `tau_matrix`, `te_profiles` (keyed by (src,tgt) tuple for lag-profile viz), and `skipped_pairs`. Unit tests: (a) 3-ROI synthetic window → EC shape (3,3) ✓, diagonal=0 ✓, all off-diag > 0 ✓, 6/6 te_profile entries ✓, EC[0→1] > EC[1→0] as expected from causal coupling ✓; (b) NaN-row guard → 4 pairs correctly skipped, EC entries = 0.0 ✓. All PASS.

- [ ] **3.4.4** Validate on Wang (2020) simulation data BEFORE applying to ASTaUND. Generate AR model (Eq. 26): `X(t) = 0.4X(t-1) - 0.6X(t-2) + ε_x`, `Y(t) = 0.9·X(t-3) + ε_y`, N=20,000 samples, δ=3, a=0.9. Expected results per paper: dSPTE_{X→Y} ≫ dSPTE_{Y→X} at lag u=3; EC matrix shows X→Y connection; accuracy AC ≈ 74% over multivariate hybrid model test. Save simulation validation report to `code/stage_e_transfer_entropy/VALIDATION_SIMULATION.md`.

- [ ] **3.4.5** *(Strongly recommended before ASTaUND)* Validate on Wang (2020)'s **real fNIRS finger-tapping dataset** (Bak et al. 2019 — see dataset note in Step 3 above). Download from figshare, write a minimal SNIRF/CSV loader, run the full Phase 1 pipeline, and check whether the EC strength between left and right parietal cortex is significantly higher in task vs rest (target: p < 0.05 replicating Wang Figure 9). This is the strongest possible sanity check — if you cannot replicate a published result on the exact dataset the paper used, something is wrong with the implementation before you ever touch ASTaUND. Save replication report to `code/stage_e_transfer_entropy/VALIDATION_WANG2020_REPLICATION.md`.

---

### SECTION 4: PHASE 1 — SLIDING WINDOW PIPELINE

Functions in `code/stage_e_transfer_entropy/02_dspte_pipeline.py`. Mirrors `03_linear_gc.py` / `04_subject_gc_pipeline.py` in Stage D.

#### 4.1 Windowed dSPTE
- [x] **4.1.1** Implement `compute_windowed_dspte(Y_trial: np.ndarray, window_size: int = 1350, stride: int = 1350, config: dict = None) -> dict`. Same window loop as Stage D: `for w in range(window_size, T, stride)` → `Y_win = Y_trial[:, w-window_size:w+1]` (1351 samples). Calls `compute_dspte_matrix(Y_win, config)`. Returns `{'Win_1': (EC_matrix, delay_matrix, meta), 'Win_2': ...}` with 1-indexed keys. Skip guard: skip all-NaN windows with WARNING log. **Performance note:** dSPTE is significantly more expensive than GC per window due to the NG fitting loop. Log wall-clock time per window; alert if > 60 seconds per window (suggesting codebook search is too broad).
  > **Result (2026-03-24):** Implemented in `02_dspte_pipeline.py`. Identical window indexing to Stage D (`range(window_size, T, stride)`, inclusive slice `w-window_size:w+1`). Returns `{'Win_N': (EC_matrix, delay_matrix, metadata)}` with 1-indexed keys. Logs elapsed time per window (INFO) and emits WARNING if > 60 s. All-NaN window guard with WARNING log. NG module imported lazily via importlib to avoid circular imports. Unit tests: (a) 2-ROI synthetic trial T=4053 → 3 windows, correct shapes, diagonal=0, EC≥0 ✓; (b) T=500 < window → 0 windows ✓; (c) all-NaN first window skipped → 2 valid windows, keys Win_1/Win_2 ✓. Elapsed: 1.6 s for 3×2-ROI windows (fast because u_max=5 in test config). All PASS.

- [x] **4.1.2** Implement `extract_connectivity_vector(EC_matrix: np.ndarray, roi_names: list) -> np.ndarray`. Expands to canonical 5×5 layout (identical to Stage D `_canonicalize_ec_matrix()`), removes diagonal at indices `[0,6,12,18,24]`, returns 20-element vector. Ensures shape compatibility with Stage D outputs.
  > **Result (2026-03-24):** Implemented in `02_dspte_pipeline.py`. Canonical ROI order: `['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']`. Expands (n_valid × n_valid) EC matrix to (5×5) by placing values at canonical row/column positions; missing ROI pairs left as 0. `np.delete(flat, [0,6,12,18,24])` → 20-element vector. Raises ValueError for unknown ROI names and shape mismatches. Unit tests: (a) full 5×5 → 20 elements, ordering verified (LPFC→CPFC=vec[0], CPFC→LPFC=vec[4]) ✓; (b) 3-ROI (LPFC,CPFC,LM1) → 20 elements, RPFC and SMA rows correctly zero ✓; (c) unknown ROI raises ValueError ✓; (d) shape mismatch raises ValueError ✓. All PASS.

#### 4.2 Per-Subject dSPTE Pipeline
- [ ] **4.2.1** Implement `process_subject_dspte(subject_id: str, study: str = 'expert_novice', overwrite: bool = False) -> dict` in `03_subject_dspte_pipeline.py`. Orchestrates: load ROI signals → `get_trial_segments` (reused from Stage D) → `compute_windowed_dspte` per trial → `extract_connectivity_vector` per window → save `.npz`. Return structure: `{'Tr_1': {'Win_1': conn_vec_20, ...}, ...}`. Stores both connectivity vectors AND delay matrices per window. Supports `overwrite=False` caching.

- [ ] **4.2.2** Per-subject `.npz` schema (identical to Stage D `gc_connectivity` schema for connectivity vectors; adds delay matrices as new keys):
  ```
  Tr_1_connectivity:     shape (20, n_wins_tr1)      # 20 connections × n windows
  Tr_1_delays:           shape (20, n_wins_tr1)      # optimal delay per connection per window
  Tr_1_k0_values:        shape (n_rois, n_rois, n_wins_tr1)  # k₀ per pair per window (diagnostic)
  trial_ids, window_counts, roi_names_used, connection_labels, n_trials, subject_id, group, params_json
  ```

- [ ] **4.2.3** Batch all 13 subjects (R10 excluded). Log per-subject: n_trials, n_windows, mean k₀, mean optimal delay, mean dSPTE value. Create `te_batch_log.json`. Expected anomalies (same as Stage D): R9 = 2 trials; R11 = 3 ROIs; R2 = 4 ROIs; R4 = 4 trials.

#### 4.3 Group Array Construction
- [ ] **4.3.1** Implement `build_group_arrays_dspte() -> tuple[dict, dict]`. Identical structure to Stage D `build_group_arrays()`. Returns `(B_Exp, B_Nov)` dicts keyed by trial ID, each value `array(n_subs, 20, max_wins_trial)`. NaN-padding for missing trials/windows. Save `B_Exp_dspte.npz` and `B_Nov_dspte.npz`.

---

### SECTION 5: PHASE 1 — STATISTICAL TESTING

Functions in `code/stage_e_transfer_entropy/04_dspte_statistics.py`. Mirrors Stage D `05_gc_statistics.py` exactly — same four tests, same structure.

- [ ] **5.1** Implement `run_normality_tests_te(B_Exp, B_Nov, conn_labels) -> pd.DataFrame`. Identical to Stage D normality test. Save to `phase_1_dspte/statistics/normality_test.csv`.

- [ ] **5.2** Implement `test_expert_vs_novice_te(B_Exp, B_Nov, conn_labels) -> pd.DataFrame`. Identical to Stage D H1 test. FDR correction (Benjamini-Hochberg). Save to `H1_expert_vs_novice.csv`.

- [ ] **5.3** Implement `test_within_trial_stability_te(B_Exp, B_Nov, conn_labels) -> pd.DataFrame`. Identical to Stage D H0 test. Save to `H0_within_trial_stability.csv`.

- [ ] **5.4** Implement `test_across_trial_stability_te(B_Exp, B_Nov, conn_labels) -> pd.DataFrame`. Identical to Stage D HIV test. Save to `HIV_across_trial_stability.csv`.

---

### SECTION 6: PHASE 1 — VISUALIZATION

Functions in `code/stage_e_transfer_entropy/05_dspte_visualization.py`. Reuse Stage D `06_gc_visualization.py` functions where possible (pass an `ax` argument). Add two new plot types:

- [ ] **6.1** Adapt `plot_connectivity_matrix()` for dSPTE: identical call signature, different colormap (`viridis` instead of `Blues` — TE values are unbounded above unlike log-GC, so a sequential colormap reads better). Save per-subject connectivity matrices.

- [ ] **6.2** Implement `plot_optimal_delay_matrix(delay_matrix, subject_id, trial, window)`. 5×5 heatmap of `δ̂_{src→tgt}` in samples (and in seconds = samples/sfreq). Annotates each cell with `N ms`. Color by delay duration. Diagonal = `—`. Gives interpretable picture of estimated neural transmission delays across ROI pairs.

- [ ] **6.3** Implement `plot_gc_vs_te_comparison(subject_id, trial, window_idx)`. Loads both Stage D GC and Stage E dSPTE connectivity vectors for the same subject/trial/window. Side-by-side bar chart of all 20 connections, with GC in steel blue and dSPTE in teal. Include Pearson r and Spearman ρ in the title. This is the primary diagnostic for the GC↔TE comparison goal.

- [ ] **6.4** Implement `plot_group_topology_te()`. Same anatomical node layout as Stage D topology plot. Generate for Tr_1 Win_1 and for subject-averaged data.

- [ ] **6.5** Implement `plot_lag_te_profile(te_vs_lag_array, src_roi, tgt_roi, optimal_delay)`. Line plot of `dSPTE_{X→Y,u}` vs lag `u` for a single pair, with a vertical marker at the optimal delay. Helps interpret whether the multi-delay peak is sharp (confident delay estimate) or broad (uncertain).

---

### SECTION 7: PHASE 2 — HYBRID-dSPTE (MULTIVARIATE CONDITIONING)

Functions in `code/stage_e_transfer_entropy/06_hybrid_dspte.py`.

#### 7.1 Multivariate Symbolisation
- [ ] **7.1.1** Implement `symbolise_multivariate(Y_phase: np.ndarray, d: int, tau_list: list, k0_max: int, ndis_threshold: float) -> tuple[list[np.ndarray], dict]`. Input: phase matrix `(n_rois, T)` with per-ROI embedding delays. Builds joint state matrix over ALL n_rois channels: `θ_all^d` with shape `(n_rois × T, d)` (stack all embeddings). Fit single shared NG codebook. Quantise each channel's embedded vectors independently using this shared codebook. Returns `(list_of_symbol_sequences, {'k0': k0, 'nDis': nDis})`. If k₀ from nDis criterion exceeds `k0_max_multivariate`, clamp to `k0_max_multivariate` and log WARNING.

> ⚠️ **Sample scarcity is a serious risk in Phase 2.** With 1,351 samples per window (after embedding edge effects, ~1,341 usable), the 4D contingency table for one conditioning channel has shape `(k₀, k₀, k₀, k₀)`. At k₀=8 that is 4,096 cells — meaning an average of **< 1 sample per cell**. Even at k₀=4 (256 cells) you have ~5 samples/cell, which is borderline. Frequency-counting entropy estimates become unreliable when cells are sparse (bias is O(k₀/N)). Mitigation options in order of preference: (1) keep `k0_max_multivariate ≤ 4` in practice even if nDis criterion would select higher; (2) run Phase 2 with the sparsity guard set to 20% (not 40%) for ASTaUND's short windows; (3) use IDTxl's `MultivariateTE` with its built-in NUE and KSG estimator, which handles continuous data and avoids the discrete table problem entirely. **Before reporting Phase 2 results, run Phase 2 on the Wang (2020) AR simulation with artificially truncated N=1,351 samples and compare accuracy against the full N=20,000 run — this empirically characterises how much Phase 2 degrades at ASTaUND's window length.**

#### 7.2 Conditional dSPTE
- [ ] **7.2.1** Implement `compute_conditional_dspte_at_lag(S_x, S_y, S_z_list, u, k0) -> float`. S_z_list is a list of symbolic sequences for all conditioning channels. Extends Eq. 17 by adding conditioning on `{S_{t-u}^{θz} for z in S_z_list}`:
  - For 1 conditioning channel: 4D contingency table `(k₀, k₀, k₀, k₀)`
  - For n conditioning channels: `(n+2)`-dimensional table
  - Sparsity guard: if > 40% cells are zero → fallback to bivariate dSPTE and log WARNING
  - Compute conditional TE as nested probability ratios (same log-ratio structure as Eq. 17, extended dimensions)

  > 📦 **Consider IDTxl as an alternative backend for Phase 2:** IDTxl's `MultivariateTE` class implements multivariate TE with non-uniform embedding (the Faes 2011 NUE method, same reference as Wang 2022) and uses KSG estimation on continuous data — sidestepping the discrete table sparsity problem entirely. The workflow would be: run NG symbolisation as normal → pass the *pre-symbolisation* phase sequences to IDTxl's `MultivariateTE` (treating them as continuous inputs) → use IDTxl's output EC matrix in place of the manual conditional dSPTE. This is not faithful to Wang (2020)'s symbolic approach but gives you a well-validated multivariate TE with built-in significance testing. Implementing both and comparing is ideal — the NG-symbolic path is the novel contribution; the IDTxl path is the sanity check.

- [ ] **7.2.2** Implement `compute_hybrid_dspte_matrix(Y_window, config) -> tuple[np.ndarray, np.ndarray, dict]`. Drop-in replacement for `compute_dspte_matrix()`. Same signature. For each pair (src, tgt): conditioning set = `{all ROIs} \ {src, tgt}`. Returns `(EC_matrix, delay_matrix, metadata)` in same format as Phase 1.

- [ ] **7.2.3** Wire into subject pipeline. Create `process_subject_hybrid_dspte()` by cloning `process_subject_dspte()` and swapping the core function. Save to `phase_2_hybrid_dspte/te_connectivity/`. Batch all 13 subjects.

#### 7.3 Phase 2 Stats + Visualization
- [ ] **7.3.1** Run same 4 statistical tests as Phase 1 on hybrid group arrays. Save to `phase_2_hybrid_dspte/statistics/`.

- [ ] **7.3.2** Implement `plot_phase1_vs_phase2_comparison()`. For each connection: bar showing absolute difference `|dSPTE_bivariate - dSPTE_hybrid|` averaged across subjects. Connections where the multivariate conditioning substantially changes the value (>20% relative change) are highlighted — these are candidates for indirect/spurious connections in Phase 1.

---

### SECTION 8: PHASE 3 — BIMODAL dSPTE (AMPLITUDE + PHASE FUSION)

Functions in `code/stage_e_transfer_entropy/07_bimodal_dspte.py`.

#### 8.1 Amplitude Branch
- [ ] **8.1.1** Implement `symbolise_amplitude_pair(amp_x, amp_y, d, tau, k0_max, ndis_threshold) -> tuple`. Identical call to `symbolise_pair()` but takes instantaneous amplitude signals as input. Note: amplitude signals are strictly positive and smooth — verify that NG codebook selection works (nDis criterion) and that k₀ is comparable to the phase branch.

- [ ] **8.1.2** Implement `compute_amplitude_dspte_matrix(Y_window, config) -> tuple`. Extracts amplitude via `extract_amplitude_signal()`, symbolises, runs multi-delay dSPTE. Identical output format to Phase 1.

#### 8.2 Bimodal Fusion
- [ ] **8.2.1** Implement `compute_bimodal_dspte_matrix(Y_window, config) -> tuple`. Calls both `compute_hybrid_dspte_matrix` (phase, with multivariate conditioning from Phase 2) and `compute_amplitude_dspte_matrix` (amplitude, bivariate). Fuses:
  ```
  EC_final = w_phase · EC_phase + w_amplitude · EC_amplitude
  ```
  where `w_phase` and `w_amplitude` are read from config. If `fusion_mode == 'optimise'`: run `pyswarms.single.GlobalBestPSO` to maximise mean directionality `|AC|` across all connections (objective proxy). PSO is run on training windows only (Tr_1, Tr_2) and the resulting weights are applied to all windows. Saves optimised weights to `bimodal_summary.json`.

  > ⚠️ **Three concerns with the bimodal fusion as designed:**
  >
  > **1. Phase and amplitude are not independent.** Both are derived from the same analytic signal `A(t)exp(iθ(t))`. They share underlying structure — particularly near signal peaks where phase changes fastest and amplitude is largest. The linear fusion `w·TE_phase + (1-w)·TE_amplitude` may therefore double-count shared information. A principled decomposition would use partial information decomposition (PID), but that is a research project on its own. For this pipeline, the pragmatic fix is a sensitivity analysis: run Phase 3 with weight ratios 0.3/0.7, 0.5/0.5, and 0.7/0.3 and report all three. Document the non-independence as a limitation.
  >
  > **2. The default weight ratio w_phase=0.7 may be inverted for broadband fNIRS.** Phase-primacy is justified for narrowband oscillatory signals (EEG). For broadband fNIRS hemodynamic envelopes, amplitude is the primary information-carrying quantity — the BOLD/HbO magnitude directly reflects local neural activity, while phase is an indirect coupling proxy. Consider defaulting to `w_phase=0.3, w_amplitude=0.7` for ASTaUND and treating the 0.7/0.3 ratio as the "phase-emphasis" sensitivity condition, not the default.
  >
  > **3. The fusion weights have no empirical basis yet.** The fixed defaults are initial estimates. Run the PSO optimisation on one subject (A1) to get a data-driven starting point before committing to fixed weights for the batch. Even a single-subject optimised weight is more defensible than an arbitrary prior.

- [ ] **8.2.2** Wire into subject pipeline. Create `process_subject_bimodal_dspte()`. Save both fused and per-branch connectivity vectors. Batch all 13 subjects.

#### 8.3 Phase 3 Stats + Visualization
- [ ] **8.3.1** Run 4 statistical tests on bimodal group arrays.

- [ ] **8.3.2** Implement `plot_phase_vs_amplitude_contribution()`. Stacked bar chart per connection showing `w_phase · TE_phase` vs `w_amplitude · TE_amplitude` contributions to `TE_final`. Averaged across subjects × windows. Reveals which connections are phase-dominated vs amplitude-coupled.

---

### SECTION 9: CROSS-PHASE COMPARISON

Functions in `code/stage_e_transfer_entropy/08_cross_phase_comparison.py`.

- [ ] **9.1** Implement `compile_all_methods_summary()`. Loads Stage D GC and all three TE phase connectivity vectors. For each subject × trial × window × connection: records GC, dSPTE, hybrid-dSPTE, bimodal values. Saves to `cross_phase_comparison/all_methods_connectivity_summary.csv`.

- [ ] **9.2** Implement `plot_method_correlation_matrix()`. Computes pairwise Pearson correlation between the 20-connection vectors of each method, averaged across subjects × windows. Renders as 4×4 correlation heatmap (GC vs dSPTE vs hybrid vs bimodal). High correlation = methods agree; low = they capture different aspects of connectivity.

- [ ] **9.3** Implement `plot_expert_novice_delta_comparison()`. For each method, compute `Expert_mean - Novice_mean` per connection (same subject-averaged estimator as Stage D). Display all 4 methods side-by-side for the same 20 connections. This is the primary figure for the long-term GC-vs-TE comparison goal.

---

### SECTION 10: BATCH RUNNER

- [ ] **10.1** Create `09_run_te_pipeline.py`. CLI batch script accepting `--study`, `--subjects`, `--phase` (`1`, `2`, `3`, or `all`) flags. Runs the full pipeline end-to-end for the specified phase(s). Writes `te_provenance.json` with all parameters, timestamps, per-subject success/failure, and phase-specific notes.

- [ ] **10.2** Add `--validate` flag to `09_run_te_pipeline.py` that runs: (a) Wang 2020 simulation validation (Section 3.4.4), (b) shape assertions on group arrays, (c) all-values-non-negative check, (d) diagonal-zero check, (e) connectivity vector length = 20. Print PASS/FAIL per check.

---

## ⚠️ KNOWN DEVIATIONS FROM WANG (2020) — PRE-DOCUMENTED

### E1 — Entropy estimator: frequency counting, not KSG
Wang (2020) references TRENTOOL (Lindner et al., 2011) for entropy estimation, which uses the Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator. KSG is designed for **continuous** data. After symbolisation, the data is **discrete integers** in `{1,...,k₀}`. For discrete data, frequency counting (histogram-based maximum likelihood) is the correct and more efficient estimator. KSG applied to discrete data with repeated values produces undefined or noisy results due to degenerate k-NN distances. This is not a deviation from the paper's intent — Wang (2020) applies KSG to the pre-symbolisation SPTE formulation; post-symbolisation counting is implicitly assumed.

### E2 — Embedding dimension d=3 (unspecified in paper)
Wang (2020) does not specify the embedding dimension d. Value of d=3 is standard for phase-space reconstruction in neuroimaging TE literature. This is configurable in `config.yaml`. A sensitivity analysis at d=2, 3, 4 is recommended as an optional extension.

### E3 — u_max=50 vs paper's u=3–13 range
Wang (2020) tested lags 3–13 on simulated data with high sampling rates. ASTaUND is sampled at 25 Hz and hemodynamic delays are typically 0.5–3 seconds (12–75 samples). Default `u_max=50` covers ~2 seconds. If computational cost is prohibitive, reduce to `u_max=25` (1 second). Document in provenance file.

### E4 — k₀ ceiling for multivariate case (Phase 2)
The paper does not address multivariate TE directly. The `k0_max_multivariate=8` ceiling is an engineering choice to prevent sample scarcity in the 4D+ contingency tables. This trades reconstruction fidelity for statistical reliability. Documented as a limitation in Phase 2 output metadata.

### E5 — Shared codebook across Phase 2 channels
The Phase 1 paper fits a separate codebook per pair. Phase 2 fits a single shared codebook over all channels to maintain a consistent state-space definition for multivariate conditioning. This is a methodological extension, not a paper deviation — the paper does not address multivariate symbolisation.

### E6 — Bimodal fusion weights (Phase 3 — novel contribution)
Phase 3 is entirely original (not from Wang 2020 or Wang 2022). It is proposed in the researcher's own methodology report. The fixed weights `w_phase=0.7, w_amplitude=0.3` are initial estimates without empirical basis. Additional concerns: (a) phase and amplitude are derived from the same analytic signal and are not statistically independent, meaning the linear fusion may double-count shared information; (b) for broadband fNIRS the amplitude-primacy argument suggests inverting the default weight ratio to `w_phase=0.3, w_amplitude=0.7`. Run a sensitivity analysis across at least three weight ratios before interpreting Phase 3 results. All Phase 3 results should be framed as exploratory.

### E7 — Uniform embedding throughout (not NUE from Wang 2022)
Phase 1 and Phase 2 both use uniform time-delay embedding with a fixed τ per signal. Wang (2022)'s MTENUE uses non-uniform embedding (NUE), which selects embedding lags greedily via conditional mutual information — a strictly more powerful approach for non-stationary signals. Keeping uniform embedding is a simplification justified by implementation scope. This should be stated as a limitation when comparing Phase 2 results against the MTENUE method, and NUE incorporation is a natural extension if results are inconclusive.

---

## ✅ SUCCESS CRITERIA

| Check | Target |
|---|---|
| NG codebook validation | nDis < 0.05 achieved for all subject/window pairs in bivariate mode; k₀ ∈ [2, 20] |
| Wang (2020) simulation | dSPTE_{X→Y} > dSPTE_{Y→X} for AR model with δ=3, a=0.9; peak at lag u=3 |
| dSPTE matrix shape | `EC_matrix.shape == (5, 5)`, diagonal = 0, all values ≥ 0 |
| Connectivity vector | 20 elements, non-negative, no NaN |
| Group array shape | `B_Exp_dspte['Tr_1'].shape == (7, 20, n_windows)` |
| Output schema match | `Tr_1_connectivity` shape identical between Stage D and Stage E `.npz` files |
| H1 comparison | At least some overlap with Stage D significant connections, enabling GC↔TE comparison |
| Phase 2 effect | At least some connections show >10% change between bivariate (Phase 1) and multivariate (Phase 2) — confirming indirect connection suppression is operational |
| Phase 3 effect | Amplitude branch contributes non-negligibly to at least 3 connections (w_amplitude · TE_amplitude > 0.05 · TE_final) |
| Method correlation | Pearson r between GC and dSPTE 20-conn vectors: r > 0.3 (some agreement expected for strong direct connections) |

---

## 📊 COMPLETION METRICS

- **Total Tasks:** 49
- **Completed:** 18
- **In Progress:** 0
- **Remaining:** 31

### By Section
- **Section 1 (Setup):** 4/4 ✅ Complete (2026-03-23)
- **Section 2 (ROI Signals):** 2/2 ✅ Complete (2026-03-23)
- **Section 3 (Phase 1 Core):** 10/12 (3.1–3.4.3 complete 2026-03-24; 3.4.4–3.4.5 pending simulation data upload)
- **Section 4 (Phase 1 Pipeline):** 2/5 (4.1 complete 2026-03-24)
- **Section 5 (Phase 1 Stats):** 0/4
- **Section 6 (Phase 1 Viz):** 0/5
- **Section 7 (Phase 2 Hybrid):** 0/6
- **Section 8 (Phase 3 Bimodal):** 0/5
- **Section 9 (Cross-Phase):** 0/3
- **Section 10 (Batch Runner):** 0/2

---

## 🔗 REFERENCE NOTES

### Connection to Stage D (Linear GC) — Comparison Design
The single architectural decision that enables direct GC↔TE comparison is that **both pipelines produce identical output schemas**. Specifically:
- Both produce `Tr_N_connectivity: shape (20, n_windows)` in `.npz` files
- Both produce `B_Exp['Tr_N']: shape (7, 20, max_windows)` group arrays
- Both use the same 20 connection labels from `get_connection_labels()` in `utils_gc.py`
- Both subject to the same 4 statistical tests via analogous functions

This means Section 9 cross-phase comparison is straightforward array algebra — no format conversion needed.

### Stage F Interface (Future)
The long-term goal (LSTM-based nonlinear GC from Week 1 paper) will replace Stage D's `compute_pairwise_gc()`. Stage E's `compute_dspte_matrix()` uses the same interface. Stage F can run both and compare all three at once using Section 9's infrastructure.

### Wang (2022) MTENUE — Limitations in Phase 2 Design
The Phase 2 design draws on the researcher's Week 2 report summary of Wang (2022) rather than direct access to the paper at time of PRD writing. The core multivariate conditioning formula (Section 7.2.1) is derived from the researcher's proposed formula in the methodology document. Once Wang (2022) is uploaded, verify: (a) exact conditioning formula structure, (b) MTENUE's non-uniform embedding (NUE) selection method (should this replace uniform embedding in Phase 2?), (c) PSO parameters for Phase 3 weight optimisation. Update Phase 2 design accordingly with a PRD amendment note.

### Computational Cost Warning
dSPTE is substantially more expensive than GC per window:
- GC: ~2–5 seconds per subject (bivariate VAR, closed form)
- dSPTE: estimated 5–30 minutes per subject depending on NG convergence speed and u_max

Mitigation strategies (implement in order as needed):
1. Parallelise pair computation with `concurrent.futures.ProcessPoolExecutor` — pairs are independent
2. Cache NG codebook: if the same (subject, trial, window) pair appears across phase runs, reuse the codebook
3. Reduce `u_max` to 25 if wall-clock time exceeds 60 minutes per subject
4. Reduce `k0_max` to 10 bivariate if convergence is slow — acceptable nDis trade-off documented

Log wall-clock time per subject and per phase to `te_provenance.json`.