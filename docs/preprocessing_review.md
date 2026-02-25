# Preprocessing Review: ASTaUND Reproduction (Stage B / Stage C)

This review compares the current pipeline against the ASTaUND paper description and the provided reference figures (`Figure 3c`, `Figure 4i`), with emphasis on causes of HRF/CNR mismatch.

## Section 1: Discrepancies, Likely Causes, and Fixes

### 1) HRF replication target is not 1-to-1 with paper Figure 3c
- Discrepancy:
  - Paper Figure 3c shows **5 cortical regions** (`Left Lateral PFC`, `Central PFC`, `Right Lateral PFC`, `Left Medial M1`, `SMA`) and extends to about **100 s** post onset.
  - Current code uses **3 regions** (`PFC`, `M1`, `SMA`) and default epoch window `tmin=-5, tmax=30`.
- Evidence:
  - `code/stage_c_validation/01_hrf_extraction.py` and `03_hrf_visualization.py` implement 3-region plots and 30 s epochs.
  - Provided `docs/Figure 3c Kamat et al. (2023).png` is 5-region and 100 s.
- Why this matters:
  - Region merging and short time window can materially change waveform shape, peak timing, and variance bands.
- Potential solution:
  - Implement an explicit **5-region mapping** and replot exactly as Figure 3c.
  - Increase epoch/post window to match paper visualization (100 s for HRF plots).

### 2) Short-channel regression model differs from paper description
- Discrepancy:
  - Paper describes short-channel regression using a **consecutive sequence of Gaussian basis functions via OLS**.
  - Current implementation is single-regressor OLS (`L - βS*`) with one best-correlated short channel.
- Evidence:
  - `code/stage_b_preprocessing/06_ols_regression.py` implements scalar β per channel.
- Why this matters:
  - A static single-regressor model may underfit time-varying systemic contamination and leave oscillatory residuals.
- Potential solution:
  - Implement the Gaussian-basis OLS approach (Homer-equivalent) for short-channel regression.

### 3) CNR path currently includes channel pruning, but paper says exclude pruning for this validation
- Discrepancy:
  - Paper technical validation explicitly says CNR HRFs are computed **excluding channel pruning**.
  - Current CNR uses Stage B outputs where Stage A 80–140 dB exclusions are applied.
- Evidence:
  - Stage B provenance includes B1 signal-amplitude exclusion; Stage C reads those outputs.
- Why this matters:
  - Pruning changes channel population and variance structure, biasing CNR downward/upward depending on removed channels.
- Potential solution:
  - Build a dedicated “CNR-validation” preprocessing branch with no channel pruning.

### 4) CNR computation method differs from paper technical-validation description
- Discrepancy:
  - Paper description: compute **group mean HRFs**, then channel-wise CNR from HRF mean/variance.
  - Current code computes per-subject **peak amplitudes** in `5–15 s`, then CNR across subject peak distributions.
- Evidence:
  - `code/stage_c_validation/04_calculate_cnr.py` uses peak extraction per subject.
- Why this matters:
  - Peak-only compression can reduce effect magnitude and change denominator behavior, often lowering CNR.
- Potential solution:
  - Add a paper-faithful CNR mode that uses group-mean HRF traces (not peak-only subject summaries), then compare outputs.

### 5) Missing low-pass stage after rPCA/systemic correction
- Discrepancy:
  - Paper states motion/systemic interference were corrected with **rPCA and low-pass filters**.
  - Current pipeline performs initial `0.01-0.5 Hz` bandpass, but no explicit additional low-pass stage aligned to this statement.
- Evidence:
  - Stage B chain is `B1→B2→B3→B4→B5→B6`; no dedicated post-rPCA low-pass stage.
- Why this matters:
  - Remaining high-frequency physiology can create jagged HRFs and inflated variance (wider SD bands), reducing CNR.
- Potential solution:
  - Add a post-rPCA/post-regression low-pass branch (for validation outputs) and compare PSD + HRF smoothness + CNR before/after.

### 6) Potential unit/scaling mismatch with Figure 3c
- Discrepancy:
  - Reference Figure 3c y-axis is labeled as `ΔHbo2 conc. (µM*mm)`.
  - Current pipeline assumes and thresholds in `µM`.
- Evidence:
  - Provided figure axis label vs current code comments/threshold checks.
- Why this matters:
  - Unit mismatch invalidates direct amplitude threshold comparison and may explain perceived scale mismatch.
- Potential solution:
  - Verify Homer2/Homer3 output units and scaling convention for ASTaUND figures; align plot units and thresholds accordingly.

### 7) Motion detection configuration is internally inconsistent and likely over-flags filtered data
- Discrepancy:
  - Code comments acknowledge dual-criterion detection on filtered OD can over-flag; however production path still uses dual criterion (`use_amp_only=False`) for Stage B preprocessing provenance.
- Evidence:
  - `outputs/stage_b_processed/processed_data/expert-novice/expert-novice_provenance.json` shows B3 `use_amp_only=false`.
  - `03_motion_correct.py` comments note oversensitivity risk on filtered OD.
- Why this matters:
  - Over-flagging drives larger correction masks and can remove neural signal with artifacts.
- Potential solution:
  - Reproduce Homer behavior more faithfully for filtered data (or move detection point), then lock one validated criterion for production.

### 8) rPCA correction appears over-aggressive for many subjects
- Discrepancy:
  - A large fraction of subjects show very high variance removal in OLS summary outputs.
- Evidence:
  - `expert-novice_master_summary.csv`: 7/14 subjects have HbO variance removed >50%; 4/14 have HbR >30%.
  - `R10` is extreme (`HbO 87.05%`, `HbR 98.73%`).
- Why this matters:
  - Excessive correction can flatten or distort true hemodynamics, weaken HbO/HbR inverse relationship, and inflate group variability.
- Potential solution:
  - Add QC gates for over-correction, inspect per-subject corrected spectra, and fail/exclude severe outliers from group inference.

### 9) Region assignment method is heuristic, not paper-specific channel definitions
- Discrepancy:
  - Current region mapping is coordinate-threshold based (`x/y` cutoffs) and not explicit paper channel lists.
- Evidence:
  - `build_region_map()` in `01_hrf_extraction.py` uses fixed thresholds.
- Why this matters:
  - Region composition differences directly alter mean traces and SD bands.
- Potential solution:
  - Use explicit channel-to-region definitions from ASTaUND study metadata (the paper-referenced preprocessing sheet / study mapping), not geometric thresholds.

### 10) R10 artifact handling is likely contaminating group variance
- Discrepancy:
  - R10 trailing-zero issue is handled via NaN/Inf imputation during rPCA matrix prep.
- Evidence:
  - `R10_rpca_summary.json`: `variance_removed_pct = 99.99`.
  - `R10_mbll_summary.json`: extreme ranges (HbO to +3481.48 / -2629.77 µM; HbR to +1458.88 / -1099.00 µM).
- Why this matters:
  - Even when NaN channels are dropped later, finite extreme residuals can widen SD bands and suppress CNR.
- Potential solution:
  - Fix upstream raw truncation/split handling before filtering; if unresolved, exclude R10 from group-level technical validation or run robust statistics with explicit outlier policy.

### 11) Need explicit replication harness to prove mismatch closure
- Current metrics already indicate divergence:
  - `group_hrf_stats.json`: many HbO amplitude checks fail under current thresholds; region means often very small.
  - `cnr_summary.json`: median CNR `0.2395`, max `0.6339`, while reference Figure 4i Expert-vs-Novice distribution visually extends much higher.

### 12) Additional mismatch: band-pass lower cutoff
- Discrepancy:
  - Paper text states `0–0.50 Hz` band-pass (TechEn example).
  - Current implementation uses `0.01–0.5 Hz`.
- Why this matters:
  - Small but real baseline/slow-drift behavior differences, which can affect early post-onset shape.
- Potential solution:
  - Test both `0`-equivalent and `0.01` high-pass variants and lock whichever better reproduces Figure 3c morphology.

### Recommended execution order (highest impact first)
1. Build a strict Figure-3c replication path: 5 regions, 100 s, matching channel definitions and units.
2. Replace/augment short-channel regression with Gaussian-basis OLS equivalent.
3. Create a CNR-validation branch with **no channel pruning** and paper-faithful CNR calculation mode.
4. Add explicit post-rPCA low-pass stage for validation outputs.
5. Resolve unit/scaling conventions before strict amplitude parity judgments.

---

## Section 2: Expected Values and Their Source Status (ASTaUND vs Generic)

| Metric / Parameter | ASTaUND paper value or statement | Current value in repo | Source status |
|---|---|---|---|
| Signal-quality exclusion (TechEn example) | Exclude channels outside **80–140 dB** | 80–140 dB | From ASTaUND text |
| TechEn wavelengths | **690 / 830 nm** | 690 / 830 nm | From ASTaUND text |
| Band-pass filter (TechEn example) | **0–0.50 Hz** | 0.01–0.5 Hz | Near-match, not exact |
| Motion artifact detection function | Homer3 `hmrR_MotionArtifactByChannel` with params from cited ref | Implemented custom equivalent; production uses dual criterion on filtered OD | Partially aligned, implementation diverges |
| Motion/systemic correction | **rPCA + low-pass filters** | rPCA present; no explicit additional low-pass stage in Stage B chain | Incomplete alignment |
| MBLL PPF (TechEn) | **6.4 (690), 5.8 (830)** | 6.4 / 5.8 | From ASTaUND text |
| Short-channel regression model | Most-correlated short channels regressed using **Gaussian basis sequence via OLS** | Single-regressor static OLS per long channel (`L-βS*`) | Divergent |
| HRF regions in Fig 3c | **Left Lateral PFC, Central PFC, Right Lateral PFC, Left Medial M1, SMA** | PFC, M1, SMA | Divergent |
| HRF time span in Fig 3c | Plot shown to ~**100 s** post onset | Epoch/plots to 30 s | Divergent |
| HRF y-axis units in Fig 3c | Labeled **µM*mm** | Treated as µM in pipeline checks/plots | Potential unit mismatch |
| CNR computation description | Compute group mean HRFs per population, then channel-wise CNR | Peak-based per-subject summary in 5–15 s, then CNR | Divergent |
| CNR channel-pruning policy in technical validation | **Exclude channel pruning** when computing group mean HRFs for CNR | Uses pruned Stage B outputs | Divergent |
| CNR threshold (detectable/strong) | Not explicitly defined as fixed cutoffs in ASTaUND text | `1.0` detectable, `2.0` strong | Generic (not ASTaUND-specific) |
| Expected HbO/HbR amplitude thresholds (validation) | No explicit fixed numeric pass/fail ranges given in ASTaUND text | Multiple generic values present (e.g., historical 5–15 µM; config uses 0.35–1.0 for HbO) | Generic / mixed assumptions |
| Stage A SNR expected mean range | ASTaUND reports SNR distributions in figure form; no 90–120 dB requirement | `validation.stage_a.snr_mean_range = [90,120]` | Generic/inconsistent with ASTaUND figure scale |
| OD range expected [0.1, 2.0] | Not an ASTaUND-stated criterion | Present in validation config | Generic |
| Hb range expected ±100 µM | Not ASTaUND-stated as universal acceptance criterion | Present in config and MBLL checks | Generic |

### Notes on expected values to use going forward
- Keep ASTaUND-stated values as hard constraints only where explicitly stated (80–140 dB, PPFs, core processing order/statements).
- Treat all other numeric thresholds as **project hypotheses** unless tied to ASTaUND text/figures or the referenced preprocessing spreadsheet.
- For Figure 3c/4i replication, prefer **shape/ranking/distribution matching** first, then tune numeric thresholds after units and processing equivalence are confirmed.
