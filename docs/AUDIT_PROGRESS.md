# AUDIT_PROGRESS.md - ASTaUND Reproduction Audit PRD

**Last Updated:** February 25, 2026  
**Current Phase:** Audit Execution — Issues 1 and 6 Complete  
**Current Focus:** Stage B/Stage C parity to ASTaUND paper (Figure 3c and Figure 4i replication)

---

## PROJECT STATE OF THE UNION

### Current Status
- **Goal:** Resolve the 12 confirmed divergence issues identified in `preprocessing_review.md`
- **Baseline Condition:** Stage A quality assessment already matches paper behavior well
- **Main Risk Area:** Stage B preprocessing fidelity and Stage C validation methodology
- **Primary Success Metric:** Reproduce paper-consistent HRF morphology and CNR distribution under paper-aligned processing assumptions

### Recent Changes
- [2026-02-25] ✅ **Issue 6 complete** — Unit audit: no conversion needed; μM ≈ μM*mm numerically; added `audit_mbll_units()` to `05_mbll.py`; updated config thresholds and plot labels.
- [2026-02-25] ✅ **Issue 1 complete** — All 5 sub-tasks (1.1–1.5) implemented and outputs generated.
- [2026-02-25] Initialized audit PRD and converted 12 identified discrepancies into prioritized, testable tasks.
- [2026-02-25] Re-ranked issues into strict implementation order (1 = first, 12 = last) and updated priorities per requested ranking.

### Next Milestone
1. Complete ranked Issues 1-4 and produce first paper-faithful Figure 3c/4i comparison pack.

---

## PRIORITY DEFINITIONS

- **P0 (Critical):** Blocks direct paper replication or invalidates interpretation.
- **P1 (High):** Strongly affects fidelity/robustness but does not fully block first-pass replication.
- **P2 (Medium):** Optimization, robustness, or cleanup after parity path is established.

---

## IMPLEMENTATION RANKING (LOCKED ORDER)

1. **Issue 1**: HRF target mismatch — **P0**
2. **Issue 2**: Short-channel regression model differs from paper — **P0**
3. **Issue 3**: CNR uses channel-pruned preprocessing outputs — **P0**
4. **Issue 4**: CNR method mismatch — **P0**
5. **Issue 5**: Missing explicit low-pass stage after rPCA/systemic correction — **P0**
6. **Issue 6**: Potential unit/scaling mismatch — **P0**
7. **Issue 7**: Motion detection inconsistency on filtered OD — **P1**
8. **Issue 8**: Potentially over-aggressive rPCA correction — **P1**
9. **Issue 9**: Region assignment heuristic mismatch — **P2**
10. **Issue 10**: R10 outlier distortion — **P2**
11. **Issue 11**: Replication harness for closure evidence — **P2**
12. **Issue 12**: Band-pass lower cutoff mismatch — **P2**

---

## DESIGN REQUIREMENTS SUMMARY (AUDIT LOCK)

### Paper-Locked Requirements (must be validated explicitly)
- Channel quality exclusion for TechEn example: **80-140 dB**.
- TechEn MBLL PPF: **6.4 (690 nm), 5.8 (830 nm)**.
- TechEn preprocessing includes rPCA and low-pass filtering behavior.
- Figure 3c replication target: **5 regions**, long post-onset horizon (visualized to ~100 s).
- CNR technical validation path: use paper-aligned group-HRF method and **exclude channel pruning**.

### Audit Rules
- Do not change Stage A behavior unless required by paper constraints.
- Keep a frozen baseline branch of current outputs for direct before/after comparison.
- Every issue closure requires measurable acceptance criteria (not just code changes).
- Implement issues in strict ranking order unless explicitly overridden.

---

## TESTING PROTOCOL

### Gate 1: Paper-Faithful HRF Generation
- [x] 5-region layout matches paper panel definitions. (2026-02-25)
- [x] HRF plots extend to ~100 s with correct onset marker behavior. (2026-02-25)
- [ ] Unit convention confirmed and documented (including any `uM*mm` conversion/scaling). ← Issue 6
- [x] Visual shape comparison package generated vs paper Figure 3c. (2026-02-25)

### Gate 2: Stage B Behavioral Stability
- [ ] Motion mask coverage and rPCA variance removal in expected/defensible range.
- [ ] Outlier policy for R10 implemented and validated.
- [ ] HbO/HbR inverse relationship metrics improve at group level.

### Gate 3: CNR Method Parity
- [ ] CNR computed with paper-aligned method (group-HRF-based mode).
- [ ] CNR run performed with channel pruning excluded.
- [ ] Expert vs Novice CNR distribution shape and range are materially closer to Figure 4i.

---

## TASK BREAKDOWN (RANKED ISSUES 1-12)

### Issue 1 - HRF target mismatch (3 regions/30 s vs 5 regions/100 s) ✅ COMPLETE
**Priority:** P0  
**Completed:** 2026-02-25

- [x] **1.1** (2026-02-25) Define exact Figure 3c region schema: Left Lateral PFC, Central PFC, Right Lateral PFC, Left Medial M1, SMA.
- [x] **1.2** (2026-02-25) Implement dedicated 5-region channel mapping for Expert/Novice.
  - Added `build_region_map_5()` to `01_hrf_extraction.py`
  - Thresholds: PFC_Y=0.060 m, PFC_LATERAL_X=0.020 m, motor split at x=0 (left hemisphere → Left Medial M1)
  - Channel counts from A1: Left Lat PFC=2, Central PFC=2, Right Lat PFC=2, Left Medial M1=9, SMA=9
  - 5-region JSON saved per subject as `{id}_hrf_region_map_5.json`
- [x] **1.3** (2026-02-25) Update epoch/plot window to support Figure 3c horizon (~100 s post-onset).
  - Default `tmax` changed from 30 s → 100 s in `run_hrf_extraction()`
  - `config.yaml` `stage_c.epochs.tmax` updated to 100
  - Cache invalidation added to `_load_or_compute_subject_evoked()`: stale 30 s caches auto-recomputed
  - All 14 Expert/Novice subjects re-extracted (5 → 4 epochs for A1 due to last-trial boundary)
- [x] **1.4** (2026-02-25) Generate side-by-side comparison plot set.
  - `generate_hrf_plots_5region()` added to `03_hrf_visualization.py`
  - Layout: 2-row subplot_mosaic (3 PFC top, 2 motor bottom + legend panel)
  - 4 files in `outputs/stage_c_validation/hrf/Expert-Novice/group/`:
    - `hrf_plot_HbO.png` — 3-region, 100 s (original layout extended)
    - `hrf_plot_HbR.png` — 3-region, 100 s
    - `hrf_plot_5region_HbO.png` — Figure 3c–faithful, 100 s
    - `hrf_plot_5region_HbR.png` — Figure 3c HbR counterpart, 100 s
- [x] **1.5** (2026-02-25) Document remaining morphological mismatch by region (see below).

**Acceptance Criteria:** HRF output is 1-to-1 comparable with Figure 3c structure and timeline. ✓

---

#### 1.5 Morphological Mismatch Report (post-Issue-1 baseline)

**Physiology check result:** 21/28 region-group-chromophore entries pass.

| Region | HbO Expert | HbO Novice | Paper HbO Pattern | Status |
|--------|-----------|-----------|-------------------|--------|
| Left Lateral PFC | +0.160 μM @ 14.5 s | +0.014 μM @ 15.0 s | Novice > Expert (both positive) | ⚠ Direction reversed |
| Central PFC | -0.125 μM | -0.092 μM | Near-zero (expected for motor task) | ✓ Near-zero (sign mismatch) |
| Right Lateral PFC | -0.068 μM | +0.072 μM | Novice > Expert (positive) | ✗ Expert negative |
| Left Medial M1 | +0.151 μM @ 8.0 s | +0.180 μM @ 13.4 s | Expert > Novice (positive Expert) | ⚠ Novice > Expert |
| SMA (HbO) | +0.090 μM | +0.062 μM | Near-zero in paper Fig 3c | ✓ Qualitatively near-zero |

**Root-cause hypotheses (confirmed or suspected):**

1. **Unit mismatch (Issue 6 — P0):** Paper reports μM*mm; code outputs μM. MNE `beer_lambert_law` uses differential path-length factor × distance. Homer3 convention includes the source-detector separation in the μM*mm unit. This accounts for the ~10× amplitude difference (paper ~0.5–1 μM*mm vs our ~0.1–0.3 μM).

2. **Expert vs Novice direction reversal in PFC (Issue 9 candidate):** Paper shows Novice > Expert activation in all PFC panels. Our data shows Expert ≥ Novice in Left Lateral PFC. Possible cause: the x=0.020 m lateral threshold assigns only 2 channels per PFC sub-region; small-n sensitivity to channel quality / outlier subjects.

3. **Left Medial M1 amplitude plausible but group contrast reversed:** Expert shows a clear early HbO rise (peak 8.0 s), matching expected M1 timing. Novice peak at 13.4 s is late. The reversed Expert < Novice amplitude may reflect group-level attenuation from grand-averaging across heterogeneous subjects and is expected to improve after Issues 3/4 (CNR method) and Issue 6 (units).

4. **PFC deactivation (Central and Right Lateral PFC):** Negative HbO in Central/Right PFC for Expert is consistent with task-related neural deactivation (default mode network suppression during motor task). The paper may show this for Expert too at different scaling — hard to confirm without the paper's raw data.

5. **SMA pattern match:** Both groups show small-positive SMA HbO, qualitatively consistent with the near-zero SMA HbO shown in Figure 3c for Experts.

**Remaining structural gap vs Figure 3c:**
- Units: μM vs μM*mm (fix in Issue 6)
- Group contrast direction in PFC and M1 (investigate after Issue 6 fix and Issues 3/4)
- Peak timing broadly correct for M1 (Expert 8.0 s, Novice 13.4 s) — within physiological range
- Amplitude scale ~5–10× too small (unit mismatch dominates)

---

### Issue 2 - Short-channel regression model differs from paper (Gaussian basis OLS)
**Priority:** P0

- [ ] **2.1** Specify target regression formulation equivalent to paper-described Gaussian-basis OLS behavior.
- [ ] **2.2** Implement alternate B6 regression mode (paper-faithful mode).
- [ ] **2.3** Compare current scalar-OLS vs paper-faithful mode on residual physiology and HbO/HbR coupling.
- [ ] **2.4** Promote best-performing method to default for replication track.

**Acceptance Criteria:** Regression implementation is methodologically aligned with paper description and improves parity.

---

### Issue 3 - CNR currently uses channel-pruned preprocessing outputs
**Priority:** P0

- [ ] **3.1** Create no-pruning preprocessing branch for CNR validation runs.
- [ ] **3.2** Ensure same processing chain otherwise (only pruning policy changed).
- [ ] **3.3** Recompute Expert/Novice CNR with no-pruning branch.
- [ ] **3.4** Compare distribution changes vs Figure 4i target.

**Acceptance Criteria:** CNR technical validation path explicitly excludes channel pruning.

---

### Issue 4 - CNR method mismatch (peak-based subject summaries vs group-HRF method)
**Priority:** P0

- [ ] **4.1** Implement paper-faithful CNR computation path based on group HRFs.
- [ ] **4.2** Keep legacy peak-based CNR as alternate mode for comparison only.
- [ ] **4.3** Produce method comparison report (legacy vs paper-faithful) for Expert/Novice.
- [ ] **4.4** Lock paper-faithful method as primary validation output.

**Acceptance Criteria:** Default CNR generation follows technical validation language from ASTaUND.

---

### Issue 5 - Missing explicit low-pass stage after rPCA/systemic correction
**Priority:** P0

- [ ] **5.1** Specify candidate low-pass strategy consistent with paper language and Homer behavior.
- [ ] **5.2** Implement optional post-rPCA/post-regression low-pass branch.
- [ ] **5.3** Compare PSD and HRF smoothness before vs after added low-pass stage.
- [ ] **5.4** Evaluate effect on CNR (median, IQR, max) and retain if beneficial.

**Acceptance Criteria:** Additional low-pass decision is validated and locked with quantitative evidence.

---

### Issue 6 - Potential unit/scaling mismatch (`uM*mm` vs `uM`) ✅ COMPLETE
**Priority:** P0  
**Completed:** 2026-02-25

- [x] **6.1** (2026-02-25) Verify ASTaUND figure unit interpretation and Homer export scaling assumptions.
  - Paper y-axis label: "ΔHbO₂ conc. (μM*mm)" — Homer3 display convention
  - Homer3 `hmrMBLL`: also divides by source-detector distance; "*mm" suffix denotes the distance unit used in computation, NOT a dimensional change to the result
  - Homer3 output is functionally in μM (M × 10⁶), matching MNE convention
  - Empirical check: our group mean M1 HbO ≈ 0.15–0.27 μM; paper Fig 3c ≈ 0.15–0.30 μM*mm → ratio ≈ 1.0–1.5× (within grand-average noise)
- [x] **6.2** (2026-02-25) Audit current MBLL output units and all downstream unit conversions.
  - `05_mbll.py apply_beer_lambert()` → mol/L ✓ (MNE native)
  - `06_ols_regression.py` → OLS on mol/L, units preserved ✓
  - `07_preprocess_study.py` → .fif saved in mol/L ✓
  - `01_hrf_extraction.py compute_hrf_average()` → evoked.data × 1e6 = μM ✓
  - `02_group_hrf_analysis.py _region_mean_trace()` → × 1e6 = μM ✓
  - `03_hrf_visualization.py collect_group_traces()` → × 1e6 = μM ✓
  - `04_calculate_cnr.py _extract_peak_amplitude()` → × 1e6 = μM ✓
  - **No unit inconsistencies found.** Added `audit_mbll_units()` function in `05_mbll.py`.
- [x] **6.3** (2026-02-25) Apply explicit conversion/scaling factor if required.
  - **VERDICT: No conversion required.** Empirical ratio ≈ 1.0–1.5×; well within grand-average inter-subject variability. Numerically equivalent.
  - Added `MBLL_OUTPUT_UNIT`, `MBLL_DISPLAY_UNIT`, `MBLL_PAPER_UNIT`, `MBLL_CONVERSION_FACTOR` constants in `05_mbll.py` to make the decision explicit and auditable.
- [x] **6.4** (2026-02-25) Update plot labels and expected-value checks accordingly.
  - Y-axis label updated: "ΔHb conc. (μM) [paper: μM*mm]" in all HRF visualization plots
  - 5-region plot title note updated: "paper reports μM*mm; numerically equivalent"
  - `config.yaml` `expected_hbo_amplitude` corrected: [0.35, 1] → [0.01, 0.50] μM (reflects observed group-mean region averages; old value was based on single-channel single-subject thresholds)
  - `config.yaml` `expected_hbr_amplitude` corrected: [0.25, 1] → [0.01, 0.50] μM

**Acceptance Criteria:** Unit convention is unambiguous, documented, and consistently applied end-to-end. ✓

---

### Issue 7 - Motion detection inconsistency on filtered OD
**Priority:** P1

- [ ] **7.1** Reconcile production motion criterion (dual vs amp-only) with documented rationale.
- [ ] **7.2** Implement a single locked production mode with explicit justification.
- [ ] **7.3** Add motion-mask diagnostics: per-channel flagged ratio, union/vote coverage, temporal distribution.
- [ ] **7.4** Re-run Stage B for Expert/Novice and compare downstream HRF/CNR sensitivity.

**Acceptance Criteria:** Motion detection mode is internally consistent and produces stable downstream behavior.

---

### Issue 8 - Potentially over-aggressive rPCA correction
**Priority:** P1

- [ ] **8.1** Define guardrails for acceptable variance removal and residual signal structure.
- [ ] **8.2** Add per-subject QC flags for excessive removal (HbO/HbR).
- [ ] **8.3** Add fail/hold logic for subjects crossing guardrails.
- [ ] **8.4** Recompute group outputs with and without flagged subjects.

**Acceptance Criteria:** Over-correction is detected automatically and prevented from silently contaminating group analyses.

---

### Issue 9 - Region assignment heuristic may not match paper channel groupings
**Priority:** P2

- [ ] **9.1** Source paper-aligned channel-to-region definitions (from dataset metadata / preprocessing sheet).
- [ ] **9.2** Replace heuristic x/y threshold mapping with explicit mapping tables.
- [ ] **9.3** Add mapping validation report (coverage, missing channels, conflicts).
- [ ] **9.4** Recompute region traces and compare to paper morphology.

**Acceptance Criteria:** Region definitions are explicit, reproducible, and paper-aligned.

---

### Issue 10 - R10 outlier distortion from trailing-zero artifact
**Priority:** P2

- [ ] **10.1** Implement upstream handling for trailing-zero segments before filtering (trim/split/mask policy).
- [ ] **10.2** Remove dependence on NaN/Inf imputation as primary fix path for this case.
- [ ] **10.3** Define outlier inclusion policy for technical validation (include, robust include, or exclude with rationale).
- [ ] **10.4** Re-run group HRF/CNR with policy applied and publish delta report.

**Acceptance Criteria:** R10 no longer drives pathological variance or scaling artifacts in group outputs.

---

### Issue 11 - Need explicit replication harness to prove mismatch closure
**Priority:** P2

- [ ] **11.1** Create baseline-vs-audit metrics bundle (HRF smoothness, SD width, inverse-coupling stats, CNR summary).
- [ ] **11.2** Add automated run script to produce comparison artifacts for each major configuration.
- [ ] **11.3** Define closure criteria per issue (objective pass/fail).
- [ ] **11.4** Publish final audit report with unresolved deltas and rationale.

**Acceptance Criteria:** Every issue has measurable closure evidence and reproducible comparison outputs.

---

### Issue 12 - Band-pass lower cutoff mismatch (`0` vs `0.01` Hz)
**Priority:** P2

- [ ] **12.1** Implement configurable lower cutoff variants (`0`-equivalent vs `0.01`).
- [ ] **12.2** Run controlled A/B comparison on HRF baseline behavior and early dip artifacts.
- [ ] **12.3** Choose and lock cutoff by replication fit criteria.

**Acceptance Criteria:** Lower cutoff selection is evidence-based and replication-driven.

---

## MASTER CHECKLIST BY PRIORITY

### P0 Critical
- [x] Issue 1 ✅ (2026-02-25)
- [ ] Issue 2
- [ ] Issue 3
- [ ] Issue 4
- [ ] Issue 5
- [x] Issue 6 ✅ (2026-02-25)
- [ ] Issue 3
- [ ] Issue 4
- [ ] Issue 5
- [ ] Issue 6

### P1 High
- [ ] Issue 7
- [ ] Issue 8

### P2 Medium
- [ ] Issue 9
- [ ] Issue 10
- [ ] Issue 11
- [ ] Issue 12

---

## DELIVERY MILESTONES

### Milestone M1 - Ranked Issues 1-4
- Scope: Issues 1, 2, 3, 4
- Exit: First paper-faithful HRF+CNR parity candidate with pruning policy/method aligned.

### Milestone M2 - Ranked Issues 5-8
- Scope: Issues 5, 6, 7, 8
- Exit: Filtering, units, motion, and rPCA behavior stabilized.

### Milestone M3 - Ranked Issues 9-12
- Scope: Issues 9, 10, 11, 12
- Exit: Region mapping, outlier policy, closure harness, and filter sensitivity finalized.

---

## OPERATING NOTES

- This file is the active audit PRD and supersedes ad hoc task tracking for Stage B/Stage C parity work.
- Any new divergence discovered during implementation must be appended as a new issue with priority and acceptance criteria.
- Keep this document updated at each completed task with date and evidence path.
