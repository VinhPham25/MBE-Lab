# AUDIT_PROGRESS.md - ASTaUND Reproduction Audit PRD

**Last Updated:** February 25, 2026  
**Current Phase:** Audit Planning and Execution  
**Current Focus:** Stage B/Stage C parity to ASTaUND paper (Figure 3c and Figure 4i replication)

---

## PROJECT STATE OF THE UNION

### Current Status
- **Goal:** Resolve the 12 confirmed divergence issues identified in `preprocessing_review.md`
- **Baseline Condition:** Stage A quality assessment already matches paper behavior well
- **Main Risk Area:** Stage B preprocessing fidelity and Stage C validation methodology
- **Primary Success Metric:** Reproduce paper-consistent HRF morphology and CNR distribution under paper-aligned processing assumptions

### Recent Changes
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
- [ ] 5-region layout matches paper panel definitions.
- [ ] HRF plots extend to ~100 s with correct onset marker behavior.
- [ ] Unit convention confirmed and documented (including any `uM*mm` conversion/scaling).
- [ ] Visual shape comparison package generated vs paper Figure 3c.

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

### Issue 1 - HRF target mismatch (3 regions/30 s vs 5 regions/100 s)
**Priority:** P0

- [ ] **1.1** Define exact Figure 3c region schema: Left Lateral PFC, Central PFC, Right Lateral PFC, Left Medial M1, SMA.
- [ ] **1.2** Implement dedicated 5-region channel mapping for Expert/Novice.
- [ ] **1.3** Update epoch/plot window to support Figure 3c horizon (~100 s post-onset).
- [ ] **1.4** Generate side-by-side comparison plot set (current vs paper-faithful layout).
- [ ] **1.5** Document remaining morphological mismatch by region.

**Acceptance Criteria:** HRF output is 1-to-1 comparable with Figure 3c structure and timeline.

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

### Issue 6 - Potential unit/scaling mismatch (`uM*mm` vs `uM`)
**Priority:** P0

- [ ] **6.1** Verify ASTaUND figure unit interpretation and Homer export scaling assumptions.
- [ ] **6.2** Audit current MBLL output units and all downstream unit conversions.
- [ ] **6.3** Apply explicit conversion/scaling factor if required.
- [ ] **6.4** Update plot labels and expected-value checks accordingly.

**Acceptance Criteria:** Unit convention is unambiguous, documented, and consistently applied end-to-end.

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
- [ ] Issue 1
- [ ] Issue 2
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
