# CLAUDE.md — Project Context for AI Agents

This file gives any new agent session the full context of the project without
needing it re-explained. Read this first before touching code in this repo.

---

## 1. Who / What / Why

**Lab:** Mind, Brain, and Education (MBE) Lab, under Dr. Mason Borzin.
**Researcher:** Vinh Pham.
**Dataset:** ASTaUND (Assessment of Surgical Tasks Using Neuroimaging Dataset) —
open-source, BIDS-compliant fNIRS dataset of surgeons performing laparoscopic
FLS pegboard-transfer tasks. This project uses the **Expert-Novice** cohort
(7 experts, 6 novices analyzed; subject `R10` excluded for data-quality reasons
— trailing zeros, anomalous rPCA variance removal). Three other ASTaUND
sub-studies (`Learning_Curve`, `Short-Term_tES`, `Long-Term_tES`) exist in
`data/BIDS/` but are **not yet analyzed**; all current work is Expert-Novice only.

**Research question:** Can Transfer Entropy (TE), an information-theoretic,
non-linear, model-free measure of directed connectivity, serve as a viable
complement/alternative to Granger Causality (GC) for fNIRS brain connectivity,
and can it distinguish expert vs. novice surgeons across 5 ROIs (LPFC, CPFC,
RPFC, LM1, SMA covering PFC, M1, SMA)?

**A full written report exists** at the repo root (uploaded separately as a PDF,
"MBE Lab S26 Final Report", dated **April 29, 2026**). It documents the
project through the end of the **bivariate/multivariate phase-based dSPTE**
work and is a great methods reference (math, formulas, literature review,
Stage A–D details) — **but its Transfer Entropy conclusions are now
superseded** by the pivot described in Section 3 below. Treat the report as
historical/methodological background, not as the current state of results.

---

## 2. Pipeline Architecture (Stages A–E)

The codebase is organized as a linear staged pipeline under `code/`, each stage
consuming the previous stage's output. `config.yaml` is the single source of
truth for all parameters — scripts should read from it rather than hardcoding
values. `docs/` contains one detailed progress/PRD markdown per major stage.

| Stage | Folder | Purpose | Progress doc |
|---|---|---|---|
| A | `code/stage_a_raw_qa/` | Raw signal QA (SNR, motion ratio, exclusion lists) — no exclusions applied yet, just documentation | `docs/AUDIT_PROGRESS.md`, `docs/PROGRESS.md` |
| B | `code/stage_b_preprocessing/` | Intensity→OD, bandpass filter, rPCA motion correction, modified Beer-Lambert Law, short-channel OLS regression → HbO/HbR | `docs/PROGRESS.md`, `docs/reproduced_matlab_comparison.md` |
| C | `code/stage_c_validation/` | HRF extraction, CNR, statistical validation of preprocessing quality | `docs/PROGRESS.md` |
| D | `code/stage_d_granger/` | Linear Granger Causality baseline: VAR(20) model → frequency-domain GW causality → 5×5 directed connectivity matrix | `docs/LINEAR_GC_PROGRESS.md` |
| E | `code/stage_e_transfer_entropy/` | Transfer Entropy pipeline (multiple phases — see Section 3) | `docs/TRANSFER_ENTROPY_PROGRESS.md` (**the most detailed, most current doc — read this for anything TE-related**) |

Stages A–D are essentially **complete and stable**. Stage E (Transfer Entropy)
is the active area of work and has gone through several methodological
generations, described next.

---

## 3. Stage E — Transfer Entropy: the Methodological Story (READ THIS)

Stage E's design has evolved significantly. Understanding *why* it evolved is
more important than any single phase's code.

### Phase 1 — Bivariate dSPTE (phase-based)
Implements Wang & Chen (2020) delay Symbolic Phase Transfer Entropy: Hilbert
phase extraction → time-delay embedding → Neural Gas (NG) vector quantization
→ symbolic TE. Found two problems: (1) Novice TE > Expert TE on **all 20**
directed connections (suspicious — no directional structure), and (2) lag
profiles were flat/monotonically rising instead of showing a physiological
peak at 2–8s delay. Root cause: broadband HbO Hilbert phase is quasi-uniformly
distributed in phase space, so it carries little true coupling information —
it's dominated by cardiac/respiratory noise and shared background
synchronization.

### Phase 2 — Multivariate/Hybrid dSPTE (phase, conditioned)
Conditioned each bivariate calculation on the remaining 3 ROIs (PCA-compressed
to manage the curse of dimensionality in the contingency table) to strip out
shared noise. This reduced mean TE by ~45% and fixed the lag profile for some
connections (e.g., LPFC→LM1 showed a clean peak at δ≈2.76s), confirming
conditioning suppresses indirect/shared-noise connections. But most
connections still showed flat, uninformative lag profiles, and 17/20
connections remained Novice > Expert.

### Phase 3 — Bimodal dSPTE (phase + amplitude weighted fusion)
Added a Hilbert **amplitude** branch (bivariate, broadband) alongside the
phase branch and fused them as a weighted sum
(`w_phase · TE_phase + w_amplitude · TE_amplitude`). This phase is where it
became clear that the amplitude branch was carrying most of the real signal.

### **Why the project pivoted away from phase entirely (the key insight)**
The narrowband filter (0.04–0.15 Hz, or Wang 2020's 0.01–0.10 Hz) was
originally applied so the Hilbert transform would have something meaningful
to extract a phase from — Hilbert phase is only physically interpretable for
narrowband, slowly-oscillating signals. Applied to broadband fNIRS, instantaneous
phase is essentially noise-driven (spikes from any strong frequency component).
Without the filter, the NG codebook basically never converged (`nDis` floor
stuck ~0.05, needing k₀ ≈ 20–35 codebook vectors). With the filter, convergence
jumped from ~60% to ~100% of pairs and k₀ dropped to 7–13 — but at the cost of
restricting each ROI's phase trajectory to a small arc of `[-π, π]`, so the NG
codebook now mostly just encodes *where the signal is on the HRF rise/fall*,
not genuine directional coupling structure. This is evidenced by full entropy
and marginal entropy being nearly identical → conditional mutual information
(the TE itself) is tiny and uninformative. In short: **the narrowband filter
fixed the numerics but destroyed the thing we were trying to measure.**
Amplitude, by contrast, is naturally well-behaved on broadband signals because
hemodynamic amplitude directly tracks neural response magnitude (task-locked
clustering), symbolizes cleanly with few codebook vectors, and needs no
narrowband filtering.

**Conclusion: the project has moved off phase-based TE entirely.** Phase-based
methods (Phase 1/2/3's phase branch) are kept in the repo for
documentation/comparison purposes but are no longer the direction of the
project.

### Phase 4 — Multivariate Conditional Amplitude dSTE (**current primary method**)
Location: `code/stage_e_transfer_entropy/phase_4/`. This is a **novel
contribution** (not directly published elsewhere): applies the Wang (2020)
NG-symbolization + multivariate PCA-conditioning framework (previously used on
phase) entirely to the **Hilbert amplitude envelope of broadband (unfiltered)
HbO**. No phase computation, no narrowband filtering anywhere in this phase.
Naming note: dropped the "P" (dSP**T**E → dS**T**E) because there is no phase
component left at all.

- `14_amplitude_dste.py` — core: `extract_amplitude_matrix`,
  `symbolise_multivariate_amplitude`, `compute_amplitude_conditional_dste_matrix`
- `15_amplitude_dste_group.py` — group aggregation (Expert vs Novice arrays,
  3 strategies: all-windows, subject-mean, first-window)
- `16_amplitude_dste_visualization.py` — connectivity heatmaps, Expert-vs-Novice
  delta bars, bivariate-vs-multivariate suppression comparison
- `17_amplitude_dste_statistics.py` — normality (Shapiro-Wilk), H1 Expert vs
  Novice (Mann-Whitney U + FDR, compared against Phase 3's bivariate amplitude
  branch to quantify whether conditioning sharpens group separation), H0
  within-trial stability, HIV across-trial stability
- `18_asymmetry_index_statistics.py` — asymmetry index (AI = TE(X→Y) − TE(Y→X))
  diagnostics, comparing Phase 4 TE asymmetry against GC directionality
- `phase_4_debug.py` — NG convergence diagnostics for the amplitude embedding
  space (confirms amplitude data clusters non-uniformly, unlike phase)

**Status: functionally complete (Section 9 of the progress doc, 9/9 tasks).**
Outputs already exist in
`outputs/stage_e_transfer_entropy/expert-novice/phase_4_amplitude_dste/`.

**Current numeric result (as of the last stats run):** Novice > Expert mean
amplitude-dSTE on 18/20 connections (Expert > Novice on only 2/20). **Zero**
FDR-significant connections in the H1 Mann-Whitney test (n=100 tests across
5 trials × 20 connections), consistent with every other Stage D/E method so
far — this is attributed to the small cohort (n=13) rather than a pipeline
failure, matching the pattern seen in GC and all three prior TE phases.

### What's NOT done yet in Stage E
- **Section 10 (Cross-Phase Comparison)** — not started. Should compile GC +
  Phase 1–4 TE results into one comparison table/correlation matrix
  (`code/stage_e_transfer_entropy/08_cross_phase_comparison.py`, not yet created).
- **Section 11 (Batch Runner)** — not started; a single CLI entry point to run
  the whole Stage E pipeline end-to-end doesn't exist yet.
- **Interpretation.** This is the actual current focus per the researcher: the
  Phase 4 amplitude dSTE numbers exist, but their *meaning* (why Novice > Expert
  dominates, whether this reflects genuine neural effort/cognitive-load
  differences vs. a residual confound) is still being worked out. Two
  interpretive angles under consideration:
  1. **Asymmetry Index (AI)** analysis (already partially implemented in
     `18_asymmetry_index_statistics.py`) — used in Phase 1 to show experts have
     1.74–2.47× greater *directional* asymmetry even when raw TE magnitude is
     higher for novices, suggesting novices' elevated TE is more
     non-directional/noise-like.
  2. **Effective Transfer Entropy (ETE)** — surrogate-corrected TE
     (TE minus TE computed on shuffled/surrogate data) is being considered as a
     next analytical step to separate genuine coupling from estimation bias.
     Not yet implemented anywhere in the codebase as of this writing — if you
     are asked to implement it, there is no existing reference implementation
     to adapt; you'd be building it fresh (IDTxl's permutation-based
     `p_omnibus`/surrogate testing under `phase_2/10_idtxl_validation.py` is
     the closest existing infrastructure, since IDTxl already does
     surrogate-based significance testing).
- An **IDTxl parallel validation** track exists
  (`code/stage_e_transfer_entropy/phase_2/10_idtxl_validation.py`,
  `pace_idtxl_job.sh` for HPC/PACE cluster batch submission) using the KSG
  estimator as a model-free cross-check against the NG-symbolic approach. This
  was originally scoped for the phase-based multivariate TE; whether/how it
  gets extended to validate Phase 4's amplitude approach is open.

---

## 4. Repo Conventions

- **Everything is parameterized through `config.yaml`.** The
  `stage_e_transfer_entropy` block has sub-blocks per phase
  (`phase_2`, `phase_3`, `phase_4`, etc.) with extensive inline comments
  explaining *why* each parameter has its value (e.g., k₀ ceiling derivations,
  contingency-table sparsity math). Read the comments before changing values.
- **Numbered scripts within each phase folder** run in order (e.g.
  `14_amplitude_dste.py` → `15_..._group.py` → `16_..._visualization.py` →
  `17_..._statistics.py`) but each is independently runnable via CLI flags
  (typically `--smoke-test`, `--subject`, `--batch`, `--overwrite`).
- **Output schema is intentionally identical across Stage D and all Stage E
  phases**: 20-element directed-connectivity vectors (5×5 matrix, diagonal
  zeroed), `(n_subjects, 20, n_windows)` group arrays, the same 4 statistical
  tests (Shapiro-Wilk normality, H1 Expert-vs-Novice, H0 within-trial
  stability, HIV across-trial stability) — this is deliberate so results are
  directly comparable across methods without reshaping.
- **`docs/TRANSFER_ENTROPY_PROGRESS.md`** (~1,377 lines) is the living design
  doc/checklist for Stage E — it has a numbered task list per section with
  `[x]`/`[ ]` checkboxes and a `> **Result (...)**` note under each completed
  task documenting what was actually implemented and why. **This is the
  single best file to read for a deep dive into Stage E** — far more detailed
  than this CLAUDE.md.
- Progress docs generally follow the pattern: PRD-style task list up front,
  running log of issues/root-cause investigations at the bottom (see "PHASE 3
  AUDIT" section in `TRANSFER_ENTROPY_PROGRESS.md` for a good example of the
  investigative style used in this project — root cause, empirical
  confirmation, fix, impact assessment).
- `report.log*` files at the repo root are run logs from pipeline executions,
  not meaningful documentation — safe to ignore/gitignore.
- `IDTxl/` is a vendored third-party library (the IDTxl Python package),
  not project code.

---

## 5. Quick Orientation for a New Agent Session

If asked to work on this project without further context, in priority order:
1. Read `docs/TRANSFER_ENTROPY_PROGRESS.md` (especially the end, for latest status).
2. Check `git log --oneline -20` for the most recent work.
3. Assume **Phase 4 (amplitude dSTE) is the current primary method**; phase-based
   TE (Phases 1–3) is retained for comparison/documentation, not active development.
4. Assume the immediate open work is: (a) interpreting Phase 4 results
   (AI, possibly ETE), (b) cross-phase comparison (Section 10), (c) a unified
   batch runner (Section 11) — ask the researcher which of these is the
   priority before starting new implementation work.
5. The final report PDF (April 29, 2026) is useful for methodology/writing
   style and for Stages A–D, but its TE narrative is one generation behind
   the current amplitude-only approach — don't treat its TE conclusions as
   current ground truth.
