# scripts/ — Pipeline Run Scripts

This folder exists so that returning to this project after a hiatus doesn't
require re-deriving CLI flags from `argparse` blocks buried in `code/`. Every
script here is a thin `bash` wrapper around one Python entry point in `code/`
— it does nothing clever, it just `cd`s to the repo root and calls
`python3 code/.../NN_script.py "$@"`, so **any flag the underlying Python
script supports also works on the wrapper**.

Run any script from anywhere — they resolve the repo root via
`git rev-parse --show-toplevel`, so `cd`'ing into the subfolder first isn't
required. All scripts are already `chmod +x`.

For the full methodological story (why Stage E went through 4 phases, why
phase-based TE was abandoned, etc.), read `CLAUDE.md` at the repo root first,
then `docs/TRANSFER_ENTROPY_PROGRESS.md` for Stage E specifics.

## Conventions

- **Numbering matches the wrapped Python file.** `14_process.sh` wraps
  `14_amplitude_dste.py`. If a folder's Python scripts aren't numbered
  (e.g. `verify_event_markers.py`), the wrapper isn't either.
- **Every script's args below are copy-pasted from that script's own
  `--help` output** (or the underlying function signature, for the handful
  of scripts with no CLI at all), so they won't drift out of date silently
  — if a flag stops working, `--help` is the source of truth, this file is
  just a cache of it.
- **`run_all.sh`** in each folder chains that folder's scripts in pipeline
  order, calling each with *no* arguments (so each falls back to its own
  default — see that script's row below). It stops on the first failure
  (`set -euo pipefail`). It does not forward arguments itself.
- **"Batch" scripts default to running everything when called with no
  arguments**, even when the underlying Python script's own default is a
  smaller smoke test or single-subject run (`A1`). This is done with a
  small `if [ "$#" -eq 0 ]; then set -- --batch; fi` guard in the wrapper —
  called out explicitly below wherever it applies.
- Pass `--help` to any wrapper to see the real (forwarded) argparse help
  for the underlying script, e.g. `./14_process.sh --help`.
- Everything is still ultimately driven by `config.yaml` — these scripts
  don't hardcode parameters, they just save you from typing the path and
  remembering the flags.

# Example

cd "/Users/vinhpham/Desktop/Local GitHub/MBE-Lab"
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/run_all.sh"


---

## Stage A Raw QA (`code/stage_a_raw_qa/`)

Raw signal QA (SNR, motion ratio, exclusion lists) — no exclusions applied
here, just documentation. Scripts `01`–`06` in `code/` are internal modules
with only self-test blocks; `07` is the real orchestrator.

### `07_run_qa_report.sh` — wraps `07_generate_qa_report.py`
Main QA driver: SNR + motion ratio for all subjects, cumulative study report.

**Args**
- `--dataset DATASET` — dataset key from `config.yaml` (default: `expert_novice`; also accepts `learning_curve`, `short_term_tes`, `long_term_tes`)
- `--output-subdir OUTPUT_SUBDIR` — Stage A output subdirectory under `outputs/stage_a_qa/`
- `--skip-batch` — skip per-subject processing, only build cumulative outputs from existing artifacts
- `--quiet` — reduce console logging
- `--cleanup-subject-files` — after cumulative outputs are built, delete per-subject CSV/JSON/PNG artifacts

```bash
./07_run_qa_report.sh
./07_run_qa_report.sh --dataset learning_curve --quiet
```

### `08_run_signal_exclusion.sh` — wraps `08_signal_amplitude_exclusion.py`
Trailing-zero / amplitude-based exclusion check (documents the R10 exclusion).

**Args:** none — this script has no CLI, it always runs its built-in test/check.

```bash
./08_run_signal_exclusion.sh
```

### `run_all.sh`
Runs `07_run_qa_report.sh` then `08_run_signal_exclusion.sh`, no args accepted.

```bash
./run_all.sh
```

---

## Stage B Preprocessing (`code/stage_b_preprocessing/`)

Intensity→OD, bandpass filter, rPCA motion correction, modified Beer-Lambert
Law, short-channel OLS regression → HbO/HbR.

### `07_run_full_pipeline.sh` — wraps `07_preprocess_study.py`
Full B1→B6 batch for a study. Already the "run everything" script — no
`run_all.sh` needed for this folder.

**Args**
- `--study {expert_novice,learning_curve,short_term_tes,long_term_tes}` — dataset to process (default: `expert_novice`)
- `--n_components N_COMPONENTS` — rPCA variance threshold (default from `config.yaml stage_b.rpca.n_components`, currently `0.8`)
- `--min_channels MIN_CHANNELS` — rPCA min channels flagged in union mask (default from `config.yaml stage_b.rpca.min_channels_flagged`, currently `5`)
- `--subjects [SUBJECTS ...]` — process only these subjects (default: all in study)
- `--verbose` — enable verbose pipeline output

```bash
./07_run_full_pipeline.sh
./07_run_full_pipeline.sh --subjects A1 R10 --verbose
```

### `run_verify_event_markers.sh` — wraps `verify_event_markers.py`
Spot-check that event markers (1=start, 2=end) survive preprocessing and
rest/task periods are still mixed. Hardcoded to subject A1 — a debug utility,
not a batch driver.

**Args:** none.

```bash
./run_verify_event_markers.sh
```

---

## Stage C Validation (`code/stage_c_validation/`)

HRF extraction, CNR, statistical validation of preprocessing quality.

### `01_run_hrf_extraction.sh` — wraps `01_hrf_extraction.py`
Single-subject HRF epoch extraction + morphology check. Mostly a debug
tool — `02_run_group_hrf_analysis.sh` computes/caches HRFs for everyone
automatically, so you usually don't need to run this first.

**Args**
- `--subject SUBJECT` — subject ID (default: `A1`)
- `--study STUDY` — study key from config (default: `expert_novice`)

```bash
./01_run_hrf_extraction.sh
./01_run_hrf_extraction.sh --subject R7 --study expert_novice
```

### `02_run_group_hrf_analysis.sh` — wraps `02_group_hrf_analysis.py`
Group-level grand averages, Expert vs Novice. De facto "run everything"
driver for HRF extraction.

**Args:** none — always runs for all configured Expert/Novice subjects.

```bash
./02_run_group_hrf_analysis.sh
```

### `03_run_hrf_visualization.sh` — wraps `03_hrf_visualization.py`
GLM-first HRF evoked plots.

**Args**
- `--study STUDY` — study key in `config.yaml`

```bash
./03_run_hrf_visualization.sh
./03_run_hrf_visualization.sh --study expert_novice
```

### `04_run_cnr_analysis.sh` — wraps `04_calculate_cnr.py`
Main channel-wise Contrast-to-Noise Ratio analysis, Expert vs Novice.

**Args:** none.

```bash
./04_run_cnr_analysis.sh
```

### `04_run_technical_validation_cnr.sh` — wraps `04_technical_validation_cnr.py`
Alternate CNR pipeline using a minimal/independent preprocessing branch, used
as a cross-check against `04_run_cnr_analysis.sh`.

**Args**
- `--force` — force recompute all cached `.fif`/evoked files (needed after changing `config.yaml`, e.g. bandpass `lpf`)

```bash
./04_run_technical_validation_cnr.sh
./04_run_technical_validation_cnr.sh --force
```

### `run_hyperparam_sweep.sh` — wraps `hyperparam_sweep_runner.py`
Sweeps preprocessing hyperparameters against HRF/CNR quality metrics.

**Args:** none (sweep grid is defined inside the script/config, not via CLI).

```bash
./run_hyperparam_sweep.sh
```

### `run_all.sh`
Runs `02_run_group_hrf_analysis.sh` → `03_run_hrf_visualization.sh` →
`04_run_cnr_analysis.sh` → `04_run_technical_validation_cnr.sh`, no args
accepted (skips `01` and the hyperparam sweep — run those individually).

```bash
./run_all.sh
```

---

## Stage D Linear GC (`code/stage_d_granger/`)

Linear Granger Causality baseline: VAR(20) model → frequency-domain GW
causality → 5×5 directed connectivity matrix. **Currently only supports
`--study expert_novice`** — subject lists and validation assertions (7
Experts, 6 Novices) are hardcoded to that cohort in `config.yaml`.
(`03_linear_gc.py` is the core GC math library — no standalone wrapper.)

### `01_run_roi_extraction.sh` — wraps `01_extract_roi_signals.py`
Extract 5-ROI (LPFC, CPFC, RPFC, LM1, SMA) averaged signals.

**Args**
- `--subjects SUBJECTS [SUBJECTS ...]` — subject IDs to process (underlying script default: `A1` only)
- `--study STUDY` — config dataset key (default: `expert_novice`)
- `--all` — process all 13 GC-eligible subjects (ignores `--subjects`)
- `--overwrite` — re-process even if `.npz` already exists

> The wrapper defaults to `--all` when called with **no** arguments (the
> underlying Python default is A1-only).

```bash
./01_run_roi_extraction.sh
./01_run_roi_extraction.sh --subjects A1 R7 --overwrite
```

### `02_run_trial_segmentation.sh` — wraps `02_trial_segmentation.py`
Segment ROI signals into trials/windows for GC computation.

**Args**
- `--subjects SUBJECTS [SUBJECTS ...]` — (underlying script default: `A1` only)
- `--study STUDY`
- `--all` — run all 13 GC-eligible subjects

> The wrapper defaults to `--all` when called with **no** arguments.

```bash
./02_run_trial_segmentation.sh
./02_run_trial_segmentation.sh --subjects A1 R7
```

### `04_run_subject_pipeline.sh` — wraps `04_subject_gc_pipeline.py`
Main driver: batch-processes all 13 subjects (VAR(20) → GW causality → 5×5
matrix) + builds Expert/Novice group arrays, with a built-in A1 sanity check.

**Args**
- `--study STUDY` — default `expert_novice`; anything else raises an error (batch-size assertions are hardcoded to this cohort)
- `--overwrite` — recompute and overwrite cached `.npz` files (default)
- `--no-overwrite` — skip subjects that already have a cached `.npz` file

```bash
./04_run_subject_pipeline.sh
./04_run_subject_pipeline.sh --no-overwrite
```

### `05_run_statistics.sh` — wraps `05_gc_statistics.py`
Normality, H1 Expert-vs-Novice, H0 within-trial stability, HIV across-trial
stability tests.

**Args**
- `--study STUDY` — default `expert_novice`

```bash
./05_run_statistics.sh
```

### `06_run_visualization.sh` — wraps `06_gc_visualization.py`
Connectivity heatmaps, group bar charts, brain topology diagrams.

**Args**
- `--study STUDY` — default `expert_novice`

```bash
./06_run_visualization.sh
```

> Note: `04`/`05`/`06`'s `--study` flag was added as part of this `scripts/`
> effort (they previously had no CLI at all and hardcoded `expert_novice`
> internally). `06`'s `--study` only fully threads through the
> subject-averaged plots section — the underlying visualization code wasn't
> refactored beyond that.

### `run_all.sh`
Runs `01` → `02` → `04` → `05` → `06` in order, no args accepted.

```bash
./run_all.sh
```

---

## Stage E Transfer Entropy (`code/stage_e_transfer_entropy/`)

Multiple methodological generations — see `CLAUDE.md` Section 3 for the full
story of why the project moved from phase-based to amplitude-based TE.

### `run_wang2020_validation.sh` — wraps `wang2020_simulation_validation.py`
Synthetic benchmark validation (Wang 2020 Eqs. 26–28) — validates the
dSPTE/dSTE machinery against ground-truth simulated data, independent of real
subjects.

**Args**
- `--model {1,2,3,all}` — which model to run (default: `all`)
- `--N N` — number of samples (default: `20000`)
- `--u-max U_MAX` — max lag for Models 1/2 (default: `15`)
- `--u-max-mv U_MAX_MV` — max lag for Model 3 (default: `20`)
- `--d D` — embedding dimension (default: `3`)
- `--tau TAU` — override embedding delay τ for all models
- `--quiet` — suppress per-model verbose output

```bash
./run_wang2020_validation.sh
./run_wang2020_validation.sh --model 1 --N 50000
```

**Not yet created** (see CLAUDE.md "What's NOT done yet in Stage E"): Section
10 cross-phase comparison (`08_cross_phase_comparison.py`) and Section 11's
unified batch runner. Add wrapper scripts here once those exist.

### Phase 1 Bivariate dSPTE (`phase_1/`) — superseded, kept for comparison

Wang (2020) delay Symbolic Phase Transfer Entropy. Found flat lag profiles
and Novice > Expert on all 20 connections — root cause: broadband HbO
Hilbert phase is quasi-uniformly distributed.

#### `03_process.sh` — wraps `03_subject_dspte_pipeline.py`
Compute connectivity + build group arrays.

**Args**
- `--batch` — run dSPTE for all Expert+Novice subjects (R10 excluded), write `te_batch_log.json`, then build group arrays
- `--subject ID` — run dSPTE for one subject, e.g. `R7`
- `--group-only` — only rebuild group arrays from existing per-subject `.npz` files
- `--overwrite` — recompute even if output `.npz` exists
- `--no-overwrite` — skip subjects that already have output (batch mode)
- `--test` — run Section 4.2 smoke test on A1 only (**underlying script's default if no flag given**)

> The wrapper defaults to `--batch` when called with no arguments.

```bash
./03_process.sh
./03_process.sh --subject A1
./03_process.sh --group-only
```

#### `04_statistics.sh` — wraps `04_dspte_statistics.py`
**Args**
- `--overwrite` — recompute even if output CSVs already exist
- `--test {h1,h0,hiv,normality,all}` — which test to run (default: `all`)

```bash
./04_statistics.sh
./04_statistics.sh --test h1 --overwrite
```

#### `05_visualize.sh` — wraps `05_dspte_visualization.py`
**Args**
- `--overwrite` — regenerate even if PNGs already exist
- `--plot {all,heatmap,delay,gcte,topology,lag}` — which plot group to generate (default: `all`)
- `--subject SUBJECT` — subject ID for per-subject plots (default: `A1`)

```bash
./05_visualize.sh
./05_visualize.sh --plot lag --subject A1
```

#### `05a_diagnostic.sh` — wraps `05a_phase_entropy_diagnostic.py`
The entropy-collapse root-cause investigation (phase variability bias
diagnostic).

**Args**
- `--overwrite` — recompute even if cached results exist
- `--subject ID` — run for one subject only (for testing)

```bash
./05a_diagnostic.sh
```

#### `run_all.sh`
Runs `03_process.sh` → `04_statistics.sh` → `05_visualize.sh`, no args
accepted.

```bash
./run_all.sh
```

### Phase 2 Multivariate (Hybrid) dSPTE (`phase_2/`) — superseded, kept for comparison

Adds PCA-conditioning on the other 3 ROIs to suppress shared-noise
connections. Reduced mean TE ~45%, but most connections still had flat lag
profiles.

#### `06_process.sh` — wraps `06_hybrid_dspte.py`
**Args**
- `--batch` — run hybrid-dSPTE for all Expert+Novice subjects (R10 excluded), writes `hybrid_te_batch_log.json`
- `--subject ID` — run for one subject, e.g. `A1`, `R7`
- `--overwrite` — recompute and overwrite even if `.npz` already exists
- `--first-window-only` — process only the first window per trial (matches IDTxl fast-compare mode)
- `--test` — run unit smoke tests (**underlying script's default if no flag given**)

> The wrapper defaults to `--batch` when called with no arguments.

```bash
./06_process.sh
./06_process.sh --subject A1 --overwrite
./06_process.sh --batch --first-window-only
```

#### `07_visualize.sh` — wraps `07_hybrid_dspte_visualization.py`
**Args**
- `--batch` — run all Phase 2 visualizations (group matrix, group bar chart, 7.3.2 comparison chart + 20 lag profiles)
- `--group-matrix` — Phase 2 group connectivity matrix (Expert | Novice | Difference)
- `--group-bar` — Phase 2 Expert vs Novice bar chart with Phase 1 reference dots
- `--comparison` — 7.3.2 Phase 1 vs Phase 2 group comparison bar chart only
- `--lag-profiles` — 7.3.3 lag profile comparison plots (requires `--subject`, `--trial`, `--window`; optionally `--connection` for a single pair)
- `--subject ID` — subject for lag profile plots (default: `A1`)
- `--trial TR` — trial ID for lag profile plots, e.g. `Tr_1` (default: `Tr_1`)
- `--window N` — 1-indexed window number for lag profile plots (default: `1`)
- `--connection SRC→TGT` — single connection to plot, e.g. `"CPFC→LM1"` (default: all 20)
- `--no-overwrite` — skip plots whose output file already exists
- `--test` — run unit smoke tests

> The wrapper defaults to `--batch` when called with no arguments.

```bash
./07_visualize.sh
./07_visualize.sh --comparison
./07_visualize.sh --lag-profiles --subject R7 --trial Tr_2 --window 1
./07_visualize.sh --lag-profiles --subject A1 --trial Tr_1 --window 1 --connection "CPFC→LM1"
```

#### `08_statistics.sh` — wraps `08_hybrid_dspte_statistics.py`
**Args:** none — always runs the full Section 8 statistical battery.

```bash
./08_statistics.sh
```

#### `09_diagnostic_sweep.sh` — wraps `09_phase2_diagnostic_sweep.py`
Parameter sweep diagnostic (e.g. PCA component count) investigating the flat
lag profiles.

**Args:** none — sweep grid is defined inside the script/config.

```bash
./09_diagnostic_sweep.sh
```

#### `10_idtxl_validation.sh` — wraps `10_idtxl_validation.py`
IDTxl (KSG estimator) parallel validation — model-free cross-check against
the NG-symbolic hybrid-dSPTE approach. For HPC/PACE batch submission use
`pace_idtxl_job.sh` (repo root) instead.

**Args** (one of the first five is required)
- `--smoke-test` — run synthetic smoke tests
- `--subject SUBJECT` — run IDTxl for a single subject (first window only, for speed)
- `--batch` — run IDTxl for all subjects
- `--compare` — run `compare_symbolic_vs_idtxl()` (requires `--batch` to have run first)
- `--eval-md` — write `IDTXL_EVAL.md` and exit
- `--overwrite` — overwrite existing cached `.npz` files
- `--first-window-only` — process only the first window per trial (fast mode)
- `--jobs JOBS` — parallel subject workers for `--batch` (default: `1`)
- `--idtxl-max-lag IDTXL_MAX_LAG` — override IDTxl `max_lag_sources` for speed control
- `--idtxl-num-threads IDTXL_NUM_THREADS` — override IDTxl estimator `num_threads` (e.g. `1`, `2`, `USE_ALL`)
- `--idtxl-n-perm IDTXL_N_PERM` — override all IDTxl permutation counts (must be `> 1/alpha`; for α=0.05 use ≥21)
- `--idtxl-fast` — faster IDTxl permutation settings for exploratory runs
- `--first-trial-only` — process only the first trial (one IDTxl call total) — fastest single-subject cross-check
- `--idtxl-knn IDTXL_KNN` — override Kraskov KNN neighbour count (default: `4`; use `2` for ~1.5–2× speedup)

```bash
./10_idtxl_validation.sh --smoke-test
./10_idtxl_validation.sh --batch --first-window-only     # fast validation, ~5-10 min
./10_idtxl_validation.sh --compare                        # after --batch has run
./10_idtxl_validation.sh --batch                           # full batch, ~52 min estimate
./10_idtxl_validation.sh --eval-md
```

#### `run_all.sh`
Runs `06_process.sh` → `08_statistics.sh` → `07_visualize.sh`, no args
accepted (excludes the IDTxl validation and diagnostic sweep — run those
individually).

```bash
./run_all.sh
```

### Phase 3 Bimodal dSPTE (`phase_3/`) — superseded, kept for comparison

Phase + amplitude weighted fusion. This is where it became clear amplitude
carried most of the real signal — motivated the Phase 4 pivot.

#### `11_process.sh` — wraps `11_bimodal_dspte.py`
**Args** (one required)
- `--smoke-test` — run synthetic smoke tests
- `--subject SUBJECT` — run bimodal pipeline for one subject
- `--batch` — run bimodal pipeline for all subjects
- `--overwrite` — overwrite existing output files

> The wrapper defaults to `--batch` when called with no arguments.

```bash
./11_process.sh
./11_process.sh --subject A1
./11_process.sh --smoke-test
```

#### `12_statistics.sh` — wraps `12_bimodal_dspte_statistics.py`
**Args**
- `--test {all,normality,h1,h0,hiv,build-only}` — which Section 8.3.1 task to run (default: `all`)
- `--overwrite` — overwrite existing outputs (default behavior)
- `--no-overwrite` — use cached outputs when present

```bash
./12_statistics.sh
./12_statistics.sh --test h1
```

#### `13_visualize.sh` — wraps `13_bimodal_dspte_visualization.py`
**Args**
- `--subjects [SUBJECTS ...]` — explicit subject list, e.g. `A1 R3 R5` (default: all configured)
- `--plot {all,overall,grouped}` — which plot(s) to generate (default: `all`)
- `--no-save` — do not save outputs

```bash
./13_visualize.sh
./13_visualize.sh --plot grouped --subjects A1 R7
```

#### `14_weight_sweep.sh` — wraps `14_bimodal_weight_sweep.py`
Sweeps the `w_phase`/`w_amplitude` fusion weight — the diagnostic that
motivated Phase 4.

**Args**
- `--subjects [SUBJECTS ...]` — explicit subject IDs (default: all configured)
- `--weights WEIGHTS` — comma-separated candidate `w_phase` values (default: uses `--step` grid); `w_amplitude = 1 - w_phase`
- `--step STEP` — grid step for `w_phase` when `--weights` is omitted
- `--training-trials TRAINING_TRIALS` — comma-separated trial IDs for the sweep when not using `--all-trials`
- `--all-trials` — use all available trials instead of training-only trials
- `--output OUTPUT` — output JSON filename under `phase_3_group/`

```bash
./14_weight_sweep.sh
./14_weight_sweep.sh --weights 0.3,0.5,0.7 --all-trials
```

#### `run_all.sh`
Runs `11_process.sh` → `12_statistics.sh` → `13_visualize.sh`, no args
accepted.

```bash
./run_all.sh
```

### Phase 4 Multivariate Conditional Amplitude dSTE (`phase_4/`) — **current primary method**

Hilbert amplitude envelope of broadband (unfiltered) HbO, NG-symbolized,
PCA-conditioned. No phase, no narrowband filtering anywhere. Novel
contribution of this project. Functionally complete; current focus is
*interpreting* the results (Novice > Expert on 18/20 connections, zero
FDR-significant, small-cohort attributed).

#### `14_process.sh` — wraps `14_amplitude_dste.py`
**Args** (one required)
- `--smoke-test` — run self-contained smoke tests (Sections 9.2–9.4)
- `--subject SUBJECT_ID` — run Phase 4 pipeline for one subject, e.g. `A1`
- `--batch` — run Phase 4 pipeline for all subjects in config
- `--overwrite` — overwrite existing output files

> The wrapper defaults to `--batch` when called with no arguments.

```bash
./14_process.sh
./14_process.sh --subject A1
./14_process.sh --batch --overwrite
```

#### `15_group.sh` — wraps `15_amplitude_dste_group.py`
**Args**
- `--strategy {all_windows,subject_mean,first_window,metadata,all}` — which aggregation strategy to run (default: `all`)
- `--overwrite` — overwrite cached group arrays

```bash
./15_group.sh
./15_group.sh --strategy subject_mean
```

#### `16_visualize.sh` — wraps `16_amplitude_dste_visualization.py`
**Args**
- `--plot {all,heatmaps,delta,bivariate-comparison,group-te}` — which plot(s) to generate (default: `all`)
- `--no-save` — do not save figures/JSON to disk (display-only)

```bash
./16_visualize.sh
./16_visualize.sh --plot delta
```

#### `17_statistics.sh` — wraps `17_amplitude_dste_statistics.py`
**Args**
- `--test {all,normality,h1,h0,hiv,asymmetry,build-only}` — which statistical test to run (default: `all`)
- `--overwrite` — overwrite cached outputs
- `--no-overwrite` — use cached outputs when present (default)

```bash
./17_statistics.sh
./17_statistics.sh --test h1 --overwrite
```

#### `18_asymmetry.sh` — wraps `18_asymmetry_index_statistics.py`
Asymmetry Index (AI = TE(X→Y) − TE(Y→X)) diagnostics vs. GC directionality —
current interpretive focus.

**Args**
- `--test {all,per-connection,cross-pair,lm1}` — which statistical test to run (default: `all`)
- `--overwrite` — overwrite cached outputs
- `--no-overwrite` — use cached outputs when present (default)
- `--n-permutations N_PERMUTATIONS` — number of permutations for H_AI3 (default: `10000`)

```bash
./18_asymmetry.sh
./18_asymmetry.sh --test per-connection --n-permutations 5000
```

#### `debug.sh` — wraps `phase_4_debug.py`
NG convergence diagnostic for the amplitude embedding space.

**Args**
- `--sub SUB` — subject ID (default: `A1`)
- `--trial TRIAL` — trial ID (default: `Tr_1`)
- `--win WIN` — window index, 0-based (default: `0`)
- `--k_max K_MAX` — max codebook size (default: `20`)

```bash
./debug.sh
./debug.sh --sub R7 --trial Tr_2 --win 1 --k_max 25
```

#### `run_all.sh`
Runs `14_process.sh` → `15_group.sh` → `16_visualize.sh` →
`17_statistics.sh` → `18_asymmetry.sh` in order, no args accepted.

```bash
./run_all.sh
```

---

## Typical "come back after 3 months" workflow

```bash
# 1. Re-orient yourself
cat CLAUDE.md
tail -100 docs/TRANSFER_ENTROPY_PROGRESS.md

# 2. Re-run the current primary method end to end
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/run_all.sh"

# 3. Or just re-generate one thing, e.g. the asymmetry-index stats
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/18_asymmetry.sh" --test all --overwrite
```

## Known gaps / things not wrapped yet

- `code/audit/` (`08_rpca_qc_audit.py`, `audit_cnr.py`,
  `pipeline_combo_audit.py`) — cross-stage diagnostic tools that don't
  belong to a single stage. No `scripts/Audit/` folder exists yet; add one
  if these get used regularly.
- Stage E Section 10 (cross-phase comparison) and Section 11 (batch runner)
  don't exist in `code/` yet, so there's nothing to wrap.
- `run_all.sh` scripts don't forward arguments to the steps they call — if
  you want e.g. a global `--overwrite` applied to every step, edit the
  relevant `run_all.sh` or run the steps individually.
