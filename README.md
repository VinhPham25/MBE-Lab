# MBE Lab — ASTaUND Directed Connectivity

**Mind, Brain, and Education (MBE) Lab** (Dr. Mason Borzin)  
**Researcher:** Vinh Pham

Python pipeline for preprocessing the [ASTaUND](https://doi.org/10.1038/s41597-023-02603-3) fNIRS dataset and comparing **linear Granger Causality (GC)** with **Transfer Entropy (TE)** as measures of directed brain connectivity in expert vs. novice surgeons.

**Current primary method:** multivariate conditional amplitude delay Symbolic Transfer Entropy (Phase 4 dSTE), with surrogate-corrected Effective TE (ETE) and an Asymmetry Index (AI) layer on top. Phase-based TE (Phases 1–3) is retained for comparison, not active development.

---

## Contents

- [Research question](#research-question)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Repository layout](#repository-layout)
- [Documentation map](#documentation-map)
- [Status and goals](#status-and-goals)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Known caveats](#known-caveats)
- [References](#references)

---

## Research question

Can Transfer Entropy — an information-theoretic, non-linear, model-free measure of directed information flow — serve as a viable complement or alternative to Granger Causality for fNIRS connectivity, and can it distinguish expert vs. novice surgeons across five ROIs covering PFC, M1, and SMA?

The work is a linear staged pipeline. Stages A–C reproduce and validate the Kamat et al. (2023) preprocessing on the Expert–Novice cohort. Stage D is the linear GC baseline (VAR → frequency-domain Geweke–Wilson causality). Stage E is the TE track: four methodological generations that ended in a pivot **off Hilbert phase entirely** and onto the **Hilbert amplitude envelope of broadband HbO**.

A written report (“MBE Lab S26 Final Report”, 29 April 2026) covers methods through the phase-based dSPTE work and remains useful for math and Stages A–D. Its TE conclusions are superseded by the amplitude-only approach.

---

## Dataset

**ASTaUND** (Assessment of Surgical Tasks Using Neuroimaging Dataset) is an open, BIDS-compliant fNIRS collection of surgeons performing laparoscopic FLS pattern-cutting / pegboard-transfer tasks ([Kamat et al., 2023](https://doi.org/10.1038/s41597-023-02603-3)).

Four sub-studies live under `data/BIDS/`. **Only Expert–Novice is analyzed.** The other three are on disk and Stage B can process them, but Stages D–E subject lists, ROI maps, and statistics are Expert–Novice only.

| Study | System | Rate | Subjects | Status |
|---|---|---|---|---|
| **Expert–Novice** | TechEn CW6 | 25 Hz (verified; paper text says 20) | 14 recorded; **7 experts + 6 novices analyzed** | Active |
| Learning Curve | TechEn CW6 | 20 Hz in `config.yaml` (not re-verified) | 13 | Not analyzed |
| Short-Term tES | NIRx NIRScout | 7.8125 Hz | 12 | Not analyzed |
| Long-Term tES | NIRx NIRScout | 7.8125 Hz | 21 | Not analyzed |

**Expert–Novice details**

- **Experts:** A1, A2, R7, R8, R9, R11, R12  
- **Novices:** R1–R6  
- **Excluded:** `R10` — trailing zeros on 690 nm channels and anomalous rPCA variance removal (HbO ~87%, HbR ~99%)
- **Task:** 5 FLS trials per subject; stimulus 1 = trial start, 2 = trial end
- **Channels:** 25 long + 8 short; wavelengths 690 / 830 nm; PPF 6.4 / 5.8
- **ROIs (5):** LPFC, CPFC, RPFC, LM1, SMA → 5×5 directed matrix → **20 off-diagonal connections**

Connectivity methods (GC and every TE phase) share that 20-element vector so results can be compared without reshaping.

---

## Pipeline

```
Stage A  Raw QA          SNR, motion ratio, exclusion lists (documentation only)
    ↓
Stage B  Preprocessing   Intensity → OD → bandpass → rPCA → MBLL → short-channel OLS → HbO/HbR
    ↓
Stage C  Validation      HRF, CNR, group morphology checks
    ↓
Stage D  Linear GC       VAR(20) → frequency-domain pairwise GC  (linear baseline)
Stage E  Transfer Entropy  Phases 1–3 (phase; archived) → Phase 4 amplitude dSTE + ETE/AI
```

`config.yaml` is the single source of truth for parameters. Numbered scripts in each `code/stage_*` folder run in order; each is independently runnable. Thin bash wrappers live under `scripts/` (see [Running the pipeline](#running-the-pipeline)).

### Stages A–C — preprocessing (stable)

Faithful Python/MNE-NIRS reproduction of the ASTaUND technical-validation pipeline, audited against Homer3 MATLAB.

| Stage | What it does | Key outputs |
|---|---|---|
| **A** | SNR (80–140 dB band), motion ratio, signal-amplitude exclusion lists. No channels are dropped here. | `outputs/stage_a_qa/` |
| **B** | Six steps: intensity→OD, 0.01–0.5 Hz bandpass, motion detection, rPCA, modified Beer–Lambert, Gaussian-basis short-channel OLS. | `outputs/stage_b_processed/processed_data/expert-novice/*.fif` |
| **C** | Epoch HRFs (−5 to +100 s), 5-ROI morphology, CNR. | `outputs/stage_c_validation/` |

### Stage D — linear Granger Causality (complete)

Bivariate unconditional GC in the 0.01–0.07 Hz band (MATLAB `cca_pwcausal` / bsmart equivalent via `statsmodels` VAR). Sliding 54 s windows, no overlap. Same 20-connection schema as Stage E.

### Stage E — Transfer Entropy (active)

Four phases, each swapping only the connectivity estimator; ROI extraction, windowing, group arrays, and the four statistical tests (normality, H1 Expert-vs-Novice, H0 within-trial, HIV across-trial) are reused.

| Phase | Method | Role |
|---|---|---|
| 1 | Bivariate delay Symbolic **Phase** TE (Wang & Chen 2020) | Foundation. Novice > Expert on all 20 connections; lag profiles unphysiological. |
| 2 | Multivariate (PCA-conditioned) phase dSPTE | Shared-noise suppression (~45% mean TE drop). Most lag profiles still flat. |
| 3 | Bimodal: phase + amplitude weighted fusion | Showed the **amplitude branch** carried the real signal. |
| **4** | Multivariate conditional **amplitude** dSTE | **Current primary method.** No phase, no narrowband filter. NG symbolization of the Hilbert envelope of broadband HbO. |

**Why phase was abandoned.** Hilbert phase is only interpretable for narrowband signals. Broadband fNIRS phase is quasi-uniform in embedding space, so Neural-Gas codebooks never converge cleanly. Narrowband filtering (0.04–0.15 Hz or Wang’s 0.01–0.10 Hz) fixed the numerics but collapsed phase trajectories onto the HRF rise/fall — full entropy ≈ marginal entropy, so TE itself is tiny. Amplitude needs no such filter: it tracks hemodynamic response magnitude and symbolizes with few codebook vectors.

**On top of Phase 4 (Section 10, implemented)**

- **Effective TE (ETE):** `ETE = TE_raw − TE_surrogate_floor` via circular-shift surrogates (Marschinski & Kantz 2002; TRENTOOL convention). Phase 4 is batch-run; Phase 2 ETE is wired but not batch-run.
- **Asymmetry Index:** `AI = TE(X→Y) − TE(Y→X)` — directional imbalance, not raw magnitude. Registry-driven so GC, Phase 2, Phase 4 raw, and Phase 4 ETE share one statistics core.
- **Cross-method significance:** one table/figure of H1 + AI across those tracks.

IDTxl (KSG estimator) is a parallel validation track for multivariate phase TE (`phase_2/10_idtxl_validation.py`, `pace_idtxl_job.sh` for HPC). Extending it to Phase 4 amplitude is open.

---

## Repository layout

```
MBE-Lab/
├── CLAUDE.md                 # Agent / new-session context (read this first if you are an AI)
├── README.md                 # This file
├── config.yaml               # All pipeline parameters
├── requirements.txt
├── pace_idtxl_job.sh         # PACE/HPC batch for IDTxl
│
├── code/
│   ├── utils.py              # Config loader, logging, shared helpers
│   ├── visualization.py
│   ├── audit/                # Cross-stage diagnostics (rPCA QC, CNR, combo audits)
│   ├── stage_a_raw_qa/
│   ├── stage_b_preprocessing/
│   ├── stage_c_validation/
│   ├── stage_d_granger/
│   └── stage_e_transfer_entropy/
│       ├── utils_te.py                    # Neural-Gas (from scratch) + shared TE helpers
│       ├── wang2020_simulation_validation.py
│       ├── 11_symbolic_ete_core.py        # Method-agnostic circular-shift ETE
│       ├── 12_asymmetry_index_core.py     # Method-agnostic AI statistics
│       ├── 13_gc_significance.py          # Geweke analytic test (wired; batch deferred)
│       ├── 15_ete_group_stats_viz.py
│       ├── 16_cross_method_significance.py
│       ├── phase_1/   bivariate phase dSPTE
│       ├── phase_2/   multivariate phase dSPTE + IDTxl + ETE wiring
│       ├── phase_3/   bimodal fusion
│       └── phase_4/   amplitude dSTE + ETE  ← current primary
│
├── scripts/                  # chmod +x bash wrappers; see scripts/README.md
├── docs/                     # Progress docs, PRDs, output catalog
├── outputs/                  # gitignored products; see docs/OUTPUTS.md
├── data/BIDS/                # gitignored ASTaUND recordings
├── matlab_references/        # Homer3 / GCCA reference scripts
└── IDTxl/                    # Vendored third-party library (gitignored)
```

**Important files (not under `docs/`)**

| File | Why it matters |
|---|---|
| `config.yaml` | Every threshold, subject list, lag, codebook ceiling, ETE surrogate count. Read the inline comments before changing values. |
| `CLAUDE.md` | Compressed project story: who/why, stage table, why phase TE was dropped, output naming. Keep in sync when the research direction shifts. |
| `code/utils.py` | `load_config()`, paths, pyswarms log redirect. |
| `code/stage_e_transfer_entropy/utils_te.py` | Neural-Gas implementation (no suitable PyPI package). |
| `scripts/README.md` | Copy-paste CLI for every wrapper; the place to look when returning after a hiatus. |

---

## Documentation map

Progress-doc **filenames are sometimes legacy**. When a filename on disk disagrees with a progress-doc citation, **`docs/OUTPUTS.md` wins**.

| Doc | Use it for |
|---|---|
| **`docs/TRANSFER_ENTROPY_PROGRESS.md`** | Living Stage E PRD + checklist. Longest, most current. Task boxes with `> **Result**` notes. Start here for anything TE. |
| **`docs/OUTPUTS.md`** | How to read `outputs/`: folder grammar, figure naming, catalog of what each file shows. |
| `docs/LINEAR_GC_PROGRESS.md` | Stage D PRD, Geweke formula, MATLAB-parity notes, numerical-cancellation investigation. |
| `docs/PROGRESS.md` | Stages A–C implementation log (last major update ~March 2026). |
| `docs/AUDIT_PROGRESS.md` | Preprocessing audit (SNR/motion/HRF/CNR vs paper + Homer3). |
| `docs/PROJECT.md` | Original Phase-1 PRD (preprocessing reproduction). Historical. |
| `docs/reproduced_matlab_comparison.md` | Python vs Homer3 discrepancies and resolutions. |
| `docs/IDTXL_EVAL.md` | Why/how IDTxl is used as a KSG cross-check. |
| `docs/NG_LIBRARY_EVAL.md` | Why Neural-Gas is implemented in-repo. |
| `docs/Dataset_README.md` | ASTaUND file naming, known bad recordings (R9 split, etc.). |
| `outputs/README.md` | One-screen pointer into `docs/OUTPUTS.md`. |
| `scripts/README.md` | How to actually run things. |
| `update_summaries/` | Dated week notes from early preprocessing; not current. |

---

## Status and goals

### Done

| Area | State |
|---|---|
| Stages A–C | Complete and stable. Expert–Novice fully processed. |
| Stage D linear GC | Complete (ROI extract → windows → GC → stats → viz). Batch-runner section in the GC PRD was never added; `scripts/Stage D Linear GC/run_all.sh` covers the same ground. |
| Stage E Phases 1–3 | Complete; kept for comparison. |
| Stage E Phase 4 | Complete: core, group arrays, viz, H1/H0/HIV, raw AI. |
| ETE + cross-method layer | Complete for **Phase 4**. Registry, circular-shift ETE, AI core, cross-method table/figure. |

**Headline numbers (Phase 4 raw amplitude dSTE, last stats run):** Novice > Expert mean TE on 18/20 connections (Expert > Novice on 2/20). Zero FDR-significant H1 connections (n = 100 tests: 5 trials × 20 connections). The same “no FDR hits, small n=13” pattern appears in GC and Phases 1–3 — treated as a cohort-size limit, not a pipeline failure. Interpretation of *why* novices show higher magnitude TE (effort / cognitive load vs residual confound) is still open; AI is the main tool for that (experts can show larger *directional* imbalance even when raw TE is smaller).

### Open — near term

1. **Interpretation** of Phase 4 (and ETE/AI) results — the scientific question, not missing code.
2. **Cross-phase magnitude comparison** (`08_cross_phase_comparison.py`, not created). Side-by-side GC + Phases 1–4 connectivity and Expert−Novice deltas. Distinct from the already-built *significance* comparison in `16_cross_method_significance.py`.
3. **Unified Stage E batch runner** (`09_run_te_pipeline.py`, not created). Until then, use `scripts/.../Phase 4.../run_all.sh`.
4. **Deferred significance tracks** (code exists; not batch-run):
   - Phase 2 ETE (`phase_2/12_multivariate_ete.py`)
   - GC Geweke per-window test (`13_gc_significance.py` — needs a small batch runner first)
   - Commands are in the “Section 10 Deferred Work Runbook” inside `docs/TRANSFER_ENTROPY_PROGRESS.md`.
5. IDTxl validation of **amplitude** dSTE (currently phase-multivariate only).
6. Wang (2020) real finger-tapping dataset replication (simulation validation is done).

### Open — longer term

- Analyze Learning Curve and tES cohorts.
- Non-uniform embedding (Wang 2022 MTENUE) — all current TE uses uniform delay embedding.

---

## Setup

Python 3.9+ (developed on 3.12).

```bash
cd /path/to/MBE-Lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core stack: NumPy, SciPy, pandas, MNE / MNE-NIRS, scikit-learn, statsmodels, PyYAML, matplotlib, seaborn. Stage E also lists `pyswarms` (Phase 3 weight sweep; optional). Neural-Gas is **not** a pip package — it lives in `code/stage_e_transfer_entropy/utils_te.py`.

**Data.** The recordings are intentionally not versioned. Download the [ASTaUND dataset collection on Figshare](https://springernature.figshare.com/collections/Assessment_of_Surgical_TAsks_Using_Neuroimaging_Dataset_ASTaUND_/6429713) (see the accompanying [Scientific Data paper](https://www.nature.com/articles/s41597-023-02603-3#Sec11)), then place the extracted BIDS recordings at `data/BIDS/Expert-Novice/` (and the other study folders when needed). Confirm the Expert–Novice data are discoverable before running the full pipeline:

```bash
find data/BIDS/Expert-Novice -type f -iname '*.snirf' | head
```

The command should print one or more `.snirf` files. The `data/` tree is gitignored.

**IDTxl** (optional, Phase 2 KSG validation) is vendored under `IDTxl/` and gitignored; follow `docs/IDTXL_EVAL.md` if you need it.

---

## Running the pipeline

Do not hunt through `argparse` in `code/`. Use the wrappers:

### Quick smoke test (no dataset required)

After installing dependencies, verify the current primary method with synthetic inputs:

```bash
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/14_process.sh" --smoke-test
```

Success ends with `ALL SMOKE TESTS PASSED` and requires no downloaded BIDS data.

```bash
# From anywhere — wrappers resolve the repo root via git
scripts/Stage\ A\ Raw\ QA/run_all.sh
scripts/Stage\ B\ Preprocessing/07_run_full_pipeline.sh
scripts/Stage\ C\ Validation/run_all.sh
scripts/Stage\ D\ Linear\ GC/run_all.sh

# Current primary TE method (raw dSTE + ETE + AI)
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/run_all.sh"
```

Every wrapper forwards `"$@"` to the Python script, so `./14_process.sh --help` is the source of truth. `run_all.sh` in a folder chains that folder’s steps with no extra flags (`set -euo pipefail`). Full flag tables: **`scripts/README.md`**.

Typical flags on the Python/wrapper scripts: `--batch`, `--subject A1`, `--overwrite`, `--smoke-test`. Batch wrappers often default to `--batch` when called with no arguments even if the Python default is a smoke test.

**Come-back-after-a-hiatus path**

```bash
# 1. Re-orient
#    CLAUDE.md, then tail of docs/TRANSFER_ENTROPY_PROGRESS.md

# 2. Re-run Phase 4 end to end, or only ETE after a surrogate-design change:
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/19_ete_process.sh"
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/20_ete_group_stats_viz.sh"
"scripts/Stage E Transfer Entropy/Phase 4 Multivariate Conditional Amplitude dSTE/21_ete_asymmetry.sh"
```

---

## Outputs

`outputs/` is gitignored. Naming and a catalog of existing files: **`docs/OUTPUTS.md`**.

Folders already encode stage, cohort (`expert-novice`), method (`phase_4_amplitude_dste`), and artifact class (`visualizations/`, `statistics/`, `group/`, `te_connectivity/`, `ete_*`, `ai_*`). Figure names should say what the plot *is*:

```
{scope}_{measure}_{view}[_{facet}][_Tr{N}][_Win{M}].png
```

Example: `outputs/stage_e_transfer_entropy/expert-novice/phase_4_amplitude_dste/visualizations/group_te_bar_expert_vs_novice.png`

Do not prefix figures with `p4_`, `phase_4_`, or a cohort slug — the parent folders already have those. Data files (`.npz`, the four H1/H0/HIV/normality CSVs) keep a **frozen schema** shared with Stage D on purpose.

---

## Configuration

Edit `config.yaml`, not script constants.

- `datasets.expert_novice` — subject lists, sampling rate, PPF, BIDS paths. R10 is omitted from the Expert list used by Stages D–E.
- `stage_a` / `stage_b` / `stage_c` — QA thresholds, filter, rPCA, HRF/CNR windows.
- `stage_d_granger` — window length (54 s), VAR order (20), GC frequency band.
- `stage_e_transfer_entropy` — per-phase blocks (`phase_1` … `phase_4`) plus `significance_methods` (the GC / Phase 2 / Phase 4 registry for AI and ETE). Comments in-file explain k₀ ceilings, contingency-table sparsity, and surrogate settings.

Provenance for a run is also written next to the arrays (`run_metadata.json`, `*_batch_log.json`). Figure captions should read those files, not assume live `config.yaml` still matches the `.npz` on disk.

---

## Known caveats

- **Sampling rate:** data are 25 Hz, not 20 Hz as in parts of the ASTaUND paper text.
- **R9:** split recording; Stage B `.fif` is 50 Hz; Stages C–E resample to 25 Hz in memory.
- **R10:** excluded from connectivity (see Dataset).
- **R11 / R2:** some ROIs lost to all-bad channels (R11: no LM1/SMA; R2: no SMA). Pairwise GC/TE still runs on remaining ROIs.
- **Stage D GC** uses `[LPFC, CPFC, RPFC, LM1, SMA]`, not the MATLAB reference’s `[LPFC, RPFC, LPMC, RPMC, SMA]` — an anatomical choice from Stage C optode maps, not a bug.
- **Small n (13).** FDR-empty H1 tables are expected.
- **Phase TE** is not the scientific claim of this repo anymore.
- **No license file** in the repository. ASTaUND itself is governed by the *Scientific Data* / Figshare terms; this code has not been given a public license.

---

## References

- Kamat, A., Eastmond, C., Gao, Y. et al. (2023). Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND). *Sci Data* 10, 699. https://doi.org/10.1038/s41597-023-02603-3
- Wang, Y., & Chen, W. (2020). Effective brain connectivity for fNIRS data analysis based on multi-delays symbolic phase transfer entropy. *J. Neural Eng.* 17, 056024.
- Wang, Y., Chen, C., & Chen, W. (2022). Nonlinear directed information flow estimation for fNIRS brain network analysis based on the modified multivariate transfer entropy. *Biomed. Signal Process. Control* 74, 103422.
- Marschinski, R., & Kantz, H. (2002). Analysing the information flow between financial time series. *Eur. Phys. J. B* 30, 275–281. (ETE definition)
- Geweke, J. (1982). Measurement of linear dependence and feedback between multiple time series. *J. Am. Stat. Assoc.* 77, 304–313.

MATLAB references used for parity live in `matlab_references/` (Homer3 preprocessing, GCCA/bsmart GC).

---

## Contact

Vinh Pham — MBE Lab. Questions about design intent belong in `docs/TRANSFER_ENTROPY_PROGRESS.md` (Stage E) or `docs/LINEAR_GC_PROGRESS.md` (Stage D) so the next session does not have to re-derive them.
