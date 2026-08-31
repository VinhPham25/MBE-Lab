# Pipeline outputs — naming convention and catalog

How to read everything under `outputs/`: what the **folders** already tell
you, how **filenames** should be written from now on, and what the files
already on disk actually show.

`outputs/` itself is gitignored (regenerated data). This document is the
tracked source of truth. A short pointer also lives at `outputs/README.md`.

The convention below is also summarized in `CLAUDE.md` §6 so future scripts
and figures follow it.

---

## 1. How to read a path

Folders already carry stage, cohort, method, and artifact class. The
filename should **not** repeat those — it should say what *this* file is.

```
outputs /
  {stage}_{purpose} /          ← A QA, B processed, C validation, D GC, E TE
    [{substage} /]             ← e.g. linear_gc, processed_data
      {cohort} /               ← expert-novice  (kebab-case)
        [{method} /]           ← e.g. phase_4_amplitude_dste
          {artifact_class} /   ← visualizations, statistics, group, te_connectivity, …
            {filename}
```

**Example (this figure now uses the canonical name on disk):**

`outputs/stage_e_transfer_entropy/expert-novice/phase_4_amplitude_dste/visualizations/group_te_bar_expert_vs_novice.png`

| Path segment | Already tells you |
|---|---|
| `stage_e_transfer_entropy` | Stage E, Transfer Entropy |
| `expert-novice` | Expert–Novice cohort |
| `phase_4_amplitude_dste` | Phase 4 multivariate conditional amplitude dSTE |
| `visualizations/` | Human-facing figure, not a data array |
| `group_te_bar_expert_vs_novice.png` | Group-level TE, **bar** chart, Expert vs Novice per connection |

The previous stem `p4_expert_novice_grouped.png` repeated the folder (`p4_`) and used `grouped` without saying it was a paired bar chart.

---

## 2. Filename convention (going forward)

Two families. **Figures** are written for a human opening Finder or a slide
deck. **Data files** are written for the next script; their schema is already
stable across Stage D and every Stage E phase — do not invent a parallel
schema for `.npz` / stats CSVs.

### 2.1 Figures (`.png` and companion `.json`)

```
{scope}_{measure}_{view}[_{facet}...][_Tr{N}][_Win{M}].{ext}
```

All **snake_case**, lowercase except the tokens in §2.3.

| Slot | What it answers | Allowed values |
|---|---|---|
| `scope` | Whose data? | `group` (cohort aggregate), `sub-{id}` (one subject), `cross_method` (only inside `cross_method_significance/`) |
| `measure` | What quantity? | `te` `ete` `ai` `gc` `snr` `motion` `hbo` `hbr` `hrf` `cnr` `roi` |
| `view` | What kind of plot? | `heatmap` `bar` `delta` `topology` `box` `scatter` `panel` `timeseries` `boxplot` `violin` `lines` `evolution` |
| `facet` | What is being contrasted or subsetted? (omit if obvious) | `expert_vs_novice` `bivariate_vs_conditional` `phase_vs_amplitude` `te_vs_gc` `per_subject` `lm1_pairs` `null_floor_delta` … |
| `Tr{N}` / `Win{M}` | Which trial / window? | `Tr1`, `Win1` — **no extra underscore** (`Tr1` not `Tr_1`) |

Companion numeric summaries use the **same stem**:

```
group_te_bar_expert_vs_novice.png
group_te_bar_expert_vs_novice_summary.json
```

### 2.2 Do not put these in the filename

- Stage (`stage_e_…`) — folder.
- Cohort (`expert-novice`) as a prefix — folder. `expert_vs_novice` as a
  *comparison type* is different and **is** allowed.
- Method / phase abbreviations (`p4_`, `phase_4_`, `amplitude_dste_`) —
  parent method folder. Exception: a figure whose *point* is comparing
  methods (`phase_vs_amplitude`, `te_vs_gc`, `bivariate_vs_conditional`).
- Vague words that do not name the plot: `grouped`, `comparison`,
  `diagnostics` used alone. Prefer a view + facet (`bar_expert_vs_novice`,
  `panel_null_floor_delta`).

If you export a figure **out of its folder** (slides, email, paper draft),
optionally prefix the method folder so the file is self-describing:

```
phase_4_amplitude_dste__group_te_bar_expert_vs_novice.png
```

Do not store that doubled prefix on disk inside the method folder.

### 2.3 Special tokens (BIDS-like)

| Token | Form | Example |
|---|---|---|
| Subject | `sub-{id}` | `sub-A1`, `sub-R7` |
| Trial | `Tr{N}` | `Tr1` |
| Window | `Win{M}` | `Win1` |
| Directed connection | `{SRC}_to_{TGT}` | `LPFC_to_SMA` |
| Group arrays | `B_Exp_`, `B_Nov_` | keep this prefix on `.npz` group files |

Per-subject figures:

```
sub-{id}[_Tr{N}][_Win{M}][_{SRC}_to_{TGT}]_{measure}_{view}.png
```

Example: `sub-A1_Tr1_Win1_te_heatmap.png`

### 2.4 Data files (frozen schema — do not restyle)

These names are shared across GC and every TE phase on purpose, so scripts
can swap methods without reshaping. **Keep them.**

| Kind | Pattern | Example |
|---|---|---|
| Per-subject connectivity | `sub-{id}_{method}_connectivity.npz` | `sub-A1_amplitude_dste_connectivity.npz` |
| Group arrays | `B_{Exp\|Nov}_{method}_{strategy}.npz` | `B_Exp_amplitude_mean.npz` |
| Group strategies | `all` / `mean` (subject-mean) / `fw` or `first_window` | — |
| Core stats (identical four tests) | `normality_test.csv`, `H1_expert_vs_novice.csv`, `H0_within_trial_stability.csv`, `HIV_across_trial_stability.csv` | — |
| ETE stats | same four, prefixed `ete_` | `ete_H1_expert_vs_novice.csv` |
| AI stats | `ai_{what}.csv` | `ai_group_comparison.csv`, `ai_lm1_focused.csv` |
| Run logs / metadata | `{method}_{role}.json` | `run_metadata.json`, `amplitude_dste_batch_log.json` |

`method` in an `.npz` name is the **algorithm**, not the phase number:
`bivariate_dspte`, `multivariate_dspte`, `bimodal_dspte`, `amplitude_dste`,
`amplitude_ete`, `gc`.

---

## 3. Worked examples (Phase 4 / ETE / AI — canonical names now on disk)

Phase 4 visualization writers, ETE diagnostics, and AI figures have been
renamed. The left column is the **legacy** stem (still cited in older
progress-doc notes); the middle column is what is on disk now.

| On disk (legacy) | Canonical | What it actually shows |
|---|---|---|
| `phase_4_…/visualizations/p4_expert_novice_grouped.png` | `group_te_bar_expert_vs_novice.png` | Paired Expert / Novice **bars** of mean amplitude dSTE for each of the 20 directed connections (subject-mean strategy, SEM whiskers). Optional FDR stars from `H1_expert_vs_novice.csv`. This is the main “does TE differ by group?” figure. |
| `phase_4_…/visualizations/p4_expert_novice_delta.png` | `group_te_delta_expert_vs_novice.png` | Single signed bar per connection: **Expert mean − Novice mean** (trial-averaged). Blue = Expert higher; red = Novice higher. |
| `phase_4_…/visualizations/p4_connectivity_heatmap_Tr_{1–5}.png` | `group_te_heatmap_expert_vs_novice_Tr{N}.png` | Side-by-side Expert vs Novice **5×5** directed-connectivity heatmaps for one trial. Rows = source ROI, columns = target ROI. |
| `phase_4_…/visualizations/p4_bivariate_vs_multivariate.png` | `group_te_bar_bivariate_vs_conditional.png` | Phase 3 bivariate amplitude TE vs Phase 4 **conditioned** amplitude dSTE — how much multivariate conditioning suppresses each connection. |
| `phase_4_…/visualizations/p4_asymmetry_index_vs_gc.png` | `group_ai_panel_te_vs_gc.png` | Multi-panel: 5×5 AI heatmaps (Expert/Novice × TE/GC) plus bars of the 10 undirected pairs. AI = directed imbalance, not raw TE magnitude. |
| `phase_4_…/ete_visualizations/ete_diagnostics.png` | `group_ete_panel_null_floor_delta.png` | Three-panel ETE diagnostic: (1) surrogate-null histograms for the strongest connections, (2) floor-vs-signal stacked bars, (3) Expert−Novice **ETE** delta. |
| `…/ai_visualizations/ai_per_subject_grouped.png` | `group_ai_box_per_subject.png` | Box + strip plot of per-subject AI on all 10 undirected pairs, Expert vs Novice. |
| `…/ai_visualizations/ai_lm1_comparison.png` | `group_ai_bar_lm1_pairs.png` | Same AI contrast restricted to the three pre-specified PFC↔LM1 pairs. |

---

## 4. Catalog by stage

### Stage A — `outputs/stage_a_qa/`

Raw-signal QA. No exclusions applied here; lists are documentation.

| Location | Pattern | Meaning |
|---|---|---|
| `qa_reports/` | `{id}_snr_raw.csv`, `{id}_motion_od.csv`, `{id}_qa_metrics.csv`, `{id}_qa_summary.json` | Per-subject SNR, motion, roll-up |
| `qa_reports/` | `{id}_snr_boxplot.png`, `{id}_motion_boxplot.png` | Per-subject QA figures |
| `qa_reports/expert_novice_full_report/` | `expert_novice_all_subjects_*.csv/json`, `expert_novice_snr_boxplot.png`, `expert_novice_motion_boxplot.png` | Cohort roll-up |
| `channel_signal_quality_exclusion/` | `{id}_signal_amplitude_raw.csv`, `{id}_signal_amplitude_flagged.csv`, `{id}_exclusion_list_preview.json` | Amplitude-based channel flags |

### Stage B — `outputs/stage_b_processed/`

One subfolder per preprocessing step (`od_conversion`, `bandpass_filter`,
`rPCA_correction`, `mbll`, `ols_regression`, `processed_data`, …), then
usually `{subject_id}/`.

| Pattern | Meaning |
|---|---|
| `{id}_od_time_series.png` | Optical-density traces |
| `{id}_filter_comparison.png`, `{id}_psd_comparison.png` | Bandpass before/after |
| `{id}_rpca_before_after.png`, `{id}_rpca_motion_reduction.png` | Motion correction |
| `{id}_hbo_hbr_timeseries.png` | HbO/HbR after MBLL |
| `processed_data/{cohort}/sub-{id}_task-…_desc-preprocessed_nirs_raw.fif` | **Pipeline product** consumed by Stages C–E |

Filenames here are already `{id}_{what}.{ext}`. New Stage B figures should
use `sub-{id}_{measure}_{view}.png` (e.g. `sub-A1_hbo_timeseries.png`) when
touched.

### Stage C — `outputs/stage_c_validation/`

| Location | Pattern | Meaning |
|---|---|---|
| `hrf/expert-novice/` | `hrf_plot_5region_HbO.png`, `hrf_plot_5region_HbR.png` | Group HRF by ROI / chromophore |
| `cnr/` | `cnr_boxplot.png`, `cnr_Expert-Novice.csv`, `cnr_summary.json` | Contrast-to-noise |
| `techval_branch/` | paper-reproduction branch (Kamat et al. technical validation) | Keep separate from the main pipeline products |

### Stage D — `outputs/stage_d_granger/linear_gc/expert-novice/`

Linear Granger Causality baseline. Same 20-connection schema as Stage E.

| Folder | Contents |
|---|---|
| `roi_signals/` | `sub-{id}_roi_signals.npz` + `.png` — 5 ROI HbO traces; `sub-{id}_trial_durations.csv` |
| `gc_connectivity/` | `sub-{id}_gc_connectivity.npz`; `gc_batch_log.json` |
| `group/` | `B_Exp.npz`, `B_Nov.npz`, `gc_summary.json` |
| `statistics/` | the four core CSVs (`normality_test`, `H1_…`, `H0_…`, `HIV_…`) |
| `visualizations/` | see table below |

| Figure | Meaning |
|---|---|
| `group_gc_comparison_averaged.png` | Expert vs Novice GC bars, trial-averaged (same role as Phase 4 `group_te_bar_expert_vs_novice`) |
| `group_gc_comparison_Tr{N}_Win{M}.png` | Same, one window |
| `group_connectivity_matrix_averaged.png` | 5×5 GC heatmaps |
| `group_gc_topology_averaged.png` (and `_Expert_`, `_Novice_`) | Network graphs of directed GC |
| `sub-A1_Tr1_Win1_connectivity.png` | Single-subject, single-window matrix |
| `sub-A1_Tr1_connectivity_evolution.png` | One subject’s GC across windows of a trial |
| `sub-A1_Tr1_Win1_gc_spectrum.png` | Frequency-domain GC spectrum |

Canonical restyle if these are regenerated: `group_gc_bar_expert_vs_novice.png`,
`group_gc_heatmap_expert_vs_novice.png`, `group_gc_topology_expert_vs_novice.png`.
Not required until those scripts are next edited.

### Stage E — `outputs/stage_e_transfer_entropy/expert-novice/`

```
expert-novice/
  roi_signals/                          shared ROI extracts (also used by D)
  phase_1_bivariate_dspte/              Hilbert-phase bivariate dSPTE
  phase_2_multivariate_dspte/           conditioned phase dSPTE + IDTxl
  phase_3_bimodal_dspte/                phase + amplitude fusion
  phase_4_amplitude_dste/               current primary method
  gc_significance/                      GC run through the AI/significance helpers
  cross_method_significance/            GC vs TE vs ETE vs AI side-by-side
```

Inside each method folder the artifact classes are:

| Folder | What lives there |
|---|---|
| `te_connectivity/` | per-subject `.npz` connectivity |
| `group/` | `B_Exp_*` / `B_Nov_*` arrays + `*_summary.json` + `run_metadata.json` |
| `statistics/` | four core tests + method-specific extras (AI CSVs, …) |
| `visualizations/` | figures for **raw** TE / dSTE |
| `ete_connectivity/`, `ete_group/`, `ete_statistics/`, `ete_visualizations/` | same four classes for **effective** TE (surrogate-corrected) |
| `ai_statistics/`, `ai_visualizations/` | Asymmetry Index when it is not mixed into `statistics/` / `visualizations/` |
| `diagnostics/` | one-off investigations (Phase 1 phase-entropy, AI prototype) |
| `idtxl_validation/` | KSG estimator cross-check (Phase 2) |

#### Phase 1 figures (`phase_1_bivariate_dspte/visualizations/`)

| File | Meaning |
|---|---|
| `group_te_bar_expert_vs_novice.png` | Expert vs Novice TE bars (phase-based) |
| `group_te_heatmap_expert_vs_novice.png` | 5×5 TE heatmaps |
| `group_te_topology_expert_vs_novice.png` | Directed-graph view, both groups, trial-averaged |
| `group_te_topology_expert.png` / `group_te_topology_novice.png` | Same topology, one group |
| `group_te_topology_expert_vs_novice_Tr1_Win1.png` | Topology snapshot, trial 1 window 1 |
| `group_te_panel_lag_filter.png`, `group_te_bar_lag_filter_summary.png`, `group_te_heatmap_lag_flatness.png`, `group_te_lines_lag_*.png` | Delay-profile diagnostics (why phase TE looked physiologically implausible). No current writer — renamed in place. |
| `sub-{id}_Tr1_Win1_te_heatmap.png` | Per-subject matrix |
| `sub-{id}_Tr{N}_te_evolution.png` | TE across windows |
| `sub-{id}_Tr{N}_Win{M}_bar_te_vs_gc.png` | Linear GC vs phase TE, one window |
| `sub-{id}_Tr{N}_Win{M}_delay_heatmap.png` | Optimal-delay matrix |
| `sub-{id}_Tr{N}_Win{M}_{SRC}_to_{TGT}_te_lines_lag.png` | TE vs lag for one connection |
| `diagnostics/phase_entropy_bias/group_phase_{violin,lines}_entropy.png` | Hilbert-phase entropy is nearly uniform → TE is tiny |
| `diagnostics/asymmetry_index/group_ai_{bar,scatter}_expert_vs_novice.png` | Early AI prototype |

#### Phase 2 figures (`phase_2_multivariate_dspte/visualizations/`)

| File | Meaning |
|---|---|
| `group_te_bar_expert_vs_novice.png` | Expert vs Novice conditioned phase TE |
| `group_te_heatmap_expert_vs_novice.png` | 5×5 heatmaps (Expert \| Novice \| difference) |
| `group_te_bar_bivariate_vs_conditional.png` | Conditioning suppression (Phase 1 bivariate vs Phase 2 multivariate) |
| `lag_profile_comparison/sub-{id}_Tr1_Win1_{SRC}_to_{TGT}_te_lines_lag.png` | Per-connection lag profile, Phase 1 vs 2 |

#### Phase 3 figures (`phase_3_bimodal_dspte/visualizations/`)

| File | Meaning |
|---|---|
| `group_te_bar_phase_vs_amplitude.png` | Cohort-wide: how much of fused TE is phase vs amplitude |
| `group_te_bar_phase_vs_amplitude_expert_vs_novice.png` | Same split, Expert vs Novice — this is the figure that showed amplitude carrying the real signal |

#### Phase 4 — see §3

Plus `ete_*` and `ai_*` tables in `ete_statistics/` / `statistics/`:

| File | Meaning |
|---|---|
| `ete_floor_fraction.csv` | `null_mean / raw_te` per connection × group — how much of raw TE is surrogate floor |
| `ete_H1_expert_vs_novice.csv` | H1 on **ETE**, with a raw-TE comparison column |
| `ai_group_comparison.csv` | H_AI1: Expert vs Novice AI per undirected pair |
| `ai_lm1_focused.csv` | H_AI3: pre-specified PFC↔LM1 pairs |
| `ai_cross_pair_test.csv` | H_AI2: signed-rank on Expert−Novice AI across pairs |
| `run_metadata.json` | Parameters that actually produced the `.npz` on disk (figure captions read this, not live `config.yaml`) |

#### Cross-method (`cross_method_significance/`)

| File | Meaning |
|---|---|
| `significance_comparison.png` | Methods × 20 connections: direction of Expert vs Novice and whether H1/AI was significant |
| `significance_summary.csv` | Long table, one row per (method, connection) |
| `method_overview.csv` | One row per method (counts, ETE floor fraction) |
| `direction_agreement.csv` | Connections where ≥2 methods agree on direction |

Canonical restyle: `cross_method_heatmap_significance.png`.

---

## 5. Shared statistics CSVs (every connectivity method)

Same four tests, same column ideas, so GC and TE phases are comparable.

| File | Hypothesis | Read it as |
|---|---|---|
| `normality_test.csv` | Shapiro–Wilk | Whether H1 may use a t-test; usually it may not (small n) → Mann–Whitney |
| `H1_expert_vs_novice.csv` | Expert TE/GC ≠ Novice | The group-difference test. `significant` is BH-FDR across connections. With n=13 this is almost always empty — that is expected, not a pipeline failure |
| `H0_within_trial_stability.csv` | Windows of one trial agree | ANOVA across windows |
| `HIV_across_trial_stability.csv` | The five FLS trials agree | ANOVA across trials |

---

## 6. Rules for new outputs

1. Put the file in the existing artifact-class folder (`visualizations/`,
   `statistics/`, `group/`, …). Do not invent a new top-level `outputs/`
   folder without updating this document and `config.yaml`.
2. Name figures with `{scope}_{measure}_{view}_…` from §2.1.
3. Name data with the frozen schema in §2.4.
4. Hardcode the filename as a string next to `savefig` / `to_csv` / `np.savez`
   (or a small `*_FILENAME` constant at the top of the script). Do not
   assemble names from leftover abbreviations (`p4_`, `grouped`).
5. If you add a figure, add one row to the matching table in §3 or §4 of
   this file in the same PR.
6. Progress docs (`docs/TRANSFER_ENTROPY_PROGRESS.md`) may still cite
   **legacy** names from when a task was completed. This file wins when
   names disagree.
