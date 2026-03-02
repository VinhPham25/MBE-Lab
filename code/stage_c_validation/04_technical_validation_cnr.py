"""

If changing parameters from config.yaml, run:
python code/stage_c_validation/04_technical_validation_cnr.py --force

This forces a rewrite of the cached .fif and evoked files, so the new parameters will be used.

04_technical_validation_cnr.py
Stage C — Technical Validation CNR Branch (Issue 3)
Purpose: Compute Expert/Novice CNR using a minimal preprocessing branch
         (no channel pruning, no rPCA, no OLS) per ASTaUND paper technical
         validation spec.

Pipeline  : Intensity → OD → 0.01–0.5 Hz Bandpass → MBLL
Excluded  : Stage A 80–140 dB channel pruning
            B3/B4 motion detection and rPCA correction
            B6 Gaussian-basis OLS short-channel regression

Paper basis (ASTaUND, Kamat et al. 2023):
  "To give an accurate assessment of the quality of the data, no channels
  have been excluded from the analysis provided here and no denoising
  methods have been used prior to analysis."

This script replaces the CNR outputs in outputs/stage_c_validation/cnr/
with values computed from the paper-aligned technical validation branch.

Issue 3 tasks
-------------
  3.1 — Define Technical Validation branch (no Stage A pruning)
  3.2 — Exclude rPCA and OLS from CNR path
  3.3 — Confirm OD → Bandpass → MBLL chain is active
  3.4 — Recompute Expert/Novice CNR with all channels

Cache paths
-----------
  outputs/stage_c_validation/techval_branch/expert-novice/
      sub-{subject_id}_techval_raw.fif       <- minimal preprocessed .fif

  outputs/stage_c_validation/techval_branch/hrf/Expert-Novice/{subject_id}/
      {subject_id}_techval-ave.fif           <- trial-averaged evoked

Outputs (outputs/stage_c_validation/cnr/)
  cnr_Expert-Novice.csv   — per-channel CNR (replaces previous values)
  cnr_boxplot.png         — box plot (replaces previous values)
  cnr_summary.json        — group stats (replaces previous values)

Reference : AUDIT_PROGRESS.md Issue 3; preprocessing_review.md §3
Upstream  : data/BIDS/Expert-Novice raw SNIRF files (direct load)
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_snirf_files, load_config

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Load config once at import time — all tunable constants derive from here
# ---------------------------------------------------------------------------
_CONFIG = load_config()


# ---------------------------------------------------------------------------
# Dynamic imports of Stage C helpers
# ---------------------------------------------------------------------------

def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hrf = _load_module("hrf_extraction",   "01_hrf_extraction.py")
_grp = _load_module("group_hrf",        "02_group_hrf_analysis.py")
_cnr = _load_module("cnr_calculation",  "04_calculate_cnr.py")

# Re-export helpers used in this module
extract_epochs           = _hrf.extract_epochs
get_long_channel_names   = _hrf.get_long_channel_names
_harmonise_evoked        = _grp._harmonise_evoked
_extract_peak_amplitudes = _cnr._extract_peak_amplitudes
compute_cnr              = _cnr.compute_cnr              # legacy peak-based (Task 4.2)
compute_group_hrf_cnr    = _cnr.compute_group_hrf_cnr   # paper-faithful (Task 4.1 / 4.4)
plot_cnr_boxplot         = _cnr.plot_cnr_boxplot
save_cnr_csv             = _cnr.save_cnr_csv
COMPARISON_REGISTRY      = _cnr.COMPARISON_REGISTRY


# ---------------------------------------------------------------------------
# Study / group constants — sourced from config.yaml
# ---------------------------------------------------------------------------

STUDY         = "expert_novice"
_DS_CFG       = _CONFIG.get("datasets", {}).get(STUDY, {})
STUDY_FOLDER  = _DS_CFG.get("output_folder", "expert-novice")
STUDY_DISPLAY = _DS_CFG.get("name", "Expert-Novice")

# PPF from config.yaml datasets.expert_novice.ppf
_ppf_raw = _DS_CFG.get("ppf", {})
PPF_DICT: dict = (
    {int(k): float(v) for k, v in _ppf_raw.items()}
    if isinstance(_ppf_raw, dict) else {690: 6.4, 830: 5.8}
)

# Groups from config.yaml datasets.expert_novice.groups
_groups_raw = _DS_CFG.get("groups", {})
GROUPS: dict[str, list[str]] = (
    {grp: list(subs) for grp, subs in _groups_raw.items()}
    if _groups_raw else {
        "Expert": ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"],
        "Novice": ["R1", "R2", "R3", "R4", "R5", "R6"],
    }
)

# Peak window from config.yaml stage_c.cnr.peak_window
_CNR_CFG = _CONFIG.get("stage_c", {}).get("cnr", {})
_PEAK_WINDOW_DEFAULT: tuple[float, float] = tuple(
    float(v) for v in _CNR_CFG.get("peak_window", [5.0, 15.0])
)

# Bandpass filter params from config.yaml stage_b.filter
_FILT_CFG = _CONFIG.get("stage_b", {}).get("filter", {})
_BP_L_FREQ   = float(_FILT_CFG.get("l_freq",  0.01))
_BP_H_FREQ   = float(_FILT_CFG.get("h_freq",  0.5))
_BP_METHOD   = str(_FILT_CFG.get("method",   "iir"))


# ===========================================================================
# Task 3.1 / 3.2 / 3.3 — Minimal preprocessing branch
# ===========================================================================

def preprocess_subject_techval(
    subject_id: str,
    dataset_name: str,
    ppf_dict: dict,
    out_dir: Path,
    l_freq: float = _BP_L_FREQ,
    h_freq: float = _BP_H_FREQ,
    method: str = _BP_METHOD,
    force_recompute: bool = False,
) -> Path:
    """
    Apply the technical validation preprocessing chain to one subject.

    Steps (paper-aligned):
      B1  Intensity → Optical Density
      B2  Bandpass filter (l_freq–h_freq Hz, method from config.yaml)
      # Note, bandpass filter is still applied since it performs signal deconditioning:
        1. Removes DC drift (high-pass component)
        2. Eliminates non-physiological frequencies (low-pass component)
        3. Prepares data for MBLL conversion
      B5  Beer-Lambert Law → HbO / HbR

    Excluded (per paper "no denoising methods"):
      Stage A  80–140 dB channel exclusion  (all channels included)
      B3 / B4  Motion detection and rPCA correction
      B6       Gaussian-basis OLS short-channel regression

    Returns
    -------
    Path to the saved techval .fif file.
    """
    fif_path = out_dir / f"sub-{subject_id}_techval_raw.fif"
    if fif_path.exists() and not force_recompute:
        print(f"  [{subject_id}] cached: {fif_path.name}")
        return fif_path

    # Load raw SNIRF — no bad channels marked (all channels included)
    snirf_files = get_snirf_files(dataset_name, subject_id)
    raw = mne.io.read_raw_snirf(str(snirf_files[0]), preload=True, verbose=False)
    print(f"  [{subject_id}] loaded SNIRF: {len(raw.ch_names)} ch, "
          f"{raw.times[-1]:.1f} s @ {raw.info['sfreq']:.1f} Hz")

    # B1: Intensity → OD
    raw_od = mne.preprocessing.nirs.optical_density(raw, verbose=False)

    # B2: Bandpass filter
    raw_od_filt = raw_od.copy().filter(
        l_freq=l_freq, h_freq=h_freq, method=method, verbose=False
    )

    # B5: Modified Beer-Lambert Law
    wavelengths = sorted({
        int(ch.split()[-1])
        for ch in raw_od_filt.ch_names
        if ch.split()[-1].isdigit()
    })
    if len(wavelengths) != 2:
        raise ValueError(
            f"[{subject_id}] Expected 2 wavelengths, found {wavelengths}"
        )
    wl1, wl2 = wavelengths
    ppf_tuple = (ppf_dict.get(wl1, 6.0), ppf_dict.get(wl2, 6.0))
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od_filt, ppf=ppf_tuple)

    raw_haemo.save(str(fif_path), overwrite=True, verbose=False)
    n_hbo = sum(1 for c in raw_haemo.ch_names if "hbo" in c)
    n_hbr = sum(1 for c in raw_haemo.ch_names if "hbr" in c)
    print(f"  [{subject_id}] saved: {fif_path.name}  "
          f"({n_hbo} HbO, {n_hbr} HbR, ppf={ppf_tuple})")
    return fif_path


# ===========================================================================
# HRF extraction from techval branch
# ===========================================================================

def _load_or_compute_techval_evoked(
    subject_id: str,
    techval_fif_dir: Path,
    hrf_cache_dir: Path,
    dataset_name: str,
    ppf_dict: dict,
    config: dict,
    force_recompute: bool = False,
) -> mne.Evoked | None:
    """
    Return the per-subject trial-averaged Evoked from the techval branch.

    Loads from cache when available and the epoch window spans the required
    range (tmax >= 95 s, tmin <= -24 s); otherwise re-runs preprocessing
    and epoch extraction.
    """
    ave_path = hrf_cache_dir / subject_id / f"{subject_id}_techval-ave.fif"

    if ave_path.exists() and not force_recompute:
        evoked = mne.read_evokeds(ave_path, verbose=False)[0]
        if float(evoked.times[-1]) >= 95.0 and float(evoked.times[0]) <= -24.0:
            print(f"    [{subject_id}] cached techval evoked "
                  f"(tmin={evoked.times[0]:.1f} s, tmax={evoked.times[-1]:.1f} s)")
            return evoked
        print(f"    [{subject_id}] stale techval evoked — recomputing …")

    # Ensure techval preprocessed .fif exists
    fif_path = techval_fif_dir / f"sub-{subject_id}_techval_raw.fif"
    if not fif_path.exists():
        try:
            preprocess_subject_techval(
                subject_id, dataset_name, ppf_dict, techval_fif_dir
            )
        except Exception as exc:
            print(f"    [{subject_id}] ⚠ preprocessing failed — {exc}")
            return None

    # Load techval raw
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    except Exception as exc:
        print(f"    [{subject_id}] ⚠ could not load {fif_path.name} — {exc}")
        return None

    # Epoch extraction using same window as Stage C main pipeline
    epochs_cfg = config["stage_c"]["epochs"]
    tmin     = float(epochs_cfg["tmin"])
    tmax     = float(epochs_cfg["tmax"])
    baseline = tuple(float(v) for v in epochs_cfg["baseline"])

    try:
        epochs = extract_epochs(raw, tmin=tmin, tmax=tmax, baseline=baseline)
    except Exception as exc:
        print(f"    [{subject_id}] ⚠ epoch extraction failed — {exc}")
        return None

    if len(epochs) == 0:
        print(f"    [{subject_id}] ⚠ no epochs found — skipping")
        return None

    evoked = epochs.average()

    # Cache evoked
    ave_path.parent.mkdir(parents=True, exist_ok=True)
    evoked.save(str(ave_path), overwrite=True, verbose=False)
    print(f"    [{subject_id}] computed techval evoked "
          f"({len(epochs)} epochs, {len(evoked.ch_names)} channels)")

    return evoked


# ===========================================================================
# Evoked collection helpers for techval branch
# ===========================================================================

def collect_all_techval_evokeds(
    subjects: list[str],
    techval_fif_dir: Path,
    hrf_cache_dir: Path,
    dataset_name: str,
    ppf_dict: dict,
    config: dict,
    target_sfreq: float = 25.0,
    force_recompute: bool = False,
) -> list[mne.Evoked]:
    """
    Load (or compute) techval-branch Evokeds for all subjects in a group.

    Returns a list of harmonised Evoked objects ready for grand-averaging.
    Used by the paper-faithful group-HRF CNR path (Task 4.1 / 4.4).
    """
    evokeds: list[mne.Evoked] = []
    for sid in subjects:
        ev = _load_or_compute_techval_evoked(
            sid, techval_fif_dir, hrf_cache_dir,
            dataset_name, ppf_dict, config,
            force_recompute=force_recompute,
        )
        if ev is None:
            print(f"    [{sid}] ⚠ skipped (no data)")
            continue
        ev = _harmonise_evoked(ev, target_sfreq)
        evokeds.append(ev)
    return evokeds


def collect_techval_group_peaks(
    subjects: list[str],
    long_channels: list[str],
    chromophore: str,
    peak_window: tuple[float, float],
    techval_fif_dir: Path,
    hrf_cache_dir: Path,
    dataset_name: str,
    ppf_dict: dict,
    config: dict,
    target_sfreq: float = 25.0,
    force_recompute: bool = False,
) -> dict[str, list[float]]:
    """
    Load (or compute) techval-branch Evokeds for all subjects in a group
    and collect per-channel peak amplitudes.

    Legacy peak-based path — kept for Task 4.2 method comparison.
    """
    group_peaks: dict[str, list[float]] = {ch: [] for ch in long_channels}

    for sid in subjects:
        ev = _load_or_compute_techval_evoked(
            sid, techval_fif_dir, hrf_cache_dir,
            dataset_name, ppf_dict, config,
            force_recompute=force_recompute,
        )
        if ev is None:
            print(f"    [{sid}] ⚠ skipped (no data)")
            continue
        ev = _harmonise_evoked(ev, target_sfreq)

        peaks = _extract_peak_amplitudes(ev, long_channels, chromophore, peak_window)
        for ch, val in peaks.items():
            group_peaks[ch].append(val)

    return group_peaks


# ===========================================================================
# Top-level pipeline
# ===========================================================================

def run_techval_cnr_analysis(
    groups: dict[str, list[str]] = GROUPS,
    study: str = STUDY,
    study_folder: str = STUDY_FOLDER,
    study_display: str = STUDY_DISPLAY,
    ppf_dict: dict = PPF_DICT,
    peak_window: tuple[float, float] = _PEAK_WINDOW_DEFAULT,
    comparison_registry: dict | None = None,
    force_recompute: bool = False,
) -> dict:
    """
    Issues 3 + 4 technical validation CNR pipeline.

    1. Preprocess all subjects through techval branch (OD → BP → MBLL).
    2. Extract and cache per-subject trial-averaged evokeds.
    3. Compute CNR using group-HRF method (primary, Task 4.1 / 4.4).
    4. Compute legacy peak-based CNR for comparison (Task 4.2).
    5. Save method comparison report (Task 4.3).
    6. Overwrite outputs/stage_c_validation/cnr/ with new values.

    Parameters
    ----------
    force_recompute : bool
        If True, re-run preprocessing and evoked extraction even if cached.

    Returns
    -------
    summary : dict  CNR stats and output file paths.
    """
    if comparison_registry is None:
        comparison_registry = dict(COMPARISON_REGISTRY)

    config = load_config()

    # ── Output directories ────────────────────────────────────────────────────
    cnr_out_dir = PROJECT_ROOT / config["paths"]["stage_c_outputs"] / "cnr"
    cnr_out_dir.mkdir(parents=True, exist_ok=True)

    techval_fif_dir = (
        PROJECT_ROOT / config["paths"]["stage_c_outputs"]
        / "techval_branch" / study_folder
    )
    techval_fif_dir.mkdir(parents=True, exist_ok=True)

    hrf_cache_dir = (
        PROJECT_ROOT / config["paths"]["stage_c_outputs"]
        / "techval_branch" / "hrf" / study_display
    )
    hrf_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Issues 3+4 — Technical Validation CNR  (group-HRF method)")
    print(f"  Pipeline: Intensity → OD → {_BP_L_FREQ}–{_BP_H_FREQ} Hz BP ({_BP_METHOD}) → MBLL")
    print(f"  Excluded: Stage A pruning | rPCA | OLS regression")
    print(f"  CNR method: group-HRF (paper-faithful, Task 4.1)")
    print(f"  Study   : {study_display}  |  "
          f"Comparison: {' vs '.join(groups.keys())}")
    print(f"  Peak window: {peak_window[0]}–{peak_window[1]} s")
    print(f"{'='*64}")

    # ── Step 1: Preprocess all subjects ───────────────────────────────────────
    all_subjects = [s for sublist in groups.values() for s in sublist]
    print(f"\n  Step 1 — Techval preprocessing ({len(all_subjects)} subjects)")
    for sid in all_subjects:
        try:
            preprocess_subject_techval(
                sid, study, ppf_dict, techval_fif_dir,
                method=_BP_METHOD,
                force_recompute=force_recompute,
            )
        except Exception as exc:
            print(f"  [{sid}] ⚠ preprocessing failed — {exc}")

    # ── Step 2: Identify long-separation channels ─────────────────────────────
    first_subj = list(groups.values())[0][0]
    ref_fif = techval_fif_dir / f"sub-{first_subj}_techval_raw.fif"
    if not ref_fif.exists():
        raise FileNotFoundError(
            f"Reference techval .fif not found for {first_subj}: {ref_fif}"
        )
    raw_ref = mne.io.read_raw_fif(ref_fif, preload=False, verbose=False)
    hbo_long, hbr_long = get_long_channel_names(raw_ref)
    print(f"\n  Step 2 — Channel inventory (techval branch, no pruning)")
    print(f"  Total channels: {len(raw_ref.ch_names)}  |  "
          f"Long HbO: {len(hbo_long)}  Long HbR: {len(hbr_long)}")

    # ── Step 3: Load all evokeds for group-HRF CNR (Task 4.1) ────────────────
    group_names = list(groups.keys())
    g1_name, g2_name = group_names[0], group_names[1]

    print(f"\n  Step 3 — Loading evokeds for group-HRF CNR …")
    g1_evokeds = collect_all_techval_evokeds(
        groups[g1_name], techval_fif_dir, hrf_cache_dir, study, ppf_dict, config,
        force_recompute=force_recompute,
    )
    g2_evokeds = collect_all_techval_evokeds(
        groups[g2_name], techval_fif_dir, hrf_cache_dir, study, ppf_dict, config,
        force_recompute=force_recompute,
    )
    print(f"  {g1_name}: {len(g1_evokeds)} subjects | "
          f"{g2_name}: {len(g2_evokeds)} subjects")

    # ── Step 4: Compute CNR (group-HRF primary + legacy comparison) ───────────
    print(f"\n  Step 4 — CNR computation")
    rows: list[dict] = []

    for chromo, long_chs in [("hbo", hbo_long), ("hbr", hbr_long)]:
        label = "HbO" if chromo == "hbo" else "HbR"
        print(f"\n  [CNR — {label}]")

        # ── Group-HRF CNR (paper-faithful, primary) ───────────────────────────
        ghrf_cnr, ghrf_meta = compute_group_hrf_cnr(
            g1_evokeds, g2_evokeds, long_chs, chromo, peak_window,
        )

        # ── Legacy peak-based CNR (Task 4.2, comparison) ─────────────────────
        g1_peaks = collect_techval_group_peaks(
            groups[g1_name], long_chs, chromo, peak_window,
            techval_fif_dir, hrf_cache_dir, study, ppf_dict, config,
            force_recompute=force_recompute,
        )
        g2_peaks = collect_techval_group_peaks(
            groups[g2_name], long_chs, chromo, peak_window,
            techval_fif_dir, hrf_cache_dir, study, ppf_dict, config,
            force_recompute=force_recompute,
        )
        legacy_cnr_vals = compute_cnr(g1_peaks, g2_peaks)

        # ── Per-channel rows ──────────────────────────────────────────────────
        for ch in sorted(set(long_chs)):
            m = ghrf_meta.get(ch, {})
            cnr_val = ghrf_cnr.get(ch, np.nan)
            leg_val = legacy_cnr_vals.get(ch, np.nan)
            rows.append({
                "channel":              ch,
                "chromophore":          label,
                "cnr":                  round(cnr_val, 4) if np.isfinite(cnr_val) else np.nan,
                f"{g1_name}_ga_peak":   round(m.get("mu1", np.nan), 4) if np.isfinite(m.get("mu1", np.nan)) else np.nan,
                f"{g2_name}_ga_peak":   round(m.get("mu2", np.nan), 4) if np.isfinite(m.get("mu2", np.nan)) else np.nan,
                f"{g1_name}_n":         m.get("n1", 0),
                f"{g1_name}_sigma":     round(m.get("sigma1", np.nan), 4) if np.isfinite(m.get("sigma1", np.nan)) else np.nan,
                f"{g2_name}_n":         m.get("n2", 0),
                f"{g2_name}_sigma":     round(m.get("sigma2", np.nan), 4) if np.isfinite(m.get("sigma2", np.nan)) else np.nan,
                "legacy_cnr":           round(leg_val, 4) if np.isfinite(leg_val) else np.nan,
            })

        # Per-chromophore stats
        valid_ghrf = [v for v in ghrf_cnr.values() if np.isfinite(v)]
        valid_leg  = [v for v in legacy_cnr_vals.values() if np.isfinite(v)]
        if valid_ghrf:
            print(f"    {label} [group-HRF primary] : {len(valid_ghrf)} channels | "
                  f"median={np.median(valid_ghrf):.3f}  max={np.max(valid_ghrf):.3f}")
        if valid_leg:
            print(f"    {label} [legacy peak-based]  : {len(valid_leg)} channels | "
                  f"median={np.median(valid_leg):.3f}  max={np.max(valid_leg):.3f}")

    df_cnr = pd.DataFrame(rows)
    all_ghrf = df_cnr["cnr"].dropna().values
    all_leg  = df_cnr["legacy_cnr"].dropna().values

    # ── Task 4.3 — Method comparison report ───────────────────────────────────
    print(f"\n  Task 4.3 — Method Comparison (Technical Validation Branch):")
    print(f"  {'Method':<28} {'Median':>8}  {'Mean':>8}  {'Max':>8}")
    print(f"  {'-'*48}")
    g_med = round(float(np.median(all_ghrf)), 4) if len(all_ghrf) > 0 else None
    g_mn  = round(float(np.mean(all_ghrf)),   4) if len(all_ghrf) > 0 else None
    g_mx  = round(float(np.max(all_ghrf)),    4) if len(all_ghrf) > 0 else None
    l_med = round(float(np.median(all_leg)),  4) if len(all_leg) > 0 else None
    l_mn  = round(float(np.mean(all_leg)),    4) if len(all_leg) > 0 else None
    l_mx  = round(float(np.max(all_leg)),     4) if len(all_leg) > 0 else None
    print(f"  {'Group-HRF (paper-faithful)':<28} {str(g_med):>8}  {str(g_mn):>8}  {str(g_mx):>8}")
    print(f"  {'Legacy peak-based':<28} {str(l_med):>8}  {str(l_mn):>8}  {str(l_mx):>8}")
    print(f"  {'Paper target (Figure 4i)':<28} {'~0.45':>8}  {'—':>8}  {'>2.25':>8}")

    # Retrieve actual filter params used in this techval branch
    first_fif = techval_fif_dir / f"sub-{list(groups.values())[0][0]}_techval_raw.fif"
    _actual_l_freq = _BP_L_FREQ
    _actual_h_freq = _BP_H_FREQ

    cnr_meta = {
        "comparison":            f"{g1_name} vs {g2_name}",
        "group1":                g1_name,
        "group2":                g2_name,
        "group1_n_subjects":     len(groups[g1_name]),
        "group2_n_subjects":     len(groups[g2_name]),
        "peak_window_s":         list(peak_window),
        "n_channels_hbo":        len(hbo_long),
        "n_channels_hbr":        len(hbr_long),
        "cnr_method":            "group_hrf",
        "n_valid_cnr":           int(len(all_ghrf)),
        "median_cnr":            g_med,
        "mean_cnr":              g_mn,
        "max_cnr":               g_mx,
        "pct_above_1.0":         round(float((all_ghrf >= 1.0).mean() * 100), 1) if len(all_ghrf) > 0 else None,
        "preprocessing_branch":  "technical_validation",
        "channel_pruning":       False,
        "rpca_correction":       False,
        "ols_regression":        False,
        "bandpass_hz":           [_actual_l_freq, _actual_h_freq],
        "bandpass_method":       _BP_METHOD,
        "paper_quote": (
            "no channels have been excluded from the analysis provided here "
            "and no denoising methods have been used prior to analysis"
        ),
    }

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = cnr_out_dir / f"cnr_{study_display}.csv"
    save_cnr_csv(df_cnr, csv_path)

    # ── Box plot — use group-HRF CNR as primary (Task 4.4) ───────────────────
    valid_cnr_list = df_cnr["cnr"].dropna().tolist()   # group-HRF values
    comparison_label = f"{g1_name} vs {g2_name}"

    # Always include current study in plot_data regardless of registry
    plot_data: dict = {comparison_label: valid_cnr_list}
    for lbl, val in comparison_registry.items():
        if lbl != comparison_label:
            plot_data[lbl] = val

    cnr_threshold = float(
        config.get("stage_c", {}).get("cnr", {}).get("threshold_detectable", 1.0)
    )
    plot_path = cnr_out_dir / "cnr_boxplot.png"
    plot_cnr_boxplot(plot_data, plot_path, cnr_threshold=cnr_threshold)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        **cnr_meta,
        "cnr_threshold_detectable": cnr_threshold,
        "outputs": {
            "cnr_csv":     str(csv_path),
            "cnr_boxplot": str(plot_path),
        },
    }
    summary_path = cnr_out_dir / "cnr_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: cnr_summary.json")

    # ── Console report ────────────────────────────────────────────────────────
    pct_above = (all_ghrf >= cnr_threshold).mean() * 100 if len(all_ghrf) > 0 else 0
    print(f"\n{'='*64}")
    print(f"  Branch   : no pruning | no rPCA | no OLS")
    print(f"  CNR      : group-HRF method")
    print(f"  Channels : {len(hbo_long)} HbO + {len(hbr_long)} HbR = "
          f"{len(hbo_long) + len(hbr_long)} total")
    print(f"  Valid CNR: {cnr_meta['n_valid_cnr']}")
    print(f"  Median   : {cnr_meta['median_cnr']}  (paper target ~0.45)")
    print(f"  Mean     : {cnr_meta['mean_cnr']}")
    print(f"  Max      : {cnr_meta['max_cnr']}  (paper target > 2.25)")
    print(f"  ≥ {cnr_threshold:.1f}    : {pct_above:.1f}%")
    print(f"{'='*64}\n")

    return summary


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Technical Validation CNR pipeline")
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute all cached .fif and evoked files (required when "
             "config.yaml parameters change, e.g. bandpass h_freq)",
    )
    args = parser.parse_args()
    run_techval_cnr_analysis(force_recompute=args.force)
