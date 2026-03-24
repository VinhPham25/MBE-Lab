"""
Stage C — Step C1: Hemodynamic Response Function (HRF) Extraction
Purpose: Extract, average, and verify trial-locked HRF epochs from the Stage B
         preprocessed hemoglobin concentration time series.

Task coverage:
  4.1.1 — Epoch extraction (tmin=-5, tmax=100 s, baseline=(-5,0))
  4.1.2 — Single-subject HRF averaging (average across trials per channel)
  4.1.3 — HRF morphology verification for A1 (HbO↑, HbR↓)
  4.1.4 — Brain region channel selection (3-region: PFC, M1, SMA)
  1.2   — 5-region channel mapping matching Figure 3c (Kamat et al. 2023)
  1.3   — Extended epoch window: tmax = 100 s post-onset

Algorithm
---------
1.  Load Stage B preprocessed .fif (continuous HbO/HbR, all channels).
1b. Prefer Stage B GLM-HRF evoked sidecar when available:
      sub-{id}_task-*_desc-glmhrf_nirs-ave.fif
    If absent, fall back to legacy epoch-average extraction below.
2.  Convert MNE Annotations → integer event array using event_id={'1':1,'2':2}.
3.  Epoch around trial-start events (event_id=1):
      tmin = -5 s   (pre-stimulus baseline)
      tmax = +100 s (post-stimulus window capturing full 100 s Figure 3c horizon)
      baseline = (-5, 0)  (subtract mean of pre-stimulus window)
      reject_by_annotation = False  (annotations are task onsets, not artefacts)
4.  (Legacy fallback) Save raw epoch object as {subject_id}-epo.fif.
5.  (Legacy fallback) Average epochs across trials → one Evoked object per channel.
6.  Export HRF time-series matrix (time × channel) to CSV.
7.  Verify HRF morphology:
      - HbO peak should be positive (increase) in [5, 15] s window
      - HbR peak should be negative (decrease) in the same window
      - Compute mean long-channel HbO/HbR per brain region
8a. 3-region classification (PFC / M1 / SMA) using 2-D midpoint coords:
      PFC  : channel midpoint y  >  0.060 m  (frontal)
      M1   : channel midpoint |x| > 0.040 m  AND  y <= 0.060 m  (lateral motor)
      SMA  : channel midpoint |x| <= 0.040 m AND  y <= 0.060 m  (midline/medial motor)
8b. 5-region classification (Figure 3c paper-faithful):
      Left Lateral PFC  : y > 0.060 m  AND  x < -0.020 m
      Central PFC       : y > 0.060 m  AND  |x| <= 0.020 m
      Right Lateral PFC : y > 0.060 m  AND  x > 0.020 m
      Left Medial M1    : y <= 0.060 m AND  x < 0 m  (left hemisphere motor)
      SMA               : y <= 0.060 m AND  x >= 0 m (midline / right hemisphere motor)
    Short-separation channels (D17–D24 suffix, or S8_D9 at 10 mm) are excluded
    from all region analyses.

Outputs (outputs/stage_c_validation/{subject_id}/)
  {subject_id}-epo.fif                    — raw epoch object (MNE Epochs)
  {subject_id}_hrf_averages.csv           — time × channel HRF matrix (μM)
  {subject_id}_hrf_verification_plot.png  — HbO/HbR morphology verification figure
  {subject_id}_hrf_region_map.json        — 3-region channel-to-region assignment
  {subject_id}_hrf_region_map_5.json      — 5-region channel-to-region assignment

Upstream : outputs/stage_b_processed/processed_data/{study}/sub-{id}_*_raw.fif
Reference : PROGRESS.md Section 4.1; AUDIT_PROGRESS.md Issue 1; config.yaml stage_c.epochs
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHORT_SEP_SUFFIX = tuple(f"D{d}" for d in range(17, 25))   # D17–D24
SHORT_SEP_SPECIAL = {"S8_D9"}                               # 10 mm channel

# Probe-rule mapping (paper/probe-design driven)
# 3-region:
#   S1-S3 -> PFC
#   S4-S7 -> M1
#   S8    -> SMA
# 5-region:
#   S1 -> Left Lateral PFC
#   S2 -> Central PFC
#   S3 -> Right Lateral PFC
#   S4-S7 -> Left Medial M1 (single motor panel in Figure 3c)
#   S8 -> SMA
PFC_SOURCES = {1, 2, 3}
M1_SOURCES = {4, 5, 6, 7}
SMA_SOURCES = {8}

STUDY = "expert_novice"


# ===========================================================================
# 4.1.4 — Brain region channel selection
# ===========================================================================

def _is_short_sep(ch_name: str) -> bool:
    """Return True if the base channel name (without hbo/hbr suffix) is short-sep."""
    base = ch_name.split(" ")[0]    # e.g. "S1_D1" from "S1_D1 hbo"
    if base in SHORT_SEP_SPECIAL:
        return True
    detector = base.split("_")[1]   # e.g. "D17"
    return detector in SHORT_SEP_SUFFIX


def _source_number(ch_name: str) -> int | None:
    """Extract integer source ID from channel base name (e.g. S4_D6 -> 4)."""
    base = ch_name.split(" ")[0]
    src = base.split("_")[0]
    if not src.startswith("S"):
        return None
    sid = src[1:]
    return int(sid) if sid.isdigit() else None


def build_region_map(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    """
    Classify long-separation HbO channels into PFC / M1 / SMA using
    probe source rules from the study design (not coordinate thresholds).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed haemo raw object (HbO + HbR channels present).

    Returns
    -------
    region_map : dict  {region_name: [channel_base_names]}
        Keys are 'PFC', 'M1', 'SMA'.  Channel names are the base names
        without the hbo/hbr suffix, e.g. 'S1_D1'.
    """
    region_map: dict[str, list[str]] = {"PFC": [], "M1": [], "SMA": []}

    for idx, ch in enumerate(raw.info["chs"]):
        ch_name = ch["ch_name"]
        if "hbr" in ch_name:           # process each pair once (via HbO)
            continue
        if "hbo" not in ch_name:
            continue
        if _is_short_sep(ch_name):
            continue

        base = ch_name.split(" ")[0]   # strip " hbo" suffix
        src_n = _source_number(ch_name)
        if src_n in PFC_SOURCES:
            region_map["PFC"].append(base)
        elif src_n in M1_SOURCES:
            region_map["M1"].append(base)
        elif src_n in SMA_SOURCES:
            region_map["SMA"].append(base)

    return region_map


def build_region_map_5(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    """
    5-region classification matching Figure 3c (Kamat et al. 2023).

    Regions (abbreviated keys used for GC pipeline compatibility)
    -------
    LPFC : Left Lateral PFC  — source S1
    CPFC : Central PFC       — source S2
    RPFC : Right Lateral PFC — source S3
    LM1  : Left Medial M1    — sources S4-S7
    SMA  : Supplementary Motor Area — source S8

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed haemo raw object with HbO + HbR channels.

    Returns
    -------
    region_map : dict  {abbreviated_region_name: [channel_base_names]}
        Five keys: 'LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA'.
        Channel names are base names without the ' hbo' suffix (e.g. 'S1_D1').
        The Stage D GC pipeline (utils_gc.py) appends ' hbo' when selecting
        channels from raw.ch_names.

    Note
    ----
    03_hrf_visualization.py maintains its own local copy of this function
    with full-name keys ('Left Lateral PFC', etc.) required for panel labels.
    Only this canonical copy uses abbreviated keys.
    """
    region_map: dict[str, list[str]] = {
        "LPFC": [],
        "CPFC": [],
        "RPFC": [],
        "LM1":  [],
        "SMA":  [],
    }

    for ch in raw.info["chs"]:
        ch_name = ch["ch_name"]
        if "hbr" in ch_name:
            continue
        if "hbo" not in ch_name:
            continue
        if _is_short_sep(ch_name):
            continue

        base = ch_name.split(" ")[0]
        src_n = _source_number(ch_name)
        if src_n == 1:
            region_map["LPFC"].append(base)
        elif src_n == 2:
            region_map["CPFC"].append(base)
        elif src_n == 3:
            region_map["RPFC"].append(base)
        elif src_n in M1_SOURCES:
            region_map["LM1"].append(base)
        elif src_n in SMA_SOURCES:
            region_map["SMA"].append(base)

    return region_map


def get_long_channel_names(raw: mne.io.BaseRaw) -> tuple[list[str], list[str]]:
    """
    Return (hbo_long, hbr_long) — channel names for long-separation channels only.
    """
    hbo_long = [c for c in raw.ch_names if "hbo" in c and not _is_short_sep(c)]
    hbr_long = [c for c in raw.ch_names if "hbr" in c and not _is_short_sep(c)]
    return hbo_long, hbr_long


# ===========================================================================
# 4.1.1 — Epoch extraction
# ===========================================================================

def extract_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = -5.0,
    tmax: float = 30.0,
    baseline: tuple[float, float] = (-5.0, 0.0),
    event_id: int = 1,
) -> mne.Epochs:
    """
    Extract trial-locked epochs from preprocessed continuous haemo data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed HbO/HbR data with MNE Annotations for trial onsets.
    tmin, tmax : float
        Epoch window in seconds relative to trial onset.
    baseline : tuple
        Baseline correction window (start, end) in seconds.
    event_id : int
        Annotation description integer to epoch around (1 = trial start).

    Returns
    -------
    epochs : mne.Epochs
        Baseline-corrected epoch object.
    """
    events, event_id_map = mne.events_from_annotations(
        raw,
        event_id={str(event_id): event_id},
        verbose=False,
    )

    n_events = (events[:, 2] == event_id).sum()
    print(f"  Found {n_events} trial-start events (annotation '{event_id}')")

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={str(event_id): event_id},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    n_dropped = n_events - len(epochs)
    print(f"  Epochs created: {len(epochs)} (dropped: {n_dropped})")
    print(f"  Epoch shape: {epochs.get_data().shape}  "
          f"(n_trials × n_channels × n_times)")
    return epochs


# ===========================================================================
# 4.1.2 — Single-subject HRF averaging
# ===========================================================================

def compute_hrf_average(
    epochs: mne.Epochs,
    hbo_long: list[str],
    hbr_long: list[str],
) -> tuple[mne.Evoked, pd.DataFrame]:
    """
    Average epochs across trials for long-separation channels and export to CSV.

    Parameters
    ----------
    epochs : mne.Epochs
        Trial-locked epoch object.
    hbo_long, hbr_long : list[str]
        Long-separation HbO and HbR channel names.

    Returns
    -------
    evoked : mne.Evoked
        Average evoked response for long-separation channels.
    df_hrf : pd.DataFrame
        Tidy DataFrame with columns ['time_s', ch_name, …] in μM.
    """
    long_channels = hbo_long + hbr_long
    epochs_long = epochs.copy().pick(long_channels)
    evoked = epochs_long.average()

    times = evoked.times
    data_uM = evoked.data * 1e6     # mol/L → μM

    df_hrf = pd.DataFrame(data_uM.T, columns=evoked.ch_names)
    df_hrf.insert(0, "time_s", times)

    print(f"  HRF averaged: {len(evoked.ch_names)} long channels "
          f"× {len(times)} time points")
    print(f"  HbO mean peak range: "
          f"{data_uM[[i for i, c in enumerate(evoked.ch_names) if 'hbo' in c], :].max():.3f} μM")
    print(f"  HbR mean peak range: "
          f"{data_uM[[i for i, c in enumerate(evoked.ch_names) if 'hbr' in c], :].min():.3f} μM")

    return evoked, df_hrf


# ===========================================================================
# 4.1.3 — HRF morphology verification
# ===========================================================================

def verify_hrf_morphology(
    evoked: mne.Evoked,
    region_map: dict[str, list[str]],
    peak_window: tuple[float, float] = (5.0, 15.0),
) -> dict:
    """
    Verify that HbO increases and HbR decreases post-stimulus in the expected
    time window.  Compute per-region averages for the verification plot.

    Returns
    -------
    results : dict
        Morphology check results and per-region mean HRF arrays.
    """
    times = evoked.times
    t_mask = (times >= peak_window[0]) & (times <= peak_window[1])

    data_uM = evoked.data * 1e6
    ch_names = evoked.ch_names

    hbo_idx = [i for i, c in enumerate(ch_names) if "hbo" in c]
    hbr_idx = [i for i, c in enumerate(ch_names) if "hbr" in c]

    hbo_mean_trace = data_uM[hbo_idx, :].mean(axis=0)
    hbr_mean_trace = data_uM[hbr_idx, :].mean(axis=0)

    hbo_peak = hbo_mean_trace[t_mask].max()
    hbr_trough = hbr_mean_trace[t_mask].min()

    hbo_ok = hbo_peak > 0
    hbr_ok = hbr_trough < 0

    print(f"\n  --- HRF Morphology Check ---")
    print(f"  HbO peak (mean long, {peak_window[0]}-{peak_window[1]}s): "
          f"{hbo_peak:+.3f} μM  → {'✓ INCREASE' if hbo_ok else '✗ FAIL (expected positive)'}")
    print(f"  HbR trough (mean long, {peak_window[0]}-{peak_window[1]}s): "
          f"{hbr_trough:+.3f} μM  → {'✓ DECREASE' if hbr_ok else '✗ FAIL (expected negative)'}")

    # Per-region traces
    region_traces: dict[str, dict[str, np.ndarray]] = {}
    for region, bases in region_map.items():
        hbo_names = [f"{b} hbo" for b in bases]
        hbr_names = [f"{b} hbr" for b in bases]
        hbo_ri = [i for i, c in enumerate(ch_names) if c in hbo_names]
        hbr_ri = [i for i, c in enumerate(ch_names) if c in hbr_names]
        region_traces[region] = {
            "hbo": data_uM[hbo_ri, :].mean(axis=0) if hbo_ri else np.zeros_like(times),
            "hbr": data_uM[hbr_ri, :].mean(axis=0) if hbr_ri else np.zeros_like(times),
            "n_channels": len(bases),
        }
        r_hbo_peak = region_traces[region]["hbo"][t_mask].max() if hbo_ri else np.nan
        r_hbr_trough = region_traces[region]["hbr"][t_mask].min() if hbr_ri else np.nan
        print(f"  {region:5s}: {len(bases):2d} ch | "
              f"HbO peak={r_hbo_peak:+.3f} μM  HbR trough={r_hbr_trough:+.3f} μM")

    return {
        "hbo_peak_uM": float(hbo_peak),
        "hbr_trough_uM": float(hbr_trough),
        "hbo_morphology_pass": bool(hbo_ok),
        "hbr_morphology_pass": bool(hbr_ok),
        "overall_pass": bool(hbo_ok and hbr_ok),
        "region_traces": region_traces,
        "times": times,
        "hbo_mean_trace": hbo_mean_trace,
        "hbr_mean_trace": hbr_mean_trace,
    }


def plot_hrf_verification(
    morph: dict,
    region_map: dict[str, list[str]],
    subject_id: str,
    out_path: Path,
) -> None:
    """
    Generate a 2-row verification figure:
      Top row : grand-average HbO (red) + HbR (blue) across all long channels
      Bottom row : per-region HbO/HbR average traces (PFC | M1 | SMA)

    Saves PNG to out_path.
    """
    times = morph["times"]
    regions = list(region_map.keys())
    n_regions = len(regions)

    fig, axes = plt.subplots(
        2, n_regions,
        figsize=(5 * n_regions, 8),
        sharex=True,
    )
    if n_regions == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"{subject_id} — HRF Verification  "
        f"({'✓ PASS' if morph['overall_pass'] else '✗ FAIL'})",
        fontsize=14, fontweight="bold",
    )

    hbo_color, hbr_color = "#D62728", "#1F77B4"
    shade_color = "#FFDD89"

    # ── Top row: grand-average across all long channels ──────────────────────
    for col in range(n_regions):
        ax = axes[0, col]
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax.axvspan(5, 15, alpha=0.15, color=shade_color, label="Peak window", zorder=0)
        ax.axvline(0, color="black", lw=1, ls="-", zorder=2)

        ax.plot(times, morph["hbo_mean_trace"],
                color=hbo_color, lw=2.0, label="HbO (mean long)")
        ax.plot(times, morph["hbr_mean_trace"],
                color=hbr_color, lw=2.0, label="HbR (mean long)")

        if col == 0:
            ax.set_ylabel("ΔHb (μM)", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")
        ax.set_title(
            f"Grand Average (all long ch)\n"
            f"HbO peak={morph['hbo_peak_uM']:+.2f} μM  "
            f"HbR trough={morph['hbr_trough_uM']:+.2f} μM",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

    # ── Bottom row: per-region ────────────────────────────────────────────────
    for col, region in enumerate(regions):
        ax = axes[1, col]
        traces = morph["region_traces"][region]
        n_ch = traces["n_channels"]

        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax.axvspan(5, 15, alpha=0.15, color=shade_color, zorder=0)
        ax.axvline(0, color="black", lw=1, ls="-", zorder=2)

        ax.plot(times, traces["hbo"], color=hbo_color, lw=2.0,
                label=f"HbO ({n_ch} ch)")
        ax.plot(times, traces["hbr"], color=hbr_color, lw=2.0,
                label=f"HbR ({n_ch} ch)")

        t_mask = (times >= 5) & (times <= 15)
        r_hbo_peak = traces["hbo"][t_mask].max()
        r_hbr_trough = traces["hbr"][t_mask].min()

        ax.set_title(
            f"{region}  (n={n_ch} pairs)\n"
            f"HbO peak={r_hbo_peak:+.2f} μM  HbR trough={r_hbr_trough:+.2f} μM",
            fontsize=10,
        )
        ax.set_xlabel("Time (s)", fontsize=11)
        if col == 0:
            ax.set_ylabel("ΔHb (μM)", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ===========================================================================
# Top-level pipeline
# ===========================================================================

_SENTINEL = object()   # sentinel to detect "caller passed nothing"


def run_hrf_extraction(
    subject_id: str = "A1",
    study: str = STUDY,
    tmin=_SENTINEL,
    tmax=_SENTINEL,
    baseline=_SENTINEL,
    trial_event_id: int = 1,
) -> dict:
    """
    Full Section 4.1 pipeline for one subject.

    Parameters
    ----------
    subject_id : str   Subject identifier (e.g. 'A1').
    study      : str   Study key matching config (e.g. 'expert_novice').
    tmin, tmax : float Epoch window in seconds.  When not supplied the values
                       are read from config.yaml (stage_c.epochs.tmin / tmax).
    baseline   : tuple Baseline correction window.  When not supplied, read
                       from config.yaml (stage_c.epochs.baseline).
    trial_event_id : int  Annotation code for trial onset.

    Returns
    -------
    results : dict  Summary metrics and file paths (includes both 3-region and 5-region maps).
    """
    config = load_config()

    # ── Resolve epoch window from config when not explicitly passed ───────────
    epochs_cfg = config["stage_c"]["epochs"]
    if tmin is _SENTINEL:
        tmin = float(epochs_cfg["tmin"])
    if tmax is _SENTINEL:
        tmax = float(epochs_cfg["tmax"])
    if baseline is _SENTINEL:
        baseline = tuple(float(v) for v in epochs_cfg["baseline"])
    # Study display name matches the BIDS folder (e.g. "Expert-Novice")
    study_display_name = Path(config["datasets"][study]["data_path"]).name
    out_dir = (Path(config["paths"]["stage_c_outputs"])
               / "hrf" / study_display_name / subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve Stage B outputs (.fif + optional GLM-HRF sidecar) ───────────
    study_cfg = config["datasets"][study]
    task_slug = {
        "expert_novice": "ExpNovPatternCutting",
        "learning_curve": "LC",
        "short_term_tes": "tES",
        "long_term_tes": "tES",
    }.get(study, "task")

    stage_b_dir = Path(config["paths"]["stage_b_outputs"]) / "processed_data" / study.replace("_", "-")
    fif_path = stage_b_dir / f"sub-{subject_id}_task-{task_slug}_desc-preprocessed_nirs_raw.fif"
    glm_ave_path = stage_b_dir / f"sub-{subject_id}_task-{task_slug}_desc-glmhrf_nirs-ave.fif"

    if not fif_path.exists():
        # Fallback: search for any matching file
        candidates = list(stage_b_dir.glob(f"sub-{subject_id}_*_raw.fif"))
        if not candidates:
            raise FileNotFoundError(
                f"No preprocessed .fif found for {subject_id} in {stage_b_dir}"
            )
        fif_path = candidates[0]

    if not glm_ave_path.exists():
        glm_candidates = list(stage_b_dir.glob(f"sub-{subject_id}_*_desc-glmhrf_*ave.fif"))
        if glm_candidates:
            glm_ave_path = glm_candidates[0]

    print(f"\n{'='*60}")
    print(f"  HRF Extraction — Subject: {subject_id}")
    print(f"  Input : {fif_path.name}")
    if glm_ave_path.exists():
        print(f"  GLM HRF sidecar: {glm_ave_path.name}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ── Load preprocessed data ───────────────────────────────────────────────
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    print(f"\n  Loaded: {len(raw.ch_names)} channels, "
          f"{raw.times[-1]:.1f} s @ {raw.info['sfreq']} Hz")

    # ── 4.1.4 — 3-region map ─────────────────────────────────────────────────
    print(f"\n[4.1.4] Brain region channel classification (3-region)")
    region_map = build_region_map(raw)
    hbo_long, hbr_long = get_long_channel_names(raw)
    print(f"  Long channels: {len(hbo_long)} HbO, {len(hbr_long)} HbR")
    for region, bases in region_map.items():
        print(f"  {region:5s}: {len(bases)} pairs → {bases}")

    region_map_path = out_dir / f"{subject_id}_hrf_region_map.json"
    with open(region_map_path, "w") as f:
        json.dump(region_map, f, indent=2)
    print(f"  Saved: {region_map_path.name}")

    # ── 1.2 — 5-region map (Figure 3c) ───────────────────────────────────────
    print(f"\n[1.2] 5-region classification (Figure 3c — Kamat et al. 2023)")
    region_map_5 = build_region_map_5(raw)
    for region, bases in region_map_5.items():
        print(f"  {region:6s}: {len(bases)} pairs")

    region_map_5_path = out_dir / f"{subject_id}_hrf_region_map_5.json"
    with open(region_map_5_path, "w") as f:
        json.dump(region_map_5, f, indent=2)
    print(f"  Saved: {region_map_5_path.name}")

    hrf_source = "epoch_average"
    epo_path = None
    n_trials = None

    if glm_ave_path.exists():
        # Preferred path: use Stage B GLM-deconvolved HRF directly.
        print(f"\n[4.1.1/4.1.2] Loading Stage B GLM HRF (skipping epoch averaging)")
        evoked = mne.read_evokeds(glm_ave_path, verbose=False)[0]
        df_hrf = pd.DataFrame((evoked.data * 1e6).T, columns=evoked.ch_names)
        df_hrf.insert(0, "time_s", evoked.times)
        hrf_source = "stage_b_glm_hrf"
        n_trials = int(getattr(evoked, "nave", 0))
        print(f"  Loaded GLM HRF: {len(evoked.ch_names)} channels × {len(evoked.times)} time points")
        print(f"  GLM nave (trials): {n_trials}")
    else:
        # Legacy fallback: epoch and average from continuous signal.
        print(f"\n[4.1.1] Epoch extraction (tmin={tmin}s, tmax={tmax}s, baseline={baseline})")
        epochs = extract_epochs(raw, tmin=tmin, tmax=tmax, baseline=baseline,
                                event_id=trial_event_id)

        epo_path = out_dir / f"{subject_id}-epo.fif"
        epochs.save(epo_path, overwrite=True, verbose=False)
        print(f"  Saved: {epo_path.name}")

        print(f"\n[4.1.2] HRF averaging across {len(epochs)} trials")
        evoked, df_hrf = compute_hrf_average(epochs, hbo_long, hbr_long)
        n_trials = int(len(epochs))

    ave_path = out_dir / f"{subject_id}-ave.fif"
    evoked.save(ave_path, overwrite=True, verbose=False)
    print(f"  Saved: {ave_path.name}")

    csv_path = out_dir / f"{subject_id}_hrf_averages.csv"
    df_hrf.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}  "
          f"({df_hrf.shape[0]} rows × {df_hrf.shape[1]} cols)")

    # ── 4.1.3 — Morphology verification ──────────────────────────────────────
    print(f"\n[4.1.3] HRF morphology verification")
    peak_window = tuple(
        float(v) for v in config["stage_c"]["hrf"]["peak_analysis_window"]
    )
    morph = verify_hrf_morphology(evoked, region_map, peak_window=peak_window)

    plot_path = out_dir / f"{subject_id}_hrf_verification_plot.png"
    plot_hrf_verification(morph, region_map, subject_id, plot_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    results = {
        "subject_id": subject_id,
        "hrf_source": hrf_source,
        "n_trials": n_trials,
        "n_long_hbo": len(hbo_long),
        "n_long_hbr": len(hbr_long),
        "epoch_tmin": tmin,
        "epoch_tmax": tmax,
        "baseline": list(baseline),
        "hbo_peak_uM": morph["hbo_peak_uM"],
        "hbr_trough_uM": morph["hbr_trough_uM"],
        "hbo_morphology_pass": morph["hbo_morphology_pass"],
        "hbr_morphology_pass": morph["hbr_morphology_pass"],
        "overall_morphology_pass": morph["overall_pass"],
        "region_channel_counts_3": {r: len(v) for r, v in region_map.items()},
        "region_channel_counts_5": {r: len(v) for r, v in region_map_5.items()},
        "outputs": {
            "epochs_fif": str(epo_path) if epo_path is not None else None,
            "evoked_fif": str(ave_path),
            "hrf_averages_csv": str(csv_path),
            "verification_plot": str(plot_path),
            "region_map_json": str(region_map_path),
            "region_map_5_json": str(region_map_5_path),
            "stage_b_glm_hrf_input": str(glm_ave_path) if glm_ave_path.exists() else None,
        },
    }

    summary_path = out_dir / f"{subject_id}_hrf_extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Section 4.1 complete — {subject_id}")
    print(f"  Trials: {results['n_trials']}  |  "
          f"Long channels: {results['n_long_hbo']} HbO + {results['n_long_hbr']} HbR")
    print(f"  Morphology: {'✓ PASS' if results['overall_morphology_pass'] else '✗ FAIL'}")
    print(f"  HbO peak: {results['hbo_peak_uM']:+.3f} μM  "
          f"HbR trough: {results['hbr_trough_uM']:+.3f} μM")
    print(f"  3-region counts: {results['region_channel_counts_3']}")
    print(f"  5-region counts: {results['region_channel_counts_5']}")
    print(f"{'='*60}\n")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage C — Step C1: HRF Extraction"
    )
    parser.add_argument("--subject", default="A1", help="Subject ID (default: A1)")
    parser.add_argument("--study", default=STUDY,
                        help="Study key from config (default: expert_novice)")
    args = parser.parse_args()

    run_hrf_extraction(subject_id=args.subject, study=args.study)
