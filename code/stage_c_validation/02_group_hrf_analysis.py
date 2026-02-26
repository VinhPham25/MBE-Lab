"""
Stage C — Step C1 (Group): Group-Level HRF Analysis
Purpose: Compute grand-average hemodynamic response functions for the Expert
         and Novice groups, then verify HRF peak timing and amplitude.

Task coverage:
  4.2.1 — Grand average HRF for Expert group (A1, A2, R7–R12; n=8)
  4.2.2 — Grand average HRF for Novice group (R1–R6; n=6)
  4.2.3 — Verify HRF timing (peak ~8–12 s) and amplitude (5–15 μM HbO)

Algorithm
---------
1.  For each subject in both groups, run the Section 4.1 per-subject HRF
    extraction pipeline (`run_hrf_extraction` from 01_hrf_extraction.py).
    Caches: if {subject_id}-ave.fif already exists in the subject output
    directory, it is loaded directly (skips re-computation).

2.  Collect per-subject Evoked objects into Expert / Novice lists.
    MNE grand_average() is then used to compute the group mean.
    Channels not present in every subject are dropped (MNE default) to
    guarantee a consistent channel set across subjects with different bad
    channels.

3.  For each group × region (PFC / M1 / SMA) × chromophore (HbO / HbR):
      - Extract the region-mean time-series from the grand-average Evoked.
      - Find peak time and amplitude within the analysis window [5, 15] s.
      - Report results in group_hrf_stats.json.

Group definitions (Expert-Novice study):
  Expert (n=8) : A1, A2, R7, R8, R9, R10, R11, R12
  Novice (n=6) : R1, R2, R3, R4, R5, R6
  Source       : data/BIDS/Expert-Novice/README_EN_Performance.txt

Outputs (outputs/stage_c_validation/group/)
  expert_grand_average-ave.fif   — MNE Evoked, grand mean Expert HRF
  novice_grand_average-ave.fif   — MNE Evoked, grand mean Novice HRF
  group_hrf_stats.json           — Per-region peak time & amplitude report

Upstream : 01_hrf_extraction.py  (run_hrf_extraction, build_region_map)
Reference : PROGRESS.md Section 4.2; config.yaml stage_c
"""

import importlib.util
import json
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config

# ---------------------------------------------------------------------------
# Import Section 4.1 helpers via importlib (same pattern as Stage B chain)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "hrf_extraction",
    Path(__file__).parent / "01_hrf_extraction.py",
)
_hrf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hrf)

run_hrf_extraction = _hrf.run_hrf_extraction
build_region_map   = _hrf.build_region_map

# ---------------------------------------------------------------------------
# Group definitions (Expert-Novice study)
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "Expert": ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"],
    "Novice": ["R1", "R2", "R3", "R4", "R5", "R6"],
}

STUDY = "expert_novice"


# ===========================================================================
# 4.2.1 / 4.2.2 — Per-subject extraction + grand average
# ===========================================================================

def _load_or_compute_subject_evoked(
    subject_id: str,
    out_base: Path,
    study: str,
    min_tmax: float = 95.0,
    max_tmin: float = -24.0,
) -> mne.Evoked | None:
    """
    Return the per-subject Evoked object for `subject_id`.
    Loads from cached -ave.fif if available AND the cached epoch window
    spans the required range; otherwise (re-)runs the full Section 4.1
    pipeline to produce it.

    Parameters
    ----------
    min_tmax : float
        Minimum required last time point (s).  Cached evokeds that end before
        this are stale (e.g. old 30 s extractions) and are recomputed.
        Default 95 s (tolerates sample-rounding near tmax=100 s).
    max_tmin : float
        Maximum (least negative) allowed first time point (s).  Cached evokeds
        whose tmin is greater than this value have an insufficient pre-stimulus
        baseline and are recomputed.  Default -24 s (tolerates sample-rounding
        near tmin=-25 s).  Set to a large negative number (e.g. -999) to skip
        this check.

    Returns None if the subject's preprocessed .fif cannot be found.
    """
    ave_path = out_base / subject_id / f"{subject_id}-ave.fif"

    if ave_path.exists():
        evoked = mne.read_evokeds(ave_path, verbose=False)[0]
        tmin_cached = float(evoked.times[0])
        tmax_cached = float(evoked.times[-1])

        tmax_ok = tmax_cached >= min_tmax
        tmin_ok = tmin_cached <= max_tmin

        if tmax_ok and tmin_ok:
            print(f"    [{subject_id}] loaded cached {ave_path.name} "
                  f"(tmin={tmin_cached:.1f} s, tmax={tmax_cached:.1f} s)")
            return evoked

        reason = []
        if not tmax_ok:
            reason.append(f"tmax {tmax_cached:.1f} s < {min_tmax} s")
        if not tmin_ok:
            reason.append(f"tmin {tmin_cached:.1f} s > {max_tmin} s")
        print(f"    [{subject_id}] cached window stale ({'; '.join(reason)}) — recomputing …")

    # Run full Section 4.1 pipeline (reads tmin/baseline from config)
    print(f"    [{subject_id}] running HRF extraction …")
    try:
        run_hrf_extraction(subject_id=subject_id, study=study)
        evoked = mne.read_evokeds(ave_path, verbose=False)[0]
        return evoked
    except FileNotFoundError as exc:
        print(f"    [{subject_id}] ⚠ SKIPPED — {exc}")
        return None


def _harmonise_evoked(
    evoked: mne.Evoked,
    target_sfreq: float,
    target_n_times: int | None = None,
) -> mne.Evoked:
    """
    Resample `evoked` to `target_sfreq` if its sampling rate differs.
    Ex: Subject R9 from Expert-Novice study was processed at 50 Hz in Stage B due to split-recording handling.

    Also drops any channels that contain NaN values 
    Ex: Subject R10 has from Expert-Novice study has trailing-zero artifacts

    Returns the (possibly modified) Evoked in-place.
    """
    # Drop NaN channels
    nan_channels = [
        ch for ch, row in zip(evoked.ch_names, evoked.data)
        if not np.isfinite(row).all()
    ]
    if nan_channels:
        evoked = evoked.copy().drop_channels(nan_channels)
        print(f"      ⚠ dropped {len(nan_channels)} NaN channel(s): "
              f"{nan_channels[:4]}{'…' if len(nan_channels) > 4 else ''}")

    # Resample to common sfreq
    if evoked.info["sfreq"] != target_sfreq:
        original = evoked.info["sfreq"]
        evoked = evoked.copy().resample(target_sfreq)
        print(f"      ⚠ resampled {original} Hz → {target_sfreq} Hz  "
              f"({evoked.data.shape[1]} time points)")

    return evoked


def compute_grand_averages(
    groups: dict[str, list[str]],
    out_base: Path,
    study: str,
    target_sfreq: float = 25.0,
) -> dict[str, mne.Evoked]:
    """
    Run per-subject extraction for every subject in every group, then compute
    grand averages with mne.grand_average().

    Before averaging, each Evoked is harmonised:
      - Resampled to `target_sfreq` if sfreq differs (e.g. R9 at 50 Hz).
      - Channels with NaN dropped (e.g. R10 data quality flag).
    mne.grand_average then drops any channels absent in all harmonised evokeds.

    Returns
    -------
    grand_avgs : dict  {group_name: mne.Evoked}
    """
    grand_avgs: dict[str, mne.Evoked] = {}

    for group_name, subjects in groups.items():
        print(f"\n  [{group_name} group — {len(subjects)} subjects]")
        evokeds: list[mne.Evoked] = []

        for sid in subjects:
            ev = _load_or_compute_subject_evoked(sid, out_base, study)
            if ev is not None:
                ev = _harmonise_evoked(ev, target_sfreq)
                evokeds.append(ev)

        n_ok = len(evokeds)
        n_total = len(subjects)
        print(f"  → {n_ok}/{n_total} subjects available for grand average")

        if n_ok == 0:
            raise RuntimeError(
                f"No valid subjects found for group '{group_name}'"
            )

        # grand_average drops channels absent in any subject (consistent set)
        grand_avg = mne.grand_average(evokeds, interpolate_bads=False,
                                       drop_bads=False)
        grand_avg.comment = group_name
        grand_avgs[group_name] = grand_avg

        n_ch = len(grand_avg.ch_names)
        print(f"  → Grand average: {n_ch} channels, "
              f"{n_ok} subjects contributed")

    return grand_avgs


# ===========================================================================
# 4.2.3 — Peak timing and amplitude verification
# ===========================================================================

def _region_mean_trace(
    evoked: mne.Evoked,
    region_channels: list[str],
    chromophore: str,
) -> np.ndarray | None:
    """
    Return the mean time-series across all `region_channels` of type
    `chromophore` ('hbo' or 'hbr') from `evoked`, in μM.
    Returns None if no matching channels are present.
    """
    target = [f"{base} {chromophore}" for base in region_channels]
    present = [c for c in target if c in evoked.ch_names]
    if not present:
        return None
    idx = [evoked.ch_names.index(c) for c in present]
    return evoked.data[idx, :].mean(axis=0) * 1e6    # mol/L → μM


def compute_peak_stats(
    grand_avgs: dict[str, mne.Evoked],
    region_map: dict[str, list[str]],
    peak_window: tuple[float, float] = (5.0, 15.0),
    expected_peak_range: tuple[float, float] = (8.0, 12.0),
    expected_hbo_amp_range: tuple[float, float] = (5.0, 15.0),
) -> dict:
    """
    For each group × region × chromophore extract:
      - peak_time_s  : time (s) of maximum |amplitude| within peak_window
      - peak_amp_uM  : amplitude at peak_time (signed, μM)
      - timing_ok    : bool — peak_time within expected_peak_range
      - amplitude_ok : bool — for HbO: peak within expected_hbo_amp_range

    Returns
    -------
    stats : dict  flat key format "{Group}_{Region}_{Chromo}_{Metric}"
    """
    times = list(grand_avgs.values())[0].times
    t_mask = (times >= peak_window[0]) & (times <= peak_window[1])

    stats: dict = {
        "analysis_peak_window_s": list(peak_window),
        "expected_peak_range_s": list(expected_peak_range),
        "expected_hbo_amplitude_uM": list(expected_hbo_amp_range),
        "groups": {},
    }

    print(f"\n  --- Peak Stats ({peak_window[0]}–{peak_window[1]} s window) ---")

    for group_name, evoked in grand_avgs.items():
        stats["groups"][group_name] = {}

        for region, bases in region_map.items():
            stats["groups"][group_name][region] = {}

            for chromo in ("hbo", "hbr"):
                trace = _region_mean_trace(evoked, bases, chromo)
                if trace is None:
                    stats["groups"][group_name][region][chromo] = None
                    continue

                window_trace = trace[t_mask]
                window_times = times[t_mask]

                # For HbO take positive peak; for HbR take negative trough
                if chromo == "hbo":
                    peak_idx = np.argmax(window_trace)
                else:
                    peak_idx = np.argmin(window_trace)

                peak_time  = float(window_times[peak_idx])
                peak_amp   = float(window_trace[peak_idx])
                timing_ok  = expected_peak_range[0] <= peak_time <= expected_peak_range[1]
                amp_ok     = (expected_hbo_amp_range[0] <= abs(peak_amp) <= expected_hbo_amp_range[1]
                              if chromo == "hbo" else True)

                entry = {
                    "peak_time_s": round(peak_time, 2),
                    "peak_amp_uM": round(peak_amp, 3),
                    "timing_ok":   timing_ok,
                    "amplitude_ok": amp_ok,
                    "n_channels": len(bases),
                }
                stats["groups"][group_name][region][chromo] = entry

                timing_flag = "✓" if timing_ok else "✗"
                amp_flag    = "✓" if amp_ok else "~"
                label = "hbo" if chromo == "hbo" else "hbr"
                print(f"  {group_name:7s} {region:5s} {label}: "
                      f"peak={peak_amp:+6.3f} μM @ {peak_time:.1f} s  "
                      f"timing{timing_flag}  amp{amp_flag}")

    return stats


# ===========================================================================
# Top-level pipeline
# ===========================================================================

def run_group_hrf_analysis(
    groups: dict[str, list[str]] = GROUPS,
    study: str = STUDY,
    peak_window: tuple[float, float] = (5.0, 15.0),
) -> dict:
    """
    Full Section 4.2 pipeline.

    Parameters
    ----------
    groups      : dict {group_name: [subject_ids]}
    study       : config study key
    peak_window : (start, end) seconds for peak extraction

    Returns
    -------
    results : dict  summary metrics and file paths
    """
    config = load_config()
    # Study display name matches the BIDS folder (e.g. "Expert-Novice")
    study_display_name = Path(config["datasets"][study]["data_path"]).name
    out_base = (Path(config["paths"]["stage_c_outputs"])
                / "hrf" / study_display_name)
    group_dir = out_base / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    # Stage B paths use the lowercase-dash form (e.g. "expert-novice")
    stage_b_study_name = study.replace("_", "-")

    print(f"\n{'='*60}")
    print(f"  Section 4.2 — Group-Level HRF Analysis")
    print(f"  Groups: {', '.join(f'{k} (n={len(v)})' for k, v in groups.items())}")
    print(f"{'='*60}")

    # ── 4.2.1 + 4.2.2 — Compute grand averages ──────────────────────────────
    grand_avgs = compute_grand_averages(groups, out_base, study)

    # Save grand-average Evoked files
    saved_paths: dict[str, str] = {}
    for group_name, evoked in grand_avgs.items():
        fname = f"{group_name.lower()}_grand_average-ave.fif"
        out_path = group_dir / fname
        evoked.save(out_path, overwrite=True, verbose=False)
        saved_paths[group_name] = str(out_path)
        print(f"\n  Saved: {fname}  ({len(evoked.ch_names)} channels)")

    # ── Region map: derive from first available subject's preprocessed fif ──
    first_subject = next(iter(next(iter(groups.values()))))
    ave_path = out_base / first_subject / f"{first_subject}-ave.fif"
    # We need the raw fif for channel positions; load from Stage B
    stage_b_dir = (Path(config["paths"]["stage_b_outputs"])
                   / "processed_data"
                   / stage_b_study_name)
    candidates = list(stage_b_dir.glob(f"sub-{first_subject}_*_raw.fif"))
    raw_ref = mne.io.read_raw_fif(candidates[0], preload=False, verbose=False)
    region_map = build_region_map(raw_ref)
    print(f"\n  Region map (from {first_subject}): "
          f"{', '.join(f'{r}={len(v)}ch' for r, v in region_map.items())}")

    # ── 4.2.3 — Peak stats ──────────────────────────────────────────────────
    expected_peak = tuple(
        float(v) for v in config["stage_c"]["hrf"]["expected_peak_time_min":"expected_peak_time_max"]
    ) if False else (
        float(config["stage_c"]["hrf"]["expected_peak_time_min"]),
        float(config["stage_c"]["hrf"]["expected_peak_time_max"]),
    )
    expected_hbo_amp = tuple(
        float(v) for v in config["stage_c"]["hrf"]["expected_hbo_amplitude"]
    )

    peak_stats = compute_peak_stats(
        grand_avgs,
        region_map,
        peak_window=peak_window,
        expected_peak_range=expected_peak,
        expected_hbo_amp_range=expected_hbo_amp,
    )

    # Flatten stats into the paper-style key format for the JSON report
    flat_stats: dict = {
        "analysis_peak_window_s": peak_stats["analysis_peak_window_s"],
        "expected_peak_range_s": peak_stats["expected_peak_range_s"],
        "expected_hbo_amplitude_uM": peak_stats["expected_hbo_amplitude_uM"],
    }
    for group_name, regions in peak_stats["groups"].items():
        for region, chromos in regions.items():
            for chromo, entry in chromos.items():
                if entry is None:
                    continue
                label = "HbO" if chromo == "hbo" else "HbR"
                prefix = f"{group_name}_{region}_{label}"
                flat_stats[f"{prefix}_PeakTime_s"]  = entry["peak_time_s"]
                flat_stats[f"{prefix}_PeakAmp_uM"]  = entry["peak_amp_uM"]
                flat_stats[f"{prefix}_TimingOK"]    = entry["timing_ok"]
                flat_stats[f"{prefix}_AmplitudeOK"] = entry["amplitude_ok"]
                flat_stats[f"{prefix}_N_Channels"]  = entry["n_channels"]

    stats_path = group_dir / "group_hrf_stats.json"
    with open(stats_path, "w") as f:
        json.dump(flat_stats, f, indent=2)
    print(f"\n  Saved: group_hrf_stats.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_timing_ok = sum(
        1 for k, v in flat_stats.items()
        if k.endswith("_TimingOK") and v is True
    )
    n_timing_total = sum(1 for k in flat_stats if k.endswith("_TimingOK"))

    results = {
        "groups": {g: len(s) for g, s in groups.items()},
        "subjects_processed": {
            g: [sid for sid in sids
                if (out_base / sid / f"{sid}-ave.fif").exists()]
            for g, sids in groups.items()
        },
        "peak_timing_pass": f"{n_timing_ok}/{n_timing_total}",
        "outputs": {
            "expert_grand_avg": saved_paths.get("Expert", ""),
            "novice_grand_avg": saved_paths.get("Novice", ""),
            "group_hrf_stats": str(stats_path),
        },
    }

    print(f"\n{'='*60}")
    print(f"  Section 4.2 complete")
    print(f"  Peak timing within {expected_peak[0]}–{expected_peak[1]} s: "
          f"{n_timing_ok}/{n_timing_total} region×group×chromo entries")
    print(f"{'='*60}\n")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_group_hrf_analysis()
