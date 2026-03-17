"""
Stage D — Step D2: Trial Segmentation

Extract per-trial sample index pairs ``(onset_sample, offset_sample)`` from the
Stage B .fif annotations, and validate their durations against the GC sliding-
window minimum.

MATLAB equivalent
-----------------
``GC_freq_main_TRGC_NovExp.m`` lines 73–81::

    st = find(da.s);           % all trigger sample indices
    size_st = size(st, 1);
    for stt = 1:2:(size_st-1) % step by 2 → pair start/end
        duration_task(d) = (st(stt+1) - st(stt)) / 25;
        Y = Data_Reg(:, st(stt):st(stt+1));

Window-size note
----------------
The sliding window is ``window_size_s × sfreq`` samples.  For most subjects
this is 54 × 25 = 1350 samples; for R9 (50 Hz .fif) it becomes 54 × 50 = 2700
samples so the window always represents 54 seconds of actual neural data.

Task coverage
-------------
3.1.1 — get_trial_segments(raw)
3.1.2 — A1 validation: trial count, durations, MATLAB comparison
3.1.3 — trial duration logging → sub-{id}_trial_durations.csv

Inputs
------
``outputs/stage_b_processed/processed_data/expert-novice/
  sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif``

Outputs
-------
``outputs/stage_d_granger/linear_gc/expert-novice/roi_signals/``
  sub-{id}_trial_durations.csv   — per-trial duration + window count + flags
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import mne
import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config
from stage_d_granger.utils_gc import (
    get_event_onsets,
    get_gc_output_dir,
    load_preprocessed_fif,
)

logger = logging.getLogger(__name__)

_TRIAL_START_ID = 1
_TRIAL_END_ID   = 2


# ===========================================================================
# 3.1.1 — get_trial_segments
# ===========================================================================

def get_trial_segments(
    raw: mne.io.BaseRaw,
    config: Optional[dict] = None,
) -> list[tuple[int, int]]:
    """
    Return per-trial ``(onset_sample, offset_sample)`` index pairs.

    Mirrors MATLAB ``st = find(da.s)`` + ``for stt = 1:2:(size_st-1)`` loop.
    All trigger sample indices (both start=1 and end=2) are extracted, sorted
    by time, then stepped through in pairs: ``(st[0], st[1])``,
    ``(st[2], st[3])``, etc.

    Each pair is validated to be a ``(start_id, end_id)`` sequence.  Improperly
    ordered annotations (two consecutive starts or two consecutive ends) raise a
    ``RuntimeError`` rather than silently producing garbage trial boundaries.

    Window feasibility check
    ------------------------
    Trials shorter than ``window_size_s`` seconds (config
    ``stage_d_granger.linear_gc.window.size_s``, default 54 s) produce zero
    windows in the GC loop.  These are returned but flagged with a logged
    warning so the downstream windowing code can skip them cleanly.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed .fif with Annotations for trial start/end.
    config : dict, optional
        Parsed config.yaml. Loaded if None.

    Returns
    -------
    segments : list of (int, int)
        Each element is ``(onset_sample, offset_sample)`` in the global
        continuous time series (same frame as ``Data_Reg`` columns).
        Inclusive on both ends, matching MATLAB ``Y = Data_Reg(:,st(stt):st(stt+1))``.

    Raises
    ------
    RuntimeError
        If no events are found, an odd number of triggers exist, or any pair
        is not a (start, end) sequence.
    """
    if config is None:
        config = load_config()

    events = get_event_onsets(raw)          # shape (n_events, 3)
    sfreq  = float(raw.info["sfreq"])

    # Sort by sample index (should already be sorted, but be safe)
    events = events[np.argsort(events[:, 0])]

    n_events = len(events)
    if n_events == 0:
        raise RuntimeError("get_trial_segments: no events found in .fif")
    if n_events % 2 != 0:
        logger.warning(
            "get_trial_segments: odd number of triggers (%d) — last trigger "
            "has no matching end; it will be ignored.", n_events
        )

    window_size_s = (
        config.get("stage_d_granger", {})
        .get("linear_gc", {})
        .get("window", {})
        .get("size_s", 54)
    )
    min_samples = int(window_size_s * sfreq)

    segments: list[tuple[int, int]] = []
    n_pairs = n_events // 2

    for k in range(n_pairs):
        onset_sample,  _, id_on  = events[2 * k]
        offset_sample, _, id_off = events[2 * k + 1]

        if id_on != _TRIAL_START_ID or id_off != _TRIAL_END_ID:
            raise RuntimeError(
                f"get_trial_segments: trial {k+1} — expected event pair "
                f"(id={_TRIAL_START_ID}, id={_TRIAL_END_ID}), got "
                f"(id={id_on}, id={id_off}).  Check Annotation ordering."
            )

        dur_samples = int(offset_sample - onset_sample)
        dur_s = dur_samples / sfreq

        if dur_samples < min_samples:
            logger.warning(
                "get_trial_segments: trial %d is only %.1f s (< %.0f s "
                "window) — it will yield 0 GC windows and will be skipped "
                "in the windowing loop.",
                k + 1, dur_s, window_size_s,
            )

        segments.append((int(onset_sample), int(offset_sample)))

    logger.info(
        "get_trial_segments: %d trials found  (sfreq=%.1f Hz, "
        "window_min=%.0f s = %d samples)",
        len(segments), sfreq, window_size_s, min_samples,
    )
    return segments


# ===========================================================================
# 3.1.3 — trial duration logging
# ===========================================================================

def save_trial_durations(
    subject_id: str,
    segments: list[tuple[int, int]],
    sfreq: float,
    config: Optional[dict] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Compute per-trial duration and window count; save to CSV.

    Mirrors MATLAB ``duration_task(d) = (st(stt+1)-st(stt))/25``.

    The window count is the number of non-overlapping 54-second windows that
    fit inside the trial (same arithmetic as the MATLAB loop
    ``for w = window_size+1:sT:LY``).

    Parameters
    ----------
    subject_id : str
    segments : list of (onset_sample, offset_sample)
    sfreq : float
    config : dict, optional
    out_dir : Path, optional
        Defaults to ``stage_d_granger.linear_gc.output_subdirs.roi_signals``.

    Returns
    -------
    csv_path : Path
    """
    if config is None:
        config = load_config()
    if out_dir is None:
        out_dir = get_gc_output_dir("roi_signals", config=config)

    window_size_s = (
        config.get("stage_d_granger", {})
        .get("linear_gc", {})
        .get("window", {})
        .get("size_s", 54)
    )
    window_samples = int(window_size_s * sfreq)

    rows = []
    for k, (onset, offset) in enumerate(segments, 1):
        dur_samples = offset - onset
        dur_s       = dur_samples / sfreq
        # MATLAB window loop: w = window_size+1 : sT : LY
        # LY = dur_samples + 1 (inclusive slice), sT = window_samples
        # n_windows = floor((LY - window_samples) / sT) + 1  ← when LY > window_samples
        LY = dur_samples + 1   # inclusive sample count (MATLAB convention)
        n_windows = max(0, (LY - window_samples - 1) // window_samples + 1)
        short_flag = "SHORT" if dur_samples < window_samples else ""
        rows.append({
            "trial":          k,
            "onset_sample":   onset,
            "offset_sample":  offset,
            "duration_s":     round(dur_s, 3),
            "duration_samples": dur_samples,
            "n_windows":      n_windows,
            "short_flag":     short_flag,
        })
        if short_flag:
            logger.warning(
                "save_trial_durations [%s] trial %d: %.1f s < %.0f s window "
                "→ 0 windows (SHORT flag)",
                subject_id, k, dur_s, window_size_s,
            )

    csv_path = out_dir / f"sub-{subject_id}_trial_durations.csv"
    fieldnames = ["trial","onset_sample","offset_sample","duration_s",
                  "duration_samples","n_windows","short_flag"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "save_trial_durations [%s]: %d trials, %d total windows → %s",
        subject_id, len(rows), sum(r["n_windows"] for r in rows), csv_path.name,
    )
    return csv_path


# ===========================================================================
# Batch runner
# ===========================================================================

def run_trial_segmentation(
    study: str = "expert_novice",
    subjects: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> list[dict]:
    """
    Run trial segmentation for all GC-eligible subjects and save CSV logs.

    Parameters
    ----------
    study, subjects, config : see ``run_roi_extraction``

    Returns
    -------
    results : list[dict]
        One dict per subject with keys:
        ``subject_id, status, n_trials, total_windows, sfreq, csv_path``.
    """
    if config is None:
        config = load_config()
    if subjects is None:
        gc_subj = config["stage_d_granger"]["linear_gc"]["subjects"]
        subjects = gc_subj["Expert"] + gc_subj["Novice"]

    results = []
    for idx, sid in enumerate(subjects, 1):
        print(f"  [{idx:2d}/{len(subjects)}] {sid:4s} ... ", end="", flush=True)
        try:
            raw = load_preprocessed_fif(sid, study=study, config=config, preload=False)
            segs = get_trial_segments(raw, config=config)
            csv_path = save_trial_durations(sid, segs, float(raw.info["sfreq"]), config=config)

            total_win = sum(
                max(0, ((s[1]-s[0]+1) - int(54*raw.info["sfreq"]) - 1)
                    // int(54*raw.info["sfreq"]) + 1)
                for s in segs
            )
            durations = [(s[1]-s[0])/raw.info["sfreq"] for s in segs]
            print(
                f"{len(segs)} trials  "
                f"dur=[{min(durations):.0f}–{max(durations):.0f}]s  "
                f"total_windows={total_win}  "
                f"sfreq={raw.info['sfreq']:.0f}Hz"
            )
            results.append({
                "subject_id":    sid,
                "status":        "ok",
                "n_trials":      len(segs),
                "total_windows": total_win,
                "sfreq":         float(raw.info["sfreq"]),
                "csv_path":      str(csv_path),
            })
        except Exception as exc:
            logger.exception("run_trial_segmentation [%s]: FAILED — %s", sid, exc)
            print(f"FAILED — {exc}")
            results.append({"subject_id": sid, "status": "failed", "error": str(exc)})

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n  Summary: {n_ok}/{len(subjects)} subjects OK")
    return results


# ===========================================================================
# Main — 3.1.2 validation on A1 + optional batch
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Stage D: Trial segmentation")
    parser.add_argument("--subjects", nargs="+", default=["A1"])
    parser.add_argument("--study", default="expert_novice")
    parser.add_argument("--all", action="store_true",
                        help="Run all 13 GC-eligible subjects")
    args = parser.parse_args()

    cfg = load_config()

    if args.all:
        print("\n=== Trial segmentation — all subjects ===\n")
        run_trial_segmentation(study=args.study, config=cfg)
    else:
        print(f"\n=== Trial segmentation: {args.subjects} ===\n")
        for sid in args.subjects:
            raw = load_preprocessed_fif(sid, study=args.study, config=cfg, preload=False)
            sfreq = float(raw.info["sfreq"])
            segs  = get_trial_segments(raw, config=cfg)
            csv_path = save_trial_durations(sid, segs, sfreq, config=cfg)

            win_s  = cfg["stage_d_granger"]["linear_gc"]["window"]["size_s"]
            win_n  = int(win_s * sfreq)

            print(f"\nsub-{sid}  (sfreq={sfreq:.0f} Hz)")
            print(f"  Window size : {win_s} s × {sfreq:.0f} Hz = {win_n} samples")
            print(f"  {'Trial':6s}  {'onset_s':>9s}  {'offset_s':>9s}  "
                  f"{'dur_s':>7s}  {'dur_MATLAB':>10s}  windows")
            print(f"  {'-'*60}")
            for k, (onset, offset) in enumerate(segs, 1):
                dur_s = (offset - onset) / sfreq
                dur_matlab = (offset - onset) / 25.0   # MATLAB hardcoded /25
                LY = (offset - onset) + 1
                n_win = max(0, (LY - win_n - 1) // win_n + 1)
                flag = "  ← SHORT" if (offset-onset) < win_n else ""
                print(f"  Tr_{k}   {onset/sfreq:9.2f}  {offset/sfreq:9.2f}  "
                      f"{dur_s:7.1f}  {dur_matlab:10.1f}  {n_win}{flag}")
            print(f"\n  CSV saved: {csv_path.name}")
