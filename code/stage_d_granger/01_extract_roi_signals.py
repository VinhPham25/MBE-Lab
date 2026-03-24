"""
Stage D — Step D1: ROI Signal Extraction

Extract region-of-interest (ROI) averaged HbO time series from the Stage B
preprocessed .fif file, producing the ``Data_Reg`` (5×T) array used as input
to the GC computation.

MATLAB equivalent
-----------------
``GC_freq_main_TRGC_NovExp.m`` lines 89–108::

    LPFC = D([1 2],:);
    LPFC(any(isnan(LPFC),2),:) = [];
    LPFC = mean(LPFC,1);
    ...
    Data_Reg = [LPFC; RPFC; LPMC; RPMC; SMA];
    Data_Reg(any(isnan(Data_Reg),2),:) = [];

Task coverage
-------------
2.2.1 — extract_roi_signals(raw) → (Data_Reg, valid_roi_names)
2.2.2 — NaN-row removal for empty/all-bad ROIs
2.2.3 — A1 validation: shape check + ROI time-series plot
2.2.4 — run_roi_extraction() batch over all 13 GC-eligible subjects

Signal units
------------
MNE stores hemoglobin in **mol/L** (not μM). GC is computed on raw mol/L
values — the Geweke log-ratio formulation is unit-agnostic. No conversion
is applied here.

Inputs
------
``outputs/stage_b_processed/processed_data/expert-novice/
  sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif``

Outputs
-------
``outputs/stage_d_granger/linear_gc/expert-novice/roi_signals/``
  sub-{id}_roi_signals.npz        — keys: data (n_rois×T), roi_names, sfreq
  sub-{id}_roi_signals.png        — 5-panel ROI time-series plot
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config
from stage_d_granger.utils_gc import (
    ROI_NAMES,
    get_gc_output_dir,
    get_roi_channel_map,
    get_subject_group,
    load_preprocessed_fif,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 2.2.1 + 2.2.2 — extract_roi_signals
# ===========================================================================

def extract_roi_signals(
    raw: mne.io.BaseRaw,
) -> tuple[np.ndarray, list[str]]:
    """
    Average HbO channels within each ROI to produce ``Data_Reg`` (n_rois × T).

    Mirrors the MATLAB channel-averaging block::

        LPFC = mean(D([1 2],:), 1);   % average 2 channels
        Data_Reg = [LPFC; RPFC; LPMC; RPMC; SMA];
        Data_Reg(any(isnan(Data_Reg),2),:) = [];

    Processing steps
    ----------------
    1. Call ``get_roi_channel_map(raw)`` → per-ROI HbO channel name lists.
    2. For each ROI, extract the channel data matrix (n_ch × T) and compute
       the channel mean (row mean → shape (T,)).
    3. If a ROI has zero valid channels (all marked bad), its row is filled
       with NaN — it will be removed in step 4.
    4. Remove any all-NaN rows (ROIs where every sample is NaN) and record
       which ROIs were dropped in ``valid_roi_names``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Stage B preprocessed .fif loaded with ``preload=True``.

    Returns
    -------
    Data_Reg : np.ndarray, shape (n_valid_rois, T)
        ROI-averaged HbO signals in mol/L (MNE native units).
        Rows correspond to entries in ``valid_roi_names``.
    valid_roi_names : list[str]
        ROI names retained after NaN-row removal (subset of
        ``['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']``).
        Usually all 5; may be fewer if all channels in a ROI are bad.

    Raises
    ------
    RuntimeError
        If all 5 ROIs are dropped (no usable signal at all).
    """
    if not raw.preload:
        raw.load_data(verbose=False)

    roi_map = get_roi_channel_map(raw)
    n_times = len(raw.times)

    rows: list[np.ndarray] = []
    row_names: list[str] = []

    for roi in ROI_NAMES:
        channels = roi_map.get(roi, [])
        if not channels:
            # ROI has no valid channels — insert an all-NaN placeholder
            logger.warning(
                "extract_roi_signals [%s]: no valid HbO channels, "
                "inserting NaN row (will be dropped)", roi
            )
            rows.append(np.full(n_times, np.nan))
            row_names.append(roi)
            continue

        # Pick only this ROI's channels, extract data (n_ch × T)
        ch_data = raw.get_data(picks=channels)   # shape (n_ch, T)
        roi_mean = np.nanmean(ch_data, axis=0)   # shape (T,)
        rows.append(roi_mean)
        row_names.append(roi)

    Data_Reg = np.vstack(rows)   # (5, T) initially

    # 2.2.2 — NaN-row removal (mirrors MATLAB Data_Reg(any(isnan(...),2),:)=[])
    nan_mask = np.all(np.isnan(Data_Reg), axis=1)   # True if entire row is NaN
    n_dropped = int(nan_mask.sum())

    if n_dropped:
        dropped_names = [row_names[i] for i in range(len(row_names)) if nan_mask[i]]
        logger.warning(
            "extract_roi_signals: removing %d all-NaN ROI(s): %s",
            n_dropped, dropped_names,
        )
        Data_Reg = Data_Reg[~nan_mask]
        valid_roi_names = [row_names[i] for i in range(len(row_names)) if not nan_mask[i]]
    else:
        valid_roi_names = list(row_names)

    if len(valid_roi_names) == 0:
        raise RuntimeError(
            "extract_roi_signals: all 5 ROIs dropped — no usable signal. "
            "Check Stage B bad-channel list for this subject."
        )

    logger.info(
        "extract_roi_signals: Data_Reg shape %s, valid ROIs: %s",
        Data_Reg.shape, valid_roi_names,
    )
    return Data_Reg, valid_roi_names


# ===========================================================================
# 2.2.3 — ROI time-series plot
# ===========================================================================

def plot_roi_signals(
    Data_Reg: np.ndarray,
    valid_roi_names: list[str],
    sfreq: float,
    subject_id: str,
    save_path: Path,
) -> None:
    """
    Plot each ROI's averaged HbO trace in a stacked multi-panel figure.

    Parameters
    ----------
    Data_Reg : np.ndarray, shape (n_rois, T)
    valid_roi_names : list[str]
    sfreq : float
    subject_id : str
    save_path : Path
    """
    n_rois, n_times = Data_Reg.shape
    times = np.arange(n_times) / sfreq          # seconds

    # Convert mol/L → μM for display only (×1e6)
    display_data = Data_Reg * 1e6

    fig, axes = plt.subplots(
        n_rois, 1,
        figsize=(14, 2.2 * n_rois),
        sharex=True,
        gridspec_kw={"hspace": 0.45},
    )
    if n_rois == 1:
        axes = [axes]

    roi_colors = {
        "LPFC": "#2166AC",
        "CPFC": "#4DAC26",
        "RPFC": "#D7191C",
        "LM1":  "#762A83",
        "SMA":  "#E66101",
    }

    for ax, roi, trace in zip(axes, valid_roi_names, display_data):
        color = roi_colors.get(roi, "#333333")
        ax.plot(times, trace, lw=0.8, color=color, alpha=0.9)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(f"{roi}\n(μM)", fontsize=8)
        ax.tick_params(labelsize=7)
        # Shade ±2 SD band to help visually assess scale
        mu, sd = float(np.nanmean(trace)), float(np.nanstd(trace))
        ax.axhline(mu + 2 * sd, color=color, lw=0.5, ls=":", alpha=0.5)
        ax.axhline(mu - 2 * sd, color=color, lw=0.5, ls=":", alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(
        f"ROI-averaged HbO signals — sub-{subject_id}\n"
        f"Stage D input (Data_Reg, {n_rois} ROIs × {n_times} samples @ {sfreq:.0f} Hz)",
        fontsize=10, fontweight="bold",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_roi_signals: saved %s", save_path.name)


# ===========================================================================
# 2.2.4 — per-subject runner + batch
# ===========================================================================

def run_subject_roi_extraction(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = True,
) -> dict:
    """
    Full ROI extraction pipeline for one subject.

    Loads Stage B .fif → extracts Data_Reg → saves .npz + plot.

    Parameters
    ----------
    subject_id : str
    study : str
    config : dict, optional
    overwrite : bool
        If False, skip subjects whose .npz already exists.

    Returns
    -------
    result : dict
        ``{'subject_id', 'status', 'shape', 'valid_roi_names', 'sfreq',
           'npz_path', 'plot_path'}``
    """
    if config is None:
        config = load_config()

    out_dir = get_gc_output_dir("roi_signals", config=config)
    npz_path = out_dir / f"sub-{subject_id}_roi_signals.npz"
    plot_path = out_dir / f"sub-{subject_id}_roi_signals.png"

    if npz_path.exists() and not overwrite:
        logger.info("run_subject_roi_extraction [%s]: cached .npz found, skipping", subject_id)
        cached = np.load(npz_path, allow_pickle=True)
        return {
            "subject_id": subject_id,
            "status": "cached",
            "shape": tuple(cached["data"].shape),
            "valid_roi_names": list(cached["roi_names"]),
            "sfreq": float(cached["sfreq"]),
            "npz_path": str(npz_path),
            "plot_path": str(plot_path),
        }

    raw = load_preprocessed_fif(subject_id, study=study, config=config, preload=True)
    sfreq = float(raw.info["sfreq"])

    Data_Reg, valid_roi_names = extract_roi_signals(raw)

    # Save .npz
    np.savez(
        npz_path,
        data=Data_Reg,
        roi_names=np.array(valid_roi_names),
        sfreq=np.float64(sfreq),
    )
    logger.info(
        "run_subject_roi_extraction [%s]: saved %s  shape=%s",
        subject_id, npz_path.name, Data_Reg.shape,
    )

    # Save plot
    plot_roi_signals(Data_Reg, valid_roi_names, sfreq, subject_id, plot_path)

    return {
        "subject_id": subject_id,
        "status": "ok",
        "shape": tuple(Data_Reg.shape),
        "valid_roi_names": valid_roi_names,
        "sfreq": sfreq,
        "npz_path": str(npz_path),
        "plot_path": str(plot_path),
    }


def run_roi_extraction(
    study: str = "expert_novice",
    subjects: Optional[list[str]] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> list[dict]:
    """
    Batch ROI extraction for all GC-eligible subjects.

    Processes all 13 Expert/Novice subjects (R10 excluded — Stage B quality
    flag). Subjects that raise an exception are logged and marked 'failed'
    in the results; processing continues for the remaining subjects.

    Parameters
    ----------
    study : str
        Config dataset key (default 'expert_novice').
    subjects : list[str], optional
        Override the subject list. If None, uses Expert + Novice lists from
        ``stage_d_granger.linear_gc.subjects`` in config.yaml.
    config : dict, optional
    overwrite : bool

    Returns
    -------
    results : list[dict]
        One result dict per subject (from ``run_subject_roi_extraction``).
    """
    if config is None:
        config = load_config()

    if subjects is None:
        gc_subj = config["stage_d_granger"]["linear_gc"]["subjects"]
        subjects = gc_subj["Expert"] + gc_subj["Novice"]

    n_total = len(subjects)
    logger.info(
        "run_roi_extraction: processing %d subjects for study '%s'",
        n_total, study,
    )

    results = []
    n_ok = n_cached = n_failed = 0

    for idx, sid in enumerate(subjects, 1):
        print(f"  [{idx:2d}/{n_total}] {sid:4s} ... ", end="", flush=True)
        try:
            res = run_subject_roi_extraction(
                sid, study=study, config=config, overwrite=overwrite
            )
            status = res["status"]
            shape = res["shape"]
            rois = res["valid_roi_names"]
            print(f"{status}  shape={shape}  ROIs={rois}")
            if status == "cached":
                n_cached += 1
            else:
                n_ok += 1
        except Exception as exc:
            logger.exception("run_roi_extraction [%s]: FAILED — %s", sid, exc)
            print(f"FAILED — {exc}")
            res = {"subject_id": sid, "status": "failed", "error": str(exc)}
            n_failed += 1
        results.append(res)

    print(
        f"\n  Summary: {n_ok} new, {n_cached} cached, {n_failed} failed "
        f"(total={n_total})"
    )
    return results


# ===========================================================================
# Main — 2.2.3 validation on A1 + optional batch
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage D: ROI signal extraction"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=["A1"],
        help="Subject IDs to process (default: A1 only for validation)",
    )
    parser.add_argument(
        "--study", default="expert_novice",
        help="Config dataset key (default: expert_novice)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all 13 GC-eligible subjects (ignores --subjects)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process even if .npz already exists",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.all:
        print("\n=== Batch ROI extraction — all 13 subjects ===\n")
        results = run_roi_extraction(study=args.study, config=cfg, overwrite=args.overwrite)
    else:
        print(f"\n=== ROI extraction: {args.subjects} ===\n")
        results = []
        for sid in args.subjects:
            res = run_subject_roi_extraction(sid, study=args.study, config=cfg, overwrite=args.overwrite)
            results.append(res)

    # Validation report for any processed subjects
    print("\n--- Validation ---")
    for res in results:
        if res.get("status") in ("ok", "cached"):
            sid = res["subject_id"]
            shape = res["shape"]
            rois = res["valid_roi_names"]
            npz = np.load(res["npz_path"], allow_pickle=True)
            data = npz["data"]
            n_rois, n_times = data.shape
            expected_times = int(res["sfreq"] * 1060) if sid == "A1" else None

            print(f"\n  sub-{sid}:")
            print(f"    shape         : {shape}  (expected: ({len(rois)}, ~T))")
            if expected_times:
                print(f"    T={n_times}  (expected ~{expected_times} for A1 @ {res['sfreq']} Hz)")
            print(f"    valid ROIs    : {rois}")
            print(f"    NaN rows      : {int(np.all(np.isnan(data), axis=1).sum())}")
            print(f"    data range    : [{np.nanmin(data)*1e6:.3f}, {np.nanmax(data)*1e6:.3f}] μM")
            print(f"    plot saved    : {Path(res['plot_path']).name}")
