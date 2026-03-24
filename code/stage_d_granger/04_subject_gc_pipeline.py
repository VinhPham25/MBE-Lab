"""
Stage D — Step D4: Per-Subject GC Pipeline and Group Array Construction

Orchestrates the full Granger Causality pipeline for each subject and builds
the group-level connectivity arrays (B_Exp, B_Nov) that feed the statistical
tests in Section 6.

Section 5 task coverage (LINEAR_GC_PROGRESS.md)
------------------------------------------------
5.1 — Per-subject GC pipeline         (process_subject_gc)
5.1.1 — Orchestrator function
5.1.2 — Diagonal removal (5×5 → 20 connections)
5.1.3 — Per-subject .npz storage

5.2 — Group array construction        (build_group_arrays)
5.2.1 — Stack per-subject files into B_Exp / B_Nov
5.2.2 — Save group arrays + gc_summary.json

5.3 — Single-subject + batch validation (run as __main__)
5.3.1 — process_subject_gc('A1') end-to-end validation
5.3.2 — Batch processing, all 13 GC-eligible subjects, gc_batch_log.json

MATLAB equivalent (outer subject loop)
---------------------------------------
``GC_freq_main_TRGC_NovExp.m``::

    for m = 1:length(Exp_sub)
        ...
        for d = 1:size_st2
            Y = Data_Reg(:, st(stt):st(stt+1));
            for Wn = 0 : sT : LY
                [GW,...] = cca_pwcausal(Y_windowed, ...);
                GC_fqmean = mean(GW(:,:,idx1:idx2), 3);
                GC_fqmean(:,[1 7 13 19 25]) = [];   % remove diagonal → 20 elements
                B_Exp.(trial).Exp(h,:,Wn+1) = GC_fqmean;

Storage design notes
---------------------
Different subjects have different trial counts (R9=2, R4=4, most=5) and
different window counts per trial (depends on trial duration).

Per-subject .npz layout
    ``Tr_1_connectivity`` : array(20, n_wins_tr1) — 20 directed connections × windows
    ``Tr_2_connectivity`` : array(20, n_wins_tr2)
    ...
    ``trial_ids``          : string array of trial IDs present
    ``window_counts``      : int array, number of windows per trial
    ``roi_names_used``     : string array, ROI names that survived bad-channel filtering
    ``connection_labels``  : string array of 20 connection labels e.g. 'LPFC→CPFC'
    ``n_trials``           : scalar int
    ``subject_id``         : scalar string
    ``group``              : 'Expert' or 'Novice'
    ``params_json``        : JSON-serialised processing parameters

Per-trial group .npz layout (B_Exp.npz, B_Nov.npz)
    ``Tr_1``               : array(n_subs, 20, max_wins_tr1), NaN-padded
    ``Tr_2``               : array(n_subs, 20, max_wins_tr2), NaN-padded
    ...
    ``subject_ids``        : string array of subject IDs in row order

NaN-padding
    A subject with fewer windows than max_wins for that trial has NaN in the
    trailing window slots.  A subject with no data for a given trial (e.g.
    R9 for Tr_3–5, R4 for Tr_5) has all NaN for that trial.
"""

import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config
from stage_d_granger.utils_gc import (
    ROI_NAMES,
    get_connection_labels,
    get_gc_output_dir,
    get_subject_group,
    load_preprocessed_fif,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy module imports (avoids matplotlib backend / side-effect issues)
# ---------------------------------------------------------------------------
_ROI_MODULE = None
_SEG_MODULE = None
_GC_MODULE = None


def _get_roi_module():
    global _ROI_MODULE
    if _ROI_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "extract_roi_signals",
            _CODE_ROOT / "stage_d_granger" / "01_extract_roi_signals.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ROI_MODULE = mod
    return _ROI_MODULE


def _get_seg_module():
    global _SEG_MODULE
    if _SEG_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "trial_segmentation",
            _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEG_MODULE = mod
    return _SEG_MODULE


def _get_gc_module():
    global _GC_MODULE
    if _GC_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "linear_gc",
            _CODE_ROOT / "stage_d_granger" / "03_linear_gc.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GC_MODULE = mod
    return _GC_MODULE


# ===========================================================================
# Section 5.1 — Per-Subject GC Pipeline
# ===========================================================================

def process_subject_gc(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = True,
) -> dict:
    """
    Run the full GC pipeline for one subject and save connectivity to .npz.

    Orchestrates: load .fif → extract ROI signals → segment trials →
    windowed GC → band average → diagonal removal → save .npz.

    Mirrors MATLAB outer loops over trials and windows, building the per-
    subject ``GC_fqmean`` (20 directed connections × windows) array.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'A1', 'R7'.
    study : str
        Config dataset key.  Default 'expert_novice' (underscore, matching
        ``load_preprocessed_fif`` convention).
    config : dict, optional
        Parsed config.yaml.  Loaded from project root if None.
    overwrite : bool
        If False and the output .npz already exists, skip and return the
        previously computed result.  Default True (always recompute).

    Returns
    -------
    result : dict
        Nested dict::

            {
              'Tr_1': {'Win_1': array(20,), 'Win_2': array(20,), ...},
              'Tr_2': {'Win_1': array(20,), ...},
              ...
            }

        Each leaf value is a 1D array of 20 directed GC connectivity values
        in the canonical connection order (LPFC→CPFC, LPFC→RPFC, ...,
        SMA→LM1) — matching MATLAB's post-diagonal-removal output.

    Notes
    -----
    **Diagonal removal (5.1.2):**
    ``average_gc_over_band`` returns a 5×5 matrix.  Flattening row-major
    (``GC_mean.flatten()``) gives 25 elements where diagonal positions are
    [0, 6, 12, 18, 24].  Removing these yields 20 off-diagonal values in
    the exact order documented in ``get_connection_labels()``.

    This mirrors MATLAB::

        B_Exp.(trial).Exp(h,:,Wn+1) = GC_fqmean;
        % where GC_fqmean(:,[1 7 13 19 25]) = [] was applied

    **NaN ROI handling:**
    Subjects R11 (missing LM1+SMA) and R2 (missing SMA) have all-zero GC
    values for connections involving those ROIs after ``compute_pairwise_gc``.
    These zeros propagate through band-averaging and diagonal removal cleanly
    — no special handling is needed here.

    **R9 sfreq:**
    R9's .fif is at 50 Hz.  ``load_preprocessed_fif`` resamples to 25 Hz
    in-memory before returning, so the window size (1350 samples = 54 s) and
    all downstream operations are uniform across subjects.
    """
    if config is None:
        config = load_config()

    gc_cfg = config["stage_d_granger"]["linear_gc"]
    freq_cfg = gc_cfg["frequency"]
    band_cfg = gc_cfg["band_avg"]
    window_cfg = gc_cfg["window"]
    var_cfg = gc_cfg["var"]

    # Check for existing output (skip if overwrite=False)
    out_dir = get_gc_output_dir("gc_connectivity", config=config)
    npz_path = out_dir / f"sub-{subject_id}_gc_connectivity.npz"
    if not overwrite and npz_path.exists():
        logger.info(
            "process_subject_gc [%s]: output already exists, loading from disk.",
            subject_id,
        )
        return _load_subject_gc(subject_id, config=config)

    # ------------------------------------------------------------------
    # Step 1 — Load .fif
    # ------------------------------------------------------------------
    logger.info("process_subject_gc [%s]: loading .fif ...", subject_id)
    raw = load_preprocessed_fif(subject_id, study=study, config=config)
    sfreq = float(raw.info["sfreq"])   # always 25 Hz after harmonisation

    # ------------------------------------------------------------------
    # Step 2 — Extract ROI signals
    # ------------------------------------------------------------------
    roi_mod = _get_roi_module()
    logger.info("process_subject_gc [%s]: extracting ROI signals ...", subject_id)
    Data_Reg, valid_roi_names = roi_mod.extract_roi_signals(raw)
    n_rois_valid = Data_Reg.shape[0]
    logger.info(
        "process_subject_gc [%s]: Data_Reg shape=%s  valid ROIs=%s",
        subject_id, Data_Reg.shape, valid_roi_names,
    )

    # ------------------------------------------------------------------
    # Step 3 — Trial segmentation
    # ------------------------------------------------------------------
    seg_mod = _get_seg_module()
    logger.info("process_subject_gc [%s]: getting trial segments ...", subject_id)
    segments = seg_mod.get_trial_segments(raw, config=config)
    logger.info(
        "process_subject_gc [%s]: %d trials found.", subject_id, len(segments)
    )

    # ------------------------------------------------------------------
    # Build frequency vector from config
    # ------------------------------------------------------------------
    freq = np.arange(
        freq_cfg["f_start"],
        freq_cfg["f_stop"] + freq_cfg["f_step"] / 2,
        freq_cfg["f_step"],
    )
    window_size = int(window_cfg["size_samples"])
    stride = int(window_cfg["stride_samples"])
    var_order = int(var_cfg["order"])
    f_low = float(band_cfg["f_low"])
    f_high = float(band_cfg["f_high"])

    # ------------------------------------------------------------------
    # Step 4 — Windowed GC + band averaging + diagonal removal, per trial
    # ------------------------------------------------------------------
    gc_mod = _get_gc_module()
    result = {}   # {trial_id: {win_id: array(20,)}}

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        # Extract trial signal — MATLAB: Y = Data_Reg(:, st(stt):st(stt+1))
        # inclusive at both ends, matching the MATLAB convention
        Y_trial = Data_Reg[:, onset : offset + 1]

        logger.info(
            "process_subject_gc [%s]: %s  samples=[%d, %d]  shape=%s",
            subject_id, trial_id, onset, offset, Y_trial.shape,
        )

        # Sliding-window GC for this trial
        windows_gc = gc_mod.compute_windowed_gc(
            Y_trial,
            window_size=window_size,
            stride=stride,
            order=var_order,
            freq=freq,
            sfreq=sfreq,
            trend="c",
        )

        trial_result = {}
        for win_id, GW in windows_gc.items():
            # Band-average: (n_rois, n_rois, n_freqs) → (n_rois, n_rois)
            GC_mean = gc_mod.average_gc_over_band(GW, freq, f_low=f_low, f_high=f_high)

            # -------------------------------------------------------
            # 5.1.2 — Expand to canonical 5×5 layout + diagonal removal
            #
            # GC_mean has shape (n_rois_valid × n_rois_valid).  For
            # subjects with bad ROIs (R11: 3 valid, R2: 4 valid) this
            # is smaller than 5×5.  To ensure a uniform 20-element
            # connectivity vector across all subjects, we expand to the
            # canonical 5×5 matrix by placing valid-ROI values at their
            # correct positions (defined by ROI_NAMES order).  Missing
            # ROI pairs are left at 0 (consistent with compute_pairwise_gc
            # returning 0 for NaN-channel pairs).
            #
            # After expansion, diagonal indices are always [0, 6, 12, 18, 24]
            # and deletion yields exactly 20 elements.
            #
            # MATLAB equivalent:
            #   B_Exp.(trial)(:,[1 7 13 19 25],:) = []
            # -------------------------------------------------------
            n_all = len(ROI_NAMES)   # 5 (canonical)
            GC_full = np.zeros((n_all, n_all), dtype=float)
            for i_v, roi_i in enumerate(valid_roi_names):
                for j_v, roi_j in enumerate(valid_roi_names):
                    if roi_i != roi_j:
                        i_c = ROI_NAMES.index(roi_i)
                        j_c = ROI_NAMES.index(roi_j)
                        GC_full[i_c, j_c] = GC_mean[i_v, j_v]

            diag_indices = [i * (n_all + 1) for i in range(n_all)]  # [0,6,12,18,24]
            connectivity_flat = GC_full.flatten()                    # 25 elements
            connectivity_20 = np.delete(connectivity_flat, diag_indices)  # 20 elements

            trial_result[win_id] = connectivity_20

        result[trial_id] = trial_result

        total_wins = sum(len(v) for v in result.values())
        logger.info(
            "process_subject_gc [%s]: %s → %d windows  (total so far: %d)",
            subject_id, trial_id, len(trial_result), total_wins,
        )

    # ------------------------------------------------------------------
    # Step 5 — Save per-subject .npz  (5.1.3)
    # ------------------------------------------------------------------
    _save_subject_gc_npz(
        subject_id=subject_id,
        result=result,
        valid_roi_names=valid_roi_names,
        params={
            "study": study,
            "var_order": var_order,
            "window_size_samples": window_size,
            "stride_samples": stride,
            "sfreq": sfreq,
            "freq_start": freq_cfg["f_start"],
            "freq_stop": freq_cfg["f_stop"],
            "freq_step": freq_cfg["f_step"],
            "f_band_low": f_low,
            "f_band_high": f_high,
            "n_connections": 20,
        },
        config=config,
    )

    # Summary log
    total_wins = sum(len(v) for v in result.values())
    logger.info(
        "process_subject_gc [%s]: complete.  Trials=%d, Total windows=%d",
        subject_id, len(result), total_wins,
    )
    return result


# ---------------------------------------------------------------------------
# Private helper — save per-subject .npz
# ---------------------------------------------------------------------------

def _save_subject_gc_npz(
    subject_id: str,
    result: dict,
    valid_roi_names: list,
    params: dict,
    config: Optional[dict] = None,
) -> Path:
    """
    Save ``process_subject_gc`` output to
    ``gc_connectivity/sub-{id}_gc_connectivity.npz``.

    Per-trial connectivity arrays are stored as separate keys (``Tr_1_connectivity``,
    ``Tr_2_connectivity``, ...) rather than a padded 3D array.  This preserves
    the ragged structure (subjects differ in trial count and windows per trial)
    while keeping the file self-describing.

    Each connectivity key has shape ``(n_connections, n_windows)`` where
    ``n_connections = 20`` for a 5-ROI subject or fewer for subjects with
    bad ROIs (R11=3-ROI→ 6 connections, R2=4-ROI→ 12 connections).

    Returns
    -------
    npz_path : Path
        Absolute path to the saved .npz file.
    """
    out_dir = get_gc_output_dir("gc_connectivity", config=config)

    trial_ids = sorted(result.keys(), key=lambda k: int(k.split("_")[1]))
    window_counts = []
    save_dict = {}

    for trial_id in trial_ids:
        trial_data = result[trial_id]
        n_wins = len(trial_data)
        window_counts.append(n_wins)

        if n_wins > 0:
            wins_sorted = sorted(trial_data.keys(), key=lambda k: int(k.split("_")[1]))
            # Stack into (n_connections, n_windows)
            conn_array = np.column_stack([trial_data[wk] for wk in wins_sorted])
            save_dict[f"{trial_id}_connectivity"] = conn_array
        else:
            # Empty trial — store an empty marker so the key exists
            n_conn = len(get_connection_labels())
            save_dict[f"{trial_id}_connectivity"] = np.empty((n_conn, 0), dtype=float)

    # Metadata fields — stored as numpy arrays for .npz compatibility
    save_dict["subject_id"] = np.array(subject_id)
    save_dict["n_trials"] = np.array(len(trial_ids))
    save_dict["trial_ids"] = np.array(trial_ids)
    save_dict["window_counts"] = np.array(window_counts, dtype=int)
    save_dict["roi_names_used"] = np.array(valid_roi_names)
    save_dict["connection_labels"] = np.array(get_connection_labels())

    # Store group membership
    try:
        group = get_subject_group(subject_id, config=config)
    except ValueError:
        group = "Unknown"
    save_dict["group"] = np.array(group)

    # Parameters as JSON string (npz doesn't support nested dicts natively)
    save_dict["params_json"] = np.array(json.dumps(params))

    npz_path = out_dir / f"sub-{subject_id}_gc_connectivity.npz"
    np.savez(npz_path, **save_dict)
    logger.info(
        "_save_subject_gc_npz [%s]: saved → %s  "
        "(trials=%d, total_windows=%d)",
        subject_id, npz_path.name, len(trial_ids), sum(window_counts),
    )
    return npz_path


# ---------------------------------------------------------------------------
# Private helper — load per-subject .npz back into result dict
# ---------------------------------------------------------------------------

def _load_subject_gc(
    subject_id: str,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Load ``sub-{id}_gc_connectivity.npz`` and return the nested result dict.

    Returns None if the file does not exist.
    """
    out_dir = get_gc_output_dir("gc_connectivity", config=config)
    npz_path = out_dir / f"sub-{subject_id}_gc_connectivity.npz"

    if not npz_path.exists():
        logger.warning("_load_subject_gc [%s]: file not found: %s", subject_id, npz_path)
        return None

    data = dict(np.load(npz_path, allow_pickle=True))
    trial_ids = list(data["trial_ids"])
    window_counts = list(data["window_counts"])

    result = {}
    for trial_id, n_wins in zip(trial_ids, window_counts):
        conn_array = data.get(f"{trial_id}_connectivity")  # (n_conn, n_wins) or empty
        trial_result = {}
        if conn_array is not None and conn_array.shape[1] > 0:
            for win_idx in range(conn_array.shape[1]):
                trial_result[f"Win_{win_idx + 1}"] = conn_array[:, win_idx]
        result[trial_id] = trial_result

    return result


# ===========================================================================
# Section 5.2 — Group Array Construction
# ===========================================================================

def build_group_arrays(
    study: str = "expert_novice",
    config: Optional[dict] = None,
) -> tuple:
    """
    Load all per-subject .npz files and stack into group-level arrays.

    Mirrors MATLAB's accumulation into ``B_Exp.(trial).Exp(h,:,Wn+1)`` and
    ``B_Nov.(trial).Nov(h,:,Wn+1)`` across subjects.

    Returns per-trial group arrays.  Different subjects may have different
    numbers of trials and windows; shorter subjects are NaN-padded.

    Parameters
    ----------
    study : str
        Config dataset key.  Default 'expert_novice'.
    config : dict, optional
        Parsed config.yaml.  Loaded if None.

    Returns
    -------
    B_Exp : dict
        ``{trial_id: array(n_exp_subjects, 20, max_windows_for_trial)}``.
        NaN-padded where a subject has fewer windows than the maximum for
        that trial, or has no data for that trial at all.
    B_Nov : dict
        Same structure for the Novice group.

    Saves
    -----
    ``group/B_Exp.npz``   — per-trial Expert group arrays
    ``group/B_Nov.npz``   — per-trial Novice group arrays
    ``group/gc_summary.json`` — processing metadata
    """
    if config is None:
        config = load_config()

    gc_cfg = config["stage_d_granger"]["linear_gc"]
    subjects_cfg = gc_cfg["subjects"]
    expert_ids: list = subjects_cfg["Expert"]
    novice_ids: list = subjects_cfg["Novice"]

    logger.info(
        "build_group_arrays: Experts n=%d %s, Novices n=%d %s",
        len(expert_ids), expert_ids, len(novice_ids), novice_ids,
    )

    # ------------------------------------------------------------------
    # 5.2.1 — Load per-subject .npz files
    # ------------------------------------------------------------------
    expert_npz_list = _load_all_subject_npz(expert_ids, config=config)
    novice_npz_list = _load_all_subject_npz(novice_ids, config=config)

    n_exp_loaded = sum(1 for d in expert_npz_list if d is not None)
    n_nov_loaded = sum(1 for d in novice_npz_list if d is not None)
    logger.info(
        "build_group_arrays: loaded %d/%d Expert, %d/%d Novice .npz files",
        n_exp_loaded, len(expert_ids), n_nov_loaded, len(novice_ids),
    )

    if n_exp_loaded == 0 and n_nov_loaded == 0:
        raise RuntimeError(
            "build_group_arrays: no per-subject .npz files found. "
            "Run process_subject_gc() for each subject first."
        )

    # ------------------------------------------------------------------
    # Determine max trials and max windows per trial across all subjects
    # ------------------------------------------------------------------
    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]

    max_trials = max(int(d["n_trials"]) for d in all_npz)

    # For each trial, find the maximum window count across all subjects
    max_wins_per_trial: dict[str, int] = {}
    for trial_idx in range(max_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        max_w = 0
        for d in all_npz:
            key = f"{trial_id}_connectivity"
            if key in d and d[key].shape[1] > 0:
                max_w = max(max_w, int(d[key].shape[1]))
        max_wins_per_trial[trial_id] = max_w

    logger.info(
        "build_group_arrays: max trials=%d, windows per trial=%s",
        max_trials, max_wins_per_trial,
    )

    # ------------------------------------------------------------------
    # 5.2.1 — Stack into group arrays with NaN-padding
    # ------------------------------------------------------------------
    n_conn = len(get_connection_labels())   # 20
    n_exp = len(expert_ids)
    n_nov = len(novice_ids)

    B_Exp: dict[str, np.ndarray] = {}
    B_Nov: dict[str, np.ndarray] = {}

    for trial_id, max_w in max_wins_per_trial.items():
        if max_w == 0:
            continue
        B_Exp[trial_id] = np.full((n_exp, n_conn, max_w), np.nan)
        B_Nov[trial_id] = np.full((n_nov, n_conn, max_w), np.nan)

        for sub_idx, npz_data in enumerate(expert_npz_list):
            if npz_data is None:
                continue
            key = f"{trial_id}_connectivity"
            if key in npz_data:
                arr = npz_data[key]   # (n_conn, n_wins_for_this_subject_trial)
                if arr.shape[1] > 0:
                    B_Exp[trial_id][sub_idx, :, : arr.shape[1]] = arr

        for sub_idx, npz_data in enumerate(novice_npz_list):
            if npz_data is None:
                continue
            key = f"{trial_id}_connectivity"
            if key in npz_data:
                arr = npz_data[key]
                if arr.shape[1] > 0:
                    B_Nov[trial_id][sub_idx, :, : arr.shape[1]] = arr

        exp_valid = np.sum(~np.all(np.isnan(B_Exp[trial_id]), axis=(1, 2)))
        nov_valid = np.sum(~np.all(np.isnan(B_Nov[trial_id]), axis=(1, 2)))
        logger.info(
            "  %s: B_Exp shape=%s (%d non-NaN subs), B_Nov shape=%s (%d non-NaN subs)",
            trial_id, B_Exp[trial_id].shape, exp_valid,
            B_Nov[trial_id].shape, nov_valid,
        )

    # ------------------------------------------------------------------
    # 5.2.2 — Save B_Exp.npz, B_Nov.npz, gc_summary.json
    # ------------------------------------------------------------------
    _save_group_npz(
        B_Exp=B_Exp,
        B_Nov=B_Nov,
        expert_ids=expert_ids,
        novice_ids=novice_ids,
        max_wins_per_trial=max_wins_per_trial,
        expert_npz_list=expert_npz_list,
        novice_npz_list=novice_npz_list,
        config=config,
    )

    return B_Exp, B_Nov


# ---------------------------------------------------------------------------
# Private helper — load raw .npz dict for a list of subjects
# ---------------------------------------------------------------------------

def _load_all_subject_npz(
    subject_ids: list,
    config: Optional[dict] = None,
) -> list:
    """
    Load gc_connectivity.npz for each subject ID.

    Returns a list of dicts (numpy.load output), one per subject.
    Missing files produce a None entry — the list length always matches
    ``len(subject_ids)``.
    """
    out_dir = get_gc_output_dir("gc_connectivity", config=config)
    results = []
    for sid in subject_ids:
        fpath = out_dir / f"sub-{sid}_gc_connectivity.npz"
        if not fpath.exists():
            logger.warning("_load_all_subject_npz: missing file for %s: %s", sid, fpath.name)
            results.append(None)
        else:
            results.append(dict(np.load(fpath, allow_pickle=True)))
    return results


# ---------------------------------------------------------------------------
# Private helper — save group .npz files and summary JSON
# ---------------------------------------------------------------------------

def _save_group_npz(
    B_Exp: dict,
    B_Nov: dict,
    expert_ids: list,
    novice_ids: list,
    max_wins_per_trial: dict,
    expert_npz_list: list,
    novice_npz_list: list,
    config: Optional[dict] = None,
) -> None:
    """Save B_Exp.npz, B_Nov.npz, and gc_summary.json to the group directory."""
    out_dir = get_gc_output_dir("group", config=config)

    # B_Exp.npz
    exp_save = {trial_id: arr for trial_id, arr in B_Exp.items()}
    exp_save["subject_ids"] = np.array(expert_ids)
    np.savez(out_dir / "B_Exp.npz", **exp_save)
    logger.info("Saved B_Exp.npz  (trials: %s)", list(B_Exp.keys()))

    # B_Nov.npz
    nov_save = {trial_id: arr for trial_id, arr in B_Nov.items()}
    nov_save["subject_ids"] = np.array(novice_ids)
    np.savez(out_dir / "B_Nov.npz", **nov_save)
    logger.info("Saved B_Nov.npz  (trials: %s)", list(B_Nov.keys()))

    # Per-subject window count summary
    def _window_summary(subject_ids, npz_list):
        summary = {}
        for sid, d in zip(subject_ids, npz_list):
            if d is None:
                summary[sid] = {"status": "missing", "trial_window_counts": {}}
                continue
            trial_ids = list(d["trial_ids"])
            window_counts = list(d["window_counts"].astype(int))
            summary[sid] = {
                "status": "ok",
                "n_trials": int(d["n_trials"]),
                "trial_window_counts": dict(zip(trial_ids, window_counts)),
                "total_windows": int(sum(window_counts)),
                "roi_names_used": list(d["roi_names_used"]),
                "group": str(d["group"]),
            }
        return summary

    gc_summary = {
        "created_at": datetime.now().isoformat(),
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "trials": list(max_wins_per_trial.keys()),
        "max_windows_per_trial": {k: int(v) for k, v in max_wins_per_trial.items()},
        "n_connections": len(get_connection_labels()),
        "connection_labels": get_connection_labels(),
        "expert_subjects": _window_summary(expert_ids, expert_npz_list),
        "novice_subjects": _window_summary(novice_ids, novice_npz_list),
    }

    class _NumpyEncoder(json.JSONEncoder):
        """Convert numpy scalar types to native Python for JSON serialization."""
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    summary_path = out_dir / "gc_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(gc_summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info("Saved gc_summary.json")


# ===========================================================================
# Section 5.3 — Single-Subject + Batch Validation
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    gc_cfg = cfg["stage_d_granger"]["linear_gc"]
    conn_labels = get_connection_labels()

    # ===================================================================
    # Section 5.3.1 — Single-subject validation on A1
    # ===================================================================
    print("\n" + "=" * 70)
    print("Section 5.3.1 — process_subject_gc('A1') end-to-end validation")
    print("=" * 70)

    print("\n[1] Running process_subject_gc('A1') ...")
    t0 = time.time()
    result_A1 = process_subject_gc("A1", study="expert_novice", config=cfg)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f} s")

    # Check output structure
    n_trials_A1 = len(result_A1)
    print(f"\n  Trials found : {n_trials_A1}  (expected: 5 for A1)")
    total_wins_A1 = sum(len(v) for v in result_A1.values())
    print(f"  Total windows: {total_wins_A1}  (expected: 8 for A1)")

    for trial_id, trial_data in sorted(result_A1.items(),
                                        key=lambda x: int(x[0].split("_")[1])):
        n_wins = len(trial_data)
        if n_wins > 0:
            # Check shape of each connectivity vector
            sample_vec = next(iter(trial_data.values()))
            shape_ok = sample_vec.shape == (20,)
            values_ok = np.all(sample_vec >= 0.0)
            print(
                f"  {trial_id}: {n_wins} windows  "
                f"conn_shape={sample_vec.shape}  "
                f"shape_ok={shape_ok}  values_ok={values_ok}  "
                f"max={sample_vec.max():.4f}"
            )
        else:
            print(f"  {trial_id}: 0 windows (SHORT trial — no data)")

    # Shape assertions
    assert n_trials_A1 == 5, f"FAIL: expected 5 trials for A1, got {n_trials_A1}"
    assert total_wins_A1 == 8, f"FAIL: expected 8 total windows for A1, got {total_wins_A1}"

    for trial_id, trial_data in result_A1.items():
        for win_id, conn_vec in trial_data.items():
            assert conn_vec.shape == (20,), (
                f"FAIL: {trial_id}/{win_id} connectivity shape {conn_vec.shape} != (20,)"
            )
            assert np.all(conn_vec >= 0.0), (
                f"FAIL: {trial_id}/{win_id} has negative GC values"
            )

    # Mean GC per connection across all windows
    all_vecs = [v for td in result_A1.values() for v in td.values()]
    all_conn = np.stack(all_vecs, axis=0)   # (n_windows_total, 20)
    mean_gc = all_conn.mean(axis=0)          # (20,)

    print(f"\n  Mean GC per connection (A1, averaged over all {total_wins_A1} windows):")
    for conn_idx, (label, val) in enumerate(zip(conn_labels, mean_gc)):
        print(f"    [{conn_idx:2d}] {label:12s}  mean={val:.4f}")

    print(f"\n  Overall mean GC : {mean_gc.mean():.4f}")
    print(f"  Overall max GC  : {mean_gc.max():.4f}  ({conn_labels[int(np.argmax(mean_gc))]})")
    print(f"  Overall min GC  : {mean_gc.min():.4f}  ({conn_labels[int(np.argmin(mean_gc))]})")

    # Verify .npz was saved correctly
    out_dir = get_gc_output_dir("gc_connectivity", config=cfg)
    npz_path = out_dir / "sub-A1_gc_connectivity.npz"
    assert npz_path.exists(), f"FAIL: .npz not found at {npz_path}"
    saved = np.load(npz_path, allow_pickle=True)
    assert "Tr_1_connectivity" in saved, "FAIL: Tr_1_connectivity not in .npz"
    assert saved["Tr_1_connectivity"].shape[0] == 20, (
        f"FAIL: connectivity rows {saved['Tr_1_connectivity'].shape[0]} != 20"
    )

    print(f"\n✓ .npz saved: {npz_path.name}")
    print(f"  Keys : {sorted(saved.keys())}")
    print(f"  Tr_1_connectivity shape : {saved['Tr_1_connectivity'].shape}")
    print(f"  roi_names_used : {list(saved['roi_names_used'])}")
    print(f"  connection_labels : {list(saved['connection_labels'])[:5]} ...")

    print("\n✓ All 5.3.1 assertions passed.")

    # ===================================================================
    # Section 5.3.2 — Batch processing: all 13 GC-eligible subjects
    # ===================================================================
    print("\n" + "=" * 70)
    print("Section 5.3.2 — Batch processing (13 GC-eligible subjects)")
    print("=" * 70)

    all_subjects = gc_cfg["subjects"]["Expert"] + gc_cfg["subjects"]["Novice"]
    print(f"\n  Subjects to process: {all_subjects}  (n={len(all_subjects)})")
    print("  (R10 already excluded from config)\n")

    batch_log = {
        "run_timestamp": datetime.now().isoformat(),
        "subjects_attempted": all_subjects,
        "n_attempted": len(all_subjects),
        "results": {},
    }

    n_success = 0
    n_fail = 0

    for sid in all_subjects:
        t_sub = time.time()
        print(f"  [{sid}] processing ...", end="", flush=True)
        try:
            res = process_subject_gc(sid, study="expert_novice", config=cfg)
            n_trials = len(res)
            total_w = sum(len(v) for v in res.values())
            elapsed_s = time.time() - t_sub

            # Per-trial window summary
            tw_counts = {tid: len(tdata) for tid, tdata in res.items()}

            # ROI names from saved .npz
            npz_data = dict(np.load(
                get_gc_output_dir("gc_connectivity", config=cfg)
                / f"sub-{sid}_gc_connectivity.npz",
                allow_pickle=True,
            ))
            roi_names = list(npz_data["roi_names_used"])
            n_rois = len(roi_names)

            print(
                f" ✓  trials={n_trials}, total_windows={total_w}, "
                f"rois={n_rois} ({', '.join(roi_names)}), "
                f"t={elapsed_s:.0f}s"
            )
            batch_log["results"][sid] = {
                "status": "ok",
                "n_trials": n_trials,
                "total_windows": total_w,
                "trial_window_counts": tw_counts,
                "roi_names_used": roi_names,
                "n_rois": n_rois,
                "elapsed_s": round(elapsed_s, 1),
                "group": get_subject_group(sid, config=cfg),
                "notes": [],
            }

            # Flag subjects with reduced ROI count
            if n_rois < 5:
                note = f"REDUCED ROIs: only {n_rois}/5 ROIs ({', '.join(roi_names)})"
                batch_log["results"][sid]["notes"].append(note)
                print(f"      ⚠ {note}")

            # Flag subjects with fewer trials than expected (5)
            if n_trials < 5:
                note = f"FEWER TRIALS: {n_trials}/5"
                batch_log["results"][sid]["notes"].append(note)
                print(f"      ⚠ {note}")

            n_success += 1

        except Exception as exc:
            elapsed_s = time.time() - t_sub
            print(f" ✗  ERROR: {exc}")
            batch_log["results"][sid] = {
                "status": "error",
                "error": str(exc),
                "elapsed_s": round(elapsed_s, 1),
            }
            n_fail += 1

    batch_log["n_success"] = n_success
    batch_log["n_fail"] = n_fail

    # Save gc_batch_log.json
    log_dir = get_gc_output_dir("gc_connectivity", config=cfg)
    log_path = log_dir / "gc_batch_log.json"
    with open(log_path, "w") as fh:
        json.dump(batch_log, fh, indent=2)

    print(f"\n  Batch complete: {n_success}/{len(all_subjects)} succeeded, {n_fail} failed")
    print(f"  Batch log saved → {log_path.relative_to(_PROJECT_ROOT)}")

    assert n_success == len(all_subjects), (
        f"FAIL: only {n_success}/{len(all_subjects)} subjects succeeded"
    )
    print("\n✓ All 5.3.2 assertions passed — 13/13 subjects processed.")

    # ===================================================================
    # Section 5.2 — Build group arrays after all subjects are processed
    # ===================================================================
    print("\n" + "=" * 70)
    print("Section 5.2 — build_group_arrays() after batch processing")
    print("=" * 70)

    print("\n[Building B_Exp and B_Nov ...]")
    B_Exp, B_Nov = build_group_arrays(study="expert_novice", config=cfg)

    print("\n  Group array shapes:")
    for trial_id in sorted(B_Exp.keys(), key=lambda k: int(k.split("_")[1])):
        print(
            f"  {trial_id}: B_Exp={B_Exp[trial_id].shape}  "
            f"B_Nov={B_Nov[trial_id].shape}"
        )

    # Success criteria check
    for trial_id in B_Exp:
        n_exp_subs, n_conn, n_wins = B_Exp[trial_id].shape
        n_nov_subs, _, _ = B_Nov[trial_id].shape
        assert n_exp_subs == 7, (
            f"FAIL: B_Exp[{trial_id}] has {n_exp_subs} rows, expected 7 Experts"
        )
        assert n_nov_subs == 6, (
            f"FAIL: B_Nov[{trial_id}] has {n_nov_subs} rows, expected 6 Novices"
        )
        assert n_conn == 20, (
            f"FAIL: B_Exp[{trial_id}] has {n_conn} connections, expected 20"
        )

    print("\n✓ B_Exp[Tr_1].shape[0] == 7 (Expert) ✓")
    print("✓ B_Nov[Tr_1].shape[0] == 6 (Novice) ✓")
    print("✓ n_connections == 20 for all trials ✓")
    print("\n✓ Section 5 (5.1 + 5.2 + 5.3) validation complete.\n")
