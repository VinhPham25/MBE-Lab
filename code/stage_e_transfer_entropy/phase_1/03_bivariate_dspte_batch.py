"""
Stage E — Step E3: Per-Subject Bivariate dSPTE Pipeline and Batch Runner

Orchestrates the full Transfer Entropy pipeline for one subject and
for the entire ASTaUND expert-novice cohort.  Mirrors Stage D's
``04_subject_gc_pipeline.py`` as closely as possible — same function
structure, same .npz schema (extended with delay and k₀ arrays), same
batch loop and batch log format.

Section 4.2 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
---------------------------------------------------------
4.2.1 — process_subject_bivariate_dspte(subject_id, study, overwrite) -> dict
           Orchestrates: load ROI signals (from Stage D cache) →
           get_trial_segments → compute_windowed_bivariate_dspte per trial →
           extract_connectivity_vector per window → save .npz.
           Supports overwrite=False caching.

4.2.2 — Per-subject .npz schema:
           Tr_N_connectivity   shape (20, n_wins)         connectivity vectors
           Tr_N_delays         shape (20, n_wins)         optimal delay per connection
           Tr_N_k0_values      shape (5, 5, n_wins)       k₀ per ROI pair per window
           Tr_N_te_profiles    shape (20, u_max, n_wins)  full lag-TE curve per connection
           + metadata keys (subject_id, group, trial_ids, window_counts,
             roi_names_used, connection_labels, n_trials, params_json)

4.2.3 — run_batch_bivariate_dspte(study, subjects, overwrite) -> dict
           Batch processes all 13 subjects (R10 excluded).
           Writes te_batch_log.json with per-subject stats.

.npz schema rationale
---------------------
The connectivity arrays are schema-identical to Stage D:
    ``Tr_N_connectivity`` : shape (20, n_windows)
This allows Section 9 cross-phase comparison to operate directly on
aligned arrays without any format conversion.

The additional delay and k₀ arrays provide diagnostic output for:
- Section 6.2 (optimal delay heatmaps)
- Section 6.5 (lag-TE profiles, stored in te_profiles metadata)
- Phase 2/3 deviation auditing

Expected anomalies (same as Stage D)
-------------------------------------
R9  : 2 trials (short recording)
R11 : 3 valid ROIs (LM1 + SMA bad channels) → RPFC/LM1/SMA connections = 0
R2  : 4 valid ROIs (SMA bad channel) → SMA connections = 0
R4  : 4 trials

Reference
---------
Wang, Y., & Chen, W. (2020). Effective brain connectivity for fNIRS data
analysis based on multi-delays symbolic phase transfer entropy.
J. Neural Eng., 17(5), 056024. doi:10.1088/1741-2552/abb4a4
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent.parent   # code/
_STAGE_E_ROOT = Path(__file__).parent.parent       # stage_e_transfer_entropy/
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
# utils_te lives in stage_e_transfer_entropy/ (one level up from phase_1/)
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy module imports — avoids matplotlib/side-effect issues on import
# ---------------------------------------------------------------------------
_SEG_MODULE = None
_PIPELINE_MODULE = None
_CORE_MODULE = None


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


def _get_pipeline_module():
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "bivariate_dspte_pipeline",
            Path(__file__).parent / "02_bivariate_dspte_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PIPELINE_MODULE = mod
    return _PIPELINE_MODULE


def _get_core_module_for_filter():
    global _CORE_MODULE
    if _CORE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "bivariate_dspte_core",
            Path(__file__).parent / "01_bivariate_dspte_core.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE_MODULE = mod
    return _CORE_MODULE


def _filter_and_extract_phase_trial(
    Y_trial: np.ndarray,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    order: int,
    core_mod,
    subject_id: str = "",
    trial_id: str = "",
) -> tuple[np.ndarray, list]:
    """
    Apply narrowband bandpass filter to the full trial then extract Hilbert phase.

    This is Option A of the narrowband phase-TE approach:
    1. Filter each ROI's full trial signal with a zero-phase Butterworth bandpass.
    2. Extract instantaneous phase via the Hilbert transform on the filtered signal.
    3. Compute embedding delay τ from the filtered phase signal.

    By operating on the full trial (not per-window), edge artefacts from the
    Hilbert transform are confined to the trial boundaries, which lie outside
    the analysis windows.

    Parameters
    ----------
    Y_trial : np.ndarray, shape (n_rois, T_trial)
        Raw ROI signal matrix for one trial.
    sfreq : float
        Sampling frequency in Hz.
    f_lo, f_hi : float
        Bandpass cutoff frequencies in Hz.
    order : int
        Butterworth filter order.
    core_mod : module
        Loaded ``01_bivariate_dspte_core`` module (provides narrowband_filter,
        extract_phase_signal, compute_embedding_delay).
    subject_id, trial_id : str
        For logging only.

    Returns
    -------
    Phi_trial : np.ndarray, shape (n_rois, T_trial)
        Instantaneous phase array for the full trial.  All-NaN rows remain
        all-NaN (bad channels preserved for downstream NaN guard).
    taus_trial : list of int, length n_rois
        Embedding delays computed from the filtered phase signals.
    """
    n_rois, T_trial = Y_trial.shape
    Phi_trial = np.full_like(Y_trial, np.nan)
    taus_trial: list = []

    for roi_idx in range(n_rois):
        row = Y_trial[roi_idx]
        if np.all(np.isnan(row)):
            taus_trial.append(1)
            logger.warning(
                "_filter_and_extract_phase_trial [%s %s]: ROI %d all-NaN — skipping.",
                subject_id, trial_id, roi_idx,
            )
            continue

        # Bandpass filter on the full trial signal
        x_filt = core_mod.narrowband_filter(
            row, f_lo=f_lo, f_hi=f_hi, sfreq=sfreq, order=order
        )

        # Hilbert phase on the filtered full-trial signal
        ph = core_mod.extract_phase_signal(x_filt)
        Phi_trial[roi_idx] = ph

        # Embedding delay from the full-trial filtered phase
        tau_i = core_mod.compute_embedding_delay(ph)
        taus_trial.append(tau_i)

    logger.info(
        "_filter_and_extract_phase_trial [%s %s]: filtered [%.3f–%.3f Hz]  "
        "n_rois=%d  taus=%s",
        subject_id, trial_id, f_lo, f_hi, n_rois, taus_trial,
    )
    return Phi_trial, taus_trial


# ===========================================================================
# Section 4.2.1 — Per-Subject dSPTE Pipeline
# ===========================================================================

def process_subject_bivariate_dspte(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = True,
) -> dict:
    """
    Run the full dSPTE pipeline for one subject and save connectivity to .npz.

    Orchestrates: load ROI signals (from Stage D cache) → trial segmentation
    → windowed dSPTE per trial → connectivity vector extraction → save .npz.

    Mirrors Stage D's ``process_subject_gc()`` in structure, with three
    differences:
    1. ROI signals are loaded from the Stage D cache via
       ``load_or_extract_roi_signals()`` (no .fif reload needed).
    2. The core computation is ``compute_windowed_bivariate_dspte()`` (not GC).
    3. The .npz stores additional ``Tr_N_delays`` and ``Tr_N_k0_values``
       arrays for diagnostic use.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'A1', 'R7'.
    study : str
        Config dataset key.  Default 'expert_novice'.
    config : dict, optional
        Parsed config.yaml.  Loaded from disk if None.
    overwrite : bool
        If False and the output .npz already exists, load and return the
        cached result.  Default False (cache-first).

    Returns
    -------
    result : dict
        Nested dict::

            {
              'Tr_1': {'Win_1': array(20,), 'Win_2': array(20,), ...},
              'Tr_2': {'Win_1': array(20,), ...},
              ...
            }

        Each leaf is a 20-element connectivity vector (identical schema to
        Stage D's ``process_subject_gc()`` return value).

    Notes
    -----
    **Trial segmentation:** ``get_trial_segments()`` from Stage D's
    ``02_trial_segmentation.py`` is reused via importlib.  It requires the
    Stage B .fif file to extract MNE annotations (event onsets).  The ROI
    signals themselves are loaded from the Stage D .npz cache.

    **R9 sfreq:** R9's .fif was resampled to 25 Hz during Stage D processing.
    The Stage D .npz cache stores the already-resampled signals, so no
    special handling is needed here.

    **Delay-matrix → 20-element delay vector:** The ``delay_matrix`` from
    ``compute_bivariate_dspte_matrix()`` is a (n_rois × n_rois) integer array.  It is
    expanded and flattened identically to the EC matrix (canonical 5×5 →
    remove diagonal → 20 elements) and stored as ``Tr_N_delays``.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    sym_cfg = te_cfg.get("symbolisation", {})
    win_cfg = te_cfg.get("window", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})

    window_size = int(win_cfg.get("size_samples", 1350))
    stride = int(win_cfg.get("stride_samples", 1350))

    # ------------------------------------------------------------------
    # Check for cached output (overwrite=False path)
    # ------------------------------------------------------------------
    from utils_te import (  # noqa: E402
        get_te_output_dir,
        get_subject_group,
        get_connection_labels,
        load_or_extract_roi_signals,
    )

    out_dir = get_te_output_dir("phase_1_connectivity", config=config)
    npz_path = out_dir / f"sub-{subject_id}_bivariate_dspte_connectivity.npz"

    if not overwrite and npz_path.exists():
        logger.info(
            "process_subject_bivariate_dspte [%s]: cached .npz found — loading.",
            subject_id,
        )
        return _load_subject_bivariate_dspte(subject_id, config=config)

    t_subject_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load ROI signals from Stage D cache
    # ------------------------------------------------------------------
    logger.info(
        "process_subject_bivariate_dspte [%s]: loading ROI signals ...", subject_id
    )
    Data_Reg, valid_roi_names, sfreq = load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    n_rois_valid = Data_Reg.shape[0]
    logger.info(
        "process_subject_bivariate_dspte [%s]: Data_Reg shape=%s  ROIs=%s  sfreq=%.1f Hz",
        subject_id, Data_Reg.shape, valid_roi_names, sfreq,
    )

    # ------------------------------------------------------------------
    # Step 2 — Trial segmentation (reuse Stage D segmenter on .fif)
    # ------------------------------------------------------------------
    # load_preprocessed_fif needed only for annotations/events
    from utils_te import load_preprocessed_fif  # noqa: E402
    logger.info(
        "process_subject_bivariate_dspte [%s]: loading .fif for trial segments ...",
        subject_id,
    )
    raw = load_preprocessed_fif(subject_id, study=study, config=config)

    seg_mod = _get_seg_module()
    segments = seg_mod.get_trial_segments(raw, config=config)
    logger.info(
        "process_subject_bivariate_dspte [%s]: %d trials found.", subject_id, len(segments)
    )

    # ------------------------------------------------------------------
    # Step 3 — Windowed dSPTE + connectivity extraction, per trial
    # ------------------------------------------------------------------
    pipe_mod = _get_pipeline_module()
    compute_windowed_bivariate_dspte = pipe_mod.compute_windowed_bivariate_dspte
    extract_connectivity_vector = pipe_mod.extract_connectivity_vector

    result: dict = {}          # {trial_id: {win_id: conn_vec_20}}
    delays_result: dict = {}   # {trial_id: {win_id: delay_vec_20}}
    k0_result: dict = {}       # {trial_id: {win_id: k0_matrix_5x5}}
    profiles_result: dict = {} # {trial_id: {win_id: lag_profiles_20 x u_max}}

    # Read narrowband filter settings once (used inside loop)
    filter_enabled = bool(filt_cfg.get("enabled", False))
    filt_f_lo = float(filt_cfg.get("f_lo", 0.04))
    filt_f_hi = float(filt_cfg.get("f_hi", 0.15))
    filt_order = int(filt_cfg.get("order", 3))

    # Import core module for narrowband_filter and extract_phase_signal
    core_mod = _get_core_module_for_filter()

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset : offset + 1]

        logger.info(
            "process_subject_bivariate_dspte [%s]: %s  samples=[%d, %d]  shape=%s",
            subject_id, trial_id, onset, offset, Y_trial.shape,
        )

        # ------------------------------------------------------------------
        # Step 2.5 — Option A: narrowband filter + full-trial Hilbert
        # ------------------------------------------------------------------
        # Apply bandpass to the FULL trial signal before windowing so that:
        #   (a) Hilbert phase is computed on a quasi-monocomponent oscillation
        #   (b) Edge artefacts are confined to trial boundaries (outside windows)
        # The resulting Phi_trial (n_rois, T_trial) and taus_trial (n_rois,)
        # are passed to compute_windowed_bivariate_dspte, which slices Phi_trial per
        # window instead of recomputing phase from raw signal slices.
        if filter_enabled:
            Phi_trial, taus_trial = _filter_and_extract_phase_trial(
                Y_trial,
                sfreq=sfreq,
                f_lo=filt_f_lo,
                f_hi=filt_f_hi,
                order=filt_order,
                core_mod=core_mod,
                subject_id=subject_id,
                trial_id=trial_id,
            )
        else:
            Phi_trial = None
            taus_trial = None

        windows_bivariate_dspte = compute_windowed_bivariate_dspte(
            Y_trial,
            window_size=window_size,
            stride=stride,
            config=config,
            Phi_trial=Phi_trial,
            taus_trial=taus_trial,
        )

        trial_result: dict = {}
        trial_delays: dict = {}
        trial_k0: dict = {}
        trial_profiles: dict = {}

        from utils_te import get_connection_labels  # noqa: E402
        conn_labels = get_connection_labels()
        u_max = int(sig_cfg.get("u_max", 50))

        for win_id, (EC_matrix, delay_matrix, metadata) in windows_bivariate_dspte.items():
            # Connectivity vector: (n_valid × n_valid) → canonical 5×5 → 20
            conn_vec_20 = extract_connectivity_vector(EC_matrix, valid_roi_names)

            # Delay vector: same expansion, integer dtype
            delay_matrix_float = delay_matrix.astype(float)
            delay_vec_20 = extract_connectivity_vector(delay_matrix_float, valid_roi_names)

            # k₀ matrix: canonical 5×5 (integer, 0 on diagonal/missing)
            k0_full = _expand_to_canonical_5x5(
                metadata["k0_matrix"].astype(float), valid_roi_names
            )

            # Lag-TE profiles: (20, u_max) array in canonical connection order.
            # metadata["te_profiles"] is a dict keyed by (src_roi_idx, tgt_roi_idx)
            # mapping to a 1D array of shape (u_max,).
            # We expand to canonical 20-connection order, zeroing missing pairs.
            profiles_20 = _extract_canonical_lag_profiles(
                metadata.get("te_profiles", {}),
                valid_roi_names,
                conn_labels,
                u_max,
            )

            trial_result[win_id] = conn_vec_20
            trial_delays[win_id] = delay_vec_20
            trial_k0[win_id] = k0_full
            trial_profiles[win_id] = profiles_20

        result[trial_id] = trial_result
        delays_result[trial_id] = trial_delays
        k0_result[trial_id] = trial_k0
        profiles_result[trial_id] = trial_profiles

        total_wins = sum(len(v) for v in result.values())
        logger.info(
            "process_subject_bivariate_dspte [%s]: %s → %d windows  (total so far: %d)",
            subject_id, trial_id, len(trial_result), total_wins,
        )

    # ------------------------------------------------------------------
    # Step 4 — Save per-subject .npz
    # ------------------------------------------------------------------
    elapsed_subject = time.perf_counter() - t_subject_start

    filter_enabled = bool(filt_cfg.get("enabled", False))
    params = {
        "study": study,
        "window_size_samples": window_size,
        "stride_samples": stride,
        "sfreq": sfreq,
        "embedding_dimension": int(sig_cfg.get("embedding_dimension", 3)),
        "u_max": int(sig_cfg.get("u_max", 50)),
        "ndis_threshold": float(sym_cfg.get("ndis_threshold", 0.05)),
        "k0_max_bivariate": int(sym_cfg.get("k0_max_bivariate", 20)),
        "n_connections": 20,
        "elapsed_s": round(elapsed_subject, 1),
        "narrowband_filter_enabled": filter_enabled,
        "narrowband_filter_f_lo": float(filt_cfg.get("f_lo", 0.0)) if filter_enabled else None,
        "narrowband_filter_f_hi": float(filt_cfg.get("f_hi", 0.0)) if filter_enabled else None,
        "narrowband_filter_order": int(filt_cfg.get("order", 3)) if filter_enabled else None,
    }

    _save_subject_bivariate_dspte_npz(
        subject_id=subject_id,
        result=result,
        delays_result=delays_result,
        k0_result=k0_result,
        profiles_result=profiles_result,
        valid_roi_names=valid_roi_names,
        params=params,
        config=config,
    )

    total_wins = sum(len(v) for v in result.values())
    logger.info(
        "process_subject_bivariate_dspte [%s]: complete.  "
        "Trials=%d, Total windows=%d, Elapsed=%.1f s",
        subject_id, len(result), total_wins, elapsed_subject,
    )
    return result


# ---------------------------------------------------------------------------
# Private helper — expand (n_valid × n_valid) to canonical 5×5
# ---------------------------------------------------------------------------

_ROI_NAMES_CANONICAL = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_N_CANONICAL = 5


def _expand_to_canonical_5x5(
    matrix: np.ndarray,
    roi_names: list[str],
) -> np.ndarray:
    """
    Place a (n_valid × n_valid) matrix into a canonical (5×5) array.

    Missing ROI positions are filled with 0.  Used for k₀ and nDis
    diagnostic matrices (same expansion logic as extract_connectivity_vector,
    but retains the full 5×5 layout rather than flattening to 20 elements).
    """
    n_valid = len(roi_names)
    full = np.zeros((_N_CANONICAL, _N_CANONICAL), dtype=matrix.dtype)
    for i_v, roi_i in enumerate(roi_names):
        for j_v, roi_j in enumerate(roi_names):
            if roi_i != roi_j:
                i_c = _ROI_NAMES_CANONICAL.index(roi_i)
                j_c = _ROI_NAMES_CANONICAL.index(roi_j)
                full[i_c, j_c] = matrix[i_v, j_v]
    return full


# ---------------------------------------------------------------------------
# Private helper — expand te_profiles dict to canonical 20×u_max array
# ---------------------------------------------------------------------------

def _extract_canonical_lag_profiles(
    te_profiles: dict,
    roi_names: list[str],
    conn_labels: list[str],
    u_max: int,
) -> np.ndarray:
    """
    Convert a ``te_profiles`` dict keyed by ``(src_roi_idx, tgt_roi_idx)``
    (local valid-ROI indices) into a canonical (20, u_max) array.

    The 20 rows correspond to the same off-diagonal canonical 5×5 ordering
    used by ``extract_connectivity_vector``.  Pairs that were skipped or
    missing produce a row of zeros.

    Parameters
    ----------
    te_profiles : dict
        ``{(src, tgt): np.ndarray(u_max,)}`` from ``compute_bivariate_dspte_matrix``
        metadata.  Keys use *local* valid-ROI indices.
    roi_names : list[str]
        Valid ROI names for this window (e.g. ``['LPFC','CPFC','RPFC','LM1','SMA']``).
    conn_labels : list[str]
        Canonical 20 connection labels (``get_connection_labels()``).
    u_max : int
        Maximum lag searched.

    Returns
    -------
    profiles : np.ndarray, shape (20, u_max), dtype float64
    """
    profiles = np.zeros((20, u_max), dtype=np.float64)

    # Build a lookup: (canonical_src_idx, canonical_tgt_idx) → row in profiles
    # conn_labels has the form "LPFC→CPFC", "LPFC→RPFC", ...
    conn_index: dict = {}
    for row_idx, label in enumerate(conn_labels):
        parts = label.split("→")
        if len(parts) == 2:
            conn_index[(parts[0], parts[1])] = row_idx

    for (src_v, tgt_v), profile in te_profiles.items():
        if src_v >= len(roi_names) or tgt_v >= len(roi_names):
            continue
        src_name = roi_names[src_v]
        tgt_name = roi_names[tgt_v]
        row_idx = conn_index.get((src_name, tgt_name))
        if row_idx is None:
            continue
        arr = np.asarray(profile, dtype=np.float64)
        length = min(len(arr), u_max)
        profiles[row_idx, :length] = arr[:length]

    return profiles


# ---------------------------------------------------------------------------
# Private helper — save per-subject .npz
# ---------------------------------------------------------------------------

def _save_subject_bivariate_dspte_npz(
    subject_id: str,
    result: dict,
    delays_result: dict,
    k0_result: dict,
    profiles_result: dict,
    valid_roi_names: list,
    params: dict,
    config: Optional[dict] = None,
) -> Path:
    """
    Save ``process_subject_bivariate_dspte`` output to
    ``phase_1_connectivity/sub-{id}_bivariate_dspte_connectivity.npz``.

    Schema
    ------
    Tr_N_connectivity  : (20, n_wins)         — directed TE connectivity vectors
    Tr_N_delays        : (20, n_wins)         — optimal delay (samples) per connection
    Tr_N_k0_values     : (5, 5, n_wins)       — k₀ per ROI pair per window (diagnostic)
    Tr_N_te_profiles   : (20, u_max, n_wins)  — full lag-TE curve per connection per window
    subject_id         : str
    group              : 'Expert' or 'Novice'
    trial_ids          : str array
    window_counts      : int array
    roi_names_used     : str array
    connection_labels  : str array (20 labels)
    n_trials           : int
    params_json        : JSON string of processing parameters

    Returns the absolute path to the saved file.
    """
    from utils_te import get_te_output_dir, get_subject_group, get_connection_labels  # noqa

    out_dir = get_te_output_dir("phase_1_connectivity", config=config)
    conn_labels = get_connection_labels()

    trial_ids = sorted(result.keys(), key=lambda k: int(k.split("_")[1]))
    window_counts = []
    save_dict: dict = {}

    n_conn = 20
    n_rois_full = _N_CANONICAL
    u_max = int(params.get("u_max", 50))

    for trial_id in trial_ids:
        trial_data = result[trial_id]
        trial_delays = delays_result.get(trial_id, {})
        trial_k0 = k0_result.get(trial_id, {})
        trial_profiles = profiles_result.get(trial_id, {})
        n_wins = len(trial_data)
        window_counts.append(n_wins)

        if n_wins > 0:
            wins_sorted = sorted(
                trial_data.keys(), key=lambda k: int(k.split("_")[1])
            )
            # (20, n_wins) connectivity array
            conn_array = np.column_stack(
                [trial_data[wk] for wk in wins_sorted]
            ).astype(np.float64)

            # (20, n_wins) delay array
            delay_array = np.column_stack(
                [trial_delays.get(wk, np.zeros(n_conn)) for wk in wins_sorted]
            ).astype(np.float64)

            # (5, 5, n_wins) k₀ array — stack along third axis
            k0_list = [
                trial_k0.get(wk, np.zeros((n_rois_full, n_rois_full)))
                for wk in wins_sorted
            ]
            k0_array = np.stack(k0_list, axis=2).astype(np.float64)  # (5,5,n_wins)

            # (20, u_max, n_wins) lag-TE profiles array
            profiles_list = [
                trial_profiles.get(wk, np.zeros((n_conn, u_max)))
                for wk in wins_sorted
            ]
            profiles_array = np.stack(profiles_list, axis=2).astype(np.float64)

        else:
            conn_array = np.empty((n_conn, 0), dtype=np.float64)
            delay_array = np.empty((n_conn, 0), dtype=np.float64)
            k0_array = np.empty((n_rois_full, n_rois_full, 0), dtype=np.float64)
            profiles_array = np.empty((n_conn, u_max, 0), dtype=np.float64)

        save_dict[f"{trial_id}_connectivity"] = conn_array
        save_dict[f"{trial_id}_delays"] = delay_array
        save_dict[f"{trial_id}_k0_values"] = k0_array
        save_dict[f"{trial_id}_te_profiles"] = profiles_array

    # Metadata
    save_dict["subject_id"] = np.array(subject_id)
    save_dict["n_trials"] = np.array(len(trial_ids))
    save_dict["trial_ids"] = np.array(trial_ids)
    save_dict["window_counts"] = np.array(window_counts, dtype=int)
    save_dict["roi_names_used"] = np.array(valid_roi_names)
    save_dict["connection_labels"] = np.array(get_connection_labels())

    try:
        group = get_subject_group(subject_id, config=config)
    except ValueError:
        group = "Unknown"
    save_dict["group"] = np.array(group)
    save_dict["params_json"] = np.array(json.dumps(params))

    npz_path = out_dir / f"sub-{subject_id}_bivariate_dspte_connectivity.npz"
    np.savez(npz_path, **save_dict)

    logger.info(
        "_save_subject_bivariate_dspte_npz [%s]: saved → %s  (trials=%d, total_wins=%d)",
        subject_id, npz_path.name, len(trial_ids), sum(window_counts),
    )
    return npz_path


# ---------------------------------------------------------------------------
# Private helper — load per-subject .npz
# ---------------------------------------------------------------------------

def _load_subject_bivariate_dspte(
    subject_id: str,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Load ``sub-{id}_bivariate_dspte_connectivity.npz`` and return the nested result dict.

    Returns a dict ``{trial_id: {win_id: conn_vec_20}}`` matching the return
    format of ``process_subject_bivariate_dspte()``.  Returns None if file not found.
    """
    from utils_te import get_te_output_dir  # noqa

    out_dir = get_te_output_dir("phase_1_connectivity", config=config)
    npz_path = out_dir / f"sub-{subject_id}_bivariate_dspte_connectivity.npz"

    if not npz_path.exists():
        logger.warning(
            "_load_subject_bivariate_dspte [%s]: file not found: %s", subject_id, npz_path
        )
        return None

    data = dict(np.load(npz_path, allow_pickle=True))
    trial_ids = list(data["trial_ids"])
    window_counts = list(data["window_counts"])

    result = {}
    for trial_id, n_wins in zip(trial_ids, window_counts):
        conn_array = data.get(f"{trial_id}_connectivity")
        trial_result = {}
        if conn_array is not None and conn_array.shape[1] > 0:
            for win_idx in range(conn_array.shape[1]):
                trial_result[f"Win_{win_idx + 1}"] = conn_array[:, win_idx]
        result[trial_id] = trial_result

    logger.info(
        "_load_subject_bivariate_dspte [%s]: loaded from cache — "
        "%d trials, %d total windows",
        subject_id,
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


# ===========================================================================
# Section 4.2.3 — Batch Runner
# ===========================================================================

def run_batch_bivariate_dspte(
    study: str = "expert_novice",
    subjects: Optional[list[str]] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Batch-process all subjects and write te_batch_log.json.

    Runs ``process_subject_bivariate_dspte()`` for every subject in the cohort
    (R10 excluded by default).  Logs per-subject statistics and saves a
    JSON batch log to ``phase_1_connectivity/te_batch_log.json``.

    Parameters
    ----------
    study : str
        Config dataset key.  Default 'expert_novice'.
    subjects : list[str], optional
        Explicit subject list.  If None, reads from config
        ``stage_e_transfer_entropy.subjects`` (Expert + Novice, excluding R10).
    config : dict, optional
        Parsed config.yaml.
    overwrite : bool
        If False, skip subjects whose .npz already exists.  Default False.

    Returns
    -------
    batch_log : dict
        ``{'subjects': {id: stats_dict}, 'summary': {...}}``
        Also written to ``te_batch_log.json``.

    Notes
    -----
    **Expected anomalies (same as Stage D):**
    - R9: 2 trials only
    - R11: 3 valid ROIs (LM1+SMA missing) → some connections = 0
    - R2: 4 valid ROIs (SMA missing) → SMA connections = 0
    - R4: 4 trials
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_connection_labels  # noqa

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})

    if subjects is None:
        experts = subj_cfg.get("Expert", [])
        novices = subj_cfg.get("Novice", [])
        subjects = experts + novices

    logger.info(
        "run_batch_bivariate_dspte: processing %d subjects: %s", len(subjects), subjects
    )

    batch_start = time.perf_counter()
    conn_labels = get_connection_labels()

    batch_log: dict = {
        "run_timestamp": datetime.now().isoformat(),
        "study": study,
        "n_subjects": len(subjects),
        "subjects": {},
    }

    n_success = 0
    n_failed = 0
    n_skipped = 0

    for subject_id in subjects:
        subj_start = time.perf_counter()
        logger.info("run_batch_bivariate_dspte: === Subject %s ===", subject_id)

        try:
            result = process_subject_bivariate_dspte(
                subject_id, study=study, config=config, overwrite=overwrite
            )

            if result is None:
                raise RuntimeError("process_subject_bivariate_dspte returned None")

            # Gather per-subject stats
            n_trials = len(result)
            n_wins_total = sum(len(v) for v in result.values())

            # Collect all connectivity values for mean dSPTE
            all_conn = []
            for trial_data in result.values():
                for conn_vec in trial_data.values():
                    all_conn.append(conn_vec)

            if all_conn:
                all_conn_arr = np.stack(all_conn)   # (n_wins_total, 20)
                mean_bivariate_dspte = float(all_conn_arr.mean())
                max_bivariate_dspte = float(all_conn_arr.max())
                # Per-connection mean (across all windows × trials)
                conn_means = all_conn_arr.mean(axis=0).tolist()
            else:
                mean_bivariate_dspte = 0.0
                max_bivariate_dspte = 0.0
                conn_means = [0.0] * 20

            # Load k₀ stats from the saved .npz
            from utils_te import get_te_output_dir  # noqa
            out_dir = get_te_output_dir("phase_1_connectivity", config=config)
            npz_path = out_dir / f"sub-{subject_id}_bivariate_dspte_connectivity.npz"
            mean_k0 = float("nan")
            mean_delay = float("nan")
            if npz_path.exists():
                npz_data = dict(np.load(npz_path, allow_pickle=True))
                k0_vals, delay_vals = [], []
                for trial_id in result.keys():
                    k0_arr = npz_data.get(f"{trial_id}_k0_values")  # (5,5,n_wins)
                    delay_arr = npz_data.get(f"{trial_id}_delays")   # (20, n_wins)
                    if k0_arr is not None and k0_arr.size > 0:
                        # Only off-diagonal non-zero k₀ values
                        k0_vals.extend(k0_arr[k0_arr > 0].tolist())
                    if delay_arr is not None and delay_arr.size > 0:
                        delay_vals.extend(delay_arr[delay_arr > 0].tolist())
                if k0_vals:
                    mean_k0 = float(np.mean(k0_vals))
                if delay_vals:
                    mean_delay = float(np.mean(delay_vals))

            elapsed_subj = time.perf_counter() - subj_start

            batch_log["subjects"][subject_id] = {
                "status": "ok",
                "n_trials": n_trials,
                "n_windows_total": n_wins_total,
                "mean_bivariate_dspte": round(mean_bivariate_dspte, 6),
                "max_bivariate_dspte": round(max_bivariate_dspte, 6),
                "mean_k0": round(mean_k0, 2),
                "mean_optimal_delay_samples": round(mean_delay, 2),
                "mean_optimal_delay_s": round(mean_delay / 25.0, 3)
                    if not np.isnan(mean_delay) else float("nan"),
                "per_connection_mean_bivariate_dspte": [round(v, 6) for v in conn_means],
                "connection_labels": conn_labels,
                "elapsed_s": round(elapsed_subj, 1),
            }
            n_success += 1

            logger.info(
                "run_batch_bivariate_dspte [%s]: OK — trials=%d  wins=%d  "
                "mean_bivariate_dspte=%.4f  mean_k0=%.1f  elapsed=%.1f s",
                subject_id, n_trials, n_wins_total, mean_bivariate_dspte, mean_k0, elapsed_subj,
            )

        except Exception as exc:
            elapsed_subj = time.perf_counter() - subj_start
            logger.error(
                "run_batch_bivariate_dspte [%s]: FAILED — %s  (elapsed=%.1f s)",
                subject_id, exc, elapsed_subj,
            )
            batch_log["subjects"][subject_id] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(elapsed_subj, 1),
            }
            n_failed += 1

    elapsed_total = time.perf_counter() - batch_start
    batch_log["summary"] = {
        "n_success": n_success,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "total_elapsed_s": round(elapsed_total, 1),
        "completed_at": datetime.now().isoformat(),
    }

    # Save batch log
    from utils_te import get_te_output_dir  # noqa
    log_dir = get_te_output_dir("phase_1_connectivity", config=config)
    log_path = log_dir / "te_batch_log.json"
    with open(log_path, "w") as f:
        json.dump(batch_log, f, indent=2)

    logger.info(
        "run_batch_bivariate_dspte: complete.  OK=%d  FAILED=%d  Total=%.1f s  "
        "Log → %s",
        n_success, n_failed, elapsed_total, log_path,
    )
    return batch_log


# ===========================================================================
# Section 4.3 — Group Array Construction
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared helpers used by all three group array builders
# ---------------------------------------------------------------------------

_N_TRIALS_CANONICAL = 5  # canonical trial count for H1/group arrays


def _load_npz_lists(
    expert_ids: list,
    novice_ids: list,
    conn_dir: Path,
    caller: str,
) -> tuple[list, list]:
    """Load per-subject .npz files for both groups; None for missing files."""
    def _load(subject_ids):
        results = []
        for sid in subject_ids:
            fpath = conn_dir / f"sub-{sid}_bivariate_dspte_connectivity.npz"
            if not fpath.exists():
                logger.warning(
                    "%s: missing .npz for %s: %s", caller, sid, fpath.name
                )
                results.append(None)
            else:
                results.append(dict(np.load(fpath, allow_pickle=True)))
        return results

    return _load(expert_ids), _load(novice_ids)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _window_summary(subject_ids, npz_list):
    out = {}
    for sid, d in zip(subject_ids, npz_list):
        if d is None:
            out[sid] = {"status": "missing", "trial_window_counts": {}}
            continue
        trial_ids_s = list(d["trial_ids"])
        window_counts_s = list(d["window_counts"].astype(int))
        out[sid] = {
            "status": "ok",
            "n_trials": int(d["n_trials"]),
            "trial_window_counts": dict(zip(trial_ids_s, window_counts_s)),
            "total_windows": int(sum(window_counts_s)),
            "roi_names_used": list(d["roi_names_used"]),
            "group": str(d["group"]),
        }
    return out


# ---------------------------------------------------------------------------
# 4.3.1c — build_group_arrays_all_windows()  (renamed from build_group_arrays_bivariate_dspte)
# ---------------------------------------------------------------------------

def build_group_arrays_all_windows(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[dict, dict]:
    """
    Load all per-subject dSPTE .npz files and stack into group-level arrays.

    All windows from every subject are included.  This is the original
    ``build_group_arrays_bivariate_dspte()`` logic, renamed.  Use for H0
    (within-trial stability) and HIV (across-trial stability) tests which
    require window-level resolution.

    .. note::
        Uses all-windows arrays — window-level resolution required for
        within-trial stability test (H0) and across-trial stability test (HIV).

    Parameters
    ----------
    study : str
        Config dataset key.  Default 'expert_novice'.
    config : dict, optional
        Parsed config.yaml.  Loaded if None.
    overwrite : bool
        If False and the output .npz files already exist, load and return
        from disk.  Default False.

    Returns
    -------
    B_Exp : dict
        ``{trial_id: np.ndarray(n_exp_subjects, 20, max_windows_for_trial)}``.
        NaN where a subject has fewer windows or no data for that trial.
    B_Nov : dict
        Same structure for the Novice group.

    Saves
    -----
    ``phase_1_group/B_Exp_bivariate_dspte_all.npz``  — Expert group arrays
    ``phase_1_group/B_Nov_bivariate_dspte_all.npz``  — Novice group arrays
    ``phase_1_group/bivariate_dspte_summary.json``   — processing metadata
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_connection_labels  # noqa

    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_path = group_dir / "B_Exp_bivariate_dspte_all.npz"
    nov_path = group_dir / "B_Nov_bivariate_dspte_all.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info(
            "build_group_arrays_all_windows: cached files found — loading from disk."
        )
        return _load_group_npz_as_dict(exp_path), _load_group_npz_as_dict(nov_path)

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids: list = subj_cfg.get("Expert", [])
    novice_ids: list = subj_cfg.get("Novice", [])

    logger.info(
        "build_group_arrays_all_windows: Experts n=%d %s, Novices n=%d %s",
        len(expert_ids), expert_ids, len(novice_ids), novice_ids,
    )

    conn_dir = get_te_output_dir("phase_1_connectivity", config=config)
    expert_npz_list, novice_npz_list = _load_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_group_arrays_all_windows"
    )

    n_exp_loaded = sum(1 for d in expert_npz_list if d is not None)
    n_nov_loaded = sum(1 for d in novice_npz_list if d is not None)
    logger.info(
        "build_group_arrays_all_windows: loaded %d/%d Expert, %d/%d Novice .npz files",
        n_exp_loaded, len(expert_ids), n_nov_loaded, len(novice_ids),
    )

    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]
    if not all_npz:
        raise RuntimeError(
            "build_group_arrays_all_windows: no per-subject .npz files found. "
            "Run process_subject_bivariate_dspte() for each subject first."
        )

    # Determine max trials and max windows per trial across all subjects
    max_trials = max(int(d["n_trials"]) for d in all_npz)

    max_wins_per_trial: dict = {}
    for trial_idx in range(max_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        max_w = 0
        for d in all_npz:
            key = f"{trial_id}_connectivity"
            if key in d and d[key].ndim == 2 and d[key].shape[1] > 0:
                max_w = max(max_w, int(d[key].shape[1]))
        max_wins_per_trial[trial_id] = max_w

    logger.info(
        "build_group_arrays_all_windows: max_trials=%d  wins_per_trial=%s",
        max_trials, max_wins_per_trial,
    )

    n_conn = len(get_connection_labels())   # 20
    n_exp = len(expert_ids)
    n_nov = len(novice_ids)

    B_Exp: dict = {}
    B_Nov: dict = {}

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
                arr = npz_data[key]   # (20, n_wins_subj)
                if arr.ndim == 2 and arr.shape[1] > 0:
                    B_Exp[trial_id][sub_idx, :, : arr.shape[1]] = arr

        for sub_idx, npz_data in enumerate(novice_npz_list):
            if npz_data is None:
                continue
            key = f"{trial_id}_connectivity"
            if key in npz_data:
                arr = npz_data[key]
                if arr.ndim == 2 and arr.shape[1] > 0:
                    B_Nov[trial_id][sub_idx, :, : arr.shape[1]] = arr

        exp_valid = int(np.sum(~np.all(np.isnan(B_Exp[trial_id]), axis=(1, 2))))
        nov_valid = int(np.sum(~np.all(np.isnan(B_Nov[trial_id]), axis=(1, 2))))
        logger.info(
            "  %s: B_Exp shape=%s (%d non-NaN subs)  B_Nov shape=%s (%d non-NaN subs)",
            trial_id, B_Exp[trial_id].shape, exp_valid,
            B_Nov[trial_id].shape, nov_valid,
        )

    # Save
    exp_save = {tid: arr for tid, arr in B_Exp.items()}
    exp_save["subject_ids"] = np.array(expert_ids)
    exp_save["strategy"] = np.array("all_windows")
    np.savez(exp_path, **exp_save)
    logger.info(
        "build_group_arrays_all_windows: saved B_Exp_bivariate_dspte_all.npz  (trials=%s)",
        list(B_Exp.keys()),
    )

    nov_save = {tid: arr for tid, arr in B_Nov.items()}
    nov_save["subject_ids"] = np.array(novice_ids)
    nov_save["strategy"] = np.array("all_windows")
    np.savez(nov_path, **nov_save)
    logger.info(
        "build_group_arrays_all_windows: saved B_Nov_bivariate_dspte_all.npz  (trials=%s)",
        list(B_Nov.keys()),
    )

    # Summary JSON (shared with all builders — written by build_all_group_arrays)
    summary = {
        "created_at": datetime.now().isoformat(),
        "strategy": "all_windows",
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "trials": list(max_wins_per_trial.keys()),
        "max_windows_per_trial": {k: int(v) for k, v in max_wins_per_trial.items()},
        "n_connections": n_conn,
        "connection_labels": get_connection_labels(),
        "expert_subjects": _window_summary(expert_ids, expert_npz_list),
        "novice_subjects": _window_summary(novice_ids, novice_npz_list),
    }
    with open(group_dir / "bivariate_dspte_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info("build_group_arrays_all_windows: saved bivariate_dspte_summary.json")

    return B_Exp, B_Nov


# ---------------------------------------------------------------------------
# 4.3.1a — build_group_arrays_subject_mean()  PRIMARY for H1
# ---------------------------------------------------------------------------

def build_group_arrays_subject_mean(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build group arrays using the subject-mean strategy (PRIMARY for H1).

    For each subject × trial, average all windows (axis=1 of the (20, n_wins)
    connectivity array) to produce a single (20,) vector.  Stack into 3-D
    arrays of shape ``(n_subs, 20, n_trials)`` with NaN-fill for missing
    trials.  This gives every subject equal weight regardless of trial
    duration, controlling for the Novice/Expert window-count confound.

    Parameters
    ----------
    study : str
        Config dataset key.  Default 'expert_novice'.
    config : dict, optional
        Parsed config.yaml.  Loaded if None.
    overwrite : bool
        If False and the output .npz files already exist, load and return
        from disk.  Default False.

    Returns
    -------
    B_Exp_mean : np.ndarray, shape (n_exp_subjects, 20, n_trials)
    B_Nov_mean : np.ndarray, shape (n_nov_subjects, 20, n_trials)

    Saves
    -----
    ``phase_1_group/B_Exp_bivariate_dspte_mean.npz``
    ``phase_1_group/B_Nov_bivariate_dspte_mean.npz``

    Each .npz stores: ``data`` (n_subs, 20, 5), ``subject_ids``,
    ``strategy="subject_mean"``.
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_connection_labels  # noqa

    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_path = group_dir / "B_Exp_bivariate_dspte_mean.npz"
    nov_path = group_dir / "B_Nov_bivariate_dspte_mean.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info(
            "build_group_arrays_subject_mean: cached files found — loading from disk."
        )
        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)
        return exp_npz["data"], nov_npz["data"]

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids: list = subj_cfg.get("Expert", [])
    novice_ids: list = subj_cfg.get("Novice", [])

    logger.info(
        "build_group_arrays_subject_mean: Experts n=%d, Novices n=%d",
        len(expert_ids), len(novice_ids),
    )

    conn_dir = get_te_output_dir("phase_1_connectivity", config=config)
    expert_npz_list, novice_npz_list = _load_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_group_arrays_subject_mean"
    )

    n_conn = len(get_connection_labels())   # 20
    n_trials = _N_TRIALS_CANONICAL

    def _build_mean_array(subject_ids, npz_list):
        n_subs = len(subject_ids)
        arr = np.full((n_subs, n_conn, n_trials), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                logger.warning(
                    "build_group_arrays_subject_mean: skipping %s (missing .npz)",
                    subject_ids[sub_idx],
                )
                continue
            for trial_idx in range(n_trials):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]   # (20, n_wins)
                if conn.ndim != 2 or conn.shape[1] == 0:
                    continue
                # Mean across windows → (20,)
                arr[sub_idx, :, trial_idx] = np.nanmean(conn, axis=1)
        return arr

    B_Exp_mean = _build_mean_array(expert_ids, expert_npz_list)
    B_Nov_mean = _build_mean_array(novice_ids, novice_npz_list)

    logger.info(
        "build_group_arrays_subject_mean: B_Exp_mean shape=%s  B_Nov_mean shape=%s",
        B_Exp_mean.shape, B_Nov_mean.shape,
    )

    np.savez(
        exp_path,
        data=B_Exp_mean,
        subject_ids=np.array(expert_ids),
        strategy=np.array("subject_mean"),
    )
    logger.info("build_group_arrays_subject_mean: saved B_Exp_bivariate_dspte_mean.npz")

    np.savez(
        nov_path,
        data=B_Nov_mean,
        subject_ids=np.array(novice_ids),
        strategy=np.array("subject_mean"),
    )
    logger.info("build_group_arrays_subject_mean: saved B_Nov_bivariate_dspte_mean.npz")

    return B_Exp_mean, B_Nov_mean


# ---------------------------------------------------------------------------
# 4.3.1b — build_group_arrays_first_window()  SENSITIVITY CHECK for H1
# ---------------------------------------------------------------------------

def build_group_arrays_first_window(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build group arrays using the first-window strategy (SENSITIVITY CHECK for H1).

    For each subject × trial, take only ``Win_1`` — the first column of the
    (20, n_wins) connectivity array (index ``[:, 0]``).  This is the most
    conservative de-confounding approach: every subject contributes exactly
    one observation per trial regardless of trial duration.

    If a subject has a trial recorded but that trial has zero windows (edge
    case), the entry is left as NaN.

    Parameters
    ----------
    study : str
        Config dataset key.  Default 'expert_novice'.
    config : dict, optional
        Parsed config.yaml.  Loaded if None.
    overwrite : bool
        If False and the output .npz files already exist, load and return
        from disk.  Default False.

    Returns
    -------
    B_Exp_fw : np.ndarray, shape (n_exp_subjects, 20, n_trials)
    B_Nov_fw : np.ndarray, shape (n_nov_subjects, 20, n_trials)

    Saves
    -----
    ``phase_1_group/B_Exp_bivariate_dspte_fw.npz``
    ``phase_1_group/B_Nov_bivariate_dspte_fw.npz``

    Each .npz stores: ``data`` (n_subs, 20, 5), ``subject_ids``,
    ``strategy="first_window"``.
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_connection_labels  # noqa

    group_dir = get_te_output_dir("phase_1_group", config=config)
    exp_path = group_dir / "B_Exp_bivariate_dspte_fw.npz"
    nov_path = group_dir / "B_Nov_bivariate_dspte_fw.npz"

    if not overwrite and exp_path.exists() and nov_path.exists():
        logger.info(
            "build_group_arrays_first_window: cached files found — loading from disk."
        )
        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)
        return exp_npz["data"], nov_npz["data"]

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids: list = subj_cfg.get("Expert", [])
    novice_ids: list = subj_cfg.get("Novice", [])

    logger.info(
        "build_group_arrays_first_window: Experts n=%d, Novices n=%d",
        len(expert_ids), len(novice_ids),
    )

    conn_dir = get_te_output_dir("phase_1_connectivity", config=config)
    expert_npz_list, novice_npz_list = _load_npz_lists(
        expert_ids, novice_ids, conn_dir, "build_group_arrays_first_window"
    )

    n_conn = len(get_connection_labels())   # 20
    n_trials = _N_TRIALS_CANONICAL

    def _build_fw_array(subject_ids, npz_list):
        n_subs = len(subject_ids)
        arr = np.full((n_subs, n_conn, n_trials), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                logger.warning(
                    "build_group_arrays_first_window: skipping %s (missing .npz)",
                    subject_ids[sub_idx],
                )
                continue
            for trial_idx in range(n_trials):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_connectivity"
                if key not in npz_data:
                    continue
                conn = npz_data[key]   # (20, n_wins)
                if conn.ndim != 2 or conn.shape[1] == 0:
                    # Trial recorded but no windows — leave as NaN
                    logger.debug(
                        "build_group_arrays_first_window: %s %s has 0 windows — NaN",
                        subject_ids[sub_idx], trial_id,
                    )
                    continue
                # First window only → (20,)
                arr[sub_idx, :, trial_idx] = conn[:, 0]
        return arr

    B_Exp_fw = _build_fw_array(expert_ids, expert_npz_list)
    B_Nov_fw = _build_fw_array(novice_ids, novice_npz_list)

    logger.info(
        "build_group_arrays_first_window: B_Exp_fw shape=%s  B_Nov_fw shape=%s",
        B_Exp_fw.shape, B_Nov_fw.shape,
    )

    np.savez(
        exp_path,
        data=B_Exp_fw,
        subject_ids=np.array(expert_ids),
        strategy=np.array("first_window"),
    )
    logger.info("build_group_arrays_first_window: saved B_Exp_bivariate_dspte_fw.npz")

    np.savez(
        nov_path,
        data=B_Nov_fw,
        subject_ids=np.array(novice_ids),
        strategy=np.array("first_window"),
    )
    logger.info("build_group_arrays_first_window: saved B_Nov_bivariate_dspte_fw.npz")

    return B_Exp_fw, B_Nov_fw


# ---------------------------------------------------------------------------
# 4.3.1d — build_all_group_arrays()  Orchestrator
# ---------------------------------------------------------------------------

def build_all_group_arrays(
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Build all three group array variants and update bivariate_dspte_summary.json.

    Calls:
    - ``build_group_arrays_subject_mean()`` — PRIMARY for H1
    - ``build_group_arrays_first_window()`` — SENSITIVITY CHECK for H1
    - ``build_group_arrays_all_windows()`` — H0 / HIV only

    Returns
    -------
    paths : dict
        ``{'mean': (B_Exp_mean, B_Nov_mean),
           'fw':   (B_Exp_fw,   B_Nov_fw),
           'all':  (B_Exp_all,  B_Nov_all)}``
    """
    if config is None:
        config = load_config()

    from utils_te import get_te_output_dir, get_connection_labels  # noqa

    logger.info("build_all_group_arrays: building subject_mean arrays ...")
    B_Exp_mean, B_Nov_mean = build_group_arrays_subject_mean(
        study=study, config=config, overwrite=overwrite
    )

    logger.info("build_all_group_arrays: building first_window arrays ...")
    B_Exp_fw, B_Nov_fw = build_group_arrays_first_window(
        study=study, config=config, overwrite=overwrite
    )

    logger.info("build_all_group_arrays: building all_windows arrays ...")
    B_Exp_all, B_Nov_all = build_group_arrays_all_windows(
        study=study, config=config, overwrite=overwrite
    )

    # ------------------------------------------------------------------
    # Update bivariate_dspte_summary.json with duration-confound strategy metadata
    # ------------------------------------------------------------------
    group_dir = get_te_output_dir("phase_1_group", config=config)
    summary_path = group_dir / "bivariate_dspte_summary.json"

    existing_summary: dict = {}
    if summary_path.exists():
        with open(summary_path) as fh:
            existing_summary = json.load(fh)

    existing_summary["duration_confound_strategy"] = {
        "h1_primary": "subject_mean",
        "h1_sensitivity": "first_window",
        "h0_hiv": "all_windows",
        "rationale": (
            "Novice subjects have longer trials than Experts "
            "(avg ~16 vs ~6 windows). Subject-level averaging gives each "
            "subject equal weight regardless of trial duration."
        ),
    }
    existing_summary["group_arrays_updated_at"] = datetime.now().isoformat()

    with open(summary_path, "w") as fh:
        json.dump(existing_summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info(
        "build_all_group_arrays: updated bivariate_dspte_summary.json with "
        "duration_confound_strategy."
    )

    logger.info(
        "build_all_group_arrays: DONE. "
        "mean=%s/%s  fw=%s/%s  all=%d trials",
        B_Exp_mean.shape, B_Nov_mean.shape,
        B_Exp_fw.shape, B_Nov_fw.shape,
        len(B_Exp_all),
    )

    return {
        "mean": (B_Exp_mean, B_Nov_mean),
        "fw": (B_Exp_fw, B_Nov_fw),
        "all": (B_Exp_all, B_Nov_all),
    }


# ---------------------------------------------------------------------------
# Backward-compat alias (kept so existing call sites don't break immediately)
# ---------------------------------------------------------------------------

def build_group_arrays_bivariate_dspte(
    study: str = "expert_novice",
    config: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Deprecated alias — calls ``build_group_arrays_all_windows()``."""
    logger.warning(
        "build_group_arrays_bivariate_dspte() is deprecated. "
        "Call build_all_group_arrays() instead, or one of the three "
        "strategy-specific builders."
    )
    return build_group_arrays_all_windows(study=study, config=config)


# ---------------------------------------------------------------------------
# Private helper — load a group .npz back as a trial-keyed dict
# (used by all_windows cache path)
# ---------------------------------------------------------------------------

def _load_group_npz_as_dict(path: Path) -> dict:
    """Load a B_Exp/B_Nov_bivariate_dspte_all.npz and return as trial-keyed dict."""
    raw = dict(np.load(path, allow_pickle=True))
    out = {k: v for k, v in raw.items() if k.startswith("Tr_")}
    return out


# ===========================================================================
# Smoke tests (run as script)
# ===========================================================================

def _run_tests() -> None:
    """
    Section 4.2 smoke tests.

    4.2.1/4.2.2 — process_subject_bivariate_dspte('A1'):
        (a) Returns nested dict with correct structure
        (b) Each connectivity vector has shape (20,), values ≥ 0
        (c) .npz exists with correct keys and schema
        (d) Delay vectors shape (20,), k₀ array shape (5,5,n_wins)
        (e) overwrite=False loads from cache correctly

    NOTE: This test runs the FULL pipeline on A1 — expect several minutes.
          Set u_max to a small value in config to speed up validation runs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("\n=== Section 4.2 — Per-Subject dSPTE Pipeline Smoke Test ===\n")
    print("Running process_subject_bivariate_dspte('A1') with config defaults.")
    print("This may take several minutes (u_max × 20 pairs × n_windows).\n")

    config = load_config()

    # Optionally reduce u_max for faster validation
    # Uncomment the following line to speed up the test at the cost of
    # reduced lag search range:
    # config["stage_e_transfer_entropy"]["signal"]["u_max"] = 10

    t0 = time.perf_counter()
    result = process_subject_bivariate_dspte("A1", config=config, overwrite=True)
    elapsed = time.perf_counter() - t0

    print(f"\n--- Results ---")
    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  Trials : {list(result.keys())}")
    for trial_id, trial_data in result.items():
        if trial_data:
            ex_vec = next(iter(trial_data.values()))
            print(f"  {trial_id}: {len(trial_data)} windows  "
                  f"conn_vec shape={ex_vec.shape}  "
                  f"range=[{ex_vec.min():.4f}, {ex_vec.max():.4f}]")
        else:
            print(f"  {trial_id}: 0 windows (trial too short)")

    # (a) Structure checks
    assert isinstance(result, dict) and len(result) > 0, "Expected non-empty result dict"
    first_trial = next(iter(result.values()))
    assert isinstance(first_trial, dict), "Trial value must be a dict"
    first_vec = next(iter(first_trial.values())) if first_trial else None

    if first_vec is not None:
        assert first_vec.shape == (20,), f"Expected (20,) vector, got {first_vec.shape}"
        assert np.all(first_vec >= 0), "Connectivity values must be non-negative"
        print(f"\n  (a) Structure OK ✓  (b) Shape (20,), non-negative ✓")

    # (c) .npz schema validation
    from utils_te import get_te_output_dir
    out_dir = get_te_output_dir("phase_1_connectivity", config=config)
    npz_path = out_dir / "sub-A1_bivariate_dspte_connectivity.npz"
    assert npz_path.exists(), f".npz not found: {npz_path}"

    data = dict(np.load(npz_path, allow_pickle=True))
    required_keys = ["subject_id", "group", "trial_ids", "window_counts",
                     "roi_names_used", "connection_labels", "n_trials", "params_json"]
    for key in required_keys:
        assert key in data, f"Missing required .npz key: {key}"
    print(f"  (c) .npz exists with all required metadata keys ✓")
    print(f"      subject_id={data['subject_id']}  group={data['group']}")

    # (d) Delay and k₀ arrays
    trial_ids_saved = list(data["trial_ids"])
    for trial_id in trial_ids_saved:
        conn_key = f"{trial_id}_connectivity"
        delay_key = f"{trial_id}_delays"
        k0_key = f"{trial_id}_k0_values"
        assert conn_key in data, f"Missing {conn_key}"
        assert delay_key in data, f"Missing {delay_key}"
        assert k0_key in data, f"Missing {k0_key}"

        conn_arr = data[conn_key]
        delay_arr = data[delay_key]
        k0_arr = data[k0_key]
        n_wins = conn_arr.shape[1] if conn_arr.ndim == 2 else 0

        if n_wins > 0:
            assert conn_arr.shape == (20, n_wins), \
                f"{trial_id}: conn shape {conn_arr.shape} != (20, {n_wins})"
            assert delay_arr.shape == (20, n_wins), \
                f"{trial_id}: delay shape {delay_arr.shape} != (20, {n_wins})"
            assert k0_arr.shape == (5, 5, n_wins), \
                f"{trial_id}: k0 shape {k0_arr.shape} != (5, 5, {n_wins})"
            assert np.all(conn_arr >= 0), f"{trial_id}: negative connectivity values"

        print(f"  (d) {trial_id}: conn={conn_arr.shape}  delays={delay_arr.shape}  "
              f"k0={k0_arr.shape}  ✓")

    # (e) Cache load — overwrite=False should load without recomputing
    print(f"\n  (e) Testing cache load (overwrite=False) ...")
    t_cache = time.perf_counter()
    result_cached = process_subject_bivariate_dspte("A1", config=config, overwrite=False)
    elapsed_cache = time.perf_counter() - t_cache
    assert result_cached is not None, "Cache load returned None"
    assert set(result_cached.keys()) == set(result.keys()), "Trial keys mismatch"
    print(f"      Cache load elapsed: {elapsed_cache:.2f} s  "
          f"(expected << {elapsed:.1f} s)  ✓")

    print("\n=== All Section 4.2 smoke tests passed. ===\n")


def _cli_main() -> None:
    """CLI: batch all subjects, one subject, smoke test, or group arrays only."""
    import argparse

    #To run on full expert-novice group, use the following command:
    # python3 code/stage_e_transfer_entropy/phase_1/03_bivariate_dspte_batch.py --batch --overwrite

    #To run on a single subject, use the following command:
    # python3 code/stage_e_transfer_entropy/phase_1/03_bivariate_dspte_batch.py --subject A1 --overwrite

    #To run the smoke test, use the following command:
    # python3 code/stage_e_transfer_entropy/phase_1/03_bivariate_dspte_batch.py --test
    p = argparse.ArgumentParser(
        description="Stage E Phase 1 — per-subject bivariate dSPTE and group arrays.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run bivariate dSPTE for all Expert+Novice subjects in config (R10 excluded), "
        "write te_batch_log.json, then build B_Exp_bivariate_dspte / B_Nov_bivariate_dspte.",
    )
    p.add_argument(
        "--subject",
        type=str,
        default=None,
        metavar="ID",
        help="Run bivariate dSPTE for one subject (e.g. R7). Use --overwrite to recompute if .npz exists.",
    )
    p.add_argument(
        "--group-only",
        action="store_true",
        help="Only run build_all_group_arrays() from existing per-subject .npz files "
             "(builds all three strategy variants: mean, fw, all_windows).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if sub-{id}_bivariate_dspte_connectivity.npz exists.",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip subjects that already have output (batch mode).",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run Section 4.2 smoke test on A1 only (default if no other flag).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()

    # Batch: respect --overwrite / --no-overwrite (matches process_subject_bivariate_dspte defaults).
    overwrite = bool(args.overwrite)
    if args.no_overwrite:
        overwrite = False

    if args.group_only:
        build_all_group_arrays(config=config, overwrite=overwrite)
        return
    if args.batch:
        run_batch_bivariate_dspte(
            study="expert_novice",
            subjects=None,
            config=config,
            overwrite=overwrite,
        )
        build_all_group_arrays(config=config, overwrite=overwrite)
        return
    if args.subject:
        process_subject_bivariate_dspte(
            args.subject.strip(),
            study="expert_novice",
            config=config,
            overwrite=overwrite,
        )
        return
    if args.test or not (args.batch or args.subject or args.group_only):
        _run_tests()


if __name__ == "__main__":
    _cli_main()
