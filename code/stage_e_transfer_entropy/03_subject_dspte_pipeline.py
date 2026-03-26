"""
Stage E — Step E3: Per-Subject dSPTE Pipeline and Batch Runner

Orchestrates the full Transfer Entropy pipeline for one subject and
for the entire ASTaUND expert-novice cohort.  Mirrors Stage D's
``04_subject_gc_pipeline.py`` as closely as possible — same function
structure, same .npz schema (extended with delay and k₀ arrays), same
batch loop and batch log format.

Section 4.2 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
---------------------------------------------------------
4.2.1 — process_subject_dspte(subject_id, study, overwrite) -> dict
           Orchestrates: load ROI signals (from Stage D cache) →
           get_trial_segments → compute_windowed_dspte per trial →
           extract_connectivity_vector per window → save .npz.
           Supports overwrite=False caching.

4.2.2 — Per-subject .npz schema:
           Tr_N_connectivity   shape (20, n_wins)      connectivity vectors
           Tr_N_delays         shape (20, n_wins)      optimal delay per connection
           Tr_N_k0_values      shape (5, 5, n_wins)    k₀ per ROI pair per window
           + metadata keys (subject_id, group, trial_ids, window_counts,
             roi_names_used, connection_labels, n_trials, params_json)

4.2.3 — run_batch_dspte(study, subjects, overwrite) -> dict
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

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent
_STAGE_E_ROOT = Path(__file__).parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
# utils_te lives in the same directory as this file
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy module imports — avoids matplotlib/side-effect issues on import
# ---------------------------------------------------------------------------
_SEG_MODULE = None
_PIPELINE_MODULE = None


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
            "dspte_pipeline",
            Path(__file__).parent / "02_dspte_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PIPELINE_MODULE = mod
    return _PIPELINE_MODULE


# ===========================================================================
# Section 4.2.1 — Per-Subject dSPTE Pipeline
# ===========================================================================

def process_subject_dspte(
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
    2. The core computation is ``compute_windowed_dspte()`` (not GC).
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
    ``compute_dspte_matrix()`` is a (n_rois × n_rois) integer array.  It is
    expanded and flattened identically to the EC matrix (canonical 5×5 →
    remove diagonal → 20 elements) and stored as ``Tr_N_delays``.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    sym_cfg = te_cfg.get("symbolisation", {})
    win_cfg = te_cfg.get("window", {})

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
    npz_path = out_dir / f"sub-{subject_id}_dspte_connectivity.npz"

    if not overwrite and npz_path.exists():
        logger.info(
            "process_subject_dspte [%s]: cached .npz found — loading.",
            subject_id,
        )
        return _load_subject_dspte(subject_id, config=config)

    t_subject_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load ROI signals from Stage D cache
    # ------------------------------------------------------------------
    logger.info(
        "process_subject_dspte [%s]: loading ROI signals ...", subject_id
    )
    Data_Reg, valid_roi_names, sfreq = load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    n_rois_valid = Data_Reg.shape[0]
    logger.info(
        "process_subject_dspte [%s]: Data_Reg shape=%s  ROIs=%s  sfreq=%.1f Hz",
        subject_id, Data_Reg.shape, valid_roi_names, sfreq,
    )

    # ------------------------------------------------------------------
    # Step 2 — Trial segmentation (reuse Stage D segmenter on .fif)
    # ------------------------------------------------------------------
    # load_preprocessed_fif needed only for annotations/events
    from utils_te import load_preprocessed_fif  # noqa: E402
    logger.info(
        "process_subject_dspte [%s]: loading .fif for trial segments ...",
        subject_id,
    )
    raw = load_preprocessed_fif(subject_id, study=study, config=config)

    seg_mod = _get_seg_module()
    segments = seg_mod.get_trial_segments(raw, config=config)
    logger.info(
        "process_subject_dspte [%s]: %d trials found.", subject_id, len(segments)
    )

    # ------------------------------------------------------------------
    # Step 3 — Windowed dSPTE + connectivity extraction, per trial
    # ------------------------------------------------------------------
    pipe_mod = _get_pipeline_module()
    compute_windowed_dspte = pipe_mod.compute_windowed_dspte
    extract_connectivity_vector = pipe_mod.extract_connectivity_vector

    result: dict = {}         # {trial_id: {win_id: conn_vec_20}}
    delays_result: dict = {}  # {trial_id: {win_id: delay_vec_20}}
    k0_result: dict = {}      # {trial_id: {win_id: k0_matrix_5x5}}

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset : offset + 1]

        logger.info(
            "process_subject_dspte [%s]: %s  samples=[%d, %d]  shape=%s",
            subject_id, trial_id, onset, offset, Y_trial.shape,
        )

        windows_dspte = compute_windowed_dspte(
            Y_trial,
            window_size=window_size,
            stride=stride,
            config=config,
        )

        trial_result: dict = {}
        trial_delays: dict = {}
        trial_k0: dict = {}

        for win_id, (EC_matrix, delay_matrix, metadata) in windows_dspte.items():
            # Connectivity vector: (n_valid × n_valid) → canonical 5×5 → 20
            conn_vec_20 = extract_connectivity_vector(EC_matrix, valid_roi_names)

            # Delay vector: same expansion, integer dtype
            delay_matrix_float = delay_matrix.astype(float)
            delay_vec_20 = extract_connectivity_vector(delay_matrix_float, valid_roi_names)

            # k₀ matrix: canonical 5×5 (integer, 0 on diagonal/missing)
            k0_full = _expand_to_canonical_5x5(
                metadata["k0_matrix"].astype(float), valid_roi_names
            )

            trial_result[win_id] = conn_vec_20
            trial_delays[win_id] = delay_vec_20
            trial_k0[win_id] = k0_full

        result[trial_id] = trial_result
        delays_result[trial_id] = trial_delays
        k0_result[trial_id] = trial_k0

        total_wins = sum(len(v) for v in result.values())
        logger.info(
            "process_subject_dspte [%s]: %s → %d windows  (total so far: %d)",
            subject_id, trial_id, len(trial_result), total_wins,
        )

    # ------------------------------------------------------------------
    # Step 4 — Save per-subject .npz
    # ------------------------------------------------------------------
    elapsed_subject = time.perf_counter() - t_subject_start

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
    }

    _save_subject_dspte_npz(
        subject_id=subject_id,
        result=result,
        delays_result=delays_result,
        k0_result=k0_result,
        valid_roi_names=valid_roi_names,
        params=params,
        config=config,
    )

    total_wins = sum(len(v) for v in result.values())
    logger.info(
        "process_subject_dspte [%s]: complete.  "
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
# Private helper — save per-subject .npz
# ---------------------------------------------------------------------------

def _save_subject_dspte_npz(
    subject_id: str,
    result: dict,
    delays_result: dict,
    k0_result: dict,
    valid_roi_names: list,
    params: dict,
    config: Optional[dict] = None,
) -> Path:
    """
    Save ``process_subject_dspte`` output to
    ``phase_1_connectivity/sub-{id}_dspte_connectivity.npz``.

    Schema
    ------
    Tr_N_connectivity : (20, n_wins)     — directed TE connectivity vectors
    Tr_N_delays       : (20, n_wins)     — optimal delay (samples) per connection
    Tr_N_k0_values    : (5, 5, n_wins)   — k₀ per ROI pair per window (diagnostic)
    subject_id        : str
    group             : 'Expert' or 'Novice'
    trial_ids         : str array
    window_counts     : int array
    roi_names_used    : str array
    connection_labels : str array (20 labels)
    n_trials          : int
    params_json       : JSON string of processing parameters

    Returns the absolute path to the saved file.
    """
    from utils_te import get_te_output_dir, get_subject_group, get_connection_labels  # noqa

    out_dir = get_te_output_dir("phase_1_connectivity", config=config)

    trial_ids = sorted(result.keys(), key=lambda k: int(k.split("_")[1]))
    window_counts = []
    save_dict: dict = {}

    n_conn = 20
    n_rois_full = _N_CANONICAL

    for trial_id in trial_ids:
        trial_data = result[trial_id]
        trial_delays = delays_result.get(trial_id, {})
        trial_k0 = k0_result.get(trial_id, {})
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

        else:
            conn_array = np.empty((n_conn, 0), dtype=np.float64)
            delay_array = np.empty((n_conn, 0), dtype=np.float64)
            k0_array = np.empty((n_rois_full, n_rois_full, 0), dtype=np.float64)

        save_dict[f"{trial_id}_connectivity"] = conn_array
        save_dict[f"{trial_id}_delays"] = delay_array
        save_dict[f"{trial_id}_k0_values"] = k0_array

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

    npz_path = out_dir / f"sub-{subject_id}_dspte_connectivity.npz"
    np.savez(npz_path, **save_dict)

    logger.info(
        "_save_subject_dspte_npz [%s]: saved → %s  (trials=%d, total_wins=%d)",
        subject_id, npz_path.name, len(trial_ids), sum(window_counts),
    )
    return npz_path


# ---------------------------------------------------------------------------
# Private helper — load per-subject .npz
# ---------------------------------------------------------------------------

def _load_subject_dspte(
    subject_id: str,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """
    Load ``sub-{id}_dspte_connectivity.npz`` and return the nested result dict.

    Returns a dict ``{trial_id: {win_id: conn_vec_20}}`` matching the return
    format of ``process_subject_dspte()``.  Returns None if file not found.
    """
    from utils_te import get_te_output_dir  # noqa

    out_dir = get_te_output_dir("phase_1_connectivity", config=config)
    npz_path = out_dir / f"sub-{subject_id}_dspte_connectivity.npz"

    if not npz_path.exists():
        logger.warning(
            "_load_subject_dspte [%s]: file not found: %s", subject_id, npz_path
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
        "_load_subject_dspte [%s]: loaded from cache — "
        "%d trials, %d total windows",
        subject_id,
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


# ===========================================================================
# Section 4.2.3 — Batch Runner
# ===========================================================================

def run_batch_dspte(
    study: str = "expert_novice",
    subjects: Optional[list[str]] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Batch-process all subjects and write te_batch_log.json.

    Runs ``process_subject_dspte()`` for every subject in the cohort
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
        "run_batch_dspte: processing %d subjects: %s", len(subjects), subjects
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
        logger.info("run_batch_dspte: === Subject %s ===", subject_id)

        try:
            result = process_subject_dspte(
                subject_id, study=study, config=config, overwrite=overwrite
            )

            if result is None:
                raise RuntimeError("process_subject_dspte returned None")

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
                mean_dspte = float(all_conn_arr.mean())
                max_dspte = float(all_conn_arr.max())
                # Per-connection mean (across all windows × trials)
                conn_means = all_conn_arr.mean(axis=0).tolist()
            else:
                mean_dspte = 0.0
                max_dspte = 0.0
                conn_means = [0.0] * 20

            # Load k₀ stats from the saved .npz
            from utils_te import get_te_output_dir  # noqa
            out_dir = get_te_output_dir("phase_1_connectivity", config=config)
            npz_path = out_dir / f"sub-{subject_id}_dspte_connectivity.npz"
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
                "mean_dspte": round(mean_dspte, 6),
                "max_dspte": round(max_dspte, 6),
                "mean_k0": round(mean_k0, 2),
                "mean_optimal_delay_samples": round(mean_delay, 2),
                "mean_optimal_delay_s": round(mean_delay / 25.0, 3)
                    if not np.isnan(mean_delay) else float("nan"),
                "per_connection_mean_dspte": [round(v, 6) for v in conn_means],
                "connection_labels": conn_labels,
                "elapsed_s": round(elapsed_subj, 1),
            }
            n_success += 1

            logger.info(
                "run_batch_dspte [%s]: OK — trials=%d  wins=%d  "
                "mean_dspte=%.4f  mean_k0=%.1f  elapsed=%.1f s",
                subject_id, n_trials, n_wins_total, mean_dspte, mean_k0, elapsed_subj,
            )

        except Exception as exc:
            elapsed_subj = time.perf_counter() - subj_start
            logger.error(
                "run_batch_dspte [%s]: FAILED — %s  (elapsed=%.1f s)",
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
        "run_batch_dspte: complete.  OK=%d  FAILED=%d  Total=%.1f s  "
        "Log → %s",
        n_success, n_failed, elapsed_total, log_path,
    )
    return batch_log


# ===========================================================================
# Smoke tests (run as script)
# ===========================================================================

def _run_tests() -> None:
    """
    Section 4.2 smoke tests.

    4.2.1/4.2.2 — process_subject_dspte('A1'):
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
    print("Running process_subject_dspte('A1') with config defaults.")
    print("This may take several minutes (u_max × 20 pairs × n_windows).\n")

    config = load_config()

    # Optionally reduce u_max for faster validation
    # Uncomment the following line to speed up the test at the cost of
    # reduced lag search range:
    # config["stage_e_transfer_entropy"]["signal"]["u_max"] = 10

    t0 = time.perf_counter()
    result = process_subject_dspte("A1", config=config, overwrite=True)
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
    npz_path = out_dir / "sub-A1_dspte_connectivity.npz"
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
    result_cached = process_subject_dspte("A1", config=config, overwrite=False)
    elapsed_cache = time.perf_counter() - t_cache
    assert result_cached is not None, "Cache load returned None"
    assert set(result_cached.keys()) == set(result.keys()), "Trial keys mismatch"
    print(f"      Cache load elapsed: {elapsed_cache:.2f} s  "
          f"(expected << {elapsed:.1f} s)  ✓")

    print("\n=== All Section 4.2 smoke tests passed. ===\n")


if __name__ == "__main__":
    _run_tests()
