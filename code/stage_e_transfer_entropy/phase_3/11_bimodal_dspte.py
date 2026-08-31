"""
Stage E — Phase 3: Bimodal dSPTE (Sections 8.1, 8.2)

Implements:
  - 8.1.1 `symbolise_amplitude_pair`
  - 8.1.2 `compute_amplitude_dspte_matrix`   (bivariate; kept as baseline reference)
  - 8.2.1 `compute_bimodal_dspte_matrix`
  - 8.2.2 `process_subject_bimodal_dspte` + `run_batch_bimodal_dspte`

Conditioning design (both branches are now multivariate)
---------------------------------------------------------
Phase branch   — `phase_2/06_multivariate_dspte.compute_multivariate_dspte_matrix()`:
    Multivariate conditional phase dSPTE with PCA-compressed conditioning on
    the n-2 remaining ROIs' phase symbols.

Amplitude branch — `phase_4/14_amplitude_dste.compute_amplitude_conditional_dste_matrix()`:
    Multivariate conditional amplitude dSTE.  Applies Hilbert amplitude +
    z-score internally (no narrowband filter), fits a shared Neural-Gas
    codebook on all ROIs, then conditions each (src, tgt) pair on the
    PCA-compressed amplitude symbols of the remaining n-2 ROIs — identical
    structural design to the phase branch.

    `compute_amplitude_dspte_matrix()` (bivariate pairwise) is retained in
    this file as a legacy baseline for diagnostic comparisons.
"""

from __future__ import annotations

import argparse
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
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

logger = logging.getLogger(__name__)

_CORE_MOD = None
_PIPELINE_MOD = None
_MULTIVARIATE_MOD = None
_SEG_MOD = None
_P4_MOD = None


def _load_module(name: str, abs_path: "Path"):
    """Load a module from an absolute path."""
    spec = importlib.util.spec_from_file_location(name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_core_module():
    """Return cached import of `phase_1/01_bivariate_dspte_core.py`."""
    global _CORE_MOD
    if _CORE_MOD is None:
        _CORE_MOD = _load_module("bivariate_dspte_core_bimodal", _STAGE_E_ROOT / "phase_1" / "01_bivariate_dspte_core.py")
    return _CORE_MOD


def _get_pipeline_module():
    """Return cached import of `phase_1/02_bivariate_dspte_pipeline.py`."""
    global _PIPELINE_MOD
    if _PIPELINE_MOD is None:
        _PIPELINE_MOD = _load_module("bivariate_dspte_pipeline_bimodal", _STAGE_E_ROOT / "phase_1" / "02_bivariate_dspte_pipeline.py")
    return _PIPELINE_MOD


def _get_multivariate_module():
    """Return cached import of `phase_2/06_multivariate_dspte.py`."""
    global _MULTIVARIATE_MOD
    if _MULTIVARIATE_MOD is None:
        _MULTIVARIATE_MOD = _load_module("multivariate_dspte_bimodal", _STAGE_E_ROOT / "phase_2" / "06_multivariate_dspte.py")
    return _MULTIVARIATE_MOD


def _get_seg_module():
    """Return cached import of Stage D trial segmentation helper."""
    global _SEG_MOD
    if _SEG_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "trial_segmentation_bimodal",
            _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SEG_MOD = mod
    return _SEG_MOD


def _get_p4_module():
    """Return cached import of `phase_4/14_amplitude_dste.py`.

    Provides `compute_amplitude_conditional_dste_matrix()` — the multivariate
    conditional amplitude dSTE function that replaces the bivariate
    `compute_amplitude_dspte_matrix()` in both branches of this module.
    """
    global _P4_MOD
    if _P4_MOD is None:
        _P4_MOD = _load_module(
            "amplitude_dste_p4_bimodal",
            _STAGE_E_ROOT / "phase_4" / "14_amplitude_dste.py",
        )
    return _P4_MOD


def _load_config_if_none(config: Optional[dict]) -> dict:
    """Load `config.yaml` when no config dict is supplied."""
    if config is not None:
        return config
    from utils import load_config
    return load_config()


def _normalise_weights(w_phase: float, w_amplitude: float) -> tuple[float, float]:
    """
    Normalise non-negative fusion weights to sum to 1.

    Returns
    -------
    (float, float)
        `(w_phase_norm, w_amplitude_norm)`.
    """
    wp = float(max(0.0, w_phase))
    wa = float(max(0.0, w_amplitude))
    s = wp + wa
    if s <= 0:
        return 0.5, 0.5
    return wp / s, wa / s


def _get_phase3_fixed_weights(config: dict) -> tuple[float, float]:
    """Read fixed Phase 3 weights from config and normalise them."""
    p3_cfg = config.get("stage_e_transfer_entropy", {}).get("phase_3", {})
    wp = float(p3_cfg.get("w_phase", 0.7))
    wa = float(p3_cfg.get("w_amplitude", 0.3))
    return _normalise_weights(wp, wa)


def _to_jsonable(obj):
    """Convert numpy/scalar containers to JSON-serialisable Python objects."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def _load_subject_phase3_params(
    subject_id: str,
    config: dict,
) -> dict:
    """
    Load `params_json` metadata from a saved fused Phase 3 subject file.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., `A1`).
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    dict
        Parsed params dictionary, or empty dict when unavailable.
    """
    from stage_e_transfer_entropy.utils_te import get_te_output_dir

    conn_dir = get_te_output_dir("phase_3_connectivity", config=config)
    npz_path = conn_dir / f"sub-{subject_id}_bimodal_dspte_connectivity.npz"
    if not npz_path.exists():
        return {}
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "params_json" not in data:
            return {}
        raw = data["params_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        return json.loads(str(raw))
    except Exception:
        return {}


def _write_bimodal_run_parameters_json(
    *,
    study: str,
    subjects: list[str],
    overwrite: bool,
    config: dict,
    batch_log: dict,
    out_path: Path,
) -> Path:
    """
    Write one JSON file describing parameters used in the current Phase 3 run.

    Parameters
    ----------
    study : str
        Study name used for this batch run.
    subjects : list[str]
        Requested subjects for this run.
    overwrite : bool
        Whether cached subject outputs were overwritten.
    config : dict
        Parsed pipeline configuration.
    batch_log : dict
        Batch summary produced by `run_batch_bimodal_dspte`.
    out_path : Path
        Output file path for run parameters JSON.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    te_cfg = config.get("stage_e_transfer_entropy", {})
    p3_cfg = te_cfg.get("phase_3", {})
    sig_cfg = te_cfg.get("signal", {})
    win_cfg = te_cfg.get("window", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})

    wp_fixed, wa_fixed = _get_phase3_fixed_weights(config)
    per_subject: dict = {}
    for sid in subjects:
        subj_info = batch_log.get("subjects", {}).get(sid, {})
        subj_params = _load_subject_phase3_params(sid, config)
        per_subject[sid] = {
            "status": subj_info.get("status", "unknown"),
            "w_phase": subj_params.get("w_phase"),
            "w_amplitude": subj_params.get("w_amplitude"),
            "fusion_mode": subj_params.get("fusion_mode"),
            "optimisation": subj_params.get("optimisation"),
            "n_trials": subj_info.get("n_trials"),
            "n_windows_total": subj_info.get("n_windows_total"),
        }

    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "phase": "phase_3_bimodal_dspte",
        "study": study,
        "overwrite": bool(overwrite),
        "subjects_requested": subjects,
        "batch_summary": batch_log.get("summary", {}),
        "fusion_configuration": {
            "fusion_mode": str(p3_cfg.get("fusion_mode", "fixed")).lower(),
            "configured_w_phase": float(p3_cfg.get("w_phase", 0.7)),
            "configured_w_amplitude": float(p3_cfg.get("w_amplitude", 0.3)),
            "normalised_fixed_w_phase": float(wp_fixed),
            "normalised_fixed_w_amplitude": float(wa_fixed),
            "pso_n_particles": p3_cfg.get("pso_n_particles"),
            "pso_iters": p3_cfg.get("pso_iters"),
            "pso_c1": p3_cfg.get("pso_c1"),
            "pso_c2": p3_cfg.get("pso_c2"),
            "pso_w": p3_cfg.get("pso_w"),
        },
        "signal_configuration": {
            "sfreq": sig_cfg.get("sfreq"),
            "embedding_dimension": sig_cfg.get("embedding_dimension"),
            "u_max": sig_cfg.get("u_max"),
        },
        "window_configuration": {
            "size_samples": win_cfg.get("size_samples"),
            "stride_samples": win_cfg.get("stride_samples"),
        },
        "narrowband_filter_configuration": {
            "enabled": filt_cfg.get("enabled"),
            "apply_to_phase": filt_cfg.get("apply_to_phase"),
            "apply_to_amplitude": filt_cfg.get("apply_to_amplitude"),
            "f_lo": filt_cfg.get("f_lo"),
            "f_hi": filt_cfg.get("f_hi"),
            "order": filt_cfg.get("order"),
        },
        "subjects": per_subject,
    }
    with open(out_path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    return out_path


def _compute_phase_window_and_taus(
    Y_window: np.ndarray,
    core_mod,
) -> tuple[np.ndarray, list[int]]:
    """Compute per-window phase and embedding delays from raw ROI windows."""
    n_rois, _ = Y_window.shape
    Phi = np.full_like(Y_window, np.nan)
    taus: list[int] = []
    for i in range(n_rois):
        row = Y_window[i]
        if np.all(np.isnan(row)) or np.any(~np.isfinite(row)):
            taus.append(1)
            continue
        ph = core_mod.extract_phase_signal(row)
        Phi[i] = ph
        taus.append(core_mod.compute_embedding_delay(ph))
    return Phi, taus


def _fuse_te_profiles(
    phase_profiles: dict,
    amp_profiles: dict,
    w_phase: float,
    w_amplitude: float,
    u_max: int,
) -> dict:
    """Build fused lag-TE profiles dictionary by weighted profile summation."""
    fused: dict = {}
    all_keys = set(phase_profiles.keys()) | set(amp_profiles.keys())
    for key in all_keys:
        p = np.asarray(phase_profiles.get(key, np.zeros(u_max)), dtype=np.float64)
        a = np.asarray(amp_profiles.get(key, np.zeros(u_max)), dtype=np.float64)
        if p.size < u_max:
            p2 = np.zeros(u_max, dtype=np.float64)
            p2[:p.size] = p
            p = p2
        if a.size < u_max:
            a2 = np.zeros(u_max, dtype=np.float64)
            a2[:a.size] = a
            a = a2
        fused[key] = (w_phase * p[:u_max]) + (w_amplitude * a[:u_max])
    return fused


def _delay_matrix_from_profiles(
    te_profiles: dict,
    n_rois: int,
) -> np.ndarray:
    """
    Build delay matrix by argmax over lag profiles.

    Returns
    -------
    np.ndarray
        Integer delay matrix `(n_rois, n_rois)`, zero on diagonal/missing keys.
    """
    delay_matrix = np.zeros((n_rois, n_rois), dtype=int)
    for (src, tgt), profile in te_profiles.items():
        if src == tgt:
            continue
        arr = np.asarray(profile, dtype=np.float64)
        if arr.size == 0:
            continue
        delay_matrix[src, tgt] = int(np.argmax(arr)) + 1
    return delay_matrix


def _extract_canonical_lag_profiles(
    te_profiles: dict,
    roi_names: list[str],
    conn_labels: list[str],
    u_max: int,
) -> np.ndarray:
    """Convert local lag-profile dict to canonical `(20, u_max)` profile array."""
    profiles = np.zeros((20, u_max), dtype=np.float64)
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
        n = min(arr.shape[0], u_max)
        profiles[row_idx, :n] = arr[:n]
    return profiles


def _matrix_directionality_score(ec_matrix: np.ndarray) -> float:
    """
    Compute mean absolute antisymmetry score for one EC matrix.

    Score = mean(|EC[i,j] - EC[j,i]|) over upper-triangle pairs.
    """
    n = ec_matrix.shape[0]
    if n < 2:
        return 0.0
    diff = np.abs(ec_matrix - ec_matrix.T)
    tri = np.triu_indices(n, k=1)
    vals = diff[tri]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))


def optimise_bimodal_weights(
    phase_mats: list[np.ndarray],
    amplitude_mats: list[np.ndarray],
    config: Optional[dict] = None,
) -> tuple[float, float, dict]:
    """
    Optimise fusion weights using PSO on training windows.

    Objective: maximise mean directionality score across training fused matrices.
    """
    config = _load_config_if_none(config)
    wp_fixed, wa_fixed = _get_phase3_fixed_weights(config)

    if len(phase_mats) == 0 or len(amplitude_mats) == 0:
        return wp_fixed, wa_fixed, {
            "status": "fallback_fixed",
            "reason": "no_training_windows",
        }

    p3_cfg = config.get("stage_e_transfer_entropy", {}).get("phase_3", {})
    n_particles = int(p3_cfg.get("pso_n_particles", 16))
    iters = int(p3_cfg.get("pso_iters", 40))
    options = {
        "c1": float(p3_cfg.get("pso_c1", 0.5)),
        "c2": float(p3_cfg.get("pso_c2", 0.3)),
        "w": float(p3_cfg.get("pso_w", 0.9)),
    }

    try:
        from utils import configure_pyswarms_logging
        configure_pyswarms_logging()  # redirect pyswarms' report.log into report_logs/
        import pyswarms as ps
    except Exception as exc:
        logger.warning("optimise_bimodal_weights: pyswarms unavailable: %s", exc)
        return wp_fixed, wa_fixed, {
            "status": "fallback_fixed",
            "reason": "pyswarms_unavailable",
        }

    phase_mats_np = [np.asarray(m, dtype=np.float64) for m in phase_mats]
    amp_mats_np = [np.asarray(m, dtype=np.float64) for m in amplitude_mats]

    def objective(x: np.ndarray) -> np.ndarray:
        """PSO objective: negative mean directionality score per particle weight."""
        costs = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            w = float(np.clip(x[i, 0], 0.0, 1.0))
            score_vals = []
            for p, a in zip(phase_mats_np, amp_mats_np):
                fused = w * p + (1.0 - w) * a
                score_vals.append(_matrix_directionality_score(fused))
            mean_score = float(np.nanmean(score_vals)) if score_vals else 0.0
            costs[i] = -mean_score  # minimize negative score == maximize score
        return costs

    bounds = (np.array([0.0]), np.array([1.0]))
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=1,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(objective, iters=iters, verbose=False)

    wp = float(np.clip(best_pos[0], 0.0, 1.0))
    wa = 1.0 - wp
    return wp, wa, {
        "status": "optimised",
        "objective_mean_abs_asymmetry": float(-best_cost),
        "n_training_windows": len(phase_mats_np),
        "n_particles": n_particles,
        "iters": iters,
    }


def symbolise_amplitude_pair(
    amp_x: np.ndarray,
    amp_y: np.ndarray,
    d: int,
    tau: int,
    k0_max: int = 20,
    ndis_threshold: float = 0.05,
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Symbolise a pair of amplitude signals with Neural-Gas.

    Parameters
    ----------
    amp_x, amp_y : np.ndarray
        1D amplitude signals with equal length.
    d, tau, k0_max, ndis_threshold : int/float
        Embedding and symbolisation hyperparameters.

    Returns
    -------
    tuple
        `(S_x, S_y, meta)` matching Phase 1 `symbolise_pair` schema.
    """
    core = _get_core_module()

    amp_x = np.asarray(amp_x, dtype=float)
    amp_y = np.asarray(amp_y, dtype=float)
    if amp_x.ndim != 1 or amp_y.ndim != 1:
        raise ValueError(
            "symbolise_amplitude_pair: both inputs must be 1D arrays. "
            f"Got shapes {amp_x.shape} and {amp_y.shape}."
        )
    if len(amp_x) != len(amp_y):
        raise ValueError(
            f"symbolise_amplitude_pair: amp_x (len={len(amp_x)}) and "
            f"amp_y (len={len(amp_y)}) must have the same length."
        )
    if np.any(~np.isfinite(amp_x)) or np.any(~np.isfinite(amp_y)):
        raise ValueError("symbolise_amplitude_pair: amplitude inputs contain NaN/Inf.")
    if np.min(amp_x) < -1e-12 or np.min(amp_y) < -1e-12:
        logger.warning(
            "symbolise_amplitude_pair: negative amplitude values detected "
            "(min_x=%.3e, min_y=%.3e).",
            float(np.min(amp_x)), float(np.min(amp_y)),
        )

    return core.symbolise_pair(
        amp_x,
        amp_y,
        d=d,
        tau=tau,
        k0_max=k0_max,
        ndis_threshold=ndis_threshold,
        config=config,
    )


def compute_amplitude_dspte_matrix(
    Y_window: np.ndarray,
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute amplitude-branch bivariate dSPTE matrix for one ROI window.

    Returns
    -------
    tuple
        `(EC_matrix, delay_matrix, metadata)` with Phase 1-compatible keys.
    """
    config = _load_config_if_none(config)
    core = _get_core_module()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    sym_cfg = te_cfg.get("symbolisation", {})

    d = int(sig_cfg.get("embedding_dimension", 3))
    u_max = int(sig_cfg.get("u_max", 50))
    ndis_threshold = float(sym_cfg.get("ndis_threshold", 0.05))
    k0_max = int(sym_cfg.get("k0_max_bivariate", 20))

    Y_window = np.asarray(Y_window, dtype=float)
    if Y_window.ndim != 2:
        raise ValueError(
            f"compute_amplitude_dspte_matrix: expected 2D input (n_rois, T), got {Y_window.shape}"
        )

    n_rois, _ = Y_window.shape
    EC_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    delay_matrix = np.zeros((n_rois, n_rois), dtype=int)
    k0_mat = np.zeros((n_rois, n_rois), dtype=int)
    ndis_mat = np.zeros((n_rois, n_rois), dtype=np.float64)
    tau_mat = np.zeros((n_rois, n_rois), dtype=int)
    te_profiles: dict = {}
    skipped_pairs: list = []

    amplitudes: list[Optional[np.ndarray]] = []
    taus: list[int] = []
    for roi_idx in range(n_rois):
        row = Y_window[roi_idx]
        if np.all(np.isnan(row)) or np.any(~np.isfinite(row)):
            amplitudes.append(None)
            taus.append(1)
            skipped_pairs.extend([(roi_idx, j) for j in range(n_rois) if j != roi_idx])
            continue
        amp = core.extract_amplitude_signal(row)
        tau_i = core.compute_embedding_delay(amp)
        amplitudes.append(amp)
        taus.append(tau_i)

    for src in range(n_rois):
        for tgt in range(n_rois):
            if src == tgt:
                continue
            if amplitudes[src] is None or amplitudes[tgt] is None:
                if (src, tgt) not in skipped_pairs:
                    skipped_pairs.append((src, tgt))
                continue

            tau_shared = min(taus[src], taus[tgt])
            S_src, S_tgt, sym_meta = symbolise_amplitude_pair(
                amplitudes[src],
                amplitudes[tgt],
                d=d,
                tau=tau_shared,
                k0_max=k0_max,
                ndis_threshold=ndis_threshold,
                config=config,
            )
            k0 = int(sym_meta["k0"])
            max_te, opt_delay, te_profile = core.compute_bivariate_dspte_multidelay(
                S_src, S_tgt, k0=k0, u_max=u_max
            )

            EC_matrix[src, tgt] = max_te
            delay_matrix[src, tgt] = opt_delay
            k0_mat[src, tgt] = k0
            ndis_mat[src, tgt] = float(sym_meta["nDis"])
            tau_mat[src, tgt] = tau_shared
            te_profiles[(src, tgt)] = te_profile

    metadata = {
        "k0_matrix": k0_mat,
        "ndis_matrix": ndis_mat,
        "tau_matrix": tau_mat,
        "te_profiles": te_profiles,
        "skipped_pairs": sorted(set(skipped_pairs)),
    }
    return EC_matrix, delay_matrix, metadata


def compute_bimodal_dspte_matrix(
    Y_window: np.ndarray,
    config: Optional[dict] = None,
    Phi_win: Optional[np.ndarray] = None,
    taus: Optional[list[int]] = None,
    roi_names: Optional[list[str]] = None,
    weights: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute fused bimodal TE matrix for one window (Section 8.2.1).

    Both branches use multivariate conditioning:
      - Phase branch: `compute_multivariate_dspte_matrix()` conditions on the n-2
        remaining ROIs' phase symbols (PCA-compressed).
      - Amplitude branch: `compute_amplitude_conditional_dste_matrix()`
        conditions on the n-2 remaining ROIs' amplitude symbols (PCA-compressed).

    Fusion:
      `EC_final = w_phase * EC_phase + w_amplitude * EC_amplitude`
    """
    config = _load_config_if_none(config)
    core = _get_core_module()
    multivariate = _get_multivariate_module()
    p4 = _get_p4_module()

    Y_window = np.asarray(Y_window, dtype=np.float64)
    if Y_window.ndim != 2:
        raise ValueError(
            f"compute_bimodal_dspte_matrix: expected shape (n_rois, T), got {Y_window.shape}"
        )
    n_rois, _ = Y_window.shape

    if roi_names is None:
        roi_names = [f"ROI_{i}" for i in range(n_rois)]

    if Phi_win is None or taus is None:
        Phi_win, taus = _compute_phase_window_and_taus(Y_window, core)
    else:
        Phi_win = np.asarray(Phi_win, dtype=np.float64)
        if Phi_win.shape != Y_window.shape:
            raise ValueError(
                f"compute_bimodal_dspte_matrix: Phi_win shape {Phi_win.shape} != Y_window shape {Y_window.shape}"
            )

    phase_ec, _phase_delay, phase_meta = multivariate.compute_multivariate_dspte_matrix(
        Phi_win=Phi_win,
        taus=taus,
        config=config,
        roi_names=roi_names,
    )
    # Amplitude branch: multivariate conditional dSTE (Phase 4).
    # Y_window is raw broadband HbO — Phase 4 applies Hilbert amplitude internally.
    amp_ec, _amp_delay, amp_meta = p4.compute_amplitude_conditional_dste_matrix(
        A_win=Y_window,
        config=config,
        roi_names=roi_names,
    )

    if weights is None:
        w_phase, w_amplitude = _get_phase3_fixed_weights(config)
    else:
        w_phase, w_amplitude = _normalise_weights(weights[0], weights[1])

    ec_fused = (w_phase * phase_ec) + (w_amplitude * amp_ec)
    te_cfg = config.get("stage_e_transfer_entropy", {})
    u_max = int(te_cfg.get("signal", {}).get("u_max", 50))
    fused_profiles = _fuse_te_profiles(
        phase_meta.get("te_profiles", {}),
        amp_meta.get("te_profiles", {}),
        w_phase=w_phase,
        w_amplitude=w_amplitude,
        u_max=u_max,
    )
    delay_fused = _delay_matrix_from_profiles(fused_profiles, n_rois=n_rois)

    metadata = {
        "w_phase": float(w_phase),
        "w_amplitude": float(w_amplitude),
        "phase_metadata": phase_meta,
        "amplitude_metadata": amp_meta,
        "phase_ec_matrix": phase_ec,
        "amplitude_ec_matrix": amp_ec,
        "te_profiles": fused_profiles,
        "skipped_pairs": sorted(
            set(phase_meta.get("skipped_pairs", [])) | set(amp_meta.get("skipped_pairs", []))
        ),
    }
    return ec_fused, delay_fused, metadata


def _filter_and_extract_phase_trial_bimodal(
    Y_trial: np.ndarray,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    order: int,
    core_mod,
) -> tuple[np.ndarray, list[int]]:
    """
    Filter full trial and extract phase/taus once for all sliding windows.

    Returns
    -------
    tuple
        `(Phi_trial, taus_trial)`.
    """
    n_rois, _ = Y_trial.shape
    Phi_trial = np.full_like(Y_trial, np.nan, dtype=np.float64)
    taus_trial: list[int] = []

    for roi_idx in range(n_rois):
        row = Y_trial[roi_idx]
        if np.all(np.isnan(row)) or np.any(~np.isfinite(row)):
            taus_trial.append(1)
            continue
        x_filt = core_mod.narrowband_filter(
            row, f_lo=f_lo, f_hi=f_hi, sfreq=sfreq, order=order
        )
        ph = core_mod.extract_phase_signal(x_filt)
        Phi_trial[roi_idx] = ph
        taus_trial.append(core_mod.compute_embedding_delay(ph))

    return Phi_trial, taus_trial


def _filter_trial_for_amplitude(
    Y_trial: np.ndarray,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    order: int,
    core_mod,
) -> np.ndarray:
    """
    Filter full trial once for the amplitude branch, then slice by windows.

    Parameters
    ----------
    Y_trial : np.ndarray
        ROI trial matrix of shape `(n_rois, T)`.
    sfreq : float
        Sampling frequency in Hz.
    f_lo : float
        Lower cutoff frequency in Hz.
    f_hi : float
        Upper cutoff frequency in Hz.
    order : int
        Butterworth filter order.
    core_mod : module
        Imported Phase 1 core module exposing `narrowband_filter`.

    Returns
    -------
    np.ndarray
        Filtered trial matrix with same shape as `Y_trial`.
    """
    Y_filt = np.full_like(Y_trial, np.nan, dtype=np.float64)
    for roi_idx in range(Y_trial.shape[0]):
        row = Y_trial[roi_idx]
        if np.all(np.isnan(row)) or np.any(~np.isfinite(row)):
            continue
        Y_filt[roi_idx] = core_mod.narrowband_filter(
            row, f_lo=f_lo, f_hi=f_hi, sfreq=sfreq, order=order
        )
    return Y_filt


def _save_subject_phase3_branch_npz(
    subject_id: str,
    result: dict,
    delays_result: dict,
    profiles_result: Optional[dict],
    valid_roi_names: list[str],
    config: dict,
    filename: str,
    params: dict,
) -> Path:
    """Save one Phase 3 branch .npz file with Stage-D-compatible connectivity keys."""
    from stage_e_transfer_entropy.utils_te import (
        get_connection_labels,
        get_subject_group,
        get_te_output_dir,
    )

    out_dir = get_te_output_dir("phase_3_connectivity", config=config)
    conn_labels = get_connection_labels()
    trial_ids = sorted(result.keys(), key=lambda k: int(k.split("_")[1]))
    save_dict: dict = {}
    window_counts = []
    u_max = int(config.get("stage_e_transfer_entropy", {}).get("signal", {}).get("u_max", 50))

    for trial_id in trial_ids:
        trial_data = result.get(trial_id, {})
        trial_delays = delays_result.get(trial_id, {})
        trial_profiles = profiles_result.get(trial_id, {}) if profiles_result is not None else {}
        n_wins = len(trial_data)
        window_counts.append(n_wins)

        if n_wins > 0:
            wins_sorted = sorted(trial_data.keys(), key=lambda k: int(k.split("_")[1]))
            conn_array = np.column_stack([trial_data[w] for w in wins_sorted]).astype(np.float64)
            delay_array = np.column_stack(
                [trial_delays.get(w, np.zeros(20)) for w in wins_sorted]
            ).astype(np.float64)
            if profiles_result is not None:
                profiles = [
                    np.asarray(trial_profiles.get(w, np.zeros((20, u_max))), dtype=np.float64)
                    for w in wins_sorted
                ]
                profiles_array = np.stack(profiles, axis=2)
            else:
                profiles_array = np.empty((20, u_max, 0), dtype=np.float64)
        else:
            conn_array = np.empty((20, 0), dtype=np.float64)
            delay_array = np.empty((20, 0), dtype=np.float64)
            profiles_array = np.empty((20, u_max, 0), dtype=np.float64)

        save_dict[f"{trial_id}_connectivity"] = conn_array
        save_dict[f"{trial_id}_delays"] = delay_array
        if profiles_result is not None:
            save_dict[f"{trial_id}_te_profiles"] = profiles_array

    save_dict["subject_id"] = np.array(subject_id)
    save_dict["trial_ids"] = np.array(trial_ids)
    save_dict["window_counts"] = np.array(window_counts, dtype=int)
    save_dict["n_trials"] = np.array(len(trial_ids), dtype=int)
    save_dict["roi_names_used"] = np.array(valid_roi_names)
    save_dict["connection_labels"] = np.array(conn_labels)
    save_dict["params_json"] = np.array(json.dumps(params))
    try:
        group = get_subject_group(subject_id, config=config)
    except ValueError:
        group = "Unknown"
    save_dict["group"] = np.array(group)

    out_path = out_dir / filename
    np.savez(out_path, **save_dict)
    return out_path


def _load_subject_branch_npz(npz_path: Path) -> Optional[dict]:
    """Load a saved subject branch .npz as `{trial_id: {win_id: vec20}}`."""
    if not npz_path.exists():
        return None
    data = dict(np.load(npz_path, allow_pickle=True))
    trial_ids = list(data["trial_ids"])
    result: dict = {}
    for trial_id in trial_ids:
        conn = data.get(f"{trial_id}_connectivity")
        trial_result: dict = {}
        if conn is not None and conn.ndim == 2 and conn.shape[1] > 0:
            for i in range(conn.shape[1]):
                trial_result[f"Win_{i + 1}"] = conn[:, i]
        result[trial_id] = trial_result
    return result


def process_subject_bimodal_dspte(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Run full Phase 3 bimodal subject pipeline (Section 8.2.2).

    Saves three files:
      - `sub-*_bimodal_dspte_connectivity.npz` (fused)
      - `sub-*_phase_branch_connectivity.npz`
      - `sub-*_amplitude_branch_connectivity.npz`
    """
    config = _load_config_if_none(config)
    core = _get_core_module()
    multivariate = _get_multivariate_module()
    p4_mod = _get_p4_module()
    pipeline = _get_pipeline_module()
    seg_mod = _get_seg_module()
    from stage_e_transfer_entropy.utils_te import (
        get_connection_labels,
        get_te_output_dir,
        load_or_extract_roi_signals,
        load_preprocessed_fif,
        get_subject_group,
    )

    out_dir = get_te_output_dir("phase_3_connectivity", config=config)
    fused_path = out_dir / f"sub-{subject_id}_bimodal_dspte_connectivity.npz"
    phase_path = out_dir / f"sub-{subject_id}_phase_branch_connectivity.npz"
    amp_path = out_dir / f"sub-{subject_id}_amplitude_branch_connectivity.npz"

    if not overwrite and fused_path.exists() and phase_path.exists() and amp_path.exists():
        logger.info("process_subject_bimodal_dspte [%s]: cache hit.", subject_id)
        loaded = _load_subject_branch_npz(fused_path)
        return loaded if loaded is not None else {}

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    win_cfg = te_cfg.get("window", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})
    p3_cfg = te_cfg.get("phase_3", {})

    window_size = int(win_cfg.get("size_samples", 1350))
    stride = int(win_cfg.get("stride_samples", 1350))
    u_max = int(sig_cfg.get("u_max", 50))
    filter_enabled = bool(filt_cfg.get("enabled", True))
    # Master override behavior:
    # `narrowband_filter.enabled` hard-disables filtering for both branches.
    phase_filter_requested = bool(filt_cfg.get("apply_to_phase", True))
    amplitude_filter_requested = bool(filt_cfg.get("apply_to_amplitude", False))
    phase_filter_enabled = bool(filter_enabled and phase_filter_requested)
    amplitude_filter_enabled = bool(filter_enabled and amplitude_filter_requested)
    f_lo = float(filt_cfg.get("f_lo", 0.04))
    f_hi = float(filt_cfg.get("f_hi", 0.15))
    filt_order = int(filt_cfg.get("order", 3))

    t0_subj = time.perf_counter()
    conn_labels = get_connection_labels()
    Data_Reg, valid_roi_names, sfreq = load_or_extract_roi_signals(
        subject_id, study=study, config=config
    )
    raw = load_preprocessed_fif(subject_id, study=study, config=config)
    segments = seg_mod.get_trial_segments(raw, config=config)

    # Branch matrices stored first, fused after final weight selection.
    window_store: list[dict] = []
    phase_training: list[np.ndarray] = []
    amp_training: list[np.ndarray] = []

    for trial_idx, (onset, offset) in enumerate(segments):
        trial_id = f"Tr_{trial_idx + 1}"
        Y_trial = Data_Reg[:, onset: offset + 1]
        if Y_trial.shape[1] < (window_size + 1):
            continue

        if phase_filter_enabled:
            Phi_trial, taus_trial = _filter_and_extract_phase_trial_bimodal(
                Y_trial=Y_trial,
                sfreq=sfreq,
                f_lo=f_lo,
                f_hi=f_hi,
                order=filt_order,
                core_mod=core,
            )
        else:
            Phi_trial = None
            taus_trial = None

        if amplitude_filter_enabled:
            Y_trial_amp = _filter_trial_for_amplitude(
                Y_trial=Y_trial,
                sfreq=sfreq,
                f_lo=f_lo,
                f_hi=f_hi,
                order=filt_order,
                core_mod=core,
            )
        else:
            Y_trial_amp = Y_trial

        win_num = 1
        for w in range(window_size, Y_trial.shape[1], stride):
            ws = w - window_size
            we = w
            win_id = f"Win_{win_num}"
            win_num += 1
            Y_win = Y_trial[:, ws: we + 1]
            Y_win_amp = Y_trial_amp[:, ws: we + 1]

            if Phi_trial is not None and taus_trial is not None:
                Phi_win = Phi_trial[:, ws: we + 1]
                taus_win = taus_trial
            else:
                Phi_win, taus_win = _compute_phase_window_and_taus(Y_win, core)

            phase_ec, phase_delay, phase_meta = multivariate.compute_multivariate_dspte_matrix(
                Phi_win=Phi_win,
                taus=taus_win,
                config=config,
                roi_names=valid_roi_names,
            )
            # Amplitude branch: multivariate conditional dSTE (Phase 4).
            # Y_win_amp is raw broadband HbO (or narrowband-filtered when
            # amplitude_filter_enabled=True).  Phase 4 applies Hilbert amplitude
            # internally — do NOT pre-extract amplitudes before this call.
            amp_ec, amp_delay, amp_meta = p4_mod.compute_amplitude_conditional_dste_matrix(
                A_win=Y_win_amp,
                config=config,
                roi_names=valid_roi_names,
            )

            if trial_id in {"Tr_1", "Tr_2"}:
                phase_training.append(phase_ec)
                amp_training.append(amp_ec)

            window_store.append(
                {
                    "trial_id": trial_id,
                    "win_id": win_id,
                    "phase_ec": phase_ec,
                    "phase_delay": phase_delay,
                    "phase_meta": phase_meta,
                    "amp_ec": amp_ec,
                    "amp_delay": amp_delay,
                    "amp_meta": amp_meta,
                }
            )

    fusion_mode = str(p3_cfg.get("fusion_mode", "fixed")).lower()
    if fusion_mode == "optimise":
        w_phase, w_amp, opt_meta = optimise_bimodal_weights(
            phase_mats=phase_training,
            amplitude_mats=amp_training,
            config=config,
        )
    else:
        w_phase, w_amp = _get_phase3_fixed_weights(config)
        opt_meta = {"status": "fixed"}

    fused_result: dict = {}
    fused_delays: dict = {}
    fused_profiles: dict = {}
    phase_result: dict = {}
    phase_delays: dict = {}
    phase_profiles: dict = {}
    amp_result: dict = {}
    amp_delays: dict = {}
    amp_profiles: dict = {}

    for item in window_store:
        trial_id = item["trial_id"]
        win_id = item["win_id"]

        te_prof_fused = _fuse_te_profiles(
            item["phase_meta"].get("te_profiles", {}),
            item["amp_meta"].get("te_profiles", {}),
            w_phase=w_phase,
            w_amplitude=w_amp,
            u_max=u_max,
        )
        delay_fused = _delay_matrix_from_profiles(te_prof_fused, n_rois=item["phase_ec"].shape[0])
        ec_fused = w_phase * item["phase_ec"] + w_amp * item["amp_ec"]

        fused_vec = pipeline.extract_connectivity_vector(ec_fused, valid_roi_names)
        fused_delay_vec = pipeline.extract_connectivity_vector(delay_fused.astype(float), valid_roi_names)
        phase_vec = pipeline.extract_connectivity_vector(item["phase_ec"], valid_roi_names)
        phase_delay_vec = pipeline.extract_connectivity_vector(item["phase_delay"].astype(float), valid_roi_names)
        amp_vec = pipeline.extract_connectivity_vector(item["amp_ec"], valid_roi_names)
        amp_delay_vec = pipeline.extract_connectivity_vector(item["amp_delay"].astype(float), valid_roi_names)

        fused_result.setdefault(trial_id, {})[win_id] = fused_vec
        fused_delays.setdefault(trial_id, {})[win_id] = fused_delay_vec
        fused_profiles.setdefault(trial_id, {})[win_id] = _extract_canonical_lag_profiles(
            te_prof_fused, valid_roi_names, conn_labels, u_max
        )

        phase_result.setdefault(trial_id, {})[win_id] = phase_vec
        phase_delays.setdefault(trial_id, {})[win_id] = phase_delay_vec
        amp_result.setdefault(trial_id, {})[win_id] = amp_vec
        amp_delays.setdefault(trial_id, {})[win_id] = amp_delay_vec

    elapsed_subj = time.perf_counter() - t0_subj
    params_common = {
        "study": study,
        "phase": "phase_3_bimodal_dspte",
        "fusion_mode": fusion_mode,
        "w_phase": float(w_phase),
        "w_amplitude": float(w_amp),
        "optimisation": opt_meta,
        "phase_branch_mode": "multivariate_conditional",
        "amplitude_branch_mode": "multivariate_conditional",
        "window_size_samples": window_size,
        "stride_samples": stride,
        "u_max": u_max,
        "sfreq": float(sfreq),
        "narrowband_filter_enabled": filter_enabled,
        "phase_filter_enabled": phase_filter_enabled,
        "amplitude_filter_enabled": amplitude_filter_enabled,
        "narrowband_filter_f_lo": (
            f_lo if (phase_filter_enabled or amplitude_filter_enabled) else None
        ),
        "narrowband_filter_f_hi": (
            f_hi if (phase_filter_enabled or amplitude_filter_enabled) else None
        ),
        "narrowband_filter_order": (
            filt_order if (phase_filter_enabled or amplitude_filter_enabled) else None
        ),
        "elapsed_s": round(elapsed_subj, 2),
    }
    try:
        params_common["group"] = get_subject_group(subject_id, config=config)
    except ValueError:
        params_common["group"] = "Unknown"

    _save_subject_phase3_branch_npz(
        subject_id=subject_id,
        result=fused_result,
        delays_result=fused_delays,
        profiles_result=fused_profiles,
        valid_roi_names=valid_roi_names,
        config=config,
        filename=f"sub-{subject_id}_bimodal_dspte_connectivity.npz",
        params=params_common,
    )
    _save_subject_phase3_branch_npz(
        subject_id=subject_id,
        result=phase_result,
        delays_result=phase_delays,
        profiles_result=phase_profiles if phase_profiles else None,
        valid_roi_names=valid_roi_names,
        config=config,
        filename=f"sub-{subject_id}_phase_branch_connectivity.npz",
        params={**params_common, "branch": "phase"},
    )
    _save_subject_phase3_branch_npz(
        subject_id=subject_id,
        result=amp_result,
        delays_result=amp_delays,
        profiles_result=amp_profiles if amp_profiles else None,
        valid_roi_names=valid_roi_names,
        config=config,
        filename=f"sub-{subject_id}_amplitude_branch_connectivity.npz",
        params={**params_common, "branch": "amplitude"},
    )

    total_wins = sum(len(v) for v in fused_result.values())
    logger.info(
        "process_subject_bimodal_dspte [%s]: complete trials=%d windows=%d elapsed=%.1fs (w_phase=%.3f)",
        subject_id,
        len(fused_result),
        total_wins,
        elapsed_subj,
        w_phase,
    )
    return fused_result


def run_batch_bimodal_dspte(
    study: str = "expert_novice",
    subjects: Optional[list[str]] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Batch-process all subjects for Phase 3 bimodal connectivity (Section 8.2.2).

    Saves:
      - `phase_3_group/bimodal_summary.json`
      - `phase_3_group/bimodal_run_parameters.json`
    """
    config = _load_config_if_none(config)
    from stage_e_transfer_entropy.utils_te import get_te_output_dir

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    if subjects is None:
        subjects = list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))

    t0 = time.perf_counter()
    batch_log = {
        "run_timestamp": datetime.now().isoformat(),
        "phase": "phase_3_bimodal_dspte",
        "study": study,
        "n_subjects": len(subjects),
        "subjects": {},
    }

    for sid in subjects:
        t_sub = time.perf_counter()
        try:
            res = process_subject_bimodal_dspte(
                subject_id=sid,
                study=study,
                config=config,
                overwrite=overwrite,
            )
            n_wins = sum(len(v) for v in res.values())
            all_vecs = [v for tr in res.values() for v in tr.values()]
            if all_vecs:
                arr = np.stack(all_vecs, axis=0)
                mean_te = float(np.nanmean(arr))
                max_te = float(np.nanmax(arr))
            else:
                mean_te = 0.0
                max_te = 0.0
            batch_log["subjects"][sid] = {
                "status": "ok",
                "n_trials": len(res),
                "n_windows_total": n_wins,
                "mean_bimodal_te": mean_te,
                "max_bimodal_te": max_te,
                "elapsed_s": time.perf_counter() - t_sub,
            }
        except Exception as exc:
            batch_log["subjects"][sid] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": time.perf_counter() - t_sub,
            }
            logger.error("run_batch_bimodal_dspte [%s]: %s", sid, exc, exc_info=True)

    n_ok = sum(1 for v in batch_log["subjects"].values() if v.get("status") == "ok")
    batch_log["summary"] = {
        "n_success": n_ok,
        "n_failed": len(subjects) - n_ok,
        "total_elapsed_s": time.perf_counter() - t0,
    }

    group_dir = get_te_output_dir("phase_3_group", config=config)
    group_dir.mkdir(parents=True, exist_ok=True)
    summary_path = group_dir / "bimodal_summary.json"
    with open(summary_path, "w") as f:
        json.dump(batch_log, f, indent=2)
    logger.info("run_batch_bimodal_dspte: saved summary -> %s", summary_path)

    run_params_path = group_dir / "bimodal_run_parameters.json"
    _write_bimodal_run_parameters_json(
        study=study,
        subjects=subjects,
        overwrite=overwrite,
        config=config,
        batch_log=batch_log,
        out_path=run_params_path,
    )
    logger.info("run_batch_bimodal_dspte: saved run parameters -> %s", run_params_path)
    return batch_log


def _run_smoke_tests() -> None:
    """Run lightweight synthetic smoke tests for Sections 8.1 and 8.2.1."""
    cfg = _load_config_if_none(None)
    cfg_smoke = dict(cfg)
    cfg_smoke.setdefault("stage_e_transfer_entropy", {})
    cfg_smoke["stage_e_transfer_entropy"] = dict(cfg_smoke["stage_e_transfer_entropy"])
    sig_cfg = dict(cfg_smoke["stage_e_transfer_entropy"].get("signal", {}))
    sig_cfg["u_max"] = 5
    sig_cfg["embedding_dimension"] = 2
    cfg_smoke["stage_e_transfer_entropy"]["signal"] = sig_cfg
    p3_cfg = dict(cfg_smoke["stage_e_transfer_entropy"].get("phase_3", {}))
    p3_cfg["fusion_mode"] = "fixed"
    p3_cfg["w_phase"] = 0.6
    p3_cfg["w_amplitude"] = 0.4
    cfg_smoke["stage_e_transfer_entropy"]["phase_3"] = p3_cfg

    rng = np.random.default_rng(42)
    T = 600
    X = rng.standard_normal(T) * 0.2
    Y = rng.standard_normal(T) * 0.2
    Z = rng.standard_normal(T) * 0.2
    for t in range(3, T):
        X[t] += 0.7 * X[t - 1]
        Y[t] += 0.8 * X[t - 3]
        Z[t] += 0.5 * Z[t - 1]
    Y_win = np.vstack([X, Y, Z])

    core = _get_core_module()
    amp_x = core.extract_amplitude_signal(X)
    amp_y = core.extract_amplitude_signal(Y)
    S_x, S_y, meta = symbolise_amplitude_pair(
        amp_x, amp_y, d=2, tau=1, k0_max=10, ndis_threshold=0.05, config=cfg_smoke
    )
    assert S_x.shape == S_y.shape
    assert meta["k0"] >= 2

    ec_amp, delay_amp, m_amp = compute_amplitude_dspte_matrix(Y_win, config=cfg_smoke)
    assert ec_amp.shape == (3, 3)
    assert delay_amp.shape == (3, 3)
    assert len(m_amp["te_profiles"]) == 6

    ec_fused, delay_fused, m_fused = compute_bimodal_dspte_matrix(
        Y_window=Y_win,
        config=cfg_smoke,
        roi_names=["LPFC", "CPFC", "RPFC"],
    )
    assert ec_fused.shape == (3, 3)
    assert delay_fused.shape == (3, 3)
    assert np.all(np.diag(ec_fused) == 0.0)
    assert "phase_metadata" in m_fused and "amplitude_metadata" in m_fused
    # Amplitude branch should now come from Phase 4 multivariate conditioning.
    amp_meta_keys = set(m_fused["amplitude_metadata"].keys())
    assert "conditioning_mode" in amp_meta_keys, (
        "amplitude_metadata missing 'conditioning_mode' — Phase 4 multivariate path not active"
    )

    wp, wa, opt = optimise_bimodal_weights(
        phase_mats=[np.abs(ec_amp), np.abs(ec_amp * 0.8)],
        amplitude_mats=[np.abs(ec_amp * 0.3), np.abs(ec_amp * 0.4)],
        config=cfg_smoke,
    )
    assert abs((wp + wa) - 1.0) < 1e-8
    assert 0.0 <= wp <= 1.0

    print("Section 8.1/8.2 smoke tests PASS")
    print(
        f"amp_shape={ec_amp.shape}, fused_shape={ec_fused.shape}, "
        f"weights=({wp:.3f},{wa:.3f}), opt_status={opt.get('status')}"
    )


def _cli_main() -> None:
    """CLI entrypoint for bimodal Section 8.1/8.2 development workflows."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Phase 3 bimodal dSPTE")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke tests")
    group.add_argument("--subject", type=str, help="Run bimodal pipeline for one subject")
    group.add_argument("--batch", action="store_true", help="Run bimodal pipeline for all subjects")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    config = _load_config_if_none(None)

    if args.smoke_test:
        _run_smoke_tests()
        return

    if args.subject:
        res = process_subject_bimodal_dspte(
            subject_id=args.subject.strip(),
            config=config,
            overwrite=args.overwrite,
        )
        n_wins = sum(len(v) for v in res.values())
        logger.info("subject=%s trials=%d windows=%d", args.subject, len(res), n_wins)
        return

    if args.batch:
        summary = run_batch_bimodal_dspte(config=config, overwrite=args.overwrite)
        logger.info("batch done: %s", summary.get("summary", {}))


if __name__ == "__main__":
    _cli_main()
