"""
Stage E — Phase 3 Section 8.1: Amplitude Branch

Implements TRANSFER_ENTROPY_PROGRESS.md tasks:
  - 8.1.1 symbolise_amplitude_pair()
  - 8.1.2 compute_amplitude_dspte_matrix()

Design
------
This module intentionally mirrors Phase 1's output contract so downstream
subject/group/statistics code can consume amplitude-branch TE without schema
changes:
  return (EC_matrix, delay_matrix, metadata)
where metadata contains:
  k0_matrix, ndis_matrix, tau_matrix, te_profiles, skipped_pairs
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

logger = logging.getLogger(__name__)


def _load_core_module():
    """Load Stage E Phase 1 core module via importlib."""
    core_path = Path(__file__).parent / "01_dspte_core.py"
    spec = importlib.util.spec_from_file_location("dspte_core_amp", core_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    This is a thin amplitude-specific wrapper around Phase 1's `symbolise_pair`.
    Interface and return schema are intentionally identical.
    """
    core = _load_core_module()

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

    # Amplitude should be non-negative by construction of |hilbert(x)|; allow
    # tiny numerical negatives only if explicitly passed from external code.
    if np.min(amp_x) < -1e-12 or np.min(amp_y) < -1e-12:
        logger.warning(
            "symbolise_amplitude_pair: negative amplitude values detected "
            "(min_x=%.3e, min_y=%.3e). Proceeding with symbolisation.",
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
    Compute amplitude-branch dSPTE matrix for one ROI window.

    Steps:
    1. Extract instantaneous amplitude per ROI via `extract_amplitude_signal`.
    2. Compute per-ROI embedding delays on amplitude signals.
    3. Symbolise each directed ROI pair using `symbolise_amplitude_pair`.
    4. Run multi-delay TE search via Phase 1 `compute_dspte_multidelay`.

    Returns the same `(EC_matrix, delay_matrix, metadata)` schema as Phase 1
    `compute_dspte_matrix`.
    """
    if config is None:
        from utils import load_config
        config = load_config()

    core = _load_core_module()

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

    # Precompute amplitude and tau per ROI once per window.
    amplitudes: list[Optional[np.ndarray]] = []
    taus: list[int] = []
    for roi_idx in range(n_rois):
        row = Y_window[roi_idx]
        if np.all(np.isnan(row)):
            amplitudes.append(None)
            taus.append(1)
            logger.warning(
                "compute_amplitude_dspte_matrix: ROI %d is all-NaN — "
                "skipping all pairs involving this ROI.",
                roi_idx,
            )
            continue

        if np.any(~np.isfinite(row)):
            amplitudes.append(None)
            taus.append(1)
            logger.warning(
                "compute_amplitude_dspte_matrix: ROI %d contains NaN/Inf — "
                "skipping all pairs involving this ROI.",
                roi_idx,
            )
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

            max_te, opt_delay, te_profile = core.compute_dspte_multidelay(
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
        "skipped_pairs": skipped_pairs,
    }
    return EC_matrix, delay_matrix, metadata


def _run_smoke_tests() -> None:
    """Quick Section 8.1 smoke tests on synthetic signals."""
    from utils import load_config

    cfg = load_config()
    cfg_smoke = dict(cfg)
    cfg_smoke.setdefault("stage_e_transfer_entropy", {})
    cfg_smoke["stage_e_transfer_entropy"] = dict(cfg_smoke["stage_e_transfer_entropy"])
    sig_cfg = dict(cfg_smoke["stage_e_transfer_entropy"].get("signal", {}))
    sig_cfg["u_max"] = 5
    sig_cfg["embedding_dimension"] = 2
    cfg_smoke["stage_e_transfer_entropy"]["signal"] = sig_cfg

    # Synthetic 3-ROI data, one directed coupling X->Y with lag 3.
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

    # 8.1.1: symbolise_amplitude_pair
    core = _load_core_module()
    amp_x = core.extract_amplitude_signal(X)
    amp_y = core.extract_amplitude_signal(Y)
    S_x, S_y, meta = symbolise_amplitude_pair(
        amp_x, amp_y, d=2, tau=1, k0_max=10, ndis_threshold=0.05, config=cfg_smoke
    )
    assert S_x.shape == S_y.shape
    assert meta["k0"] >= 2
    assert np.min(S_x) >= 0 and np.max(S_x) < meta["k0"]
    assert np.min(S_y) >= 0 and np.max(S_y) < meta["k0"]

    # 8.1.2: compute_amplitude_dspte_matrix
    ec, delay, m = compute_amplitude_dspte_matrix(Y_win, config=cfg_smoke)
    assert ec.shape == (3, 3)
    assert delay.shape == (3, 3)
    assert np.all(np.diag(ec) == 0.0)
    assert np.all(np.diag(delay) == 0)
    assert len(m["te_profiles"]) == 6
    assert np.all(ec >= 0)

    print("Section 8.1 smoke tests PASS")
    print(f"EC shape={ec.shape}, te_profiles={len(m['te_profiles'])}, k0 range={m['k0_matrix'].min()}..{m['k0_matrix'].max()}")


def _cli_main() -> None:
    """
    CLI entrypoint for Section 8.1 amplitude-branch utilities.

    Supports
    --------
    --smoke-test : run synthetic checks for symbolisation and matrix output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Phase 3 Section 8.1 amplitude branch")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke tests")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_tests()


if __name__ == "__main__":
    _cli_main()
