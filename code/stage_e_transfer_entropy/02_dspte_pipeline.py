"""
Stage E — Step E2: dSPTE Sliding Window Pipeline

Implements the windowed dSPTE computation and connectivity vector extraction
for one trial's worth of ROI signals.  Mirrors Stage D's ``compute_windowed_gc``
and the diagonal-removal logic in ``04_subject_gc_pipeline.py``.

Section 4.1 task coverage (TRANSFER_ENTROPY_PROGRESS.md)
---------------------------------------------------------
4.1.1 — compute_windowed_dspte(Y_trial, window_size, stride, config) -> dict
           Sliding window loop identical to Stage D's compute_windowed_gc().
           Each window calls compute_dspte_matrix() and returns the tuple
           (EC_matrix, delay_matrix, metadata) keyed by 'Win_N' strings.
           Logs wall-clock time per window; alerts if > 60 s.

4.1.2 — extract_connectivity_vector(EC_matrix, roi_names) -> np.ndarray
           Expands a (n_valid_rois × n_valid_rois) EC matrix to the canonical
           5×5 layout (filling zeros for missing ROIs), then removes the
           diagonal to produce a 20-element connectivity vector.
           Identical logic to Stage D's expansion + ``np.delete(diag_indices)``.

Window convention (matches Stage D exactly)
-------------------------------------------
    window_size  = 1350   (stride samples)
    actual slice = Y_trial[:, w - window_size : w + 1]
                 = 1351 samples   (inclusive, matching MATLAB)
    for w in range(window_size, T, stride)

Reference
---------
Wang, Y., & Chen, W. (2020). Effective brain connectivity for fNIRS data
analysis based on multi-delays symbolic phase transfer entropy.
J. Neural Eng., 17(5), 056024. doi:10.1088/1741-2552/abb4a4
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config                         

logger = logging.getLogger(__name__)

# Canonical ROI order (matches Stage D ROI_NAMES)
_ROI_NAMES_CANONICAL = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
_N_CANONICAL = len(_ROI_NAMES_CANONICAL)          # 5
_DIAG_INDICES_FLAT = [i * (_N_CANONICAL + 1) for i in range(_N_CANONICAL)]
# = [0, 6, 12, 18, 24] — diagonal positions in a flattened 5×5 matrix

# Lazy import of core dSPTE functions (avoids circular/side-effect imports)
_CORE_MODULE = None


def _get_core_module():
    global _CORE_MODULE
    if _CORE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "dspte_core",
            Path(__file__).parent / "01_dspte_core.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE_MODULE = mod
    return _CORE_MODULE


# ===========================================================================
# Section 4.1.1 — Sliding Window Loop
# ===========================================================================

def compute_windowed_dspte(
    Y_trial: np.ndarray,
    window_size: int = 1350,
    stride: int = 1350,
    config: Optional[dict] = None,
    Phi_trial: Optional[np.ndarray] = None,
    taus_trial: Optional[list] = None,
) -> dict:
    """
    Slide a fixed-size window over one trial and compute dSPTE per window.

    Mirrors Stage D's ``compute_windowed_gc()`` exactly — same window
    indexing convention, same skip logic, same 1-indexed key naming.

    Window slicing (matches MATLAB inclusive slice)::

        for w in range(window_size, T, stride):
            Y_win = Y_trial[:, w - window_size : w + 1]   # 1351 samples

    Parameters
    ----------
    Y_trial : np.ndarray, shape (n_rois, T)
        Full ROI signal matrix for one trial (e.g. shape (5, ~6750) for a
        4.5-minute trial at 25 Hz).  May contain all-NaN rows for bad
        ROI channels (handled gracefully per window by compute_dspte_matrix).
    window_size : int
        Stride step in samples.  Actual window length is ``window_size + 1``
        to match the MATLAB inclusive slice.  Default 1350 (54 s × 25 Hz).
    stride : int
        Step between consecutive window starts.  Default 1350 (no overlap).
    config : dict or None
        Parsed ``config.yaml``.  Forwarded to ``compute_dspte_matrix()`` for
        embedding dimension, u_max, nDis threshold, and NG hyperparameters.
        Loaded from disk if None.
    Phi_trial : np.ndarray, shape (n_rois, T), optional
        Pre-computed instantaneous phase array for the full trial (Option A:
        narrowband filter → Hilbert on full trial).  When provided, each
        window's phase slice ``Phi_trial[:, w-window_size : w+1]`` is passed
        directly to ``compute_dspte_matrix`` instead of re-extracting phase
        from the raw signal window.  ``None`` entries in ``taus_trial`` are
        preserved as skipped ROIs.
    taus_trial : list of int, length n_rois, optional
        Pre-computed embedding delays (one per ROI) computed on the full-trial
        phase signal.  Must be supplied together with ``Phi_trial``; ignored
        otherwise.

    Returns
    -------
    windows : dict
        ``{'Win_1': (EC_matrix, delay_matrix, metadata),
           'Win_2': (EC_matrix, delay_matrix, metadata), ...}``

        Each value is the 3-tuple returned by ``compute_dspte_matrix()``:

        - ``EC_matrix``:   np.ndarray, shape (n_rois, n_rois) — directed
          connectivity matrix, values ≥ 0, diagonal = 0.
        - ``delay_matrix``: np.ndarray, shape (n_rois, n_rois), dtype int —
          optimal causal delay in samples per pair.
        - ``metadata``:    dict — k0_matrix, ndis_matrix, tau_matrix,
          te_profiles, skipped_pairs (see compute_dspte_matrix docstring).

        Keys are 1-indexed strings ``'Win_1'``, ``'Win_2'``, … matching
        MATLAB's ``W = sprintf('Win_%d', Wn+1)`` convention.
        Empty dict if the trial is shorter than one full window.

    Notes
    -----
    **Skip logic (identical to Stage D):** A window is skipped *only* if
    every sample of every ROI is all-NaN (completely empty signal matrix).
    Structurally NaN rows for bad individual channels are *not* a skip
    condition — ``compute_dspte_matrix()`` handles those per-pair.

    **Performance alert:** dSPTE is significantly more expensive than GC
    per window (~30–60 s vs ~2–5 s) due to the Neural-Gas codebook search.
    This function logs elapsed time per window and emits a WARNING if a
    single window exceeds 60 seconds, suggesting the codebook search is
    too broad (consider reducing ``k0_max_bivariate`` or ``u_max`` in
    ``config.yaml``).

    **Wall-clock logging:** Even when no alert fires, elapsed time per
    window is logged at INFO level for provenance tracking in
    ``te_provenance.json`` (Section 10).
    """
    if config is None:
        config = load_config()

    core = _get_core_module()
    compute_dspte_matrix = core.compute_dspte_matrix

    n_rois, T = Y_trial.shape
    windows: dict = {}
    win_num = 0

    for w in range(window_size, T, stride):
        Y_win = Y_trial[:, w - window_size : w + 1]   # (n_rois, window_size+1)

        # Skip only if every sample of every ROI is NaN
        if np.all(np.isnan(Y_win)):
            logger.warning(
                "compute_windowed_dspte: Window %d skipped — all-NaN data "
                "(samples %d–%d).",
                win_num + 1, w - window_size, w,
            )
            continue

        label = f"Win_{win_num + 1}"
        t_win_start = time.perf_counter()

        logger.info(
            "  %s: samples [%d, %d]  shape=%s",
            label, w - window_size, w, Y_win.shape,
        )

        # Option A: pass pre-computed phase window slices if available.
        # Phi_trial is (n_rois, T); NaN rows remain NaN in the slice and
        # compute_dspte_matrix's own NaN guard handles them.
        phi_win = (
            [Phi_trial[roi, w - window_size : w + 1] for roi in range(n_rois)]
            if Phi_trial is not None else None
        )

        EC_matrix, delay_matrix, metadata = compute_dspte_matrix(
            Y_win, config=config,
            phases=phi_win,
            taus=taus_trial,
        )

        elapsed = time.perf_counter() - t_win_start

        logger.info(
            "  %s: elapsed=%.1f s  max_EC=%.4f  mean_EC=%.4f  "
            "mean_k0=%.1f  n_skipped=%d",
            label,
            elapsed,
            float(EC_matrix[EC_matrix > 0].max()) if np.any(EC_matrix > 0) else 0.0,
            float(EC_matrix[np.eye(n_rois, dtype=bool) == False].mean()),
            float(metadata["k0_matrix"][metadata["k0_matrix"] > 0].mean())
            if np.any(metadata["k0_matrix"] > 0) else 0.0,
            len(metadata["skipped_pairs"]),
        )

        if elapsed > 60.0:
            logger.warning(
                "compute_windowed_dspte: %s took %.1f s > 60 s threshold. "
                "Consider reducing k0_max_bivariate or u_max in config.yaml.",
                label, elapsed,
            )

        windows[label] = (EC_matrix, delay_matrix, metadata)
        win_num += 1

    logger.info(
        "compute_windowed_dspte: %d windows extracted from trial of %d samples",
        len(windows), T,
    )
    return windows


# ===========================================================================
# Section 4.1.2 — Connectivity Vector Extraction
# ===========================================================================

def extract_connectivity_vector(
    EC_matrix: np.ndarray,
    roi_names: list[str],
) -> np.ndarray:
    """
    Expand EC matrix to canonical 5×5 layout and extract 20-element vector.

    Converts a (n_valid_rois × n_valid_rois) directed connectivity matrix
    into a canonical 20-element connectivity vector that is schema-compatible
    with Stage D's ``Tr_N_connectivity`` arrays.

    The procedure is identical to Stage D's Section 5.1.2 expansion::

        n_all = 5   (canonical ROIs: LPFC, CPFC, RPFC, LM1, SMA)
        GC_full = np.zeros((5, 5))
        for each valid (roi_i, roi_j) pair:
            GC_full[canonical_i, canonical_j] = EC_matrix[valid_i, valid_j]
        diag_indices = [0, 6, 12, 18, 24]
        connectivity_20 = np.delete(GC_full.flatten(), diag_indices)

    For subjects with all 5 ROIs, ``EC_matrix`` is already 5×5 and the
    expansion is a no-op.  For subjects with bad ROIs (R11: 3 valid,
    R2: 4 valid), the missing rows/columns remain 0, preserving the
    20-element schema across all subjects.

    Parameters
    ----------
    EC_matrix : np.ndarray, shape (n_valid_rois, n_valid_rois)
        Directed connectivity matrix from ``compute_dspte_matrix()``.
        Diagonal is 0.  Values ≥ 0.  ``n_valid_rois`` ≤ 5.
    roi_names : list[str]
        ROI names corresponding to rows/columns of ``EC_matrix``, in order.
        Must be a subset of ``['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA']``.

    Returns
    -------
    connectivity_20 : np.ndarray, shape (20,), dtype float64
        Off-diagonal elements of the canonical 5×5 EC matrix in row-major
        order (after diagonal removal).  The ordering is:

            LPFC→CPFC, LPFC→RPFC, LPFC→LM1, LPFC→SMA,
            CPFC→LPFC, CPFC→RPFC, CPFC→LM1, CPFC→SMA,
            RPFC→LPFC, RPFC→CPFC, RPFC→LM1, RPFC→SMA,
            LM1→LPFC,  LM1→CPFC,  LM1→RPFC,  LM1→SMA,
            SMA→LPFC,  SMA→CPFC,  SMA→RPFC,  SMA→LM1

        This is identical to Stage D's 20-element vector order.  All values
        ≥ 0.  Missing ROI pairs (bad channels) are 0.

    Raises
    ------
    ValueError
        If ``EC_matrix`` shape does not match ``len(roi_names)``, or if
        any name in ``roi_names`` is not in the canonical ROI list.

    Notes
    -----
    **Diagonal removal indices [0, 6, 12, 18, 24]:** These are the flat
    indices of the 5×5 diagonal after ``GC_full.flatten()``:
    ``[i*(5+1) for i in range(5)] = [0, 6, 12, 18, 24]``.
    Removing them from 25 elements yields exactly 20.
    """
    EC_matrix = np.asarray(EC_matrix, dtype=float)
    n_valid = len(roi_names)

    if EC_matrix.shape != (n_valid, n_valid):
        raise ValueError(
            f"extract_connectivity_vector: EC_matrix shape {EC_matrix.shape} "
            f"does not match len(roi_names)={n_valid}. "
            f"Expected ({n_valid}, {n_valid})."
        )

    for name in roi_names:
        if name not in _ROI_NAMES_CANONICAL:
            raise ValueError(
                f"extract_connectivity_vector: ROI name '{name}' not in "
                f"canonical list {_ROI_NAMES_CANONICAL}."
            )

    # Expand to canonical 5×5 layout
    EC_full = np.zeros((_N_CANONICAL, _N_CANONICAL), dtype=float)
    for i_v, roi_i in enumerate(roi_names):
        for j_v, roi_j in enumerate(roi_names):
            if roi_i != roi_j:
                i_c = _ROI_NAMES_CANONICAL.index(roi_i)
                j_c = _ROI_NAMES_CANONICAL.index(roi_j)
                EC_full[i_c, j_c] = EC_matrix[i_v, j_v]

    # Remove diagonal → 20-element vector
    connectivity_flat = EC_full.flatten()                              # 25 elements
    connectivity_20 = np.delete(connectivity_flat, _DIAG_INDICES_FLAT)  # 20 elements

    logger.debug(
        "extract_connectivity_vector: valid_rois=%s  "
        "EC shape %s → canonical 5×5 → 20-vec  "
        "range=[%.4f, %.4f]",
        roi_names, EC_matrix.shape,
        float(connectivity_20.min()), float(connectivity_20.max()),
    )
    return connectivity_20


# ===========================================================================
# Smoke tests (run as script)
# ===========================================================================

def _run_tests() -> None:
    """
    Section 4.1 tests.

    4.1.1  compute_windowed_dspte:
           (a) 2-ROI synthetic trial → correct number of windows, correct shapes
           (b) Trial shorter than one window → empty dict
           (c) All-NaN window skipped

    4.1.2  extract_connectivity_vector:
           (a) Full 5-ROI matrix → 20-element vector, correct ordering
           (b) 3-ROI matrix (missing RPFC, SMA) → 20 elements, correct zeros
           (c) Unknown ROI name → ValueError
           (d) Shape mismatch → ValueError
    """
    import warnings
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("\n=== Section 4.1 — Windowed dSPTE Pipeline Tests ===\n")

    # ------------------------------------------------------------------
    # 4.1.2 — extract_connectivity_vector  (fast; test first)
    # ------------------------------------------------------------------
    print("--- 4.1.2  extract_connectivity_vector ---")

    # (a) Full 5-ROI 5×5 matrix → 20 elements
    rng = np.random.default_rng(0)
    EC5 = rng.uniform(0, 1, (5, 5))
    np.fill_diagonal(EC5, 0.0)
    vec5 = extract_connectivity_vector(EC5, _ROI_NAMES_CANONICAL)
    assert vec5.shape == (20,), f"Expected (20,), got {vec5.shape}"
    assert np.all(vec5 >= 0), "Connectivity values must be non-negative"
    # Verify ordering: first element should be LPFC→CPFC = EC5[0,1]
    assert vec5[0] == EC5[0, 1], (
        f"vec5[0] (LPFC→CPFC) mismatch: {vec5[0]} vs EC5[0,1]={EC5[0,1]}"
    )
    # Element 4 should be CPFC→LPFC = EC5[1,0]
    assert vec5[4] == EC5[1, 0], (
        f"vec5[4] (CPFC→LPFC) mismatch: {vec5[4]} vs EC5[1,0]={EC5[1,0]}"
    )
    print(f"  (a) Full 5×5 → 20 elements  ✓  range=[{vec5.min():.4f}, {vec5.max():.4f}]")

    # (b) 3-ROI matrix (LPFC, CPFC, LM1 only — missing RPFC and SMA)
    roi_3 = ["LPFC", "CPFC", "LM1"]
    EC3 = np.array([[0, 0.4, 0.3],
                    [0.2, 0, 0.5],
                    [0.1, 0.6, 0]], dtype=float)
    vec3 = extract_connectivity_vector(EC3, roi_3)
    assert vec3.shape == (20,), f"Expected (20,), got {vec3.shape}"
    # LPFC→CPFC = EC3[0,1] = 0.4
    assert vec3[0] == 0.4, f"vec3[0] (LPFC→CPFC) expected 0.4, got {vec3[0]}"
    # RPFC→LPFC (index 8) should be 0 (missing ROI)
    assert vec3[8] == 0.0, f"vec3[8] (RPFC→LPFC) expected 0.0 (missing), got {vec3[8]}"
    # LM1→LPFC: LM1 is at canonical index 3; LPFC at 0. After diagonal removal:
    # flat index for LM1→LPFC = 3*5+0 = 15; after removing [0,6,12,18,24] → index 12
    assert vec3[12] == 0.1, f"vec3[12] (LM1→LPFC) expected 0.1, got {vec3[12]}"
    # Check RPFC row (indices 8-11) and SMA row (indices 16-19) are all 0
    assert np.all(vec3[8:12] == 0.0), f"RPFC row should be 0: {vec3[8:12]}"
    assert np.all(vec3[16:20] == 0.0), f"SMA row should be 0: {vec3[16:20]}"
    print(f"  (b) 3-ROI (missing RPFC, SMA) → 20 elements, correct zeros  ✓")
    print(f"      LPFC→CPFC={vec3[0]:.2f}  RPFC row={vec3[8:12]}  SMA row={vec3[16:20]}")

    # (c) Unknown ROI name → ValueError
    try:
        extract_connectivity_vector(np.eye(2), ["LPFC", "UNKNOWN"])
        assert False, "Should raise ValueError for unknown ROI"
    except ValueError:
        pass
    print("  (c) Unknown ROI name raises ValueError  ✓")

    # (d) Shape mismatch → ValueError
    try:
        extract_connectivity_vector(np.eye(5), ["LPFC", "CPFC"])  # 5×5 but only 2 names
        assert False, "Should raise ValueError for shape mismatch"
    except ValueError:
        pass
    print("  (d) Shape mismatch raises ValueError  ✓")

    print("  [PASS] 4.1.2 extract_connectivity_vector\n")

    # ------------------------------------------------------------------
    # 4.1.1 — compute_windowed_dspte  (uses compute_dspte_matrix internally)
    # ------------------------------------------------------------------
    print("--- 4.1.1  compute_windowed_dspte ---")

    # Build a fast test config (small u_max, small k0_max, relaxed nDis)
    test_config = {
        "stage_e_transfer_entropy": {
            "signal": {"embedding_dimension": 3, "u_max": 5},
            "symbolisation": {
                "ndis_threshold": 0.15,
                "k0_max_bivariate": 6,
                "ng_max_iterations": 150,
                "ng_learning_rate_i": 0.5,
                "ng_learning_rate_f": 0.01,
                "ng_neighbourhood_i": 6.0,
                "ng_neighbourhood_f": 0.1,
            },
        }
    }

    # (a) Synthetic 2-ROI trial: 3 full windows (T = 3 × 1351 = 4053 samples)
    N_trial = 4053
    rng_t = np.random.default_rng(5)
    ch0 = np.zeros(N_trial)
    ch1 = np.zeros(N_trial)
    for tt in range(2, N_trial):
        ch0[tt] = 0.5 * ch0[tt-1] + 0.4 * rng_t.standard_normal()
        ch1[tt] = 0.7 * ch0[tt-2] + 0.3 * rng_t.standard_normal()
    Y_trial_test = np.vstack([ch0, ch1])   # shape (2, 4053)

    print("  (a) 2-ROI synthetic trial, 3 windows (T=4053, window=1350, stride=1350)")
    t0_a = time.perf_counter()
    windows_a = compute_windowed_dspte(Y_trial_test, window_size=1350, stride=1350,
                                       config=test_config)
    elapsed_a = time.perf_counter() - t0_a

    assert len(windows_a) == 3, f"Expected 3 windows, got {len(windows_a)}"
    for win_key, (EC, delay, meta) in windows_a.items():
        assert EC.shape == (2, 2), f"{win_key}: expected EC shape (2,2), got {EC.shape}"
        assert delay.shape == (2, 2)
        assert np.all(np.diag(EC) == 0), f"{win_key}: diagonal not zero"
        assert np.all(EC >= 0), f"{win_key}: negative EC values"
        assert "te_profiles" in meta
    print(f"  Got {len(windows_a)} windows  ✓  (elapsed: {elapsed_a:.1f} s)")
    print(f"  Window keys: {list(windows_a.keys())}")
    # Sample values from Win_1
    EC1, delay1, _ = windows_a["Win_1"]
    print(f"  Win_1: EC[0→1]={EC1[0,1]:.4f}  EC[1→0]={EC1[1,0]:.4f}  "
          f"delay[0→1]={delay1[0,1]}")

    # (b) Trial shorter than one window → empty dict
    Y_short = np.random.randn(2, 500)
    windows_b = compute_windowed_dspte(Y_short, window_size=1350, stride=1350,
                                       config=test_config)
    assert len(windows_b) == 0, f"Expected 0 windows for short trial, got {len(windows_b)}"
    print(f"  (b) Short trial (T=500 < window=1351) → 0 windows  ✓")

    # (c) First window all-NaN → skipped; second window valid → 1 window returned
    Y_nan_win = np.vstack([ch0, ch1]).copy()
    Y_nan_win[:, :1351] = np.nan    # first window all-NaN
    windows_c = compute_windowed_dspte(Y_nan_win, window_size=1350, stride=1350,
                                       config=test_config)
    # T=4053: windows would start at 1350, 2700, 4050.  w=4050 → w+1=4051, only 3 samples.
    # Actually w=4050: Y_trial[:, 4050-1350:4051] = Y[:, 2700:4051] → 1351 samples. Valid.
    # So windows at w=1350 (all-NaN → skip), w=2700 (valid), w=4050 (valid if T≥4051).
    # T=4053 ≥ 4051, so range(1350, 4053, 1350) = [1350, 2700, 4050] → 3 candidates.
    # w=1350: all-NaN → skip. w=2700 and w=4050: valid → 2 windows.
    expected_c = 2  # two valid windows after skipping the NaN-first one
    assert len(windows_c) == expected_c, (
        f"Expected {expected_c} windows after NaN-skip, got {len(windows_c)}"
    )
    print(f"  (c) All-NaN first window skipped → {len(windows_c)} valid windows  ✓")
    print(f"  Window keys: {list(windows_c.keys())}  (Win_1 skipped, Win_2 and Win_3)")

    # Window numbering continues sequentially from first *valid* window
    # Note: win_num increments only for valid windows, so keys start at Win_1
    assert "Win_1" in windows_c, (
        f"Expected Win_1 in result after skip (numbering resets to first valid), "
        f"got keys: {list(windows_c.keys())}"
    )

    print("  [PASS] 4.1.1 compute_windowed_dspte\n")

    print("=== All Section 4.1 tests passed. ===\n")


if __name__ == "__main__":
    _run_tests()
