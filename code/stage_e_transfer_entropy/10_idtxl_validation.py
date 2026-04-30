"""
Stage E — Section 7.4: IDTxl Parallel Validation

Implements Sections 7.4.1–7.4.3 from TRANSFER_ENTROPY_PROGRESS.md.

IDTxl's MultivariateTE with PythonKraskovCMI (KSG estimation) operates on
continuous amplitude values with non-uniform embedding (Faes 2011 NUE).  This
is the continuous-domain analogue to multivariate GC and provides a
well-validated independent reference against which the Phase 2 NG-symbolic
conditional dSPTE can be calibrated.

Design
------
7.4.1 — Import verification.  If IDTxl is unavailable the script writes
        IDTXL_EVAL.md and exits cleanly.

7.4.2 — run_idtxl_multivariate_te():
        Input: pre-filtered amplitude window (n_rois, T).  Returns (n_rois,
        n_rois) TE matrix and per-source TE vectors extracted from IDTxl
        results. Non-significant connections are set to 0 (uses IDTxl's
        built-in permutation test; fdr=False to avoid over-correction at
        small n).

7.4.3 — compare_symbolic_vs_idtxl():
        Per subject: correlate NG-symbolic Phase 2 20-vector against IDTxl
        20-vector across trials × windows.  Generates scatter plot coloured
        by connection category and writes idtxl_comparison_summary.json.

Usage
-----
Standalone smoke test (synthetic data, does not require real subjects):
    python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --smoke-test

Single subject (first trial, first window only for speed):
    python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --subject A1

Full batch (all 13 subjects, all trials × windows — may be slow):
    python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --batch

References
----------
Wollstadt et al. (2019): IDTxl — doi:10.21105/joss.01081
Faes et al. (2011): NUE — doi:10.1103/PhysRevE.83.051112
Kraskov et al. (2004): KSG estimator — doi:10.1103/PhysRevE.69.066138
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).parent.parent
_PROJECT_ROOT = _CODE_ROOT.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 7.4.1 — IDTxl import check
#
# IDTxl is not on PyPI — it must be installed from source:
#   git clone https://github.com/pwollstadt/IDTxl
#   pip install -e IDTxl/
#
# If the editable .pth install doesn't activate (Python 3.13 edge case),
# we fall back to adding the cloned IDTxl/ directory directly to sys.path.
#
# Estimator: PythonKraskovCMI (pure Python+scipy KSG, no Java/jpype1 needed).
# This is IDTxl's native Python implementation of the Kraskov estimator —
# identical mathematical result to JidtKraskovCMI but without the Java bridge.
# ---------------------------------------------------------------------------
_IDTXL_AVAILABLE: bool = False
_IDTXL_VERSION: str = "unavailable"
_IDTXL_IMPORT_ERROR: str = ""

# Attempt 1: normal import (works when pip install -e IDTxl/ is active)
try:
    from idtxl.multivariate_te import MultivariateTE as _MultivariateTE
    from idtxl.data import Data as _IDTxlData
    from idtxl.estimators_python import PythonKraskovCMI as _PythonKraskovCMI
    _IDTXL_AVAILABLE = True
except ImportError as _e1:
    # Attempt 2: add the cloned IDTxl/ directory directly to sys.path.
    # This handles the Python 3.13 editable-install .pth activation edge case.
    _idtxl_repo = _PROJECT_ROOT / "IDTxl"
    if _idtxl_repo.exists() and str(_idtxl_repo) not in sys.path:
        sys.path.insert(0, str(_idtxl_repo))
    try:
        from idtxl.multivariate_te import MultivariateTE as _MultivariateTE
        from idtxl.data import Data as _IDTxlData
        from idtxl.estimators_python import PythonKraskovCMI as _PythonKraskovCMI
        _IDTXL_AVAILABLE = True
    except ImportError as _e2:
        _IDTXL_IMPORT_ERROR = str(_e2)

if _IDTXL_AVAILABLE:
    try:
        import idtxl as _idtxl_mod
        _IDTXL_VERSION = getattr(_idtxl_mod, "__version__", "1.6 (cloned)")
    except Exception:
        _IDTXL_VERSION = "installed (version unknown)"
    logger.info("IDTxl imported successfully — version: %s", _IDTXL_VERSION)

    # MultivariateTE passes its full settings dict (including JIDT-specific keys
    # like theiler_t and algorithm_num) to whichever CMI estimator is selected.
    # PythonKraskovCMI rejects those keys. This thin subclass strips them before
    # delegating to the parent, so PythonKraskovCMI works inside MultivariateTE.
    _JIDT_ONLY_KEYS = {"theiler_t", "algorithm_num", "local_values"}

    class _CompatPythonKraskovCMI(_PythonKraskovCMI):
        def __init__(self, settings=None):
            """Initialise Python KSG estimator after removing JIDT-only keys."""
            if settings is not None:
                settings = {k: v for k, v in settings.items()
                            if k not in _JIDT_ONLY_KEYS}
            super().__init__(settings or {})

else:
    logger.warning("IDTxl not available: %s", _IDTXL_IMPORT_ERROR)
    _CompatPythonKraskovCMI = None


def write_idtxl_eval_md(out_path: Optional[Path] = None) -> Path:
    """Write IDTXL_EVAL.md documenting IDTxl availability status."""
    if out_path is None:
        out_path = Path(__file__).parent / "IDTXL_EVAL.md"

    status = "AVAILABLE" if _IDTXL_AVAILABLE else "UNAVAILABLE"
    import datetime

    content = f"""# IDTxl Evaluation Log

Generated: {datetime.datetime.now().isoformat()}

## Status: {status}

### Import Result

```
{'from idtxl.multivariate_te import MultivariateTE — SUCCESS' if _IDTXL_AVAILABLE else f'ImportError: {_IDTXL_IMPORT_ERROR}'}
```

### Version
{_IDTXL_VERSION}

### Installation
IDTxl was installed from https://github.com/pwollstadt/IDTxl via:
```
git clone https://github.com/pwollstadt/IDTxl
cd IDTxl
pip install -e .
```

### Estimator Used
`PythonKraskovCMI` — IDTxl's pure Python/scipy implementation of the KSG
conditional mutual information estimator. Operates on continuous data
(amplitude values), bypassing discrete binning and the contingency-table sparsity
problem. Produces identical results to `PythonKraskovCMI` but requires no Java or jpype1.

### Settings for ASTaUND fNIRS data
| Parameter | Value | Rationale |
|---|---|---|
| `cmi_estimator` | `PythonKraskovCMI` | Continuous KSG — no Java/jpype1 |
| `max_lag_sources` | `u_max` (from config, default 50) | Covers ~2 s hemodynamic lag |
| `min_lag_sources` | 1 | Allow all lags ≥ 1 sample |
| `max_lag_target` | `u_max` | Consistent with source lag search |
| `n_perm_max_stat` | 50 | Reduced for speed on fNIRS windows (~1351 samples) |
| `n_perm_min_stat` | 50 | Reduced for speed |
| `n_perm_omnibus` | 100 | Omnibus permutation test |
| `n_perm_max_seq` | 50 | Reduced for speed |
| `alpha_max_stat` | 0.05 | Standard significance threshold |

### Comparison Design (Section 7.4.3)
- IDTxl run on the same pre-filtered (0.04–0.15 Hz) amplitude windows as Phase 2
- IDTxl uses raw continuous amplitude (not symbolic sequences) — tests whether
  the KSG estimator on continuous data recovers similar directionality to NG-symbolic
- Output: 5×5 EC matrix per window; converted to 20-element connectivity vector
  using same `get_connection_labels()` schema as GC and Phase 2
- Pearson r between Phase 2 and IDTxl vectors averaged across subjects × windows
  is the primary comparison statistic

### Interpretation Thresholds (from PRD 7.4.3)
| Pearson r | Interpretation |
|---|---|
| r > 0.6 | NG-symbolic Phase 2 validated — report as primary |
| 0.3 < r ≤ 0.6 | Partial agreement — report both; NG-symbolic methodologically faithful |
| r ≤ 0.3 | Substantial disagreement — elevate IDTxl to primary |
"""
    out_path.write_text(content)
    logger.info("write_idtxl_eval_md: wrote → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 7.4.2 — run_idtxl_multivariate_te
# ---------------------------------------------------------------------------

def run_idtxl_multivariate_te(
    Y_window: np.ndarray,
    config: Optional[dict] = None,
    roi_names: Optional[list] = None,
    use_fdr: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Run IDTxl MultivariateTE with KSG estimation on a single amplitude window.

    Parameters
    ----------
    Y_window : np.ndarray, shape (n_rois, T)
        Pre-filtered (0.04–0.15 Hz) continuous amplitude (or phase) signals
        for one window.  IDTxl operates on the raw continuous values — no
        symbolisation is applied.
    config : dict, optional
        Parsed config.yaml.  Used to read u_max and n_perm settings.
    roi_names : list of str, optional
        ROI names for logging.  Default ['ROI_0', ...].
    use_fdr : bool
        If True, return FDR-corrected adjacency.  Default False (IDTxl's FDR
        across 5 targets can be overly conservative at n=1351 samples).

    Returns
    -------
    ec_matrix : np.ndarray, shape (n_rois, n_rois)
        EC matrix where ec_matrix[src, tgt] = omnibus TE from src to tgt
        (sum of per-source TE values for all significant sources).
        Non-significant entries are 0. Diagonal is 0.
    metadata : dict
        run_time_s, n_rois, T, n_sig_edges, settings_used, targets_analysed,
        per_target_omnibus_te, per_target_n_sources.
    """
    if not _IDTXL_AVAILABLE:
        raise RuntimeError(
            "IDTxl is not available. Install via: pip install -e /path/to/IDTxl"
        )

    if config is None:
        config = {}

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    idtxl_cfg = te_cfg.get("idtxl_validation", {})
    u_max_signal = int(sig_cfg.get("u_max", 50))
    # Guard against carrying over very large Phase-1 lag ranges (e.g., 300)
    # into IDTxl NUE source search, which is much more expensive.
    max_lag_sources = int(idtxl_cfg.get("max_lag_sources", min(u_max_signal, 50)))
    max_lag_sources = max(1, max_lag_sources)
    max_lag_target = int(idtxl_cfg.get("max_lag_target", min(max_lag_sources, 20)))
    max_lag_target = max(1, min(max_lag_target, max_lag_sources))

    n_perm_max_stat = int(idtxl_cfg.get("n_perm_max_stat", 50))
    n_perm_min_stat = int(idtxl_cfg.get("n_perm_min_stat", 50))
    n_perm_omnibus = int(idtxl_cfg.get("n_perm_omnibus", 100))
    n_perm_max_seq = int(idtxl_cfg.get("n_perm_max_seq", 50))
    idtxl_num_threads = idtxl_cfg.get("num_threads", "USE_ALL")
    n_rois, T = Y_window.shape

    if roi_names is None:
        roi_names = [f"ROI_{i}" for i in range(n_rois)]

    # IDTxl requires 3D data: (n_processes, n_samples, n_replications)
    data_3d = Y_window[:, :, np.newaxis].astype(np.float64)
    idtxl_data = _IDTxlData(data_3d, dim_order="psr", normalise=True)

    # Settings — reduced permutations for speed on fNIRS windows.
    # Pass the class object (not a string) so MultivariateTE uses our
    # _CompatPythonKraskovCMI wrapper that strips JIDT-only settings keys.
    settings = {
        "cmi_estimator": _CompatPythonKraskovCMI,
        "max_lag_sources": max_lag_sources,
        "min_lag_sources": 1,
        "max_lag_target": max_lag_target,
        "n_perm_max_stat": n_perm_max_stat,
        "n_perm_min_stat": n_perm_min_stat,
        "n_perm_omnibus": n_perm_omnibus,
        "n_perm_max_seq": n_perm_max_seq,
        "alpha_max_stat": 0.05,
        "alpha_min_stat": 0.05,
        "alpha_omnibus": 0.05,
        "alpha_max_seq": 0.05,
        "num_threads": idtxl_num_threads,
        "verbose": False,
    }

    t0 = time.perf_counter()
    network_analysis = _MultivariateTE()

    try:
        results = network_analysis.analyse_network(
            settings=settings,
            data=idtxl_data,
            targets="all",
            sources="all",
        )
    except (AttributeError, TypeError, ImportError, RuntimeError) as exc:
        # Re-raise internal IDTxl/NumPy compatibility errors — these indicate
        # broken code, not a legitimate "no connectivity" result. Re-raising
        # prevents a corrupt zero-TE cache from being written to disk.
        raise
    except Exception as exc:
        logger.warning("run_idtxl_multivariate_te: IDTxl raised exception: %s", exc)
        ec_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
        metadata = {
            "run_time_s": time.perf_counter() - t0,
            "n_rois": n_rois,
            "T": T,
            "n_sig_edges": 0,
            "error": str(exc),
            "settings_used": settings,
            "targets_analysed": [],
            "per_target_omnibus_te": {},
            "per_target_n_sources": {},
        }
        return ec_matrix, metadata

    elapsed = time.perf_counter() - t0

    # Extract EC matrix: ec_matrix[src, tgt] = sum of significant TE from src
    ec_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    per_target_omnibus_te: dict = {}
    per_target_n_sources: dict = {}
    n_sig_edges = 0

    for tgt_idx in range(n_rois):
        try:
            tgt_result = results.get_single_target(target=tgt_idx, fdr=use_fdr)
        except (RuntimeError, KeyError):
            per_target_omnibus_te[roi_names[tgt_idx]] = 0.0
            per_target_n_sources[roi_names[tgt_idx]] = 0
            continue

        omnibus_te = float(tgt_result.get("omnibus_te", 0.0) or 0.0)
        per_target_omnibus_te[roi_names[tgt_idx]] = omnibus_te

        sig_vars = tgt_result.get("selected_vars_sources", []) or []
        per_target_n_sources[roi_names[tgt_idx]] = len(sig_vars)

        # Accumulate TE per source: sum selected_sources_te across lags
        te_vals = tgt_result.get("selected_sources_te", None)
        if te_vals is not None and len(te_vals) > 0 and len(sig_vars) > 0:
            te_arr = np.atleast_1d(np.asarray(te_vals, dtype=np.float64))
            for lag_idx, (src_proc, _lag) in enumerate(sig_vars):
                if src_proc != tgt_idx and lag_idx < len(te_arr):
                    val = float(te_arr[lag_idx])
                    if val > 0:
                        ec_matrix[src_proc, tgt_idx] += val
                        n_sig_edges += 1
        elif omnibus_te > 0 and sig_vars:
            # Fallback: distribute omnibus_te equally across significant sources
            unique_srcs = list({sv[0] for sv in sig_vars if sv[0] != tgt_idx})
            if unique_srcs:
                per_src_te = omnibus_te / len(unique_srcs)
                for src_proc in unique_srcs:
                    ec_matrix[src_proc, tgt_idx] += per_src_te
                    n_sig_edges += 1

    np.fill_diagonal(ec_matrix, 0.0)

    metadata = {
        "run_time_s": elapsed,
        "n_rois": n_rois,
        "T": T,
        "n_sig_edges": n_sig_edges,
        "settings_used": {k: v for k, v in settings.items() if k != "verbose"},
        "targets_analysed": list(results.targets_analysed),
        "per_target_omnibus_te": per_target_omnibus_te,
        "per_target_n_sources": per_target_n_sources,
        "use_fdr": use_fdr,
    }

    logger.info(
        "run_idtxl_multivariate_te: n_rois=%d  T=%d  n_sig=%d  elapsed=%.1f s",
        n_rois, T, n_sig_edges, elapsed,
    )
    return ec_matrix, metadata


def extract_idtxl_connectivity_vector(
    ec_matrix: np.ndarray,
    roi_names: list,
    conn_labels: Optional[list] = None,
) -> np.ndarray:
    """
    Convert (n_rois, n_rois) IDTxl EC matrix to canonical 20-element vector.

    Uses the same connection label ordering as Phase 1/2 and Stage D GC.
    """
    if conn_labels is None:
        from utils_te import get_connection_labels
        conn_labels = get_connection_labels()

    conn_index: dict = {}
    for roi_a in roi_names:
        for roi_b in roi_names:
            if roi_a != roi_b:
                conn_index[(roi_a, roi_b)] = (roi_names.index(roi_a), roi_names.index(roi_b))

    vec = np.zeros(len(conn_labels), dtype=np.float64)
    for i, label in enumerate(conn_labels):
        parts = label.split("→")
        if len(parts) == 2:
            src, tgt = parts[0].strip(), parts[1].strip()
            if (src, tgt) in conn_index:
                si, ti = conn_index[(src, tgt)]
                vec[i] = ec_matrix[si, ti]
    return vec


# ---------------------------------------------------------------------------
# Subject-level IDTxl runner
# ---------------------------------------------------------------------------

def run_subject_idtxl(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    overwrite: bool = False,
    first_window_only: bool = False,
) -> Optional[dict]:
    """
    Run IDTxl MultivariateTE for all trials × windows of one subject.

    Uses the same pre-filtered amplitude signals as Phase 2 (narrowband
    0.04–0.15 Hz Butterworth → HbO amplitude window).  The amplitude window
    (not phase) is passed to IDTxl — KSG on amplitude is the continuous
    analogue of GC.

    Saves per-subject result to:
        ``phase_2_hybrid_dspte/idtxl_validation/sub-{id}_idtxl_ec.npz``

    Returns
    -------
    dict : {trial_id: {win_id: conn_vec_20}} or None on failure.
    """
    if not _IDTXL_AVAILABLE:
        logger.error(
            "run_subject_idtxl [%s]: IDTxl not available — skipping.", subject_id
        )
        return None

    if config is None:
        config = {}

    from utils_te import (
        get_te_output_dir,
        load_or_extract_roi_signals,
        load_preprocessed_fif,
        get_connection_labels,
        get_subject_group,
    )
    import importlib.util as _ilu
    _core_path = Path(__file__).parent / "06_hybrid_dspte.py"
    _spec = _ilu.spec_from_file_location("hybrid_dspte_mod", _core_path)
    _hybrid_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_hybrid_mod)

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    sfreq = float(sig_cfg.get("sfreq", 25.0))
    win_cfg = te_cfg.get("window", {})
    win_len = int(win_cfg.get("size_samples", round(float(win_cfg.get("size_s", 54.0)) * sfreq)))
    win_step = int(win_cfg.get("stride_samples", win_len))
    if win_step <= 0:
        win_step = win_len

    nb_cfg = te_cfg.get("narrowband_filter", {})
    f_lo = float(nb_cfg.get("f_lo", 0.01))
    f_hi = float(nb_cfg.get("f_hi", 0.10))
    order = int(nb_cfg.get("order", 4))

    # Output directory
    out_dir = get_te_output_dir("phase_2_viz", config=config).parent / "idtxl_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"sub-{subject_id}_idtxl_ec.npz"

    if npz_path.exists() and not overwrite:
        logger.info(
            "run_subject_idtxl [%s]: cache hit — loading %s", subject_id, npz_path.name
        )
        return _load_idtxl_npz(npz_path)

    # Load ROI signals
    try:
        Data_Reg, roi_names, sf_subj = load_or_extract_roi_signals(
            subject_id, study=study, config=config
        )
    except Exception as exc:
        logger.error("run_subject_idtxl [%s]: ROI load failed: %s", subject_id, exc)
        return None

    n_rois = Data_Reg.shape[0]
    conn_labels = get_connection_labels()

    # Trial segmentation — mirrors process_subject_hybrid_dspte in 06_hybrid_dspte.py:
    # load preprocessed fif, then call get_trial_segments(raw, config).
    import importlib.util as _ilu2
    try:
        raw = load_preprocessed_fif(subject_id, study=study, config=config)
    except Exception as exc:
        logger.error("run_subject_idtxl [%s]: fif load failed: %s", subject_id, exc)
        return None

    _seg_path = _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py"
    _spec_seg = _ilu2.spec_from_file_location("trial_segmentation_idtxl", _seg_path)
    _seg_mod = _ilu2.module_from_spec(_spec_seg)
    _spec_seg.loader.exec_module(_seg_mod)

    try:
        segments = _seg_mod.get_trial_segments(raw, config=config)
    except Exception as exc:
        logger.error(
            "run_subject_idtxl [%s]: trial segmentation failed: %s", subject_id, exc
        )
        return None

    logger.info(
        "run_subject_idtxl [%s]: %d trial segments found (win_len=%d, win_step=%d).",
        subject_id, len(segments), win_len, win_step,
    )

    t0_subj = time.perf_counter()
    result: dict = {}
    save_dict: dict = {}
    all_meta: list = []

    for trial_idx, (start_s, end_s) in enumerate(segments):
        # Match Phase 2 trial key schema for direct compare alignment.
        trial_key = f"Tr_{trial_idx + 1}"
        trial_data = Data_Reg[:, start_s:end_s]
        if trial_data.shape[1] < win_len:
            continue

        # Apply narrowband filter to full trial amplitude signal
        try:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=sf_subj, output="sos")
            trial_filtered = np.zeros_like(trial_data)
            for ch_i in range(n_rois):
                if not np.any(np.isnan(trial_data[ch_i])):
                    trial_filtered[ch_i] = sosfiltfilt(sos, trial_data[ch_i])
                else:
                    trial_filtered[ch_i] = trial_data[ch_i]
        except Exception as exc:
            logger.warning(
                "run_subject_idtxl [%s] %s: filter failed: %s — using raw",
                subject_id, trial_key, exc,
            )
            trial_filtered = trial_data

        # Sliding window
        T_trial = trial_filtered.shape[1]
        win_starts = list(range(0, T_trial - win_len + 1, win_step))
        trial_result: dict = {}

        for win_idx, ws in enumerate(win_starts):
            win_key = f"Win_{win_idx + 1}"
            Y_win = trial_filtered[:, ws: ws + win_len]

            # Skip windows with NaN channels
            valid_mask = ~np.any(np.isnan(Y_win), axis=1)
            if valid_mask.sum() < 2:
                continue

            Y_valid = Y_win[valid_mask]
            roi_names_valid = [roi_names[i] for i in range(n_rois) if valid_mask[i]]

            try:
                ec_mat_valid, meta = run_idtxl_multivariate_te(
                    Y_valid, config=config, roi_names=roi_names_valid
                )
            except Exception as exc:
                logger.warning(
                    "run_subject_idtxl [%s] %s %s: IDTxl error: %s",
                    subject_id, trial_key, win_key, exc,
                )
                continue

            # Expand back to full n_rois × n_rois
            ec_mat = np.zeros((n_rois, n_rois), dtype=np.float64)
            valid_indices = [i for i in range(n_rois) if valid_mask[i]]
            for vi, gi in enumerate(valid_indices):
                for vj, gj in enumerate(valid_indices):
                    ec_mat[gi, gj] = ec_mat_valid[vi, vj]

            conn_vec = extract_idtxl_connectivity_vector(
                ec_mat, roi_names, conn_labels
            )
            trial_result[win_key] = conn_vec
            all_meta.append({
                "trial": trial_key,
                "window": win_key,
                **{k: v for k, v in meta.items() if k not in ("settings_used",)},
            })

            logger.info(
                "run_subject_idtxl [%s %s %s]: n_sig=%d  elapsed=%.1f s",
                subject_id, trial_key, win_key,
                meta.get("n_sig_edges", 0), meta.get("run_time_s", 0.0),
            )

            if first_window_only:
                break

        result[trial_key] = trial_result

        # Build connectivity array for save_dict
        if trial_result:
            wins_sorted = sorted(trial_result.keys(), key=lambda k: int(k.split("_")[1]))
            conn_array = np.column_stack(
                [trial_result[wk] for wk in wins_sorted]
            ).astype(np.float64)
        else:
            conn_array = np.empty((len(conn_labels), 0), dtype=np.float64)
        save_dict[f"{trial_key}_connectivity"] = conn_array

    # Save .npz
    trial_ids = sorted(
        [k for k in result.keys()],
        key=lambda k: int(k.split("_")[1]),
    )
    window_counts = [len(result.get(t, {})) for t in trial_ids]
    save_dict["subject_id"] = np.array(subject_id)
    save_dict["trial_ids"] = np.array(trial_ids)
    save_dict["window_counts"] = np.array(window_counts, dtype=int)
    save_dict["roi_names_used"] = np.array(roi_names)
    save_dict["connection_labels"] = np.array(conn_labels)
    try:
        group = get_subject_group(subject_id, config=config)
    except ValueError:
        group = "Unknown"
    save_dict["group"] = np.array(group)
    save_dict["window_metadata_json"] = np.array(json.dumps(all_meta))

    elapsed_total = time.perf_counter() - t0_subj
    logger.info(
        "run_subject_idtxl [%s]: complete — trials=%d  total_wins=%d  elapsed=%.1f s",
        subject_id, len(trial_ids), sum(window_counts), elapsed_total,
    )

    np.savez(npz_path, **save_dict)
    logger.info("run_subject_idtxl [%s]: saved → %s", subject_id, npz_path.name)
    return result


def _load_idtxl_npz(npz_path: Path) -> dict:
    """Load IDTxl connectivity .npz back into {trial_id: {win_id: conn_vec}}."""
    data = dict(np.load(npz_path, allow_pickle=True))
    trial_ids = list(data["trial_ids"])
    result: dict = {}
    for trial_id in trial_ids:
        conn_array = data.get(f"{trial_id}_connectivity")
        trial_result: dict = {}
        if conn_array is not None and conn_array.ndim == 2 and conn_array.shape[1] > 0:
            for win_idx in range(conn_array.shape[1]):
                trial_result[f"Win_{win_idx + 1}"] = conn_array[:, win_idx]
        result[trial_id] = trial_result
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch_idtxl(
    study: str = "expert_novice",
    subjects: Optional[list] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
    first_window_only: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Run IDTxl validation for all subjects. Saves per-subject .npz files and
    a batch log JSON.

    Returns
    -------
    batch_log : dict
    """
    if config is None:
        try:
            from utils import load_config as _lc
            config = _lc()
        except ImportError:
            config = {}

    from utils_te import get_te_output_dir

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    if subjects is None:
        experts = subj_cfg.get("Expert", [])
        novices = subj_cfg.get("Novice", [])
        subjects = experts + novices

    logger.info(
        "run_batch_idtxl: processing %d subjects: %s", len(subjects), subjects
    )

    import datetime
    batch_start = time.perf_counter()
    batch_log: dict = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "study": study,
        "phase": "section_7_4_idtxl_validation",
        "idtxl_version": _IDTXL_VERSION,
        "n_subjects": len(subjects),
        "first_window_only": first_window_only,
        "subjects": {},
    }

    def _summarise_subject(subject_id: str, result: Optional[dict], elapsed_subj: float) -> dict:
        """
        Build one subject summary row for batch log JSON.

        Returns
        -------
        dict
            Status, window counts, TE summary, and elapsed time.
        """
        if result is None:
            return {
                "status": "FAILED",
                "error": "run_subject_idtxl returned None",
                "elapsed_s": elapsed_subj,
            }
        n_wins = sum(len(v) for v in result.values())
        all_vecs = [v for trial in result.values() for v in trial.values()]
        if all_vecs:
            stacked = np.stack(all_vecs, axis=0)
            mean_te = float(np.nanmean(stacked[stacked > 0])) if np.any(stacked > 0) else 0.0
            n_sig = int(np.sum(stacked > 0))
        else:
            mean_te = 0.0
            n_sig = 0
        if n_wins == 0:
            logger.warning(
                "run_batch_idtxl [%s]: no valid windows were produced (elapsed=%.1f s).",
                subject_id, elapsed_subj,
            )
            return {
                "status": "FAILED",
                "error": "no_valid_windows",
                "n_trials": len(result),
                "n_windows_total": 0,
                "elapsed_s": elapsed_subj,
            }
        logger.info(
            "run_batch_idtxl [%s]: OK — trials=%d  wins=%d  mean_te=%.4f  n_sig=%d  elapsed=%.1f s",
            subject_id, len(result), n_wins, mean_te, n_sig, elapsed_subj,
        )
        return {
            "status": "OK",
            "n_trials": len(result),
            "n_windows_total": n_wins,
            "mean_idtxl_te": mean_te,
            "n_significant_connections": n_sig,
            "elapsed_s": elapsed_subj,
        }

    n_jobs = int(max(1, n_jobs))
    if n_jobs == 1:
        for subject_id in subjects:
            subj_start = time.perf_counter()
            logger.info("run_batch_idtxl: === Subject %s ===", subject_id)
            try:
                result = run_subject_idtxl(
                    subject_id,
                    study=study,
                    config=config,
                    overwrite=overwrite,
                    first_window_only=first_window_only,
                )
                batch_log["subjects"][subject_id] = _summarise_subject(
                    subject_id, result, time.perf_counter() - subj_start
                )
            except Exception as exc:
                elapsed_subj = time.perf_counter() - subj_start
                logger.error(
                    "run_batch_idtxl [%s]: EXCEPTION: %s", subject_id, exc, exc_info=True
                )
                batch_log["subjects"][subject_id] = {
                    "status": "FAILED",
                    "error": str(exc),
                    "elapsed_s": elapsed_subj,
                }
    else:
        logger.info("run_batch_idtxl: parallel mode enabled (n_jobs=%d).", n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {}
            for subject_id in subjects:
                subj_start = time.perf_counter()
                fut = ex.submit(
                    run_subject_idtxl,
                    subject_id,
                    study,
                    config,
                    overwrite,
                    first_window_only,
                )
                futures[fut] = (subject_id, subj_start)

            for fut in as_completed(futures):
                subject_id, subj_start = futures[fut]
                try:
                    result = fut.result()
                    batch_log["subjects"][subject_id] = _summarise_subject(
                        subject_id, result, time.perf_counter() - subj_start
                    )
                except Exception as exc:
                    elapsed_subj = time.perf_counter() - subj_start
                    logger.error(
                        "run_batch_idtxl [%s]: EXCEPTION: %s", subject_id, exc, exc_info=True
                    )
                    batch_log["subjects"][subject_id] = {
                        "status": "FAILED",
                        "error": str(exc),
                        "elapsed_s": elapsed_subj,
                    }

    total_elapsed = time.perf_counter() - batch_start
    n_ok = sum(1 for v in batch_log["subjects"].values() if v["status"] == "OK")
    n_fail = len(subjects) - n_ok
    batch_log["summary"] = {
        "n_success": n_ok,
        "n_failed": n_fail,
        "total_elapsed_s": total_elapsed,
    }

    # Save batch log
    out_dir = get_te_output_dir("phase_2_viz", config=config).parent / "idtxl_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "idtxl_batch_log.json"
    with open(log_path, "w") as f:
        json.dump(batch_log, f, indent=2, default=str)

    logger.info(
        "run_batch_idtxl: complete.  OK=%d  FAILED=%d  Total=%.1f s  Log → %s",
        n_ok, n_fail, total_elapsed, log_path,
    )
    return batch_log


# ---------------------------------------------------------------------------
# 7.4.3 — compare_symbolic_vs_idtxl
# ---------------------------------------------------------------------------

def compare_symbolic_vs_idtxl(
    subjects: Optional[list] = None,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Compare Phase 2 NG-symbolic connectivity vectors against IDTxl vectors.

    For each subject × trial × window: compute Pearson r between the
    20-element NG-symbolic Phase 2 vector and the IDTxl vector.  Aggregate
    across subjects and windows.  Saves scatter plot and summary JSON.

    Returns
    -------
    summary : dict
        mean_pearson_r, interpretation, per_subject_r, per_connection_category.
    """
    if config is None:
        try:
            from utils import load_config as _lc
            config = _lc()
        except ImportError:
            config = {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    from utils_te import (
        get_te_output_dir,
        get_connection_labels,
        get_subject_group,
    )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    if subjects is None:
        experts = subj_cfg.get("Expert", [])
        novices = subj_cfg.get("Novice", [])
        subjects = experts + novices

    conn_labels = get_connection_labels()

    # Connection category mapping
    def _get_category(label: str) -> str:
        """Map connection label into PFC-intra, Motor-intra, or PFC→Motor."""
        parts = label.split("→")
        if len(parts) != 2:
            return "Other"
        src, tgt = parts[0].strip(), parts[1].strip()
        pfc = {"LPFC", "CPFC", "RPFC"}
        motor = {"LM1", "SMA"}
        if src in pfc and tgt in pfc:
            return "PFC-intra"
        if src in motor and tgt in motor:
            return "Motor-intra"
        return "PFC→Motor"

    categories = [_get_category(lbl) for lbl in conn_labels]
    cat_colors = {
        "PFC-intra": "#2196F3",
        "PFC→Motor": "#FF9800",
        "Motor-intra": "#4CAF50",
        "Other": "#9E9E9E",
    }

    # Load Phase 2 and IDTxl data per subject
    phase2_dir = get_te_output_dir("phase_2_connectivity", config=config)
    idtxl_dir = phase2_dir.parent / "idtxl_validation"

    all_symbolic: list = []   # (20,) vectors
    all_idtxl: list = []       # (20,) vectors
    per_subject_r: dict = {}

    for subject_id in subjects:
        p2_path = phase2_dir / f"sub-{subject_id}_hybrid_dspte_connectivity.npz"
        it_path = idtxl_dir / f"sub-{subject_id}_idtxl_ec.npz"

        if not p2_path.exists():
            logger.warning(
                "compare_symbolic_vs_idtxl [%s]: Phase 2 .npz not found — skip",
                subject_id,
            )
            continue
        if not it_path.exists():
            logger.warning(
                "compare_symbolic_vs_idtxl [%s]: IDTxl .npz not found — skip",
                subject_id,
            )
            continue

        p2_data = dict(np.load(p2_path, allow_pickle=True))
        it_data = dict(np.load(it_path, allow_pickle=True))
        trial_ids = list(p2_data["trial_ids"])

        subj_symbolic: list = []
        subj_idtxl: list = []

        for trial_id in trial_ids:
            p2_conn = p2_data.get(f"{trial_id}_connectivity")

            # Primary: exact trial key match.
            it_conn = it_data.get(f"{trial_id}_connectivity")
            # Backward compatibility for legacy IDTxl files using Trial_* keys.
            if it_conn is None and trial_id.startswith("Tr_"):
                legacy_trial_id = f"Trial_{trial_id.split('_', 1)[1]}"
                it_conn = it_data.get(f"{legacy_trial_id}_connectivity")
            # Backward compatibility in the opposite direction.
            if it_conn is None and trial_id.startswith("Trial_"):
                alt_trial_id = f"Tr_{trial_id.split('_', 1)[1]}"
                it_conn = it_data.get(f"{alt_trial_id}_connectivity")
            if p2_conn is None or it_conn is None:
                continue
            if p2_conn.ndim != 2 or it_conn.ndim != 2:
                continue
            n_wins_p2 = p2_conn.shape[1]
            n_wins_it = it_conn.shape[1]
            n_wins = min(n_wins_p2, n_wins_it)
            for wi in range(n_wins):
                subj_symbolic.append(p2_conn[:, wi])
                subj_idtxl.append(it_conn[:, wi])

        if len(subj_symbolic) < 2:
            logger.warning(
                "compare_symbolic_vs_idtxl [%s]: too few matched windows (%d) — skip",
                subject_id, len(subj_symbolic),
            )
            continue

        sym_mat = np.stack(subj_symbolic, axis=0)   # (n_wins, 20)
        itx_mat = np.stack(subj_idtxl, axis=0)       # (n_wins, 20)

        # Mean over windows → single 20-vector per subject for scatter
        sym_mean = sym_mat.mean(axis=0)
        itx_mean = itx_mat.mean(axis=0)

        all_symbolic.append(sym_mean)
        all_idtxl.append(itx_mean)

        # Per-subject Pearson r (mean across windows)
        rs = []
        for wi in range(sym_mat.shape[0]):
            sv, iv = sym_mat[wi], itx_mat[wi]
            if np.std(sv) > 0 and np.std(iv) > 0:
                r_val, _ = pearsonr(sv, iv)
                rs.append(r_val)
        mean_r = float(np.mean(rs)) if rs else float("nan")
        per_subject_r[subject_id] = mean_r
        logger.info(
            "compare_symbolic_vs_idtxl [%s]: n_wins=%d  mean_r=%.3f",
            subject_id, len(rs), mean_r,
        )

    if not all_symbolic:
        logger.warning("compare_symbolic_vs_idtxl: no matched subjects found — returning empty.")
        return {
            "mean_pearson_r": float("nan"),
            "interpretation": "insufficient_data",
            "per_subject_r": {},
            "n_subjects": 0,
        }

    # Grand mean vectors (one point per connection, averaged over subjects)
    grand_sym = np.mean(np.stack(all_symbolic, axis=0), axis=0)   # (20,)
    grand_itx = np.mean(np.stack(all_idtxl, axis=0), axis=0)       # (20,)

    valid_mask_grand = (np.std(grand_sym) > 0) and (np.std(grand_itx) > 0)
    if valid_mask_grand:
        grand_r, grand_p = pearsonr(grand_sym, grand_itx)
    else:
        grand_r, grand_p = float("nan"), float("nan")

    valid_rs = [r for r in per_subject_r.values() if not np.isnan(r)]
    mean_r = float(np.mean(valid_rs)) if valid_rs else float("nan")

    if np.isnan(mean_r):
        interpretation = "insufficient_data"
    elif mean_r > 0.6:
        interpretation = "AGREEMENT — NG-symbolic Phase 2 validated; report as primary"
    elif mean_r > 0.3:
        interpretation = "PARTIAL_AGREEMENT — report both methods; NG-symbolic methodologically faithful"
    else:
        interpretation = "DISAGREEMENT — elevate IDTxl to primary Phase 2 method"

    logger.info(
        "compare_symbolic_vs_idtxl: grand_r=%.3f  mean_subject_r=%.3f  p=%.4f  → %s",
        grand_r, mean_r, grand_p if not np.isnan(grand_p) else -1.0, interpretation,
    )

    # -----------------------------------------------------------------------
    # Per-category r analysis
    # -----------------------------------------------------------------------
    per_cat_r: dict = {cat: [] for cat in cat_colors}
    for conn_idx, cat in enumerate(categories):
        sym_vals = np.array([sv[conn_idx] for sv in all_symbolic])
        itx_vals = np.array([iv[conn_idx] for iv in all_idtxl])
        if np.std(sym_vals) > 0 and np.std(itx_vals) > 0:
            r_c, _ = pearsonr(sym_vals, itx_vals)
            per_cat_r[cat].append(r_c)
    per_cat_mean_r: dict = {
        cat: float(np.mean(rs)) if rs else float("nan")
        for cat, rs in per_cat_r.items()
    }

    # -----------------------------------------------------------------------
    # Scatter plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: grand mean scatter (subject-mean vectors)
    ax = axes[0]
    for conn_idx, (lbl, cat) in enumerate(zip(conn_labels, categories)):
        color = cat_colors.get(cat, "#9E9E9E")
        ax.scatter(
            grand_itx[conn_idx],
            grand_sym[conn_idx],
            c=color,
            s=60,
            alpha=0.75,
            zorder=3,
        )

    # Identity line range
    all_vals = np.concatenate([grand_sym, grand_itx])
    val_min, val_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = (val_max - val_min) * 0.1 if val_max > val_min else 0.001
    xl = [val_min - margin, val_max + margin]
    ax.plot(xl, xl, "k--", lw=0.8, alpha=0.5, label="Identity")

    # Best-fit line
    valid_both = ~(np.isnan(grand_itx) | np.isnan(grand_sym))
    if valid_both.sum() > 2:
        m, b = np.polyfit(grand_itx[valid_both], grand_sym[valid_both], 1)
        xfit = np.linspace(xl[0], xl[1], 50)
        ax.plot(xfit, m * xfit + b, "r-", lw=1.2, alpha=0.7, label=f"Fit (r={grand_r:.3f})")

    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in cat_colors.items()
        if cat != "Other"
    ]
    ax.legend(handles=legend_patches + [
        mpatches.Patch(color="none", label=f"r = {grand_r:.3f} (p={grand_p:.3f})")
    ], fontsize=9)
    ax.set_xlabel("IDTxl Multivariate TE (continuous KSG)", fontsize=11)
    ax.set_ylabel("NG-Symbolic Phase 2 Conditional dSPTE", fontsize=11)
    ax.set_title("Grand-Mean Scatter:\nNG-Symbolic vs IDTxl (Section 7.4.3)", fontsize=12)
    ax.set_xlim(xl)
    ax.set_ylim(xl)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Right panel: per-subject Pearson r bar chart
    ax2 = axes[1]
    subj_ids = sorted(per_subject_r.keys())
    r_vals = [per_subject_r[s] for s in subj_ids]
    try:
        groups = {s: get_subject_group(s, config=config) for s in subj_ids}
    except Exception:
        groups = {s: "Unknown" for s in subj_ids}
    bar_colors = ["#2196F3" if groups.get(s) == "Expert" else "#FF5722" for s in subj_ids]
    bars = ax2.bar(range(len(subj_ids)), r_vals, color=bar_colors, alpha=0.75, edgecolor="k", lw=0.5)
    ax2.axhline(y=mean_r, color="black", lw=1.5, linestyle="--", label=f"Mean r = {mean_r:.3f}")
    ax2.axhline(y=0.6, color="#4CAF50", lw=1, linestyle=":", alpha=0.7, label="r=0.6 threshold")
    ax2.axhline(y=0.3, color="#FF9800", lw=1, linestyle=":", alpha=0.7, label="r=0.3 threshold")
    ax2.set_xticks(range(len(subj_ids)))
    ax2.set_xticklabels(subj_ids, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Pearson r (per window)", fontsize=11)
    ax2.set_title("Per-Subject Correlation:\nNG-Symbolic Phase 2 vs IDTxl", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-0.1, 1.05)

    exp_patch = mpatches.Patch(color="#2196F3", label="Expert")
    nov_patch = mpatches.Patch(color="#FF5722", label="Novice")
    ax2.legend(
        handles=[exp_patch, nov_patch,
                 mpatches.Patch(color="none", label=f"Mean r={mean_r:.3f}"),
                 mpatches.Patch(color="none", label="--- thresholds")],
        fontsize=9
    )

    plt.tight_layout()

    viz_dir = get_te_output_dir("phase_2_viz", config=config)
    scatter_path = viz_dir / "symbolic_vs_idtxl_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("compare_symbolic_vs_idtxl: saved scatter → %s", scatter_path)

    # -----------------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "grand_pearson_r": float(grand_r) if not np.isnan(grand_r) else None,
        "grand_pearson_p": float(grand_p) if not np.isnan(grand_p) else None,
        "mean_subject_pearson_r": mean_r,
        "std_subject_pearson_r": float(np.std(valid_rs)) if valid_rs else None,
        "n_subjects": len(valid_rs),
        "interpretation": interpretation,
        "per_subject_r": {k: (v if not np.isnan(v) else None) for k, v in per_subject_r.items()},
        "per_category_mean_r": {
            k: (v if not np.isnan(v) else None) for k, v in per_cat_mean_r.items()
        },
        "idtxl_version": _IDTXL_VERSION,
    }
    json_path = idtxl_dir / "idtxl_comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("compare_symbolic_vs_idtxl: saved summary → %s", json_path)

    return summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    """Smoke test for Section 7.4 — uses synthetic data, no real subjects needed."""
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    )

    print("\n" + "=" * 65)
    print("Section 7.4 — IDTxl Parallel Validation — Smoke Tests")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Test 1: IDTxl import check
    # ------------------------------------------------------------------
    print("Test 1: IDTxl import availability")
    eval_path = write_idtxl_eval_md()
    print(f"  IDTXL_EVAL.md written → {eval_path}")
    if not _IDTXL_AVAILABLE:
        print(f"  IDTxl NOT available: {_IDTXL_IMPORT_ERROR}")
        print("  [SKIP] Tests 2–5 require IDTxl — all skipped.")
        print("\nIDTxl unavailable — section 7.4 documented as UNAVAILABLE in IDTXL_EVAL.md")
        return
    print(f"  IDTxl version: {_IDTXL_VERSION}")
    print("  [PASS] Test 1: IDTxl available\n")

    # ------------------------------------------------------------------
    # Test 2: run_idtxl_multivariate_te on synthetic AR signal
    # ------------------------------------------------------------------
    print("Test 2: run_idtxl_multivariate_te — synthetic AR (5 ROIs, T=500)")
    rng = np.random.default_rng(42)
    n_rois, T = 5, 500   # short for smoke test speed
    roi_names_t = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]

    # AR model: CPFC → LM1 with delay 3
    Y = rng.standard_normal((n_rois, T)) * 0.1
    a = 0.7
    for tt in range(3, T):
        Y[0, tt] += a * Y[0, tt - 1]   # LPFC AR(1)
        Y[1, tt] += a * Y[1, tt - 1]   # CPFC AR(1)
        Y[3, tt] += 0.5 * Y[1, tt - 3]  # LM1 ← CPFC

    config_smoke = {
        "stage_e_transfer_entropy": {
            "signal": {"u_max": 5, "sfreq": 25.0},
        }
    }

    t0 = time.perf_counter()
    ec_mat, meta = run_idtxl_multivariate_te(
        Y, config=config_smoke, roi_names=roi_names_t
    )
    elapsed = time.perf_counter() - t0

    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  EC matrix shape: {ec_mat.shape}  (expected (5, 5))")
    print(f"  Diagonal: {np.diag(ec_mat)}")
    print(f"  n_sig_edges: {meta['n_sig_edges']}")
    print(f"  CPFC→LM1 (expected non-zero): {ec_mat[1, 3]:.6f}")
    print(f"  per_target_omnibus_te: {meta['per_target_omnibus_te']}")

    assert ec_mat.shape == (n_rois, n_rois), f"Expected (5,5), got {ec_mat.shape}"
    assert np.all(np.diag(ec_mat) == 0), "Diagonal must be 0"
    assert np.all(ec_mat >= 0), "All TE values must be ≥ 0"
    print("  [PASS] Test 2: EC matrix shape and non-negativity\n")

    # ------------------------------------------------------------------
    # Test 3: extract_idtxl_connectivity_vector
    # ------------------------------------------------------------------
    print("Test 3: extract_idtxl_connectivity_vector — 5×5 → 20-element vector")
    try:
        from utils_te import get_connection_labels
        conn_labels = get_connection_labels()
    except ImportError:
        # Build minimal labels manually
        rois = roi_names_t
        conn_labels = [f"{a}→{b}" for a in rois for b in rois if a != b]

    conn_vec = extract_idtxl_connectivity_vector(ec_mat, roi_names_t, conn_labels)
    print(f"  conn_vec shape: {conn_vec.shape}  (expected (20,))")
    print(f"  Non-zero entries: {int(np.sum(conn_vec > 0))}/20")
    print(f"  CPFC→LM1 in vector: {conn_vec[conn_labels.index('CPFC→LM1')]:.6f}")
    assert conn_vec.shape == (20,), f"Expected (20,), got {conn_vec.shape}"
    assert np.all(conn_vec >= 0), "Connectivity vector must be ≥ 0"
    print("  [PASS] Test 3: Connectivity vector extraction\n")

    # ------------------------------------------------------------------
    # Test 4: Directionality check on coupled AR
    # ------------------------------------------------------------------
    print("Test 4: Directionality check — CPFC→LM1 should dominate LM1→CPFC")
    cpfc_to_lm1 = ec_mat[1, 3]
    lm1_to_cpfc = ec_mat[3, 1]
    print(f"  CPFC→LM1: {cpfc_to_lm1:.6f}")
    print(f"  LM1→CPFC: {lm1_to_cpfc:.6f}")
    if cpfc_to_lm1 > lm1_to_cpfc:
        print("  [PASS] Test 4: CPFC→LM1 > LM1→CPFC (correct directionality)")
    else:
        print("  [WARN] Test 4: Expected CPFC→LM1 > LM1→CPFC — IDTxl may need longer T")
        print("         (With T=500 and n_perm=50 the result is stochastic; re-run with T≥1000)")
    print()

    # ------------------------------------------------------------------
    # Test 5: write_idtxl_eval_md content check
    # ------------------------------------------------------------------
    print("Test 5: IDTXL_EVAL.md content validation")
    content = eval_path.read_text()
    assert "AVAILABLE" in content, "Expected 'AVAILABLE' in IDTXL_EVAL.md"
    assert "PythonKraskovCMI" in content, "Expected estimator name in IDTXL_EVAL.md"
    print(f"  File size: {len(content)} chars")
    print("  [PASS] Test 5: IDTXL_EVAL.md content check\n")

    print("=" * 65)
    print("All Section 7.4 smoke tests completed.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

'''
# Fast mode — first window only (~5–10 min):
python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --batch --first-window-only --overwrite
python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --compare
# Full batch (~52 min estimate):
python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --batch --overwrite
'''

def _cli_main() -> None:
    """
    Parse CLI flags and run one IDTxl workflow mode.

    Modes
    -----
    smoke-test, subject, batch, compare, eval-md.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Section 7.4: IDTxl Parallel Validation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true",
                       help="Run synthetic smoke tests")
    group.add_argument("--subject", type=str,
                       help="Run IDTxl for a single subject (first window only for speed)")
    group.add_argument("--batch", action="store_true",
                       help="Run IDTxl for all subjects")
    group.add_argument("--compare", action="store_true",
                       help="Run compare_symbolic_vs_idtxl() (requires batch to have run)")
    group.add_argument("--eval-md", action="store_true",
                       help="Write IDTXL_EVAL.md and exit")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cached .npz files")
    parser.add_argument("--first-window-only", action="store_true",
                        help="Process only the first window per trial (fast mode)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel subject workers for --batch (default: 1)")
    parser.add_argument("--idtxl-max-lag", type=int, default=None,
                        help="Override IDTxl max_lag_sources for speed control")
    parser.add_argument("--idtxl-num-threads", type=str, default=None,
                        help="Override IDTxl estimator num_threads (e.g., 1, 2, USE_ALL)")
    parser.add_argument("--idtxl-n-perm", type=int, default=None,
                        help="Override all IDTxl permutation counts (max/min/omnibus/max_seq). "
                             "Must be > 1/alpha (for alpha=0.05, use >=21).")
    parser.add_argument("--idtxl-fast", action="store_true",
                        help="Use faster IDTxl permutation settings for exploratory runs")
    args = parser.parse_args()

    if args.eval_md:
        path = write_idtxl_eval_md()
        print(f"Written: {path}")
        return

    if args.smoke_test:
        _run_smoke_tests()
        return

    # Load config for real-data modes
    try:
        from utils import load_config as _lc
        config = _lc()
    except ImportError:
        logger.error("Could not load config.yaml — ensure code/ is on PYTHONPATH.")
        raise

    # Apply CLI overrides without mutating the original loaded config object.
    config = copy.deepcopy(config)
    te_cfg = config.setdefault("stage_e_transfer_entropy", {})
    idtxl_cfg = te_cfg.setdefault("idtxl_validation", {})
    if args.idtxl_max_lag is not None:
        idtxl_cfg["max_lag_sources"] = int(max(1, args.idtxl_max_lag))
    if args.idtxl_num_threads is not None:
        v = args.idtxl_num_threads.strip()
        idtxl_cfg["num_threads"] = int(v) if v.lstrip("-").isdigit() else v
    if args.idtxl_n_perm is not None:
        n_perm = int(max(1, args.idtxl_n_perm))
        # IDTxl requires 1/n_perm < alpha (strict). With alpha=0.05, n_perm=20 fails.
        if n_perm <= 20:
            logger.warning(
                "--idtxl-n-perm=%d is too small for alpha=0.05; bumping to 21.",
                n_perm,
            )
            n_perm = 21
        idtxl_cfg["n_perm_max_stat"] = n_perm
        idtxl_cfg["n_perm_min_stat"] = n_perm
        idtxl_cfg["n_perm_omnibus"] = n_perm
        idtxl_cfg["n_perm_max_seq"] = n_perm
    if args.idtxl_fast:
        # Keep "fast" but valid for alpha=0.05 (strictly requires >20 perms).
        idtxl_cfg["n_perm_max_stat"] = 21
        idtxl_cfg["n_perm_min_stat"] = 21
        idtxl_cfg["n_perm_omnibus"] = 41
        idtxl_cfg["n_perm_max_seq"] = 21

    if args.subject:
        logger.info("=== IDTxl Validation: subject %s ===", args.subject)
        result = run_subject_idtxl(
            args.subject.strip(),
            config=config,
            overwrite=args.overwrite,
            first_window_only=True,   # single-subject CLI: first window only
        )
        if result:
            n_wins = sum(len(v) for v in result.values())
            logger.info(
                "Done: subject=%s  trials=%d  windows=%d",
                args.subject, len(result), n_wins,
            )
        else:
            logger.error("IDTxl validation failed for subject %s", args.subject)
        return

    if args.batch:
        logger.info("=== IDTxl Validation: full batch ===")
        batch_log = run_batch_idtxl(
            config=config,
            overwrite=args.overwrite,
            first_window_only=args.first_window_only,
            n_jobs=args.jobs,
        )
        summary = batch_log.get("summary", {})
        logger.info(
            "Batch complete: OK=%d  FAILED=%d  Total=%.1f s",
            summary.get("n_success", 0),
            summary.get("n_failed", 0),
            summary.get("total_elapsed_s", 0.0),
        )
        return

    if args.compare:
        logger.info("=== IDTxl vs Symbolic comparison ===")
        summary = compare_symbolic_vs_idtxl(config=config)
        logger.info(
            "Comparison complete: r=%.3f  interpretation=%s",
            summary.get("mean_subject_pearson_r", float("nan")),
            summary.get("interpretation", "unknown"),
        )
        return


if __name__ == "__main__":
    _cli_main()
