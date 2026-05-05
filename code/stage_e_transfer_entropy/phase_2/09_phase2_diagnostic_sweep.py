"""
Stage E — Phase 2 Diagnostic Parameter Sweep

Tests combinations of Phase 2 hyperparameters to find configurations that
maximise group differentiation (Expert vs Novice) in hybrid-dSPTE.

**Ranking policy (recommended shortlist for full-cohort runs):**
  1. **Quality gates** — drop configs with unreliable conditional estimation
     (default: ``mean_fallback_pairs >= 5`` mean across subjects).
  2. **Robust effect size** — rank survivors by **trimmed mean** (10 % / side)
     of |Cohen's d| across the 20 connections, with **median** |d| as tie-breaker.
     ``max_abs_cohens_d`` is kept for diagnostics only (single-edge lottery).

**Problem:** The current configuration produces nearly uniform TE values
(~0.01–0.02) across all connections and both groups. This script sweeps
the key parameters and reports:
  - Mean TE per group
  - Expert vs Novice absolute difference
  - Cohen's d effect size (per-connection + robust aggregates)
  - Sparsity / fallback diagnostics

**Parameters swept:**
  1. k0_max_multivariate        (2, 3, 4, 6, 8)
  2. conditional_pca.enabled    (True, False)
  3. conditional_pca.n_components  (1, 2)
  4. sparsity_fallback_threshold   (0.20, 0.50, 0.80, 0.95)
  5. embedding_dimension           (2, 3)
  6. narrowband_filter             (Wang2020: 0.01–0.10, Lobier2014: 0.04–0.15, Wider: 0.01–0.15)
  7. u_max                         (50, 150, 300)

Strategy: Run on 2 Expert (A1, R7) + 2 Novice (R1, R3) subjects for speed.
Uses Trial 1 Window 1 from each subject.
Phase/filter extracted once per (subject, filter) combo, then TE params swept.

Output: outputs/stage_e_transfer_entropy/diagnostics/phase2_param_sweep.csv
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import trim_mean

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).parent              # phase_2/
_STAGE_E_ROOT = _SCRIPT_DIR.parent                # stage_e_transfer_entropy/
_CODE_ROOT    = _STAGE_E_ROOT.parent              # code/
_PROJECT_ROOT = _CODE_ROOT.parent                 # repo root

for _p in [str(_CODE_ROOT), str(_STAGE_E_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_config  # noqa: E402

logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s %(name)s: %(message)s")
# Suppress all child loggers (hybrid_diag, utils_te_mod, etc.) that fire per-lag warnings
for _noisy in ["hybrid_diag", "utils_te_mod", "utils_te_diag", "dspte_core_diag"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
logger = logging.getLogger("phase2_diag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERTS      = ["A1", "R7"]
NOVICES      = ["R1", "R3"]
ALL_SUBJECTS = EXPERTS + NOVICES
GROUPS       = {sid: ("Expert" if sid in EXPERTS else "Novice") for sid in ALL_SUBJECTS}

ROI_NAMES     = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
WINDOW_SIZE   = 1350  # samples (MATLAB: 54 s × 25 Hz)
DIAG_OUT_DIR  = _PROJECT_ROOT / "outputs" / "stage_e_transfer_entropy" / "diagnostics"
DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Quality gate: mean n_pairs_fallback averaged across subjects (float).
MAX_MEAN_FALLBACK_PAIRS_EXCLUSIVE = 5.0  # keep rows with mean_fallback_pairs < this
TRIM_MEAN_PROPORTION = 0.1  # 10 % trimmed from each end of |d| vector (20 connections)

# ---------------------------------------------------------------------------
# Lazy module loaders
# ---------------------------------------------------------------------------

def _load_mod(name, rel_path):
    """Load a Stage E module from `rel_path` and return the imported module."""
    spec = importlib.util.spec_from_file_location(name, _STAGE_E_ROOT / rel_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_seg_mod():
    """Load Stage D trial segmentation module used for event-based trial slicing."""
    spec = importlib.util.spec_from_file_location(
        "trial_seg_diag",
        _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# Load subject ROI signals + trial segments
# ---------------------------------------------------------------------------

def load_subject_trials(subject_id: str, config: dict) -> dict | None:
    """
    Load ROI signals and segment into trials using the same pipeline as Stage E.
    Returns dict with 'trials' (list of np.ndarray (n_rois, T_trial)), 'sfreq',
    'roi_names'. Returns None on failure.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        utils_te = _load_mod("utils_te_diag", "utils_te.py")  # at stage_e_transfer_entropy/ root

        # Load ROI signals from Stage D cache
        try:
            Data_Reg, valid_roi_names, sfreq = utils_te.load_or_extract_roi_signals(
                subject_id, study="expert_novice", config=config
            )
        except Exception as exc:
            logger.error("load_subject_trials [%s]: ROI load failed: %s", subject_id, exc)
            return None

        # Load raw .fif for trial segmentation (events/annotations only)
        try:
            raw = utils_te.load_preprocessed_fif(
                subject_id, study="expert_novice", config=config
            )
        except Exception as exc:
            logger.error("load_subject_trials [%s]: FIF load failed: %s", subject_id, exc)
            return None

        # Trial segmentation
        seg_mod = _load_seg_mod()
        try:
            segments = seg_mod.get_trial_segments(raw, config=config)
        except Exception as exc:
            logger.error("load_subject_trials [%s]: segmentation failed: %s", subject_id, exc)
            return None

    trials = []
    for onset, offset in segments:
        seg = Data_Reg[:, onset: offset + 1]
        if seg.shape[1] >= WINDOW_SIZE + 1:
            trials.append(seg)

    if not trials:
        logger.warning("load_subject_trials [%s]: no valid trials", subject_id)
        return None

    return {
        "subject_id": subject_id,
        "sfreq": float(sfreq),
        "roi_names": [str(r) for r in valid_roi_names],
        "trials": trials,
        "n_trials": len(trials),
    }


# ---------------------------------------------------------------------------
# Narrowband filter + Hilbert phase extraction (full trial)
# ---------------------------------------------------------------------------

def extract_phase_trial(
    Y_trial: np.ndarray,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    order: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """
    Apply narrowband Butterworth filter to full trial then extract Hilbert phase.
    Returns (Phi_trial: (n_rois, T), taus: list[int]).
    """
    n_rois, T = Y_trial.shape
    sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=sfreq, output="sos")
    Phi = np.full_like(Y_trial, np.nan)
    for ri in range(n_rois):
        sig = Y_trial[ri]
        if np.all(np.isnan(sig)):
            continue
        try:
            filtered = sosfiltfilt(sos, sig)
            Phi[ri] = np.angle(hilbert(filtered))
        except Exception:
            pass

    # Compute ACF 1/e embedding delay per ROI
    taus = []
    for ri in range(n_rois):
        sig = Phi[ri]
        if np.all(np.isnan(sig)):
            taus.append(5)
            continue
        taus.append(_compute_tau_1e(sig))

    return Phi, taus


def _compute_tau_1e(signal: np.ndarray, max_lag: int = 300) -> int:
    """Autocorrelation 1/e embedding delay (mirrors 01_dspte_core logic)."""
    x = signal - np.nanmean(signal)
    n = len(x)
    var = np.nanvar(x)
    if var < 1e-12:
        return 1
    threshold = np.exp(-1) * var
    for lag in range(1, min(max_lag, n // 2)):
        acf = float(np.nansum(x[:n - lag] * x[lag:]) / ((n - lag) * var))
        if acf <= threshold:
            return lag
    return min(max_lag, n // 2 - 1)


# ---------------------------------------------------------------------------
# Core: compute Phase 2 TE for one phase window with given params
# ---------------------------------------------------------------------------

def _make_cfg(k0_max, d, u_max, pca_enabled, pca_n_comp, sparsity_thr) -> dict:
    """
    Build a minimal in-memory config dict for one sweep parameter combination.

    Returns
    -------
    dict
        Config subset consumed by `compute_hybrid_dspte_matrix`.
    """
    return {
        "stage_e_transfer_entropy": {
            "signal": {
                "u_max": u_max,
                "embedding_dimension": d,
            },
            "symbolisation": {
                "ndis_threshold": 0.05,
                "k0_max_multivariate": k0_max,
                "ng_max_iterations": 300,
                "ng_learning_rate_i": 0.5,
                "ng_learning_rate_f": 0.01,
                "ng_neighbourhood_i": 10.0,
                "ng_neighbourhood_f": 0.1,
            },
            "phase_2": {
                "sparsity_fallback_threshold": sparsity_thr,
                "tau_divergence_ratio": 2.0,
                "degenerate_entropy_threshold": 1.0,
                "conditional_pca": {
                    "enabled": pca_enabled,
                    "n_components": pca_n_comp,
                    "min_explained_var_warning_pct": 60.0,
                },
            },
        }
    }


_HYBRID_MODULE_CACHE = None


def _get_hybrid_cached():
    """Return cached import of `06_hybrid_dspte.py` to avoid repeated reloads."""
    global _HYBRID_MODULE_CACHE
    if _HYBRID_MODULE_CACHE is None:
        _HYBRID_MODULE_CACHE = _load_mod("hybrid_diag", "phase_2/06_hybrid_dspte.py")
        # Suppress any loggers registered inside the freshly loaded module
        for name in list(logging.Logger.manager.loggerDict.keys()):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    return _HYBRID_MODULE_CACHE


def compute_phase2_te_window(
    Phi_win: np.ndarray,
    taus: list[int],
    k0_max: int,
    d: int,
    u_max: int,
    pca_enabled: bool,
    pca_n_comp: int,
    sparsity_thr: float,
) -> dict:
    """Compute hybrid-dSPTE for one window. Returns metrics dict."""
    hybrid = _get_hybrid_cached()
    cfg = _make_cfg(k0_max, d, u_max, pca_enabled, pca_n_comp, sparsity_thr)

    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            EC_matrix, delay_matrix, meta = hybrid.compute_hybrid_dspte_matrix(
                Phi_win, taus=taus, config=cfg, roi_names=ROI_NAMES,
            )
        elapsed = time.time() - t0

        mask = ~np.eye(5, dtype=bool)
        ec_vec = EC_matrix[mask]

        return {
            "success": True,
            "ec_vec": ec_vec,
            "mean_te": float(np.nanmean(ec_vec)),
            "max_te": float(np.nanmax(ec_vec)),
            "std_te": float(np.nanstd(ec_vec)),
            "elapsed_s": elapsed,
            "k0": meta.get("k0", k0_max),
            "conditioning_mode": meta.get("conditioning_mode", "?"),
            "mean_sparsity_pct": meta.get("mean_sparsity_pct", float("nan")),
            "mean_count_occupied": meta.get("mean_count_occupied", float("nan")),
            "n_pairs_fallback": meta.get("n_pairs_fallback", 0),
            "n_pairs_total": meta.get("n_pairs_total", 20),
            "majority_fallback": meta.get("majority_pairs_fallback", False),
            "n_degenerate_rois": len(meta.get("degenerate_rois", [])),
            "tau_divergence": meta.get("tau_divergence_warning", False),
            "nDis": meta.get("nDis", float("nan")),
        }
    except Exception as exc:
        return {
            "success": False,
            "ec_vec": np.zeros(20),
            "mean_te": float("nan"),
            "max_te": float("nan"),
            "std_te": float("nan"),
            "elapsed_s": time.time() - t0,
            "error": str(exc)[:150],
        }


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d for two arrays after dropping NaNs.

    Returns
    -------
    float
        Standardised mean difference; NaN if either sample has <2 values.
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled_std < 1e-14:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _robust_abs_d_aggregates(conn_d: np.ndarray, proportiontocut: float = TRIM_MEAN_PROPORTION):
    """
    Median and trimmed mean of |Cohen's d| across connections.

    Parameters
    ----------
    conn_d : (n_conn,) float
    proportiontocut : float
        Passed to ``scipy.stats.trim_mean`` (fraction trimmed from each tail).

    Returns
    -------
    median_abs, trimmed_mean_abs : float
        NaN if no finite |d| values.
    """
    x = np.abs(np.asarray(conn_d, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    # trim_mean requires proportiontocut in [0, 0.5)
    pc = float(np.clip(proportiontocut, 0.0, 0.49))
    if x.size < 3 or pc <= 0.0:
        tmean = float(np.mean(x))
    else:
        tmean = float(trim_mean(x, proportiontocut=pc))
    return med, tmean


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    """
    Execute the full Phase 2 diagnostic parameter sweep and write CSV output.

    Returns
    -------
    None
        Results are saved to `outputs/stage_e_transfer_entropy/diagnostics/`.
    """
    # Suppress all loggers upfront (some modules register loggers on import)
    logging.disable(logging.WARNING)

    config = load_config()

    # -------------------------------------------------------------------
    # Parameter grid
    # -------------------------------------------------------------------
    PARAM_GRID = {
        "k0_max":             [2, 3, 4, 6, 8],
        "pca_enabled":        [True, False],
        "pca_n_components":   [1, 2],
        "sparsity_threshold": [0.20, 0.50, 0.80, 0.95],
        "d":                  [2, 3],
        "filter_band": [
            ("Lobier2014", 0.04, 0.15),  # current default
            ("Wang2020",   0.01, 0.10),  # Wang (2020) original
            ("Wider",      0.01, 0.15),  # combined range
        ],
        "u_max": [50, 150, 300],
    }

    # -------------------------------------------------------------------
    # Load subject data
    # -------------------------------------------------------------------
    print("Loading subject data (ROI signals + trial segments)...")
    subject_data: dict = {}
    for sid in ALL_SUBJECTS:
        data = load_subject_trials(sid, config)
        if data:
            subject_data[sid] = data
            print(f"  {sid} ({GROUPS[sid]}): {data['n_trials']} trials, "
                  f"sfreq={data['sfreq']:.0f} Hz")
        else:
            print(f"  {sid}: FAILED")

    if len(subject_data) < 2:
        print("ERROR: Fewer than 2 subjects loaded. Aborting.")
        return

    # -------------------------------------------------------------------
    # Pre-cache phase matrices per (subject, filter_band)
    # Filtering done on full trial, then Window 1 sliced for each combo
    # -------------------------------------------------------------------
    print("\nPre-computing phase matrices per (subject, filter_band)...")
    phase_cache: dict = {}  # key: (sid, band_name) → {trial_phases, taus}

    for sid, sdata in subject_data.items():
        sfreq = sdata["sfreq"]
        for band_name, f_lo, f_hi in PARAM_GRID["filter_band"]:
            key = (sid, band_name)
            trial_phases_list = []
            taus_list_per_trial = []
            for seg in sdata["trials"]:
                Phi, taus = extract_phase_trial(seg, sfreq, f_lo, f_hi, order=3)
                trial_phases_list.append(Phi)
                taus_list_per_trial.append(taus)
            phase_cache[key] = {
                "trial_phases": trial_phases_list,
                "taus_per_trial": taus_list_per_trial,
            }
        print(f"  {sid}: done ({len(PARAM_GRID['filter_band'])} bands)")

    # -------------------------------------------------------------------
    # Build parameter combinations (prune redundant ones)
    # -------------------------------------------------------------------
    combos = []
    for (k0_max, pca_enabled, pca_n_comp, sparsity_thr, d,
         filter_band_tuple, u_max) in product(
        PARAM_GRID["k0_max"],
        PARAM_GRID["pca_enabled"],
        PARAM_GRID["pca_n_components"],
        PARAM_GRID["sparsity_threshold"],
        PARAM_GRID["d"],
        PARAM_GRID["filter_band"],
        PARAM_GRID["u_max"],
    ):
        band_name, f_lo, f_hi = filter_band_tuple

        # pca_n_components irrelevant when PCA disabled
        if not pca_enabled and pca_n_comp > 1:
            continue
        # pca_n_components must be < k0_max to be meaningful
        if pca_enabled and pca_n_comp >= k0_max:
            continue

        combos.append({
            "k0_max": k0_max,
            "pca_enabled": pca_enabled,
            "pca_n_components": pca_n_comp if pca_enabled else 0,
            "sparsity_threshold": sparsity_thr,
            "d": d,
            "band_name": band_name,
            "f_lo": f_lo,
            "f_hi": f_hi,
            "u_max": u_max,
        })

    print(f"\nTotal parameter combinations: {len(combos)}")
    print("Running on Trial 1, Window 1 per subject...")
    t_sweep_start = time.time()

    # -------------------------------------------------------------------
    # Sweep
    # -------------------------------------------------------------------
    results = []
    n_combos = len(combos)

    for combo_idx, params in enumerate(combos):
        k0_max       = params["k0_max"]
        pca_enabled  = params["pca_enabled"]
        pca_n_comp   = params["pca_n_components"]
        sparsity_thr = params["sparsity_threshold"]
        d            = params["d"]
        band_name    = params["band_name"]
        f_lo         = params["f_lo"]
        f_hi         = params["f_hi"]
        u_max        = params["u_max"]

        if (combo_idx + 1) % 100 == 0:
            elapsed_min = (time.time() - t_sweep_start) / 60
            eta_min = elapsed_min / (combo_idx + 1) * (n_combos - combo_idx - 1)
            print(f"  [{combo_idx+1}/{n_combos}] elapsed={elapsed_min:.1f}min  "
                  f"eta={eta_min:.1f}min  "
                  f"k0={k0_max}|pca={'Y' if pca_enabled else 'N'}|"
                  f"sp={sparsity_thr}|d={d}|{band_name}|u={u_max}")

        exp_mean_tes, nov_mean_tes = [], []
        exp_ec_vecs,  nov_ec_vecs  = [], []
        diag_rows = []

        for sid, sdata in subject_data.items():
            band_key = (sid, band_name)
            trial_phases  = phase_cache[band_key]["trial_phases"]
            taus_per_trial = phase_cache[band_key]["taus_per_trial"]

            if not trial_phases:
                continue

            # Use Trial 1, Window 1
            Phi_trial = trial_phases[0]
            taus = taus_per_trial[0]
            T_trial = Phi_trial.shape[1]

            if T_trial < WINDOW_SIZE + 1:
                continue

            # MATLAB-style window 1: samples [0 : WINDOW_SIZE+1] (1351 samples)
            Phi_win = Phi_trial[:, : WINDOW_SIZE + 1]

            res = compute_phase2_te_window(
                Phi_win, taus, k0_max, d, u_max,
                pca_enabled, pca_n_comp, sparsity_thr,
            )

            group = GROUPS[sid]
            mean_te = res["mean_te"]

            row = {
                "subject": sid,
                "group": group,
                "mean_te": mean_te,
                "max_te": res.get("max_te", float("nan")),
                "std_te": res.get("std_te", float("nan")),
                "k0_used": res.get("k0", k0_max),
                "conditioning_mode": res.get("conditioning_mode", "?"),
                "mean_sparsity_pct": res.get("mean_sparsity_pct", float("nan")),
                "mean_count_occupied": res.get("mean_count_occupied", float("nan")),
                "n_pairs_fallback": res.get("n_pairs_fallback", 0),
                "elapsed_s": res.get("elapsed_s", float("nan")),
            }
            if not res["success"]:
                row["error"] = res.get("error", "?")
            diag_rows.append(row)

            if res["success"] and not np.isnan(mean_te):
                if group == "Expert":
                    exp_mean_tes.append(mean_te)
                    exp_ec_vecs.append(res["ec_vec"])
                else:
                    nov_mean_tes.append(mean_te)
                    nov_ec_vecs.append(res["ec_vec"])

        # -------------------------------------------------------------------
        # Aggregate stats
        # -------------------------------------------------------------------
        exp_arr = np.array(exp_mean_tes) if exp_mean_tes else np.array([float("nan")])
        nov_arr = np.array(nov_mean_tes) if nov_mean_tes else np.array([float("nan")])

        exp_global_mean = float(np.nanmean(exp_arr))
        nov_global_mean = float(np.nanmean(nov_arr))
        exp_nov_diff    = exp_global_mean - nov_global_mean

        if exp_ec_vecs and nov_ec_vecs:
            exp_mat = np.vstack(exp_ec_vecs)
            nov_mat = np.vstack(nov_ec_vecs)
            conn_d  = np.array([cohens_d(exp_mat[:, c], nov_mat[:, c]) for c in range(20)])
            mean_abs_d = float(np.nanmean(np.abs(conn_d)))
            max_abs_d  = float(np.nanmax(np.abs(conn_d)))
            median_abs_d, trim_mean_abs_d = _robust_abs_d_aggregates(conn_d)
            n_d_gt_02  = int(np.sum(np.abs(conn_d) > 0.2))
            n_d_gt_05  = int(np.sum(np.abs(conn_d) > 0.5))

            # Connection with highest |d| (diagnostic only — not used for ranking)
            best_conn_idx = int(np.nanargmax(np.abs(conn_d)))
            CONN_LABELS = [
                f"{s}→{t}" for s in ROI_NAMES for t in ROI_NAMES if s != t
            ]
            best_conn_label = CONN_LABELS[best_conn_idx] if best_conn_idx < len(CONN_LABELS) else "?"
            best_conn_d = float(conn_d[best_conn_idx])
        else:
            mean_abs_d = max_abs_d = float("nan")
            median_abs_d = trim_mean_abs_d = float("nan")
            n_d_gt_02 = n_d_gt_05 = 0
            best_conn_label = "?"
            best_conn_d = float("nan")

        valid_diag = [r for r in diag_rows if "mean_sparsity_pct" in r
                      and not np.isnan(r.get("mean_sparsity_pct", float("nan")))]
        mean_sparsity_all = float(np.mean([r["mean_sparsity_pct"] for r in valid_diag])) if valid_diag else float("nan")
        mean_count_occ    = float(np.mean([r["mean_count_occupied"] for r in valid_diag])) if valid_diag else float("nan")
        mean_fallback_pairs = float(np.mean([r["n_pairs_fallback"] for r in valid_diag])) if valid_diag else float("nan")
        mean_elapsed      = float(np.mean([r["elapsed_s"] for r in valid_diag])) if valid_diag else float("nan")

        # k0 actually used (may differ from k0_max due to nDis criterion)
        k0_vals = [r["k0_used"] for r in diag_rows if "k0_used" in r]
        modal_k0 = int(np.round(np.mean(k0_vals))) if k0_vals else k0_max

        results.append({
            # Parameters
            "k0_max":               k0_max,
            "pca_enabled":          pca_enabled,
            "pca_n_components":     pca_n_comp,
            "sparsity_threshold":   sparsity_thr,
            "embedding_dim_d":      d,
            "filter_band":          band_name,
            "f_lo_Hz":              f_lo,
            "f_hi_Hz":              f_hi,
            "u_max":                u_max,

            # Outcome metrics
            "expert_mean_te":       exp_global_mean,
            "novice_mean_te":       nov_global_mean,
            "exp_nov_diff":         exp_nov_diff,
            "exp_nov_abs_diff":     abs(exp_nov_diff),
            "mean_abs_cohens_d":    mean_abs_d,
            "median_abs_cohens_d":  median_abs_d,
            "trimmed_mean_abs_cohens_d": trim_mean_abs_d,
            "max_abs_cohens_d":     max_abs_d,
            "n_conn_d_gt_0.2":      n_d_gt_02,
            "n_conn_d_gt_0.5":      n_d_gt_05,
            "best_conn":            best_conn_label,
            "best_conn_d":          best_conn_d,

            # Diagnostics
            "modal_k0_used":        modal_k0,
            "mean_sparsity_pct":    mean_sparsity_all,
            "mean_count_occupied":  mean_count_occ,
            "mean_fallback_pairs":  mean_fallback_pairs,
            "mean_elapsed_s":       mean_elapsed,
            "n_experts_ok":         len(exp_mean_tes),
            "n_novices_ok":         len(nov_mean_tes),
        })

    # -------------------------------------------------------------------
    # Save and summarise
    # -------------------------------------------------------------------
    logging.disable(logging.NOTSET)  # re-enable for summary output

    df = pd.DataFrame(results)

    # -------------------------------------------------------------------
    # Quality gates + recommended ranking (trimmed mean |d|, then median |d|)
    # -------------------------------------------------------------------
    gate_fb = df["mean_fallback_pairs"] < MAX_MEAN_FALLBACK_PAIRS_EXCLUSIVE
    gate_finite = np.isfinite(df["trimmed_mean_abs_cohens_d"].to_numpy(dtype=float))
    df["quality_pass"] = gate_fb & gate_finite

    df["rank_recommended"] = np.nan
    pass_order = (
        df.loc[df["quality_pass"]]
        .sort_values(
            ["trimmed_mean_abs_cohens_d", "median_abs_cohens_d"],
            ascending=[False, False],
        )
        .index.to_list()
    )
    for rnk, idx in enumerate(pass_order, start=1):
        df.loc[idx, "rank_recommended"] = float(rnk)

    # Gated configs first (best rank first), then failed gate for audit
    df = df.sort_values(
        ["quality_pass", "rank_recommended"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    # Column order: recommendation flags first, then parameters, then metrics
    _first = [
        "rank_recommended",
        "quality_pass",
        "k0_max",
        "pca_enabled",
        "pca_n_components",
        "sparsity_threshold",
        "embedding_dim_d",
        "filter_band",
        "f_lo_Hz",
        "f_hi_Hz",
        "u_max",
    ]
    _rest = [c for c in df.columns if c not in _first]
    df = df[_first + _rest]

    n_pass = int(df["quality_pass"].sum())
    n_total = len(df)
    out_csv = DIAG_OUT_DIR / "phase2_param_sweep.csv"
    df.to_csv(out_csv, index=False)
    total_min = (time.time() - t_sweep_start) / 60
    print(f"\nSaved {len(df)} rows to: {out_csv}  (total time: {total_min:.1f} min)")
    print(
        f"Quality gate: mean_fallback_pairs < {MAX_MEAN_FALLBACK_PAIRS_EXCLUSIVE:g} "
        f"→ {n_pass}/{n_total} configs pass (rank_recommended 1..{n_pass})."
    )

    # -------------------------------------------------------------------
    # Summary prints
    # -------------------------------------------------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)

    df_g = df[df["quality_pass"]]
    print("\n" + "=" * 90)
    print(
        "TOP 20 CONFIGURATIONS (quality-pass only) by "
        "trimmed mean |Cohen's d|, then median |d|"
    )
    print("=" * 90)
    top_cols = [
        "rank_recommended", "k0_max", "pca_enabled", "pca_n_components",
        "sparsity_threshold", "embedding_dim_d", "filter_band", "u_max",
        "expert_mean_te", "novice_mean_te", "exp_nov_abs_diff",
        "trimmed_mean_abs_cohens_d", "median_abs_cohens_d", "mean_abs_cohens_d",
        "max_abs_cohens_d", "n_conn_d_gt_0.2",
        "best_conn", "mean_fallback_pairs", "mean_sparsity_pct", "mean_count_occupied",
    ]
    if len(df_g) == 0:
        print("WARNING: No configs passed quality gates. Showing top 20 by trimmed mean |d| (all).")
        df_show = df.sort_values(
            "trimmed_mean_abs_cohens_d", ascending=False, na_position="last"
        ).head(20)
        show_cols = [c for c in top_cols if c != "rank_recommended"]
        print(df_show[show_cols].to_string(index=False))
    else:
        print(df_g.head(20)[top_cols].to_string(index=False))

    agg_trim = ("trimmed_mean_abs_cohens_d", "mean")
    agg_med = ("median_abs_cohens_d", "mean")
    agg_max = ("max_abs_cohens_d", "mean")
    agg_mean = ("mean_abs_cohens_d", "mean")

    print("\n" + "=" * 90)
    print("FILTER BAND SUMMARY (quality-pass configs only)")
    print("=" * 90)
    if len(df_g):
        print(df_g.groupby("filter_band").agg(
            mean_trim_d=agg_trim,
            mean_median_d=agg_med,
            mean_max_d=agg_max,
            mean_exp_te=("expert_mean_te", "mean"),
            mean_nov_te=("novice_mean_te", "mean"),
        ).sort_values("mean_trim_d", ascending=False).to_string())
    else:
        print("(no rows)")

    print("\n" + "=" * 90)
    print("PCA vs NO-PCA SUMMARY (quality-pass only)")
    print("=" * 90)
    if len(df_g):
        print(df_g.groupby("pca_enabled").agg(
            mean_trim_d=agg_trim,
            mean_median_d=agg_med,
            mean_max_d=agg_max,
            mean_sparsity=("mean_sparsity_pct", "mean"),
            mean_count_occ=("mean_count_occupied", "mean"),
            mean_fallback=("mean_fallback_pairs", "mean"),
        ).sort_values("mean_trim_d", ascending=False).to_string())
    else:
        print("(no rows)")

    print("\n" + "=" * 90)
    print("k0_max SUMMARY (quality-pass only)")
    print("=" * 90)
    if len(df_g):
        print(df_g.groupby("k0_max").agg(
            mean_trim_d=agg_trim,
            mean_median_d=agg_med,
            mean_max_d=agg_max,
            mean_sparsity=("mean_sparsity_pct", "mean"),
            mean_count_occ=("mean_count_occupied", "mean"),
            mean_fallback=("mean_fallback_pairs", "mean"),
        ).sort_values("mean_trim_d", ascending=False).to_string())
    else:
        print("(no rows)")

    print("\n" + "=" * 90)
    print("u_max SUMMARY (quality-pass only)")
    print("=" * 90)
    if len(df_g):
        print(df_g.groupby("u_max").agg(
            mean_trim_d=agg_trim,
            mean_median_d=agg_med,
            mean_max_d=agg_max,
        ).sort_values("mean_trim_d", ascending=False).to_string())
    else:
        print("(no rows)")

    print("\n" + "=" * 90)
    print("SPARSITY_THRESHOLD SUMMARY (quality-pass only)")
    print("=" * 90)
    if len(df_g):
        print(df_g.groupby("sparsity_threshold").agg(
            mean_trim_d=agg_trim,
            mean_median_d=agg_med,
            mean_max_d=agg_max,
            mean_fallback=("mean_fallback_pairs", "mean"),
        ).sort_values("mean_trim_d", ascending=False).to_string())
    else:
        print("(no rows)")

    print(f"\n{'='*90}")
    print(f"Sweep complete. {len(df)} configs tested in {total_min:.1f} min.")
    print(f"Results: {out_csv}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_sweep()
