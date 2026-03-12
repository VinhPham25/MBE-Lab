"""
Stage B - Step B6: Homer3-style GLM/OLS regression (hmrS_GLM parity core)

This implementation replaces the prior sequential short-channel regression with a
unified GLM deconvolution model:

    y = [HRF Gaussian basis convolved with stimulus onsets]
      + [polynomial drift]
      + [short-separation nuisance]
      + residual

Key parity points with Homer3 hmrS_GLM (idxBasis=1, glmSolveMethod=1):
- Gaussian basis models the HRF shape (not the short-channel signal).
- Basis is convolved with stimulus onset vectors.
- Short-channel nuisance is a single raw regressor (most-correlated by default).
- Drift polynomial is included in the same OLS solve.
- HRF output is reconstructed directly from Gaussian beta weights.

The module keeps the existing run_ols_regression(...) entry point so downstream
scripts can keep calling it. It now also emits GLM HRF evoked outputs for Stage C.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, load_config


# ---------------------------------------------------------------------------
# Dynamic import of Step B5 runner
# ---------------------------------------------------------------------------
_spec_b5 = importlib.util.spec_from_file_location(
    "mbll",
    Path(__file__).parent / "05_mbll.py",
)
_b5 = importlib.util.module_from_spec(_spec_b5)
_spec_b5.loader.exec_module(_b5)
run_mbll = _b5.run_mbll


# ---------------------------------------------------------------------------
# Doc-grounded defaults (docs/reproduced_matlab_comparison.md + hmrS_GLM.m)
# ---------------------------------------------------------------------------
_DOC_DEFAULTS = {
    "trange": (-25.0, 100.0),
    "gaussian_std_s": 1.0,
    "gaussian_step_s": 1.0,
    "short_separation_threshold_mm": 15.0,
    "flag_nuisance_r_method": 1,
    "drift_order": 3,
    "orthogonalize_drift_nuisance_against_hrf": True,
    "glm_solve_method": 1,
    "event_ids": [1],
}


class GlmParams:
    """Container object for GLM parameter values used by Step B6."""

    def __init__(
        self,
        *,
        trange: tuple[float, float],
        gaussian_std_s: float,
        gaussian_step_s: float,
        short_sep_threshold_mm: float,
        flag_nuisance_r_method: int,
        drift_order: int,
        orthogonalize_drift_nuisance_against_hrf: bool,
        glm_solve_method: int,
        event_ids: list[int],
    ) -> None:
        """
        Build a GLM parameter bundle.

        Parameters
        ----------
        trange : tuple[float, float]
            HRF time window in seconds as `(t_pre, t_post)`.
        gaussian_std_s : float
            Gaussian basis function standard deviation in seconds.
        gaussian_step_s : float
            Spacing between consecutive Gaussian basis centers in seconds.
        short_sep_threshold_mm : float
            Distance threshold in millimeters for short-separation channels.
        flag_nuisance_r_method : int
            Short-channel selection mode: `0=nearest`, `1=max correlation`, `2=mean`.
        drift_order : int
            Polynomial drift order for nuisance drift regressors.
        orthogonalize_drift_nuisance_against_hrf : bool
            If True, orthogonalize drift/nuisance regressors to the HRF basis.
        glm_solve_method : int
            GLM solver code (currently OLS = 1).
        event_ids : list[int]
            Event IDs used as stimulus conditions.

        Returns
        -------
        None
        """
        self.trange = trange
        self.gaussian_std_s = gaussian_std_s
        self.gaussian_step_s = gaussian_step_s
        self.short_sep_threshold_mm = short_sep_threshold_mm
        self.flag_nuisance_r_method = flag_nuisance_r_method
        self.drift_order = drift_order
        self.orthogonalize_drift_nuisance_against_hrf = orthogonalize_drift_nuisance_against_hrf
        self.glm_solve_method = glm_solve_method
        self.event_ids = event_ids

    @property
    def n_basis(self) -> int:
        """
        Compute the number of Gaussian basis functions implied by settings.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of basis functions used in the HRF model.
        """
        span = self.trange[1] - self.trange[0]
        return int(math.floor(span / self.gaussian_step_s) - 1)


def _safe_float(v: Any, default: float) -> float:
    """
    Safely cast a value to float with fallback.

    Parameters
    ----------
    v : Any
        Value to parse.
    default : float
        Fallback value when parsing fails.

    Returns
    -------
    float
        Parsed float, or `default` on failure.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(v: Any, default: int) -> int:
    """
    Safely cast a value to int with fallback.

    Parameters
    ----------
    v : Any
        Value to parse.
    default : int
        Fallback value when parsing fails.

    Returns
    -------
    int
        Parsed int, or `default` on failure.
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(v: Any, default: bool) -> bool:
    """Safely cast common config values to bool with fallback."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _resolve_event_ids_from_config(cfg: dict) -> list[int]:
    """
    Resolve condition event IDs from config with sensible fallbacks.

    Parameters
    ----------
    cfg : dict
        Parsed `config.yaml` dictionary.

    Returns
    -------
    list[int]
        Sorted event IDs for GLM stimulus onsets.
    """
    # Preferred location (new GLM-specific key)
    scr_cfg = cfg.get("stage_b", {}).get("short_channel_regression", {})
    event_ids = scr_cfg.get("event_ids", None)
    if isinstance(event_ids, (list, tuple)) and len(event_ids) > 0:
        vals = []
        for x in event_ids:
            try:
                vals.append(int(x))
            except (TypeError, ValueError):
                continue
        if vals:
            return sorted(set(vals))

    # Fallback: Stage C trial-start event definition
    epochs_cfg = cfg.get("stage_c", {}).get("epochs", {})
    ev_map = epochs_cfg.get("event_id", {})
    if isinstance(ev_map, dict) and "trial_start" in ev_map:
        try:
            return [int(ev_map["trial_start"])]
        except (TypeError, ValueError):
            pass

    # Last fallback: docs default
    return list(_DOC_DEFAULTS["event_ids"])


def get_glm_params(
    *,
    short_threshold_mm: float | None = None,
    gaussian_std_s: float | None = None,
    gaussian_step_s: float | None = None,
) -> GlmParams:
    """
    Resolve B6 GLM parameters from overrides and `config.yaml`.

    Parameters
    ----------
    short_threshold_mm : float | None
        Optional override for short-separation threshold in mm.
    gaussian_std_s : float | None
        Optional override for Gaussian standard deviation in seconds.
    gaussian_step_s : float | None
        Optional override for Gaussian basis spacing in seconds.

    Returns
    -------
    GlmParams
        Validated GLM parameter bundle used by this module.
    """
    cfg = load_config()
    scr_cfg = cfg.get("stage_b", {}).get("short_channel_regression", {})
    epochs_cfg = cfg.get("stage_c", {}).get("epochs", {})

    # trange defaults to stage_c epoch window if present, else doc default
    trange_cfg = scr_cfg.get("trange", None)
    if isinstance(trange_cfg, (list, tuple)) and len(trange_cfg) == 2:
        trange = (
            _safe_float(trange_cfg[0], _DOC_DEFAULTS["trange"][0]),
            _safe_float(trange_cfg[1], _DOC_DEFAULTS["trange"][1]),
        )
    else:
        trange = (
            _safe_float(epochs_cfg.get("tmin", _DOC_DEFAULTS["trange"][0]), _DOC_DEFAULTS["trange"][0]),
            _safe_float(epochs_cfg.get("tmax", _DOC_DEFAULTS["trange"][1]), _DOC_DEFAULTS["trange"][1]),
        )

    std_s = (
        float(gaussian_std_s)
        if gaussian_std_s is not None
        else _safe_float(scr_cfg.get("gaussian_std_s", _DOC_DEFAULTS["gaussian_std_s"]), _DOC_DEFAULTS["gaussian_std_s"])
    )

    step_from_cfg = scr_cfg.get("gaussian_step_s", _DOC_DEFAULTS["gaussian_step_s"])
    step_s = float(gaussian_step_s) if gaussian_step_s is not None else _safe_float(step_from_cfg, _DOC_DEFAULTS["gaussian_step_s"])

    short_sep = (
        float(short_threshold_mm)
        if short_threshold_mm is not None
        else _safe_float(
            scr_cfg.get("short_separation_threshold", _DOC_DEFAULTS["short_separation_threshold_mm"]),
            _DOC_DEFAULTS["short_separation_threshold_mm"],
        )
    )

    flag_nuis = _safe_int(
        scr_cfg.get("flag_nuisance_r_method", scr_cfg.get("flagNuisanceRMethod", _DOC_DEFAULTS["flag_nuisance_r_method"])),
        _DOC_DEFAULTS["flag_nuisance_r_method"],
    )
    drift_order = _safe_int(scr_cfg.get("drift_order", _DOC_DEFAULTS["drift_order"]), _DOC_DEFAULTS["drift_order"])
    orth_drift_nuis = _safe_bool(
        scr_cfg.get(
            "orthogonalize_drift_nuisance_against_hrf",
            _DOC_DEFAULTS["orthogonalize_drift_nuisance_against_hrf"],
        ),
        _DOC_DEFAULTS["orthogonalize_drift_nuisance_against_hrf"],
    )
    glm_method = _safe_int(scr_cfg.get("glm_solve_method", _DOC_DEFAULTS["glm_solve_method"]), _DOC_DEFAULTS["glm_solve_method"])
    event_ids = _resolve_event_ids_from_config(cfg)

    params = GlmParams(
        trange=trange,
        gaussian_std_s=max(std_s, 1e-6),
        gaussian_step_s=max(step_s, 1e-6),
        short_sep_threshold_mm=max(short_sep, 0.0),
        flag_nuisance_r_method=flag_nuis,
        drift_order=max(drift_order, 0),
        orthogonalize_drift_nuisance_against_hrf=orth_drift_nuis,
        glm_solve_method=glm_method,
        event_ids=event_ids,
    )

    if params.n_basis <= 0:
        raise ValueError(
            "Invalid GLM basis configuration: computed n_basis <= 0. "
            f"trange={params.trange}, gaussian_step_s={params.gaussian_step_s}"
        )

    return params


def identify_channel_separation(
    raw_haemo: mne.io.BaseRaw,
    short_threshold_mm: float,
) -> tuple[list[int], list[int], np.ndarray]:
    """
    Classify channels as short- vs long-separation using optode distance.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Hemoglobin raw object containing probe geometry.
    short_threshold_mm : float
        Short-separation distance threshold in millimeters.

    Returns
    -------
    tuple[list[int], list[int], np.ndarray]
        `short_idx`, `long_idx`, and per-channel separation distances in mm.
    """
    dists_mm = mne.preprocessing.nirs.source_detector_distances(raw_haemo.info) * 1000.0
    short_idx = [i for i, d in enumerate(dists_mm) if d < short_threshold_mm]
    long_idx = [i for i, d in enumerate(dists_mm) if d >= short_threshold_mm]
    return short_idx, long_idx, dists_mm


def _channel_midpoints(raw_haemo: mne.io.BaseRaw) -> np.ndarray:
    """
    Extract channel midpoint coordinates from MNE channel `loc` fields.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Hemoglobin raw object with channel metadata.

    Returns
    -------
    np.ndarray
        Shape `(n_channels, 3)` array of XYZ midpoints.
    """
    return np.array([ch["loc"][:3] for ch in raw_haemo.info["chs"]], dtype=float)


def _extract_condition_onsets(
    raw_haemo: mne.io.BaseRaw,
    event_ids: list[int],
    n_pre: int,
    n_post: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Build condition onset matrix in Homer3 GLM convention.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Hemoglobin raw object containing annotations/events.
    event_ids : list[int]
        Event IDs to include as GLM conditions.
    n_pre : int
        Pre-window sample offset (can be negative).
    n_post : int
        Post-window sample offset.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        `onset` matrix with shape `(n_times, n_conditions)` and per-condition
        trial counts `n_trials`.
    """
    n_t = raw_haemo.n_times
    onset = np.zeros((n_t, len(event_ids)), dtype=float)
    n_trials = [0 for _ in event_ids]

    events, _ = mne.events_from_annotations(raw_haemo, verbose=False)
    if len(events) == 0:
        return onset, n_trials

    samples = events[:, 0] - int(raw_haemo.first_samp)
    codes = events[:, 2]

    for i_cond, cond_id in enumerate(event_ids):
        cond_samples = samples[codes == cond_id]
        valid = cond_samples[(cond_samples + n_pre >= 0) & (cond_samples + n_post < n_t)]
        if len(valid) == 0:
            continue
        onset[valid + n_pre, i_cond] = 1.0
        n_trials[i_cond] = int(len(valid))

    return onset, n_trials


def build_gaussian_basis(
    sfreq: float,
    trange: tuple[float, float],
    gaussian_step_s: float,
    gaussian_std_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct Gaussian HRF basis functions over `trange`.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    trange : tuple[float, float]
        HRF window `(t_pre, t_post)` in seconds.
    gaussian_step_s : float
        Basis center spacing in seconds.
    gaussian_std_s : float
        Basis standard deviation in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `t_hrf` time vector (seconds) and basis matrix `tbasis` of shape
        `(n_hrf_timepoints, n_basis)`.
    """
    dt = 1.0 / sfreq
    n_pre = int(round(trange[0] / dt))
    n_post = int(round(trange[1] / dt))

    t_hrf = np.arange(n_pre, n_post + 1, dtype=float) * dt
    n_basis = int(math.floor((trange[1] - trange[0]) / gaussian_step_s) - 1)
    if n_basis <= 0:
        raise ValueError(
            "Computed n_basis <= 0; adjust trange/gaussian_step_s. "
            f"trange={trange}, step={gaussian_step_s}"
        )

    tbasis = np.zeros((len(t_hrf), n_basis), dtype=float)
    for b in range(1, n_basis + 1):
        center = trange[0] + b * gaussian_step_s
        g = np.exp(-((t_hrf - center) ** 2) / (2.0 * (gaussian_std_s ** 2)))
        gmax = np.max(g)
        if gmax > 0:
            g = g / gmax
        tbasis[:, b - 1] = g

    return t_hrf, tbasis


def build_hrf_design(onset: np.ndarray, tbasis: np.ndarray) -> np.ndarray:
    """
    Build HRF design matrix by convolving condition onsets with each basis column.

    Parameters
    ----------
    onset : np.ndarray
        Binary onset matrix of shape `(n_times, n_conditions)`.
    tbasis : np.ndarray
        Gaussian basis matrix of shape `(n_hrf_timepoints, n_basis)`.

    Returns
    -------
    np.ndarray
        Design matrix `dA` of shape `(n_times, n_conditions * n_basis)`.
    """
    n_t, n_cond = onset.shape
    n_basis = tbasis.shape[1]
    dA = np.zeros((n_t, n_cond * n_basis), dtype=float)

    col = 0
    for i_cond in range(n_cond):
        for b in range(n_basis):
            conv_col = np.convolve(onset[:, i_cond], tbasis[:, b], mode="full")[:n_t]
            dA[:, col] = conv_col
            col += 1

    return dA


def build_drift_design(n_t: int, drift_order: int) -> np.ndarray:
    """
    Construct polynomial drift regressors with column-wise scaling.

    Parameters
    ----------
    n_t : int
        Number of time points.
    drift_order : int
        Maximum polynomial order.

    Returns
    -------
    np.ndarray
        Drift design matrix of shape `(n_t, drift_order + 1)`.
    """
    x = np.ones((n_t, drift_order + 1), dtype=float)
    t = np.arange(1, n_t + 1, dtype=float)
    for p in range(1, drift_order + 1):
        col = t ** p
        x[:, p] = col / col[-1]
    return x


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation robustly for 1D arrays.

    Parameters
    ----------
    a : np.ndarray
        First signal.
    b : np.ndarray
        Second signal.

    Returns
    -------
    float
        Correlation coefficient, or `nan` for invalid/degenerate input.
    """
    if a.size != b.size:
        return float("nan")
    if np.std(a, ddof=1) < 1e-15 or np.std(b, ddof=1) < 1e-15:
        return float("nan")
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def select_short_regressor(
    raw_haemo: mne.io.BaseRaw,
    y: np.ndarray,
    long_idx: int,
    short_candidates: list[int],
    method: int,
    midpoints: np.ndarray,
) -> tuple[np.ndarray | None, int | None, float]:
    """
    Select short-channel nuisance regressor using Homer3 method codes.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Hemoglobin raw object.
    y : np.ndarray
        Long-channel target signal.
    long_idx : int
        Index of the current long channel.
    short_candidates : list[int]
        Candidate short-channel indices for this chromophore.
    method : int
        Selection method (`0` nearest, `1` max-correlation, `2` mean-short).
    midpoints : np.ndarray
        Channel midpoint coordinates, shape `(n_channels, 3)`.

    Returns
    -------
    tuple[np.ndarray | None, int | None, float]
        Selected nuisance regressor vector, selected short index
        (`-1` for mean-short), and pre-regression correlation.
    """
    if len(short_candidates) == 0:
        return None, None, float("nan")

    data = raw_haemo.get_data()

    if method == 2:
        Xs = data[short_candidates, :]
        s = np.nanmean(Xs, axis=0)
        return s, -1, _corr(y, s)

    if method == 0:
        m_long = midpoints[long_idx]
        d = np.linalg.norm(midpoints[short_candidates] - m_long[None, :], axis=1)
        j = int(np.argmin(d))
        si = int(short_candidates[j])
        s = data[si, :]
        return s, si, _corr(y, s)

    # Default / method 1: greatest positive correlation (not abs), as in hmrS_GLM case 1.
    # Guard: if no positively-correlated short channel exists, skip nuisance regression
    # for this long channel to avoid injecting anti-correlated components.
    best_si = None
    best_r = -np.inf
    for si in short_candidates:
        s = data[si, :]
        r = _corr(y, s)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_si = int(si)

    # Require a minimum positive coupling strength before nuisance regression.
    min_pos_corr = 0.10
    if best_si is None or not np.isfinite(best_r) or best_r < min_pos_corr:
        return None, None, float("nan")

    s = data[best_si, :]
    return s, best_si, float(best_r)


def _solve_ols(
    At_inc: np.ndarray,
    y_inc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve OLS coefficients for one channel on included samples.

    Parameters
    ----------
    At_inc : np.ndarray
        Included-sample design matrix.
    y_inc : np.ndarray
        Included-sample response vector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `beta` coefficients and pseudo-inverse-like operator mapping `y_inc -> beta`.
    """
    ATA = At_inc.T @ At_inc
    try:
        pinvA = np.linalg.solve(ATA, At_inc.T)
    except np.linalg.LinAlgError:
        pinvA = np.linalg.pinv(At_inc)
    beta = pinvA @ y_inc
    return beta, pinvA


def _orthogonalize_against(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Remove components of columns in x that lie in the column space of ref.
    """
    if ref.size == 0 or x.size == 0:
        return x
    coef, *_ = np.linalg.lstsq(ref, x, rcond=None)
    return x - ref @ coef


def apply_glm_regression(
    raw_haemo: mne.io.BaseRaw,
    glm_params: GlmParams,
    *,
    verbose: bool = True,
) -> tuple[mne.io.BaseRaw, pd.DataFrame, dict[str, mne.Evoked], dict]:
    """
    Apply Homer3-style unified GLM deconvolution to all long channels.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Stage B5 hemoglobin signal (`hbo`/`hbr`) for one subject.
    glm_params : GlmParams
        GLM parameter bundle (basis, drift, nuisance, event IDs).
    verbose : bool
        If True, print detailed progress and summary logs.

    Returns
    -------
    tuple[mne.io.BaseRaw, pd.DataFrame, dict[str, mne.Evoked], dict]
        `raw_corrected`, `reg_results`, `evoked_by_cond`, and `glm_meta`.
    """
    sfreq = float(raw_haemo.info["sfreq"])
    ch_names = raw_haemo.ch_names
    ch_types = raw_haemo.get_channel_types()

    short_idx, long_idx, dists_mm = identify_channel_separation(
        raw_haemo,
        short_threshold_mm=glm_params.short_sep_threshold_mm,
    )

    short_by_type = {
        "hbo": [i for i in short_idx if ch_types[i] == "hbo"],
        "hbr": [i for i in short_idx if ch_types[i] == "hbr"],
    }

    t_hrf, tbasis = build_gaussian_basis(
        sfreq,
        glm_params.trange,
        glm_params.gaussian_step_s,
        glm_params.gaussian_std_s,
    )
    n_basis = tbasis.shape[1]

    dt = 1.0 / sfreq
    n_pre = int(round(glm_params.trange[0] / dt))
    n_post = int(round(glm_params.trange[1] / dt))
    onset, n_trials = _extract_condition_onsets(raw_haemo, glm_params.event_ids, n_pre, n_post)

    # Keep only conditions that actually have trials.
    valid_cond = [i for i, n in enumerate(n_trials) if n > 0]
    if len(valid_cond) == 0:
        raise RuntimeError(
            "No valid stimulus onsets found for GLM. "
            f"Requested event_ids={glm_params.event_ids}"
        )

    onset = onset[:, valid_cond]
    used_event_ids = [glm_params.event_ids[i] for i in valid_cond]
    used_n_trials = [n_trials[i] for i in valid_cond]
    n_cond = onset.shape[1]

    dA = build_hrf_design(onset, tbasis)
    x_drift = build_drift_design(raw_haemo.n_times, glm_params.drift_order)

    # tInc from upstream motion masking is not carried in Raw here; use all samples.
    t_inc = np.ones(raw_haemo.n_times, dtype=bool)
    lst_inc = np.where(t_inc)[0]

    data_before = raw_haemo.get_data()
    data_after = data_before.copy()
    midpoints = _channel_midpoints(raw_haemo)

    # Storage for HRF outputs (long channels only)
    long_idx_sorted = sorted(long_idx)
    idx_to_long_col = {idx: j for j, idx in enumerate(long_idx_sorted)}
    hrf_by_cond = {
        f"cond_{eid}": np.zeros((len(long_idx_sorted), len(t_hrf)), dtype=float)
        for eid in used_event_ids
    }

    records: list[dict] = []

    if verbose:
        print("\n" + "=" * 70)
        print("TASK B6 - HOMER3-STYLE GLM/OLS DECONVOLUTION")
        print("=" * 70)
        print(f"  sfreq                : {sfreq:.3f} Hz")
        print(f"  trange               : [{glm_params.trange[0]}, {glm_params.trange[1]}] s")
        print(f"  gaussian std / step  : {glm_params.gaussian_std_s} s / {glm_params.gaussian_step_s} s")
        print(f"  n_basis              : {n_basis}")
        print(f"  drift_order          : {glm_params.drift_order}")
        print(f"  nuisance method      : {glm_params.flag_nuisance_r_method}")
        print(
            "  orthogonalize drift/nuisance vs HRF : "
            f"{glm_params.orthogonalize_drift_nuisance_against_hrf}"
        )
        print(f"  event_ids requested  : {glm_params.event_ids}")
        print(f"  event_ids used       : {used_event_ids}")
        print(f"  trials per condition : {used_n_trials}")
        print(f"  channels             : {len(long_idx_sorted)} long, {len(short_idx)} short")

    for li in long_idx_sorted:
        y = data_before[li, :]
        ct = ch_types[li]
        short_candidates = short_by_type.get(ct, [])

        s_reg, s_idx, r_before = select_short_regressor(
            raw_haemo,
            y,
            long_idx=li,
            short_candidates=short_candidates,
            method=glm_params.flag_nuisance_r_method,
            midpoints=midpoints,
        )

        # Collinearity diagnostic: short-channel nuisance vs HRF design columns.
        if s_reg is not None:
            col_corr = np.array([_corr(s_reg, dA[:, j]) for j in range(dA.shape[1])], dtype=float)
            col_abs = np.abs(col_corr[np.isfinite(col_corr)])
            if col_abs.size:
                col_max_abs = float(np.max(col_abs))
                col_mean_abs = float(np.mean(col_abs))
                col_n_over_05 = int(np.sum(col_abs > 0.5))
            else:
                col_max_abs = np.nan
                col_mean_abs = np.nan
                col_n_over_05 = 0
        else:
            col_max_abs = np.nan
            col_mean_abs = np.nan
            col_n_over_05 = 0

        if glm_params.orthogonalize_drift_nuisance_against_hrf:
            x_drift_use = _orthogonalize_against(dA, x_drift)
        else:
            x_drift_use = x_drift
        At_parts = [dA, x_drift_use]
        if s_reg is not None:
            if glm_params.orthogonalize_drift_nuisance_against_hrf:
                s_use = _orthogonalize_against(dA, s_reg[:, None])
            else:
                s_use = s_reg[:, None]
            At_parts.append(s_use)
        At = np.concatenate(At_parts, axis=1)

        At_inc = At[lst_inc, :]
        y_inc = y[lst_inc]

        beta, pinvA = _solve_ols(At_inc, y_inc)

        y_fit = At @ beta
        resid = y - y_fit
        y_hrf_only = dA @ beta[: (n_basis * n_cond)]
        y_new = y_hrf_only + resid

        data_after[li, :] = y_new

        # HRF reconstruction per condition
        long_col = idx_to_long_col[li]
        for c, eid in enumerate(used_event_ids):
            i0 = c * n_basis
            i1 = (c + 1) * n_basis
            beta_hrf = beta[i0:i1]
            hrf_trace = tbasis @ beta_hrf
            # Homer3-style yavg baseline: subtract pre-stimulus mean (< 0 s)
            pre_mask = t_hrf < 0
            if np.any(pre_mask):
                hrf_trace = hrf_trace - float(np.mean(hrf_trace[pre_mask]))
            hrf_by_cond[f"cond_{eid}"][long_col, :] = hrf_trace

        # Stats
        y_fit_inc = At_inc @ beta
        r_model = _corr(y_inc, y_fit_inc)
        r_after = _corr(y_new, s_reg) if s_reg is not None else float("nan")

        var_before = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
        var_after = float(np.var(y_new, ddof=1)) if y_new.size > 1 else 0.0
        var_removed = ((var_before - var_after) / var_before) if var_before > 0 else float("nan")

        short_name = None
        if s_idx is None:
            short_name = None
        elif s_idx == -1:
            short_name = "AVG_SHORT"
        else:
            short_name = ch_names[s_idx]

        records.append(
            {
                "channel": ch_names[li],
                "ch_type": ct,
                "dist_mm": round(float(dists_mm[li]), 3),
                "best_short_ch": short_name,
                "pearson_r_before": round(float(r_before), 6) if np.isfinite(r_before) else np.nan,
                "pearson_r_after": round(float(r_after), 6) if np.isfinite(r_after) else np.nan,
                "beta_norm": float(np.linalg.norm(beta)),
                "var_removed_frac": float(var_removed) if np.isfinite(var_removed) else np.nan,
                "glm_r": float(r_model) if np.isfinite(r_model) else np.nan,
                "short_hrf_max_abs_corr": float(col_max_abs) if np.isfinite(col_max_abs) else np.nan,
                "short_hrf_mean_abs_corr": float(col_mean_abs) if np.isfinite(col_mean_abs) else np.nan,
                "short_hrf_n_cols_abs_corr_gt_0p5": int(col_n_over_05),
                "design_rank": int(np.linalg.matrix_rank(At_inc)),
                "n_basis": int(n_basis),
                "n_conditions": int(n_cond),
                "n_trials_total": int(sum(used_n_trials)),
            }
        )

    reg_results = pd.DataFrame(records)

    raw_corrected = raw_haemo.copy()
    raw_corrected._data = data_after

    # Build per-condition Evoked outputs (long channels only)
    info_long = mne.pick_info(raw_haemo.info, long_idx_sorted, copy=True)
    evoked_by_cond: dict[str, mne.Evoked] = {}
    for c, eid in enumerate(used_event_ids):
        key = f"cond_{eid}"
        ev = mne.EvokedArray(
            hrf_by_cond[key],
            info=info_long,
            tmin=float(t_hrf[0]),
            nave=max(1, int(used_n_trials[c])),
            comment=key,
            verbose=False,
        )
        evoked_by_cond[key] = ev

    glm_meta = {
        "trange": [float(glm_params.trange[0]), float(glm_params.trange[1])],
        "gaussian_std_s": float(glm_params.gaussian_std_s),
        "gaussian_step_s": float(glm_params.gaussian_step_s),
        "n_basis": int(n_basis),
        "drift_order": int(glm_params.drift_order),
        "orthogonalize_drift_nuisance_against_hrf": bool(
            glm_params.orthogonalize_drift_nuisance_against_hrf
        ),
        "short_separation_threshold_mm": float(glm_params.short_sep_threshold_mm),
        "flag_nuisance_r_method": int(glm_params.flag_nuisance_r_method),
        "glm_solve_method": int(glm_params.glm_solve_method),
        "event_ids": [int(x) for x in used_event_ids],
        "n_trials": [int(x) for x in used_n_trials],
        "n_long_channels": int(len(long_idx_sorted)),
        "n_short_channels": int(len(short_idx)),
        "long_channel_names": [ch_names[i] for i in long_idx_sorted],
    }

    return raw_corrected, reg_results, evoked_by_cond, glm_meta


def calculate_correlation_reduction(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    reg_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute before/after correlation summaries used by Stage B reporting.

    Parameters
    ----------
    raw_before : mne.io.BaseRaw
        Input hemoglobin signal before B6 GLM correction.
    raw_after : mne.io.BaseRaw
        Hemoglobin signal after B6 GLM correction.
    reg_results : pd.DataFrame
        Per-channel regression results from `apply_glm_regression`.

    Returns
    -------
    pd.DataFrame
        Per-channel correlation comparison table.
    """
    ch_names = raw_before.ch_names
    ch_types = raw_before.get_channel_types()
    d0 = raw_before.get_data()
    d1 = raw_after.get_data()

    def _base(ch: str) -> str:
        return ch.replace(" hbo", "").replace(" hbr", "")

    hbo_by_base = {_base(ch_names[i]): i for i, t in enumerate(ch_types) if t == "hbo"}
    hbr_by_base = {_base(ch_names[i]): i for i, t in enumerate(ch_types) if t == "hbr"}

    out = []
    for _, row in reg_results.iterrows():
        ch = row["channel"]
        ct = row["ch_type"]
        short_ch = row.get("best_short_ch", None)
        if ch not in ch_names:
            continue
        i = ch_names.index(ch)

        hbo_hbr_before = np.nan
        hbo_hbr_after = np.nan

        base = _base(ch)
        if ct == "hbo" and base in hbr_by_base:
            j = hbr_by_base[base]
            hbo_hbr_before = _corr(d0[i, :], d0[j, :])
            hbo_hbr_after = _corr(d1[i, :], d1[j, :])
        elif ct == "hbr" and base in hbo_by_base:
            j = hbo_by_base[base]
            hbo_hbr_before = _corr(d0[i, :], d0[j, :])
            hbo_hbr_after = _corr(d1[i, :], d1[j, :])

        r_ls_before = row.get("pearson_r_before", np.nan)
        r_ls_after = row.get("pearson_r_after", np.nan)

        out.append(
            {
                "channel": ch,
                "ch_type": ct,
                "best_short_ch": short_ch,
                "r_long_short_before": r_ls_before,
                "r_long_short_after": r_ls_after,
                "hbo_hbr_r_before": hbo_hbr_before,
                "hbo_hbr_r_after": hbo_hbr_after,
            }
        )

    return pd.DataFrame(out)


def _hrf_dataframe_from_evoked(evoked: mne.Evoked) -> pd.DataFrame:
    """
    Convert an MNE Evoked HRF object to a CSV-ready dataframe in micro-molar units.

    Parameters
    ----------
    evoked : mne.Evoked
        GLM HRF evoked object.

    Returns
    -------
    pd.DataFrame
        Dataframe with `time_s` and channel columns in μM.
    """
    df = pd.DataFrame((evoked.data * 1e6).T, columns=evoked.ch_names)  # mol/L -> uM
    df.insert(0, "time_s", evoked.times)
    return df


def save_regression_data(
    raw_corrected: mne.io.BaseRaw,
    reg_results: pd.DataFrame,
    corr_df: pd.DataFrame,
    evoked_by_cond: dict[str, mne.Evoked],
    glm_meta: dict,
    subject_id: str,
    output_dir: Path,
    *,
    verbose: bool = True,
) -> dict:
    """
    Save B6 outputs (continuous corrected signal, GLM HRFs, and summaries).

    Parameters
    ----------
    raw_corrected : mne.io.BaseRaw
        Continuous hemoglobin signal after B6 correction.
    reg_results : pd.DataFrame
        Per-channel GLM regression metrics.
    corr_df : pd.DataFrame
        Correlation before/after summary table.
    evoked_by_cond : dict[str, mne.Evoked]
        Per-condition GLM HRF evoked objects.
    glm_meta : dict
        Metadata describing GLM settings and channel counts.
    subject_id : str
        Subject identifier.
    output_dir : Path
        Directory where B6 artifacts are written.
    verbose : bool
        If True, print saved-file paths.

    Returns
    -------
    dict
        Dictionary of key saved paths for downstream orchestration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fif_path = output_dir / f"{subject_id}_ols_haemo_raw.fif"
    raw_corrected.save(str(fif_path), overwrite=True, verbose=False)

    reg_csv = output_dir / f"{subject_id}_ols_reg_results.csv"
    reg_results.to_csv(reg_csv, index=False)

    corr_csv = output_dir / f"{subject_id}_ols_corr_reduction.csv"
    corr_df.to_csv(corr_csv, index=False)

    evoked_paths: dict[str, str] = {}
    csv_paths: dict[str, str] = {}
    for cond_key, ev in evoked_by_cond.items():
        ev_path = output_dir / f"{subject_id}_glm_hrf_{cond_key}-ave.fif"
        ev.save(ev_path, overwrite=True, verbose=False)
        evoked_paths[cond_key] = str(ev_path)

        df_hrf = _hrf_dataframe_from_evoked(ev)
        csv_path = output_dir / f"{subject_id}_glm_hrf_{cond_key}.csv"
        df_hrf.to_csv(csv_path, index=False)
        csv_paths[cond_key] = str(csv_path)

    # Primary condition = first saved condition, for downstream convenience.
    first_cond = sorted(evoked_by_cond.keys())[0]
    primary_evoked = evoked_by_cond[first_cond]

    primary_ev_path = output_dir / f"{subject_id}_glm_hrf-ave.fif"
    primary_evoked.save(primary_ev_path, overwrite=True, verbose=False)

    primary_csv_path = output_dir / f"{subject_id}_glm_hrf_averages.csv"
    _hrf_dataframe_from_evoked(primary_evoked).to_csv(primary_csv_path, index=False)

    hbo_df = reg_results[reg_results["ch_type"] == "hbo"]
    hbr_df = reg_results[reg_results["ch_type"] == "hbr"]

    summary = {
        "regression_method": "homer3_style_glm_ols",
        "glm_meta": glm_meta,
        "n_long_channels_corrected": int(len(reg_results)),
        "HbO_regression": {
            "mean_abs_r_before": float(np.nanmean(np.abs(hbo_df.get("pearson_r_before", np.nan)))) if len(hbo_df) else None,
            "mean_abs_r_after": float(np.nanmean(np.abs(hbo_df.get("pearson_r_after", np.nan)))) if len(hbo_df) else None,
            "mean_var_removed_pct": float(np.nanmean(hbo_df.get("var_removed_frac", np.nan)) * 100.0) if len(hbo_df) else None,
            "mean_glm_r": float(np.nanmean(hbo_df.get("glm_r", np.nan))) if len(hbo_df) else None,
        },
        "HbR_regression": {
            "mean_abs_r_before": float(np.nanmean(np.abs(hbr_df.get("pearson_r_before", np.nan)))) if len(hbr_df) else None,
            "mean_abs_r_after": float(np.nanmean(np.abs(hbr_df.get("pearson_r_after", np.nan)))) if len(hbr_df) else None,
            "mean_var_removed_pct": float(np.nanmean(hbr_df.get("var_removed_frac", np.nan)) * 100.0) if len(hbr_df) else None,
            "mean_glm_r": float(np.nanmean(hbr_df.get("glm_r", np.nan))) if len(hbr_df) else None,
        },
        "outputs": {
            "corrected_raw_fif": str(fif_path),
            "regression_csv": str(reg_csv),
            "correlation_csv": str(corr_csv),
            "glm_evoked_primary": str(primary_ev_path),
            "glm_hrf_primary_csv": str(primary_csv_path),
            "glm_evokeds_by_condition": evoked_paths,
            "glm_hrf_csv_by_condition": csv_paths,
        },
    }

    summary_path = output_dir / f"{subject_id}_ols_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\nSaved outputs:")
        print(f"  corrected raw fif  : {fif_path}")
        print(f"  reg results csv    : {reg_csv}")
        print(f"  corr summary csv   : {corr_csv}")
        print(f"  GLM HRF evoked     : {primary_ev_path}")
        print(f"  GLM HRF csv        : {primary_csv_path}")
        print(f"  summary json       : {summary_path}")

    return {
        "fif_path": fif_path,
        "reg_csv": reg_csv,
        "corr_csv": corr_csv,
        "summary_json": summary_path,
        "glm_primary_evoked_path": primary_ev_path,
        "glm_primary_csv_path": primary_csv_path,
        "glm_evoked_paths": evoked_paths,
        "glm_hrf_csv_paths": csv_paths,
    }


def run_ols_regression(
    dataset_name: str,
    subject_id: str,
    short_threshold_mm: float | None = None,
    n_components: float | None = None,
    min_channels_flagged: int | None = None,
    ppf_dict: dict | None = None,
    gaussian_std_s: float | None = None,
    gaussian_step_s: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run full Stage B6 GLM/OLS pipeline for one subject.

    Parameters
    ----------
    dataset_name : str
        Dataset key from config (e.g., `expert_novice`).
    subject_id : str
        Subject ID to process.
    short_threshold_mm : float | None
        Optional override for short-separation threshold in mm.
    n_components : float | None
        Optional B4 rPCA variance threshold override.
    min_channels_flagged : int | None
        Optional B4 rPCA motion vote-threshold override.
    ppf_dict : dict | None
        Optional B5 PPF override map `{wavelength_nm: ppf}`.
    gaussian_std_s : float | None
        Optional GLM Gaussian basis std override (seconds).
    gaussian_step_s : float | None
        Optional GLM Gaussian basis spacing override (seconds).
    verbose : bool
        If True, print detailed execution logs.

    Returns
    -------
    dict
        Pipeline outputs including corrected raw, GLM HRF evoked, summaries,
        and saved-path references.
    """

    # Pull defaults from current config for upstream B4/B5 if not provided.
    cfg = load_config()
    rpca_cfg = cfg.get("stage_b", {}).get("rpca", {})
    if n_components is None:
        n_components = float(rpca_cfg.get("n_components", 0.97))
    if min_channels_flagged is None:
        # Homer3-faithful default: OR-union (min_channels_flagged=1)
        min_channels_flagged = int(rpca_cfg.get("min_channels_flagged", 1))

    glm_params = get_glm_params(
        short_threshold_mm=short_threshold_mm,
        gaussian_std_s=gaussian_std_s,
        gaussian_step_s=gaussian_step_s,
    )

    print("\n" + "=" * 70)
    print(f"STAGE B - STEP B6: GLM/OLS REGRESSION  |  Subject: {subject_id}")
    print("=" * 70)

    print("\n[1/4] Running upstream pipeline (B1->B5)...")
    b5_results = run_mbll(
        dataset_name,
        subject_id,
        ppf_dict=ppf_dict,
        n_components=float(n_components),
        min_channels_flagged=int(min_channels_flagged),
        verbose=False,
    )
    raw_haemo = b5_results["raw_haemo"]

    print("[2/4] Applying unified GLM deconvolution...")
    raw_corrected, reg_results, evoked_by_cond, glm_meta = apply_glm_regression(
        raw_haemo,
        glm_params,
        verbose=verbose,
    )

    print("[3/4] Computing regression summary metrics...")
    corr_df = calculate_correlation_reduction(raw_haemo, raw_corrected, reg_results)

    print("[4/4] Saving outputs...")
    output_dir = get_output_path("stage_b", f"ols_regression/{subject_id}")
    saved = save_regression_data(
        raw_corrected,
        reg_results,
        corr_df,
        evoked_by_cond,
        glm_meta,
        subject_id,
        output_dir,
        verbose=verbose,
    )

    # Primary outputs for downstream code
    primary_key = sorted(evoked_by_cond.keys())[0]
    glm_primary_evoked = evoked_by_cond[primary_key]
    glm_primary_df = _hrf_dataframe_from_evoked(glm_primary_evoked)

    if verbose:
        hbo = reg_results[reg_results["ch_type"] == "hbo"]
        hbr = reg_results[reg_results["ch_type"] == "hbr"]
        print("\n" + "=" * 70)
        print(f"[OK] Step B6 complete for {subject_id}")
        print(f"  GLM trange                 : {glm_meta['trange']}")
        print(f"  Gaussian std / step (s)    : {glm_meta['gaussian_std_s']} / {glm_meta['gaussian_step_s']}")
        print(f"  Basis count                : {glm_meta['n_basis']}")
        print(f"  Conditions (event ids)     : {glm_meta['event_ids']}  trials={glm_meta['n_trials']}")
        print(f"  Long channels corrected    : {len(reg_results)}")
        print(f"  HbO mean var removed (%)   : {np.nanmean(hbo['var_removed_frac']) * 100.0:.2f}")
        print(f"  HbR mean var removed (%)   : {np.nanmean(hbr['var_removed_frac']) * 100.0:.2f}")
        print(f"  Output dir                 : {output_dir}")
        print("=" * 70 + "\n")

    return {
        "raw_haemo": raw_haemo,
        "raw_corrected": raw_corrected,
        "reg_results": reg_results,
        "corr_df": corr_df,
        "glm_evokeds": evoked_by_cond,
        "glm_primary_evoked": glm_primary_evoked,
        "glm_primary_df": glm_primary_df,
        "glm_meta": glm_meta,
        "output_dir": output_dir,
        "fif_path": saved["fif_path"],
        "glm_primary_evoked_path": saved["glm_primary_evoked_path"],
        "glm_primary_csv_path": saved["glm_primary_csv_path"],
    }


if __name__ == "__main__":
    run_ols_regression(
        dataset_name="expert_novice",
        subject_id="A1",
        verbose=True,
    )
