"""
Stage D — Step D3: Core GC Computation (Linear VAR / Granger Causality)

This module implements the Python equivalent of MATLAB's ``cca_pwcausal()``
from the GCCA toolbox, adapted for the ASTaUND dataset with 5 ROIs and the
validated Stage C channel assignments.

Section 4 task coverage (LINEAR_GC_PROGRESS.md)
------------------------------------------------
4.1 — VAR model fitting                    (fit_var_model)
4.2 — Frequency-domain GC spectrum         (compute_gc_spectrum)
4.3 — Pairwise conditional GC entry point  (compute_pairwise_conditional_gc)
4.4 — Sliding window loop                  (compute_windowed_gc)
4.5 — Frequency band averaging             (average_gc_over_band)

MATLAB reference
----------------
``GC_freq_main_TRGC_NovExp.m`` line 132::

    [GW,COH,pp,waut,cons] = cca_pwcausal(Y_windowed, 1, NL, 20, 20, freq, 1)

Python equivalent::

    GW = compute_pairwise_conditional_gc(Y_windowed, order=20,
                                          freq=freq, sfreq=25.0)

Sign / index convention
-----------------------
``GW[src, tgt, freq_idx]`` = GC from ``src`` ROI → ``tgt`` ROI at the given
frequency. Diagonal (src == tgt) is always 0. Off-diagonal values are
non-negative (log-GC ≥ 0). This matches the MATLAB GCCA toolbox output where
``GW(i, j, f)`` is the influence of region i on region j.

References
----------
Geweke J (1982). "Measurement of Linear Dependence and Feedback Between
Multiple Time Series." JASA 77(378):304–313.

Barnett L & Seth AK (2014). "The MVGC Multivariate Granger Causality
Toolbox." J Neurosci Methods 223:50–68.  §2.3 describes pairwise-conditional
GC and demonstrates that fitting a *joint* multivariate VAR on all k variables
implicitly conditions the i→j measure on all remaining k−2 variables. This
is why ``compute_pairwise_conditional_gc()`` does NOT require separate bivariate
models — the multivariate VAR is the conditioning mechanism.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

logger = logging.getLogger(__name__)


# ===========================================================================
# Section 4.1.1 + 4.1.2 — VAR Model Fitting
# ===========================================================================

def fit_var_model(
    Y: np.ndarray,
    order: int = 20,
    stationarity_check: bool = True,
    adf_pvalue_threshold: float = 0.05,
    trend: str = "c",
):
    """
    Fit a VAR(order) model on the ROI time-series matrix Y.

    MATLAB equivalent
    -----------------
    ``cca_pwcausal(Y_windowed, 1, NL, 20, ...)`` internally calls OLS
    regression on stacked lags of Y_windowed to estimate VAR(20) coefficients.

    Parameters
    ----------
    Y : np.ndarray, shape (n_rois, T)
        ROI time series in channels-first layout (matches MATLAB row-per-ROI
        convention).  T must satisfy T - order > order * n_rois for reliable
        OLS estimation.
    order : int
        VAR model order p.  Matches the ``20`` in MATLAB ``cca_pwcausal(...,
        20, 20, ...)``.  The model regresses each variable on its own past
        values and those of all other variables up to lag p.
    stationarity_check : bool
        If True, apply Augmented Dickey-Fuller (ADF) test to each ROI signal
        before fitting.  Any series whose ADF p-value exceeds
        ``adf_pvalue_threshold`` is considered non-stationary, and first-
        differencing is applied to that series before VAR fitting.  The VAR
        then models the dynamics of the differenced data.

        Linear VAR / GC assumes weak stationarity (constant mean and
        covariance).  fNIRS signals after Stage B preprocessing (bandpass
        0–0.5 Hz, OLS short-channel regression) are typically stationary in
        short (54 s) windows, but checking is good practice.
    adf_pvalue_threshold : float
        ADF p-value cutoff.  Matches config.yaml
        ``stage_d_granger.linear_gc.var.adf_pvalue_threshold``.
    trend : str
        statsmodels trend parameter: ``'c'`` = constant (default), ``'nc'`` =
        no constant.  ``'c'`` is used to match the GCCA toolbox convention
        which includes an intercept in the regression.

    Returns
    -------
    var_result : statsmodels.tsa.vector_ar.var_model.VARResults
        Fitted VAR model.  Key attributes used downstream:

        - ``var_result.coefs``   — shape ``(order, n_rois, n_rois)``.
          ``coefs[k, i, j]`` is the coefficient of variable j at lag k+1 in
          the equation for variable i.  Used to build the transfer function
          H(f) in Section 4.2.
        - ``var_result.sigma_u`` — shape ``(n_rois, n_rois)``.  Noise
          covariance matrix Σ estimated from OLS residuals.  Used in the
          spectral GC formula.
        - ``var_result.resid``   — shape ``(T - order, n_rois)``.
        - ``var_result.aic``, ``var_result.bic`` — information criteria for
          validation.

    differenced_rois : list[int]
        Indices of ROI rows that were first-differenced before fitting
        (typically empty for short pre-processed windows).

    Raises
    ------
    ValueError
        If Y has too few time points to fit VAR(order) reliably.

    Notes
    -----
    **statsmodels transpose**: statsmodels VAR.fit() expects input shape
    (T, n_rois) — the transpose of our channels-first (n_rois, T) convention.
    The transpose is performed internally; Y is not modified.

    **Differencing**: When first-differencing is applied to a series, the
    first sample is replaced with 0 so that T stays the same and all arrays
    remain aligned.  In practice, the first ``order`` samples are discarded by
    the VAR lagger anyway, so this padding has no effect on the fitted model.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import adfuller

    n_rois, T = Y.shape

    # Minimum data requirement: effective obs per equation must exceed
    # parameters per equation.  With trend='c': params = n_rois*order + 1.
    min_T = order + order * n_rois + 1
    if T < min_T:
        raise ValueError(
            f"fit_var_model: T={T} is too short for VAR({order}) with "
            f"{n_rois} ROIs.  Need at least T ≥ {min_T}.  "
            f"Consider reducing window size or VAR order."
        )

    Y_fit = Y.copy()  # (n_rois, T) — will be modified if differencing needed
    differenced_rois: list[int] = []

    # -------------------------------------------------------------------
    # 4.1.2 — ADF stationarity pre-check
    #
    # The ADF null hypothesis is: the series has a unit root (non-stationary).
    # Rejecting H0 (p ≤ threshold) → stationary.
    # Failing to reject H0 (p > threshold) → likely non-stationary → difference.
    # -------------------------------------------------------------------
    if stationarity_check:
        for roi_idx in range(n_rois):
            series = Y_fit[roi_idx, :]
            try:
                adf_stat, adf_pval, _, _, _, _ = adfuller(series, autolag="AIC")
            except Exception as exc:
                logger.warning(
                    "fit_var_model: ADF test failed for ROI %d — %s.  "
                    "Assuming stationary and continuing.",
                    roi_idx, exc,
                )
                continue

            if adf_pval > adf_pvalue_threshold:
                logger.info(
                    "fit_var_model: ROI %d non-stationary "
                    "(ADF p=%.4f > threshold=%.2f) "
                    "— applying first-differencing.",
                    roi_idx, adf_pval, adf_pvalue_threshold,
                )
                # First-difference: new[t] = original[t] - original[t-1]
                # Pad index 0 with 0 to keep T unchanged (discarded by VAR lagger)
                diff_series = np.empty_like(series)
                diff_series[0] = 0.0
                diff_series[1:] = np.diff(series)
                Y_fit[roi_idx, :] = diff_series
                differenced_rois.append(roi_idx)
            else:
                logger.debug(
                    "fit_var_model: ROI %d stationary (ADF p=%.4f ≤ %.2f).",
                    roi_idx, adf_pval, adf_pvalue_threshold,
                )

        if differenced_rois:
            logger.info(
                "fit_var_model: %d/%d ROI(s) first-differenced before VAR fit: %s",
                len(differenced_rois), n_rois, differenced_rois,
            )
        else:
            logger.debug("fit_var_model: all %d ROIs passed ADF → no differencing.", n_rois)

    # -------------------------------------------------------------------
    # 4.1.1 — VAR(order) fitting
    #
    # statsmodels VAR expects (T, n_rois) — transpose from channels-first.
    # model.fit(maxlags=order, ic=None) fixes the model order at exactly
    # `order` lags (ic=None suppresses automatic order selection).
    # -------------------------------------------------------------------
    Y_T = Y_fit.T  # (T, n_rois)
    model = VAR(Y_T)
    var_result = model.fit(maxlags=order, ic=None, trend=trend, verbose=False)

    # Validate coefficient shape: (order, n_rois, n_rois)
    expected_coef_shape = (order, n_rois, n_rois)
    if var_result.coefs.shape != expected_coef_shape:
        raise RuntimeError(
            f"fit_var_model: unexpected coefs shape {var_result.coefs.shape}. "
            f"Expected {expected_coef_shape}."
        )

    logger.info(
        "fit_var_model: VAR(%d) fitted on (%d ROIs × %d samples).  "
        "coefs=%s  AIC=%.2f  BIC=%.2f  sigma_u_diag=%s",
        order, n_rois, T,
        var_result.coefs.shape,
        var_result.aic,
        var_result.bic,
        np.round(np.diag(var_result.sigma_u), 4),
    )

    return var_result, differenced_rois


# ===========================================================================
# Quick standalone validation — Section 4.1.3
# Run as: python code/stage_d_granger/03_linear_gc.py
# ===========================================================================

if __name__ == "__main__":
    import importlib.util
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    def _import_module(name, fpath):
        """Load a Python file as a module by absolute path."""
        spec = importlib.util.spec_from_file_location(name, fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    from utils import load_config
    from stage_d_granger.utils_gc import load_preprocessed_fif

    roi_mod = _import_module(
        "extract_roi_signals",
        _CODE_ROOT / "stage_d_granger" / "01_extract_roi_signals.py",
    )
    seg_mod = _import_module(
        "trial_segmentation",
        _CODE_ROOT / "stage_d_granger" / "02_trial_segmentation.py",
    )

    cfg = load_config()
    gc_cfg = cfg["stage_d_granger"]["linear_gc"]
    var_order = gc_cfg["var"]["order"]
    stationarity = gc_cfg["var"]["stationarity_check"]
    adf_thresh = gc_cfg["var"]["adf_pvalue_threshold"]
    window_cfg = gc_cfg["window"]
    window_size = window_cfg["size_samples"]   # 1350

    print("\n" + "="*65)
    print("Section 4.1.3 — VAR Model Validation on A1, Trial 1, Window 1")
    print("="*65)

    # ---------------------------------------------------------------
    # Load A1 preprocessed data and extract ROI signals
    # ---------------------------------------------------------------
    print("\n[1/4] Loading A1 and extracting ROI signals...")
    raw = load_preprocessed_fif("A1", config=cfg)
    Data_Reg, valid_roi_names = roi_mod.extract_roi_signals(raw)
    n_rois = Data_Reg.shape[0]
    T_total = Data_Reg.shape[1]
    print(f"  Data_Reg shape : {Data_Reg.shape}  (n_rois={n_rois}, T={T_total})")
    print(f"  Valid ROIs     : {valid_roi_names}")

    # ---------------------------------------------------------------
    # Get trial segments and carve out Trial 1, Window 1
    # ---------------------------------------------------------------
    print("\n[2/4] Getting trial segments...")
    segments = seg_mod.get_trial_segments(raw, config=cfg)
    print(f"  Trials found   : {len(segments)}")
    for i, (on, off) in enumerate(segments):
        dur_s = (off - on) / raw.info["sfreq"]
        print(f"    Trial {i+1}: samples [{on}, {off}]  ({dur_s:.1f} s)")

    # Trial 1 window 1 (MATLAB: w = window_size+1 → Y(:, 1:window_size+1))
    onset, offset = segments[0]
    Y_trial = Data_Reg[:, onset:offset + 1]   # inclusive, as in MATLAB
    LY = Y_trial.shape[1]
    print(f"\n  Trial 1 data shape : {Y_trial.shape}  (LY={LY})")

    # MATLAB loop: for w = window_size+1 : sT : LY
    # First window: Y(:, w-window_size : w) = Y(:, 1 : window_size+1) → 1351 samples
    w = window_size   # 0-indexed: w=1350 corresponds to MATLAB w=1351
    Y_window = Y_trial[:, w - window_size : w + 1]   # 1351 samples (MATLAB inclusive)
    print(f"\n  Window 1 slice : Y_trial[:, {w-window_size}:{w+1}]")
    print(f"  Window shape   : {Y_window.shape}  (expected: ({n_rois}, 1351))")

    # ---------------------------------------------------------------
    # Fit VAR model on this window
    # ---------------------------------------------------------------
    print(f"\n[3/4] Fitting VAR({var_order}) on Window 1...")
    var_result, differenced = fit_var_model(
        Y_window,
        order=var_order,
        stationarity_check=stationarity,
        adf_pvalue_threshold=adf_thresh,
    )

    print(f"\n[4/4] VAR model summary:")
    print(f"  coefs.shape    : {var_result.coefs.shape}  "
          f"(expected: ({var_order}, {n_rois}, {n_rois}))")
    print(f"  sigma_u.shape  : {var_result.sigma_u.shape}  "
          f"(expected: ({n_rois}, {n_rois}))")
    print(f"  resid.shape    : {var_result.resid.shape}  "
          f"(expected: (1351-{var_order}, {n_rois}) = "
          f"({1351-var_order}, {n_rois}))")
    print(f"  AIC            : {var_result.aic:.4f}")
    print(f"  BIC            : {var_result.bic:.4f}")
    print(f"  Differenced ROIs: {differenced} "
          f"({'none' if not differenced else 'differenced'})")

    print("\n  sigma_u (noise covariance matrix):")
    for i, row_name in enumerate(valid_roi_names):
        row_str = "  ".join(f"{var_result.sigma_u[i, j]:+.4e}" for j in range(n_rois))
        print(f"    [{row_name}]  {row_str}")

    # ---------------------------------------------------------------
    # Shape assertions
    # ---------------------------------------------------------------
    assert var_result.coefs.shape == (var_order, n_rois, n_rois), \
        f"FAIL: coefs shape {var_result.coefs.shape} != ({var_order},{n_rois},{n_rois})"
    assert var_result.sigma_u.shape == (n_rois, n_rois), \
        f"FAIL: sigma_u shape {var_result.sigma_u.shape} != ({n_rois},{n_rois})"
    expected_resid_rows = Y_window.shape[1] - var_order
    assert var_result.resid.shape == (expected_resid_rows, n_rois), \
        f"FAIL: resid shape {var_result.resid.shape} != ({expected_resid_rows},{n_rois})"

    print("\n✓ All shape assertions passed.")
    print("✓ Section 4.1 validation complete.\n")
