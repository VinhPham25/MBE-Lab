"""
Stage D — Step D3: Core GC Computation (Linear VAR / Granger Causality)

This module implements the Python equivalent of MATLAB's ``cca_pwcausal()``
from the GCCA toolbox, adapted for the ASTaUND dataset with 5 ROIs and the
validated Stage C channel assignments.

Section 4 task coverage (LINEAR_GC_PROGRESS.md)
------------------------------------------------
4.1 — VAR model fitting                    (fit_var_model)
4.2 — Frequency-domain GC spectrum         (compute_gc_spectrum,
                                            compute_gc_spectrum_bivariate)
4.3 — Pairwise GC entry point              (compute_pairwise_gc)
4.4 — Sliding window loop                  (compute_windowed_gc)
4.5 — Frequency band averaging             (average_gc_over_band)

MATLAB reference
----------------
``GC_freq_main_TRGC_NovExp.m`` line 132::

    [GW,COH,pp,waut,cons] = cca_pwcausal(Y_windowed, 1, NL, 20, 20, freq, 1)

Python equivalent (bivariate, matching MATLAB)::

    GW = compute_pairwise_gc(Y_windowed, order=20, freq=freq, sfreq=25.0)

How cca_pwcausal actually works
--------------------------------
``cca_pwcausal`` is a thin wrapper around the bsmart toolbox ``pwcausal``
function (Ding, Chen & Bressler 2006).  For each unique pair (i, j) it:

1. Extracts the two-channel slice ``Y_pair = Y[[i,j], :]``
2. Fits a **bivariate** VAR(p) model (using the Morf-LWR algorithm, ``armorf``)
3. Computes the 2×2 spectral density ``S2(f) = H2(f) Σ2 H2*(f)``
   — only 4 product terms, numerically stable even near unit-root
4. Uses the **Schur-complement** of Σ2 as the effective source variance::

       e_src_given_tgt = Σ2[src,src] − Σ2[src,tgt]² / Σ2[tgt,tgt]

   This equals ``L[src,src]²`` from the lower Cholesky of Σ2, so the user's
   intuition ("Cholesky of Σ") is algebraically correct for the 2×2 case.
5. Spectral GC from src→tgt::

       GW[src, tgt, f] = log( S2_tgt(f) / (S2_tgt(f) − |H2_f[tgt,src]|² · e_src_given_tgt) )

This is **bivariate unconditional GC** — it does NOT condition on the
remaining n−2 channels.  The Schur complement accounts for instantaneous
innovation cross-correlation within each pair; this is a weaker but
numerically stable form of conditioning compared to full multivariate
conditioning on all other ROIs.

Sign / index convention
-----------------------
Python GW uses **source-major** ordering:
``GW[src, tgt, freq_idx]`` = GC from ``src`` ROI → ``tgt`` ROI.
MATLAB GW uses **target-major** ordering: ``GW_MATLAB(i, j, f)`` stores
GC from j→i, so ``GW_Python = GW_MATLAB.T`` (transposed in first two dims).

References
----------
Geweke J (1982). "Measurement of Linear Dependence and Feedback Between
Multiple Time Series." JASA 77(378):304–313.

Chen Y, Bressler SL & Ding M (2006). "Frequency decomposition of conditional
Granger causality and application to multivariate neural field potential data."
J Neurosci Methods 150:228–237.

Seth AK (2010). "A MATLAB toolbox for Granger causal connectivity analysis."
J Neurosci Methods 186:262–273.
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
# Section 4.2.2 — Transfer Function Helper
# ===========================================================================

def _compute_transfer_function(
    coefs: np.ndarray,
    freq: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Convert VAR coefficients to the spectral transfer function matrix H(f).

    For a VAR(p) model  X(t) = Σ_{k=1}^{p} A_k X(t-k) + ε(t), the lag-
    polynomial evaluated at complex frequency  z = exp(-2πi f / sfreq)  is::

        A(f) = I  -  Σ_{k=1}^{p}  A_k · exp(-2πi · f · k / sfreq)

    The transfer (Green's) function is then  H(f) = A(f)^{-1}.

    H(f)[j, i] is the linear response of variable j to a unit impulse in
    variable i at frequency f.  The sign convention matches the GCCA toolbox:
    GC from i → j uses H_f[j, i].

    Parameters
    ----------
    coefs : np.ndarray, shape (order, n_rois, n_rois)
        VAR coefficient matrices.  ``coefs[k, i, j]`` = coefficient of
        variable j at lag k+1 in the ODE for variable i (statsmodels layout).
    freq : np.ndarray, shape (n_freqs,)
        Frequency vector in Hz.  For this pipeline: ``np.arange(0, 0.11, 0.01)``
        → 11 points [0.00, 0.01, …, 0.10].
    sfreq : float
        Sampling frequency in Hz (25 Hz for this dataset).

    Returns
    -------
    H : np.ndarray, shape (n_freqs, n_rois, n_rois), dtype complex128
        Transfer function matrix evaluated at each frequency.
        ``H[f_idx, j, i]`` = response of variable j to variable i at
        frequency ``freq[f_idx]``.

    Notes
    -----
    Implemented using NumPy broadcasting to avoid a Python loop over
    frequencies (vectorised over the n_freqs axis).  For n_freqs = 11 the
    speed difference is negligible, but the notation is cleaner.

    At f = 0 (DC), the formula reduces to
    ``H(0) = (I - Σ_k A_k)^{-1}`` — the static gain matrix of the VAR.
    For a stationary process this is always finite and well-conditioned.
    """
    order, n_rois, _ = coefs.shape
    n_freqs = len(freq)

    # Phase factors exp(-2πi f k / sfreq), shape (order, n_freqs)
    k_arr = np.arange(1, order + 1, dtype=float)[:, np.newaxis]   # (p, 1)
    f_arr = freq[np.newaxis, :]                                     # (1, n_freqs)
    phase = np.exp(-2j * np.pi * f_arr * k_arr / sfreq)            # (p, n_freqs)

    # Σ_{k=1}^{p} A_k exp(-2πi f k / sfreq) — result shape (n_freqs, n, n)
    # einsum: k=lag index, i/j=ROI indices, f=freq index
    A_sum = np.einsum("kij,kf->fij", coefs.astype(complex), phase)  # (n_freqs, n, n)

    # A(f) = I - A_sum
    I_broadcast = np.eye(n_rois, dtype=complex)[np.newaxis]          # (1, n, n)
    A_f = I_broadcast - A_sum                                         # (n_freqs, n, n)

    # H(f) = A(f)^{-1}  — np.linalg.inv broadcasts over leading dimension
    H = np.linalg.inv(A_f)                                           # (n_freqs, n, n)

    return H


# ===========================================================================
# Section 4.2.1 — Frequency-Domain GC Spectrum
# ===========================================================================

def compute_gc_spectrum(
    var_result,
    freq: np.ndarray,
    sfreq: float = 25.0,
) -> np.ndarray:
    """
    Compute the spectral Granger Causality matrix GW from a fitted VAR model.

    This is the core spectral step that mirrors the inner computation of
    MATLAB's ``cca_pwcausal()`` — producing the ``GW`` (5×5×11) output.

    Theory
    ------
    Given the VAR(p) model and its transfer function H(f) = A(f)^{-1}:

    The cross-spectral density matrix is::

        S(f) = H(f) · Σ · H*(f)      [Σ = noise covariance from OLS residuals]

    The spectral GC from source ROI i to target ROI j is (Geweke 1982)::

        GW[i, j, f] = log( S_jj(f) / (S_jj(f) - σ_ii · |H_ji(f)|²) )

    where:
    - ``S_jj(f) = [H(f) Σ H*(f)]_{jj}`` is the total power of j at frequency f
    - ``σ_ii = Σ[i, i]`` is the noise variance of source ROI i
    - ``H_ji(f) = H(f)[j, i]`` is the transfer from i to j at frequency f
    - ``σ_ii · |H_ji(f)|²`` is i's contribution to j's spectrum at f

    The log-ratio is the fraction of j's power at frequency f that is caused
    by the past of i.  When i has no causal influence on j at f,
    ``H_ji(f) ≈ 0`` and ``GW[i, j, f] ≈ 0``.

    Sign convention
    ---------------
    ``GW[src=i, tgt=j, f]`` = GC **from** ROI i **to** ROI j at frequency f.
    Diagonal (i == j) is forced to 0 (no self-causality).
    All values are non-negative (log of ratio ≥ 1).

    Conditioning
    ------------
    **Why this formula gives pairwise *conditional* GC** (Barnett & Seth 2014
    §2.3): the VAR model was fitted on *all n ROIs jointly*, so H(f) and Σ
    already encode the full multivariate dynamics.  The transfer H_ji(f)
    represents the direct pathway from i to j *after* all shared influence
    from other variables is captured in the model.  No separate bivariate
    models are needed — the multivariate VAR is the conditioning mechanism.

    Parameters
    ----------
    var_result : statsmodels VARResults
        Fitted VAR model returned by ``fit_var_model()``.
    freq : np.ndarray, shape (n_freqs,)
        Frequency vector in Hz.  Matches MATLAB ``freq = [0:0.01:0.1]`` → 11
        points.  Must be ≤ sfreq/2 (Nyquist).
    sfreq : float
        Sampling frequency in Hz.  Default 25 Hz.

    Returns
    -------
    GW : np.ndarray, shape (n_rois, n_rois, n_freqs), dtype float64
        Spectral GC matrix.
        ``GW[i, j, f_idx]`` = GC from ROI i → ROI j at ``freq[f_idx]`` Hz.
        Diagonal entries are 0.  Off-diagonal values are ≥ 0.

    Raises
    ------
    RuntimeError
        If the transfer function inversion fails (singular A(f) — pathological
        VAR fit, e.g. all-constant signal).

    Notes
    -----
    **Diagonal-Σ approximation (multivariate)**: This function uses a single
    multivariate VAR fitted on all n channels jointly (via ``fit_var_model``).
    It approximates S_jj(f) using only the diagonal of Σ to avoid catastrophic
    numerical cancellation.  The source contribution uses ``σ_ii`` directly
    rather than the Schur complement.

    **Relationship to MATLAB cca_pwcausal**: The actual MATLAB implementation
    uses per-pair bivariate VAR models (bsmart ``pwcausal``), not a single
    multivariate model.  Use ``compute_gc_spectrum_bivariate()`` for a
    faithful match to MATLAB.  Both functions preserve the directional
    GC relationships; they differ in absolute magnitude because of different
    Σ decompositions.

    **Numerical stability**: The diagonal-Σ S_jj is computed as
    ``Σ_k σ_kk |H_jk|²``, which avoids the catastrophic cancellation of the
    full ``[H Σ H*]_jj`` product.
    """
    coefs = var_result.coefs   # (order, n_rois, n_rois)
    Sigma = var_result.sigma_u  # (n_rois, n_rois)
    order, n_rois, _ = coefs.shape
    n_freqs = len(freq)

    # 4.2.2 — Transfer function at all frequencies
    H = _compute_transfer_function(coefs, freq, sfreq)  # (n_freqs, n, n)

    # 4.2.1 — Spectral GC computation
    #
    # Numerical stability note
    # ------------------------
    # The naive formula  S_jj = [H(f) Σ H*(f)]_jj  suffers catastrophic
    # cancellation for this dataset.  The fNIRS ROI innovations are near-
    # rank-1 (all channels share a strong common hemodynamic mode), and the
    # VAR transfer function is near-unit-root (|H| ~ 7×10⁶ at 0.05 Hz due to
    # the slow hemodynamic timescale in 54-second windows).  Their product
    # H_ji Σ_ij H_il* sums to near-zero due to off-diagonal cancellation,
    # while the diagonal term σ_ii |H_ji|² is ~10⁹ × larger, making the
    # denominator negative.
    #
    # Stable alternative — "diagonal-Σ approximation":
    #
    #   S_jj^{diag}(f) = Σ_k  σ_kk · |H(f)[j,k]|²
    #
    # This uses only the diagonal of Σ (each ROI's own noise variance) and
    # is exactly  |H_L(f)[j,:]|²  where H_L = H diag(√σ).  No cancellation.
    # The formula is exact when innovations are uncorrelated.  For correlated
    # innovations (as here) it approximates the contribution of each ROI's
    # *unique* noise to the target spectrum, consistent with the directional
    # GC intent and with the notation in docs/LINEAR_GC_PROGRESS.md §4.2.1.
    #
    # Reference: Geweke (1982) Eq. 13; Seth (2010) Eq. 6/8 (diagonal Σ case).

    sigma_diag = np.diag(Sigma)                          # (n_rois,)  σ_kk values

    GW = np.zeros((n_rois, n_rois, n_freqs), dtype=float)

    for f_idx in range(n_freqs):
        H_f = H[f_idx]                                    # (n_rois, n_rois)
        H_abs2 = np.abs(H_f) ** 2                         # (n_rois, n_rois)

        # Diagonal-Σ spectral power for each target ROI j:
        #   S_jj^{diag}(f) = Σ_k  σ_kk · |H[j,k]|²  = H_abs2[j,:] · sigma_diag
        S_jj_all = H_abs2 @ sigma_diag                    # (n_rois,)

        for src in range(n_rois):
            sigma_src = sigma_diag[src]                   # σ_ii

            for tgt in range(n_rois):
                if src == tgt:
                    continue

                S_jj = S_jj_all[tgt]
                # Source i's contribution: σ_ii · |H[j,i]|²
                source_contrib = sigma_src * H_abs2[tgt, src]
                # Intrinsic power of j not caused by i
                intrinsic = S_jj - source_contrib

                if S_jj <= 0.0 or intrinsic <= 0.0:
                    GW[src, tgt, f_idx] = 0.0
                else:
                    GW[src, tgt, f_idx] = float(np.log(S_jj / intrinsic))

    # Ensure diagonal is exactly zero (floating-point safety)
    for k in range(n_rois):
        GW[k, k, :] = 0.0

    logger.debug(
        "compute_gc_spectrum: GW shape=%s  off-diag range=[%.4f, %.4f]",
        GW.shape,
        float(np.min(GW[GW > 0])) if np.any(GW > 0) else 0.0,
        float(np.max(GW)),
    )

    return GW


# ===========================================================================
# Section 4.2 (revised) — Bivariate bsmart-faithful GC Spectrum
# ===========================================================================

def compute_gc_spectrum_bivariate(
    Y: np.ndarray,
    order: int = 20,
    freq: np.ndarray = None,
    sfreq: float = 25.0,
    trend: str = "c",
) -> np.ndarray:
    """
    Compute spectral GC by fitting a bivariate VAR for each channel pair.

    This is a faithful Python translation of the bsmart ``pwcausal`` function
    that underlies MATLAB's ``cca_pwcausal``.  For each unique pair (src, tgt)
    it fits an independent bivariate VAR(order) model, then applies the
    Schur-complement spectral GC formula (Geweke 1982; Chen et al. 2006).

    Why bivariate, not multivariate?
    ---------------------------------
    The full 5×5 spectral density  S_jj(f) = [H Σ H*]_jj  suffers
    catastrophic numerical cancellation when Σ is near-rank-1 (which occurs
    in fNIRS data because all ROIs share a strong common hemodynamic mode).
    Fitting separate 2×2 models avoids this entirely: the bivariate S2(f)
    matrix has only 4 product terms and the 2×2 Σ2 is much better conditioned
    than the full 5×5 Σ.

    Formula (per pair, for GC from src→tgt)
    -----------------------------------------
    ::

        e_src = Σ2[0,0] − Σ2[0,1]² / Σ2[1,1]   # Schur complement (≡ L[0,0]² in Cholesky)
        S2_f  = H2_f @ Σ2 @ H2_f.conj().T        # 2×2 spectral density (4 terms, stable)
        GW[src, tgt, f] = log( Re(S2_f[1,1]) / (Re(S2_f[1,1]) − |H2_f[1,0]|² · e_src) )

    where indices 0/1 correspond to src/tgt within the pair slice.

    Parameters
    ----------
    Y : np.ndarray, shape (n_rois, T)
        ROI signal matrix for one window.  NaN channels are skipped and the
        corresponding GW rows/columns remain 0.
    order : int
        VAR model order (lags).  Default 20.
    freq : np.ndarray, shape (n_freqs,)
        Frequencies in Hz.  If None, defaults to ``np.arange(0, 0.11, 0.01)``.
    sfreq : float
        Sampling frequency in Hz.  Default 25.0.
    trend : str
        Trend term for VAR fitting (passed to statsmodels).  Default ``"c"``.

    Returns
    -------
    GW : np.ndarray, shape (n_rois, n_rois, n_freqs), dtype float64
        ``GW[src, tgt, f_idx]`` = GC from ROI src → ROI tgt at ``freq[f_idx]``.
        Diagonal is 0.  Values are non-negative.

    Notes
    -----
    This is **bivariate unconditional GC** — it does not condition on other
    ROIs.  The Schur complement term accounts for instantaneous innovation
    cross-correlation within each pair.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR as _VAR

    if freq is None:
        freq = np.arange(0.0, 0.11, 0.01)

    n_rois, T = Y.shape
    n_freqs = len(freq)
    GW = np.zeros((n_rois, n_rois, n_freqs), dtype=float)

    for src in range(n_rois):
        for tgt in range(src + 1, n_rois):
            # Skip any NaN channel (R11 / R2 padding)
            Y_pair = Y[[src, tgt], :]
            if np.any(np.isnan(Y_pair)):
                logger.debug("bivariate VAR skipped for pair (%d,%d): NaN channel", src, tgt)
                continue

            # Fit bivariate VAR(order) on [src, tgt] channels
            try:
                var2 = _VAR(Y_pair.T).fit(maxlags=order, ic=None, trend=trend, verbose=False)
            except Exception as exc:
                logger.warning("bivariate VAR failed for pair (%d,%d): %s", src, tgt, exc)
                continue

            A2 = var2.coefs   # (order, 2, 2)
            Z2 = var2.sigma_u  # (2, 2)

            # Schur complements — conditional variances of each channel's
            # innovation given the other channel's innovation.
            # For a 2×2 Z2, this equals L[0,0]² / L[1,1]² from the lower
            # Cholesky factor, so it is always ≥ 0.
            e_src = Z2[0, 0] - Z2[0, 1] ** 2 / Z2[1, 1]   # Var(ε_src | ε_tgt)
            e_tgt = Z2[1, 1] - Z2[1, 0] ** 2 / Z2[0, 0]   # Var(ε_tgt | ε_src)

            # Transfer function for the bivariate model: shape (n_freqs, 2, 2)
            H2 = _compute_transfer_function(A2, freq, sfreq)

            for f_idx in range(n_freqs):
                H2_f = H2[f_idx]                            # (2, 2)

                # 2×2 spectral density — only 4 product terms, no cancellation
                S2_f = H2_f @ Z2 @ H2_f.conj().T           # (2, 2)

                S_tgt = float(np.real(S2_f[1, 1]))          # power of target channel
                S_src = float(np.real(S2_f[0, 0]))          # power of source channel

                # GC from src (channel 0) → tgt (channel 1)
                # H2_f[1, 0] = transfer from src to tgt in bivariate model
                contrib_src = float(np.abs(H2_f[1, 0]) ** 2) * e_src
                intrinsic_tgt = S_tgt - contrib_src
                if S_tgt > 0.0 and intrinsic_tgt > 0.0:
                    GW[src, tgt, f_idx] = float(np.log(S_tgt / intrinsic_tgt))

                # GC from tgt (channel 1) → src (channel 0)
                # H2_f[0, 1] = transfer from tgt to src in bivariate model
                contrib_tgt = float(np.abs(H2_f[0, 1]) ** 2) * e_tgt
                intrinsic_src = S_src - contrib_tgt
                if S_src > 0.0 and intrinsic_src > 0.0:
                    GW[tgt, src, f_idx] = float(np.log(S_src / intrinsic_src))

    logger.debug(
        "compute_gc_spectrum_bivariate: GW shape=%s  off-diag range=[%.4f, %.4f]",
        GW.shape,
        float(np.min(GW[GW > 0])) if np.any(GW > 0) else 0.0,
        float(np.max(GW)),
    )

    return GW


# ===========================================================================
# Section 4.3 — Pairwise GC Entry Point (mirror of cca_pwcausal)
# ===========================================================================

def compute_pairwise_gc(
    Y_window: np.ndarray,
    order: int = 20,
    freq: np.ndarray = None,
    sfreq: float = 25.0,
    trend: str = "c",
) -> np.ndarray:
    """
    Compute bivariate unconditional spectral GC for one window.

    This is the public entry-point mirroring MATLAB's::

        [GW,COH,pp,waut,cons] = cca_pwcausal(Y_windowed, 1, NL, 20, 20, freq, 1)

    Internally delegates to ``compute_gc_spectrum_bivariate()``, which fits a
    separate bivariate VAR(order) for each unique channel pair and applies the
    bsmart Schur-complement spectral GC formula.

    This function computes **bivariate unconditional GC** — it does not
    condition on the remaining n−2 channels.  The Schur-complement term
    ``e_src = Σ2[src,src] − Σ2[src,tgt]² / Σ2[tgt,tgt]``
    accounts for instantaneous noise cross-correlation within each pair;
    this is a weaker but numerically stable form of conditioning compared to
    full multivariate conditioning on all other ROIs.

    Parameters
    ----------
    Y_window : np.ndarray, shape (n_rois, T)
        ROI signal matrix for one 1351-sample window.  NaN rows (e.g. for
        subjects R11 or R2 with bad ROI channels) are handled gracefully —
        the corresponding GW rows/columns remain 0.
    order : int
        VAR model order.  Default 20 (matches MATLAB reference).
    freq : np.ndarray, shape (n_freqs,), optional
        Frequency vector in Hz.  Default ``np.arange(0, 0.11, 0.01)`` → 11
        points matching MATLAB ``freq = [0:0.01:0.1]``.
    sfreq : float
        Sampling frequency in Hz.  Default 25.0.
    trend : str
        Trend term for bivariate VAR fits.  Default ``"c"`` (constant).

    Returns
    -------
    GW : np.ndarray, shape (n_rois, n_rois, n_freqs), dtype float64
        Spectral GC matrix.  ``GW[src, tgt, f_idx]`` = GC from ROI src to ROI
        tgt at ``freq[f_idx]`` Hz.  Diagonal is 0, off-diagonal values ≥ 0.

    Notes
    -----
    **Sign convention:** Python uses source-major ordering (``GW[src, tgt]``),
    which is the transpose of MATLAB's target-major ``GW_MATLAB(tgt, src)``.

    **NaN handling:** Any pair involving a NaN channel is skipped and both
    ``GW[src, tgt]`` and ``GW[tgt, src]`` remain 0 for all frequencies.
    Downstream band-averaging and diagonal-removal handle this cleanly.
    """
    if freq is None:
        freq = np.arange(0.0, 0.11, 0.01)

    GW = compute_gc_spectrum_bivariate(
        Y_window,
        order=order,
        freq=freq,
        sfreq=sfreq,
        trend=trend,
    )

    logger.info(
        "compute_pairwise_gc: GW shape=%s  off-diag range=[%.4f, %.4f]",
        GW.shape,
        float(np.min(GW[GW > 0])) if np.any(GW > 0) else 0.0,
        float(np.max(GW)),
    )

    return GW


# ===========================================================================
# Section 4.4 — Sliding Window Loop
# ===========================================================================

def compute_windowed_gc(
    Y_trial: np.ndarray,
    window_size: int = 1350,
    stride: int = 1350,
    order: int = 20,
    freq: np.ndarray = None,
    sfreq: float = 25.0,
    trend: str = "c",
) -> dict:
    """
    Slide a fixed-size window over one trial and compute GC per window.

    Mirrors MATLAB's inner loop::

        for w = window_size+1 : sT : LY
            Y_windowed = Y(:, w-window_size : w);   % 1351 samples (inclusive)
            [GW,...] = cca_pwcausal(Y_windowed, ...);

    The Python indexing ``Y_trial[:, w-window_size : w+1]`` replicates the
    MATLAB inclusive slice, producing exactly ``window_size + 1 = 1351``
    samples per window (see Deviation D3 in docs/LINEAR_GC_PROGRESS.md).

    Parameters
    ----------
    Y_trial : np.ndarray, shape (n_rois, T)
        Full ROI signal matrix for one trial.  May contain all-NaN rows for
        bad ROI channels (e.g. R11, R2) — handled gracefully per window.
    window_size : int
        Number of stride samples (base step).  Actual window length is
        ``window_size + 1`` to match the MATLAB inclusive slice.  Default 1350
        (54 s × 25 Hz).
    stride : int
        Step between window starts in samples.  Default 1350 (no overlap).
    order : int
        VAR model order passed to ``compute_pairwise_gc()``.  Default 20.
    freq : np.ndarray, optional
        Frequency vector in Hz.  Default ``np.arange(0, 0.11, 0.01)``.
    sfreq : float
        Sampling frequency in Hz.  Default 25.0.
    trend : str
        Trend term for VAR fits.  Default ``"c"``.

    Returns
    -------
    windows : dict
        ``{'Win_1': GW_array, 'Win_2': GW_array, ...}``
        Each value is ``np.ndarray`` of shape ``(n_rois, n_rois, n_freqs)``.
        Keys are 1-indexed strings matching MATLAB ``W = sprintf('Win_%d', Wn+1)``.
        Empty dict if the trial is shorter than one full window.

    Notes
    -----
    **Skipping logic:** A window is skipped (and not included in the output
    dict) only if every ROI channel is all-NaN — meaning there is literally
    no signal to model.  Structurally NaN ROIs (bad channels, as in R11/R2)
    are *not* a skip condition; ``compute_pairwise_gc`` handles them per-pair.
    """
    if freq is None:
        freq = np.arange(0.0, 0.11, 0.01)

    n_rois, T = Y_trial.shape
    windows: dict = {}
    win_num = 0          # 0-indexed counter → label uses win_num+1

    # Mirror: for w = window_size+1 : stride : LY  (MATLAB 1-indexed)
    # Python: w starts at window_size (0-indexed), increments by stride,
    #         while w < T so that Y_trial[:, w-window_size : w+1] is in-bounds.
    for w in range(window_size, T, stride):
        Y_win = Y_trial[:, w - window_size : w + 1]   # (n_rois, window_size+1)

        # Skip only if every sample of every ROI is NaN (completely empty)
        if np.all(np.isnan(Y_win)):
            logger.warning(
                "Window %d skipped: all-NaN data (samples %d–%d)",
                win_num + 1, w - window_size, w,
            )
            continue

        label = f"Win_{win_num + 1}"
        logger.info(
            "  %s: samples [%d, %d]  shape=%s",
            label, w - window_size, w, Y_win.shape,
        )

        GW = compute_pairwise_gc(
            Y_win, order=order, freq=freq, sfreq=sfreq, trend=trend
        )
        windows[label] = GW
        win_num += 1

    logger.info(
        "compute_windowed_gc: %d windows extracted from trial of %d samples",
        len(windows), T,
    )
    return windows


# ===========================================================================
# Section 4.5 — Frequency Band Averaging
# ===========================================================================

def average_gc_over_band(
    GW: np.ndarray,
    freq: np.ndarray,
    f_low: float = 0.01,
    f_high: float = 0.07,
) -> np.ndarray:
    """
    Average the spectral GC matrix over the neurophysiological frequency band.

    Mirrors MATLAB::

        idx1 = find(freq == 0.01);   % → index 2 (1-indexed)
        idx2 = find(freq == 0.07);   % → index 8 (1-indexed)
        GC_fqmean = mean(GW(:,:,idx1:idx2), 3);

    MATLAB's ``idx1:idx2`` (inclusive both ends, 1-indexed) corresponds to
    Python's ``idx1 : idx2+1`` (exclusive right end, 0-indexed), so the same
    7 frequency bins (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07 Hz) are
    averaged in both implementations.

    Parameters
    ----------
    GW : np.ndarray, shape (n_rois, n_rois, n_freqs)
        Spectral GC matrix from ``compute_pairwise_gc()`` or
        ``compute_windowed_gc()``.
    freq : np.ndarray, shape (n_freqs,)
        Frequency vector in Hz used to produce ``GW``.
    f_low : float
        Lower bound of the band in Hz.  Default 0.01 Hz.
    f_high : float
        Upper bound of the band in Hz.  Default 0.07 Hz.

    Returns
    -------
    GC_mean : np.ndarray, shape (n_rois, n_rois), dtype float64
        Band-averaged GC matrix.  ``GC_mean[src, tgt]`` = mean GC from src
        to tgt over [f_low, f_high].  Diagonal is 0, off-diagonal values ≥ 0.

    Raises
    ------
    ValueError
        If ``f_low`` or ``f_high`` are not found in ``freq`` (within floating-
        point tolerance of 1e-9 Hz).
    """
    # Find indices using nearest-match to guard against float rounding
    idx1 = int(np.argmin(np.abs(freq - f_low)))
    idx2 = int(np.argmin(np.abs(freq - f_high)))

    if abs(freq[idx1] - f_low) > 1e-9:
        raise ValueError(
            f"f_low={f_low} Hz not found in freq vector "
            f"(nearest={freq[idx1]:.4f}). Check config frequency settings."
        )
    if abs(freq[idx2] - f_high) > 1e-9:
        raise ValueError(
            f"f_high={f_high} Hz not found in freq vector "
            f"(nearest={freq[idx2]:.4f}). Check config frequency settings."
        )

    # Python slice idx1:idx2+1 is inclusive on both ends — mirrors MATLAB idx1:idx2
    GC_mean = np.mean(GW[:, :, idx1 : idx2 + 1], axis=2)

    logger.debug(
        "average_gc_over_band: [%.2f–%.2f Hz]  idx=[%d,%d]  "
        "%d bins  off-diag range=[%.4f, %.4f]",
        f_low, f_high, idx1, idx2,
        idx2 - idx1 + 1,
        float(np.min(GC_mean[GC_mean > 0])) if np.any(GC_mean > 0) else 0.0,
        float(np.max(GC_mean)),
    )

    return GC_mean


# ===========================================================================
# Quick standalone validation — Sections 4.1.3, 4.2.3, 4.3, 4.4, 4.5
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
    # 4.1.3 — Shape assertions
    # ---------------------------------------------------------------
    assert var_result.coefs.shape == (var_order, n_rois, n_rois), \
        f"FAIL: coefs shape {var_result.coefs.shape} != ({var_order},{n_rois},{n_rois})"
    assert var_result.sigma_u.shape == (n_rois, n_rois), \
        f"FAIL: sigma_u shape {var_result.sigma_u.shape} != ({n_rois},{n_rois})"
    expected_resid_rows = Y_window.shape[1] - var_order
    assert var_result.resid.shape == (expected_resid_rows, n_rois), \
        f"FAIL: resid shape {var_result.resid.shape} != ({expected_resid_rows},{n_rois})"

    print("\n✓ All 4.1 shape assertions passed.")

    # ===================================================================
    # Section 4.2.3 — GC Spectrum Validation on A1 Trial 1 Window 1
    # ===================================================================
    print("\n" + "="*65)
    print("Section 4.2.3 — GC Spectrum Validation on A1, Trial 1, Window 1")
    print("="*65)

    freq_cfg = gc_cfg["frequency"]
    freq = np.arange(
        freq_cfg["f_start"],
        freq_cfg["f_stop"] + freq_cfg["f_step"] / 2,
        freq_cfg["f_step"],
    )
    sfreq_val = float(window_cfg["sfreq"])

    print(f"\n  Frequency vector : {np.round(freq, 3)}")
    print(f"  n_freqs          : {len(freq)}  (expected: 11)")
    print(f"  sfreq            : {sfreq_val} Hz")

    print(f"\n[5/6] Computing GC spectrum — diagonal-Σ (multivariate) approach...")
    GW = compute_gc_spectrum(var_result, freq=freq, sfreq=sfreq_val)

    print(f"\n[6/6] GC Spectrum summary:")
    print(f"  GW.shape          : {GW.shape}  (expected: ({n_rois}, {n_rois}, 11))")
    print(f"  Diagonal all-zero : {np.all(GW[np.eye(n_rois, dtype=bool)] == 0.0)}")
    print(f"  All values ≥ 0    : {np.all(GW >= 0.0)}")
    print(f"  GW min            : {GW.min():.6f}")
    print(f"  GW max            : {GW.max():.6f}")
    print(f"  GW off-diag mean  : {GW[~np.eye(n_rois, dtype=bool)].mean():.4f}")

    print(f"\n  Off-diagonal GC values at each frequency (src→tgt):")
    conn_labels = [
        f"{valid_roi_names[s]}→{valid_roi_names[t]}"
        for s in range(n_rois) for t in range(n_rois) if s != t
    ]
    for conn_idx, (s, t) in enumerate(
        (s, t) for s in range(n_rois) for t in range(n_rois) if s != t
    ):
        vals = GW[s, t, :]
        print(
            f"    [{conn_labels[conn_idx]:12s}]  "
            f"f=[{', '.join(f'{v:.3f}' for v in vals)}]  "
            f"mean={vals.mean():.4f}"
        )

    # 4.2.3 — Shape and property assertions (diagonal-Σ multivariate)
    assert GW.shape == (n_rois, n_rois, 11), \
        f"FAIL: GW shape {GW.shape} != ({n_rois}, {n_rois}, 11)"
    assert np.all(GW[np.eye(n_rois, dtype=bool)] == 0.0), \
        "FAIL: diagonal contains non-zero values"
    assert np.all(GW >= 0.0), \
        f"FAIL: GW has negative values (min={GW.min()})"

    # ---------------------------------------------------------------
    # Bivariate bsmart approach (matches cca_pwcausal exactly)
    # ---------------------------------------------------------------
    print(f"\n[5b/6] Computing GC spectrum — bivariate Schur-complement (bsmart) approach...")
    GW_biv = compute_gc_spectrum_bivariate(
        Y_window, order=var_order, freq=freq, sfreq=sfreq_val
    )

    print(f"\n  Bivariate GW summary:")
    print(f"  GW_biv.shape       : {GW_biv.shape}")
    print(f"  Diagonal all-zero  : {np.all(GW_biv[np.eye(n_rois, dtype=bool)] == 0.0)}")
    print(f"  All values ≥ 0     : {np.all(GW_biv >= 0.0)}")
    print(f"  GW_biv min         : {GW_biv.min():.6f}")
    print(f"  GW_biv max         : {GW_biv.max():.6f}")
    print(f"  GW_biv off-diag mean: {GW_biv[~np.eye(n_rois, dtype=bool)].mean():.4f}")

    print(f"\n  Off-diagonal GC (bivariate) at each frequency (src→tgt):")
    for conn_idx, (s, t) in enumerate(
        (s, t) for s in range(n_rois) for t in range(n_rois) if s != t
    ):
        vals = GW_biv[s, t, :]
        print(
            f"    [{conn_labels[conn_idx]:12s}]  "
            f"f=[{', '.join(f'{v:.3f}' for v in vals)}]  "
            f"mean={vals.mean():.4f}"
        )

    assert GW_biv.shape == (n_rois, n_rois, 11)
    assert np.all(GW_biv[np.eye(n_rois, dtype=bool)] == 0.0)
    assert np.all(GW_biv >= 0.0)

    # ---------------------------------------------------------------
    # 4.2.3 — Plot both methods side by side
    # ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stage_d_granger.utils_gc import get_gc_output_dir
    vis_dir = get_gc_output_dir("visualizations", config=cfg)

    n_conns = n_rois * (n_rois - 1)  # 20 off-diagonal connections
    pairs = [(s, t) for s in range(n_rois) for t in range(n_rois) if s != t]

    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharey=False)
    axes = axes.flatten()
    for conn_plot_idx, (s, t) in enumerate(pairs):
        ax = axes[conn_plot_idx]
        ax.plot(freq, GW[s, t, :], "b-o", markersize=3, linewidth=1.2,
                label="diag-Σ (multiv.)")
        ax.plot(freq, GW_biv[s, t, :], "r--s", markersize=3, linewidth=1.2,
                label="bsmart (biv.)")
        ax.axvspan(0.01, 0.07, alpha=0.12, color="green")
        ax.set_title(f"{valid_roi_names[s]}→{valid_roi_names[t]}", fontsize=8)
        ax.set_xlabel("Hz", fontsize=7)
        ax.set_ylabel("GC", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if conn_plot_idx == 0:
            ax.legend(fontsize=6)

    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "A1 — Spectral GC: diagonal-Σ multivariate (blue) vs bsmart bivariate (red)\n"
        "Trial 1, Window 1  |  Green band = neurophysio 0.01–0.07 Hz",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = vis_dir / "sub-A1_Tr1_Win1_gc_spectrum_comparison.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\n  Comparison plot saved → {plot_path.relative_to(_CODE_ROOT.parent)}")

    # ---------------------------------------------------------------
    # 4.2.3 — Debug trace: intermediate values for LPFC→CPFC at 0.05 Hz
    # ---------------------------------------------------------------
    print("\n  DEBUG — intermediate values for LPFC→CPFC at f=0.05 Hz:")
    H_all_debug = _compute_transfer_function(var_result.coefs, freq, sfreq_val)
    f_mid_idx = 5  # freq[5] = 0.05 Hz
    H_debug = H_all_debug[f_mid_idx]
    Sigma_debug = var_result.sigma_u
    sigma_diag_d = np.diag(Sigma_debug)
    src_d, tgt_d = 0, 1  # LPFC → CPFC
    H_abs2_d = np.abs(H_debug)**2
    S_jj_diag_d = float((H_abs2_d @ sigma_diag_d)[tgt_d])   # stable
    S_jj_full_d = float(np.real((H_debug @ Sigma_debug @ H_debug.conj().T)[tgt_d, tgt_d]))
    sigma_src_d = float(sigma_diag_d[src_d])
    contrib_d  = sigma_src_d * float(H_abs2_d[tgt_d, src_d])
    print(f"    H[CPFC, LPFC] at 0.05 Hz          : {H_debug[tgt_d,src_d]:.4e}  |H|={abs(H_debug[tgt_d,src_d]):.4e}")
    print(f"    S_jj FULL [H Σ H*]_jj              : {S_jj_full_d:.4e}  (catastrophic cancellation)")
    print(f"    S_jj DIAGONAL Σ_k σ_kk|H_jk|^2    : {S_jj_diag_d:.4e}  (numerically stable)")
    print(f"    sigma_u[LPFC, LPFC]                : {sigma_src_d:.4e}")
    print(f"    sigma_src * |H[CPFC,LPFC]|^2       : {contrib_d:.4e}")
    print(f"    intrinsic (S_diag - contrib)       : {S_jj_diag_d - contrib_d:.4e}")
    if S_jj_diag_d > contrib_d > 0:
        print(f"    GC(LPFC→CPFC, 0.05 Hz)            : {np.log(S_jj_diag_d / (S_jj_diag_d - contrib_d)):.6f}")
    print(f"    Y_window std per ROI               : {np.std(Y_window, axis=1)}")
    print(f"    var_result.resid std               : {np.std(var_result.resid, axis=0)}")

    print("\n✓ All 4.2 assertions passed (both diagonal-Σ multivariate and bsmart bivariate).")

    # ===================================================================
    # Section 4.3 — compute_pairwise_gc Validation
    # ===================================================================
    print("\n" + "="*65)
    print("Section 4.3 — compute_pairwise_gc Validation")
    print("="*65)

    print(f"\n[7/8] Calling compute_pairwise_gc on Window 1...")
    GW_entry = compute_pairwise_gc(
        Y_window, order=var_order, freq=freq, sfreq=sfreq_val
    )

    print(f"\n  Entry-point GW summary:")
    print(f"  GW_entry.shape         : {GW_entry.shape}  (expected: ({n_rois}, {n_rois}, 11))")
    print(f"  Diagonal all-zero      : {np.all(GW_entry[np.eye(n_rois, dtype=bool)] == 0.0)}")
    print(f"  All values ≥ 0         : {np.all(GW_entry >= 0.0)}")
    print(f"  GW_entry matches GW_biv: {np.allclose(GW_entry, GW_biv)}")
    print(f"  GW_entry off-diag mean : {GW_entry[~np.eye(n_rois, dtype=bool)].mean():.4f}")
    print(f"  GW_entry off-diag max  : {GW_entry.max():.4f}")

    # 4.3 assertions
    assert GW_entry.shape == (n_rois, n_rois, 11), \
        f"FAIL: GW shape {GW_entry.shape} != ({n_rois}, {n_rois}, 11)"
    assert np.all(GW_entry[np.eye(n_rois, dtype=bool)] == 0.0), \
        "FAIL: diagonal contains non-zero values"
    assert np.all(GW_entry >= 0.0), \
        f"FAIL: GW has negative values"
    assert np.allclose(GW_entry, GW_biv), \
        "FAIL: entry-point output differs from compute_gc_spectrum_bivariate"

    # ---------------------------------------------------------------
    # 4.3 — NaN-channel handling check (simulates R11/R2 padding)
    # ---------------------------------------------------------------
    print(f"\n[8/8] NaN-channel handling (simulates R11/R2 — pad LM1+SMA with NaN)...")
    Y_nan = Y_window.copy().astype(float)
    Y_nan[3, :] = np.nan   # LM1 → NaN (as in R11)
    Y_nan[4, :] = np.nan   # SMA → NaN (as in R11)
    GW_nan = compute_pairwise_gc(
        Y_nan, order=var_order, freq=freq, sfreq=sfreq_val
    )

    # Pairs involving NaN ROIs (3=LM1, 4=SMA) must be all-zero
    nan_rois = [3, 4]
    non_nan_rois = [0, 1, 2]
    nan_rows_zero = np.all(GW_nan[nan_rois, :, :] == 0.0)
    nan_cols_zero = np.all(GW_nan[:, nan_rois, :] == 0.0)
    valid_nonzero = np.any(GW_nan[np.ix_(non_nan_rois, non_nan_rois)] > 0)
    print(f"  GW_nan[LM1+SMA rows] all-zero  : {nan_rows_zero}")
    print(f"  GW_nan[LM1+SMA cols] all-zero  : {nan_cols_zero}")
    print(f"  GW_nan[LPFC/CPFC/RPFC] has GC  : {valid_nonzero}")
    print(f"  LPFC→CPFC mean (valid pair)    : {GW_nan[0,1,:].mean():.4f}")

    assert nan_rows_zero, "FAIL: NaN ROI rows contain non-zero GC"
    assert nan_cols_zero, "FAIL: NaN ROI cols contain non-zero GC"
    assert valid_nonzero, "FAIL: valid ROI pairs all-zero even without NaN channels"

    print("\n✓ All 4.3 assertions passed.")

    # ===================================================================
    # Section 4.4 — Sliding Window Loop Validation
    # ===================================================================
    print("\n" + "="*65)
    print("Section 4.4 — compute_windowed_gc Validation")
    print("="*65)

    window_cfg = gc_cfg["window"]
    win_size   = window_cfg["size_samples"]    # 1350
    win_stride = window_cfg["stride_samples"]  # 1350

    # ---------------------------------------------------------------
    # Trial 1: expected 2 windows (2901 samples / 1350 stride)
    # ---------------------------------------------------------------
    print(f"\n[9/10] Trial 1 — {Y_trial.shape[1]} samples "
          f"(window={win_size}, stride={win_stride})...")
    windows_t1 = compute_windowed_gc(
        Y_trial,
        window_size=win_size,
        stride=win_stride,
        order=var_order,
        freq=freq,
        sfreq=sfreq_val,
    )

    # Expected window count: number of valid w in range(win_size, T, stride)
    T_t1 = Y_trial.shape[1]
    expected_wins_t1 = len(range(win_size, T_t1, win_stride))
    print(f"\n  Windows found      : {len(windows_t1)}  (expected: {expected_wins_t1})")
    for label, GW_w in windows_t1.items():
        diag_ok   = np.all(GW_w[np.eye(n_rois, dtype=bool)] == 0.0)
        values_ok = np.all(GW_w >= 0.0)
        print(f"  {label}: shape={GW_w.shape}  diag=0:{diag_ok}  "
              f"≥0:{values_ok}  max={GW_w.max():.4f}  mean_off={GW_w[~np.eye(n_rois,dtype=bool)].mean():.4f}")

    assert len(windows_t1) == expected_wins_t1, \
        f"FAIL: {len(windows_t1)} windows != expected {expected_wins_t1}"
    for label, GW_w in windows_t1.items():
        assert GW_w.shape == (n_rois, n_rois, len(freq)), \
            f"FAIL: {label} shape {GW_w.shape}"
        assert np.all(GW_w >= 0.0), f"FAIL: {label} has negative GC"
        assert np.all(GW_w[np.eye(n_rois, dtype=bool)] == 0.0), \
            f"FAIL: {label} diagonal non-zero"

    # ---------------------------------------------------------------
    # All 5 trials — check window counts match expected totals
    # ---------------------------------------------------------------
    print(f"\n[10/10] All 5 trials — sliding window counts...")
    total_wins = 0
    for trial_idx, (on, off) in enumerate(segments):
        Y_tr = Data_Reg[:, on : off + 1]
        wins_tr = compute_windowed_gc(
            Y_tr,
            window_size=win_size,
            stride=win_stride,
            order=var_order,
            freq=freq,
            sfreq=sfreq_val,
        )
        n_w = len(wins_tr)
        total_wins += n_w
        dur_s = Y_tr.shape[1] / sfreq_val
        expected_w = len(range(win_size, Y_tr.shape[1], win_stride))
        ok = "✓" if n_w == expected_w else "✗"
        print(f"  Trial {trial_idx+1}: {dur_s:.0f} s  →  {n_w}/{expected_w} windows {ok}")

    print(f"\n  Total windows (all trials): {total_wins}  (expected: 8 for A1)")
    assert total_wins == 8, f"FAIL: expected 8 total windows for A1, got {total_wins}"

    print("\n✓ All 4.4 assertions passed.")

    # ===================================================================
    # Section 4.5 — Frequency Band Averaging Validation
    # ===================================================================
    print("\n" + "="*65)
    print("Section 4.5 — average_gc_over_band Validation")
    print("="*65)

    band_cfg = gc_cfg["band_avg"]
    f_low  = band_cfg["f_low"]    # 0.01 Hz
    f_high = band_cfg["f_high"]   # 0.07 Hz

    # ---------------------------------------------------------------
    # 4.5.1 — Band-average Window 1 of Trial 1
    # ---------------------------------------------------------------
    GW_win1 = windows_t1["Win_1"]
    print(f"\n[11/11] Band-averaging Win_1  [{f_low}–{f_high} Hz]...")
    GC_mean = average_gc_over_band(GW_win1, freq, f_low=f_low, f_high=f_high)

    # Index check — confirm same bins as MATLAB idx1:idx2
    idx1 = int(np.argmin(np.abs(freq - f_low)))
    idx2 = int(np.argmin(np.abs(freq - f_high)))
    n_bins = idx2 - idx1 + 1
    manual_mean = np.mean(GW_win1[:, :, idx1 : idx2 + 1], axis=2)

    print(f"\n  Output shape       : {GC_mean.shape}  (expected: ({n_rois}, {n_rois}))")
    print(f"  Band idx           : [{idx1}, {idx2}]  ({n_bins} bins: "
          f"{freq[idx1]:.2f}–{freq[idx2]:.2f} Hz)")
    print(f"  Matches manual mean: {np.allclose(GC_mean, manual_mean)}")
    print(f"  Diagonal all-zero  : {np.all(GC_mean[np.eye(n_rois, dtype=bool)] == 0.0)}")
    print(f"  All values ≥ 0     : {np.all(GC_mean >= 0.0)}")
    print(f"  GC_mean min        : {GC_mean.min():.4f}")
    print(f"  GC_mean max        : {GC_mean.max():.4f}")

    # ---------------------------------------------------------------
    # 4.5.2 — Print all 20 directed connections
    # ---------------------------------------------------------------
    print(f"\n  Band-averaged GC ({f_low}–{f_high} Hz) — all 20 directed connections:")
    for s in range(n_rois):
        for t in range(n_rois):
            if s == t:
                continue
            print(f"    {valid_roi_names[s]:4s}→{valid_roi_names[t]:4s}  :  {GC_mean[s,t]:.4f}")

    # ---------------------------------------------------------------
    # 4.5.2 — Apply to all 8 windows across all 5 trials
    # ---------------------------------------------------------------
    print(f"\n  Band-averaged GC across all trials/windows (max per window):")
    for trial_idx, (on, off) in enumerate(segments):
        Y_tr = Data_Reg[:, on : off + 1]
        wins_tr = compute_windowed_gc(
            Y_tr, window_size=win_size, stride=win_stride,
            order=var_order, freq=freq, sfreq=sfreq_val,
        )
        for label, GW_w in wins_tr.items():
            gc_band = average_gc_over_band(GW_w, freq, f_low=f_low, f_high=f_high)
            off_diag = gc_band[~np.eye(n_rois, dtype=bool)]
            print(f"    Trial {trial_idx+1} {label}: "
                  f"max={gc_band.max():.4f}  mean={off_diag.mean():.4f}  "
                  f"min_off={off_diag.min():.4f}")

    # 4.5 assertions
    assert GC_mean.shape == (n_rois, n_rois), \
        f"FAIL: shape {GC_mean.shape} != ({n_rois}, {n_rois})"
    assert np.allclose(GC_mean, manual_mean), \
        "FAIL: result differs from manual slice-mean"
    assert np.all(GC_mean[np.eye(n_rois, dtype=bool)] == 0.0), \
        "FAIL: diagonal non-zero"
    assert np.all(GC_mean >= 0.0), \
        f"FAIL: negative values (min={GC_mean.min()})"
    assert n_bins == 7, \
        f"FAIL: expected 7 bins (0.01–0.07 Hz), got {n_bins}"

    print("\n✓ All 4.5 assertions passed.")
    print("✓ Section 4 (4.1–4.5) validation complete.\n")
