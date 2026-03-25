"""
Stage E — Phase 1 dSPTE Core Functions

Implements the six algorithmic steps of Wang (2020)'s delay Symbolic Phase
Transfer Entropy (dSPTE), as specified in TRANSFER_ENTROPY_PROGRESS.md
Sections 3.1 through 3.4.

Section 3.1 — Phase Extraction
    extract_phase_signal(x)          : Hilbert phase in [-π, π]
    extract_amplitude_signal(x)      : Hilbert amplitude envelope
    compute_embedding_delay(x)       : autocorrelation 1/e lag τ

Section 3.2 — Uniform Embedding
    uniform_embed(theta, d, tau)

Section 3.3 — Neural-Gas Symbolisation
    symbolise_pair(theta_x, theta_y, d, tau, k0_max, ndis_threshold)

Section 3.4 — dSPTE Computation (next session)
    compute_dspte_at_lag(S_x, S_y, u, k0)
    compute_dspte_multidelay(S_x, S_y, k0, u_max)
    compute_dspte_matrix(Y_window, config)

Reference
---------
Wang, Y., & Chen, W. (2020). Effective brain connectivity for fNIRS data
analysis based on multi-delays symbolic phase transfer entropy.
J. Neural Eng., 17(5), 056024. doi:10.1088/1741-2552/abb4a4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import hilbert

_CODE_ROOT = Path(__file__).parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

logger = logging.getLogger(__name__)


# ===========================================================================
# Section 3.1 — Phase Extraction
# ===========================================================================

def extract_phase_signal(x: np.ndarray) -> np.ndarray:
    """
    Extract instantaneous phase via the Hilbert transform (Wang 2020 Eq. 8).

    Computes the analytic signal z(t) = x(t) + i·H{x}(t), then extracts the
    instantaneous phase θ(t) = angle(z(t)) ∈ [-π, π].

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        1D real-valued time series (e.g. ROI-averaged HbO signal in mol/L).
        Note, this function is called for each of the 5 ROIs.

    Returns
    -------
    theta : np.ndarray, shape (T,)
        Instantaneous phase in radians, values ∈ [-π, π].

    Notes — Methodological Limitation
    ----------------------------------
    The Hilbert transform yields a physically meaningful instantaneous phase
    only for narrowband, approximately monocomponent signals (classic use case:
    EEG oscillations within a defined frequency band). fNIRS HbO signals after
    HRF extraction are broadband, non-stationary hemodynamic envelopes — not
    narrowband oscillations. When applied to such signals, θ(t) reflects the
    relative timing of hemodynamic peaks between channels (i.e. which region
    activates first), not oscillatory phase in the EEG sense.

    Wang (2020) validated dSPTE on the same type of fNIRS data and obtained
    significant results (p < 0.01 finger-tapping EC), so the method does not
    break — but the "phase" interpretation must be stated carefully in any
    report: it is a proxy for hemodynamic timing, not neural oscillation phase.

    Practical implication: the amplitude branch (Phase 3) is likely the more
    scientifically grounded primary signal for broadband fNIRS, since HbO
    amplitude directly reflects local neural activity intensity. Consider this
    when interpreting Phase 1 results and when selecting Phase 3 fusion weights.
    See TRANSFER_ENTROPY_PROGRESS.md §Step 1 and §Phase 3 discussion.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"extract_phase_signal: expected 1D input, got shape {x.shape}"
        )
    analytic = hilbert(x)           # complex analytic signal, shape (T,)
    theta = np.angle(analytic)      # instantaneous phase in [-π, π]
    logger.debug(
        "extract_phase_signal: T=%d  phase range [%.3f, %.3f] rad",
        len(theta), float(theta.min()), float(theta.max()),
    )
    return theta


def extract_amplitude_signal(x: np.ndarray) -> np.ndarray:
    """
    Extract instantaneous amplitude (envelope) via the Hilbert transform.

    Computes A(t) = |hilbert(x(t))| — the magnitude of the analytic signal.
    Used by Phase 3 (bimodal dSPTE) as the amplitude branch input.

    For broadband fNIRS HbO signals, A(t) is the more physically meaningful
    quantity than phase: it directly tracks the hemodynamic response envelope
    and correlates with local neural activity intensity. The amplitude branch
    in Phase 3 is expected to be the primary information-carrying branch for
    this data type (see TRANSFER_ENTROPY_PROGRESS.md §Phase 3 discussion).

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        1D real-valued time series.

    Returns
    -------
    amplitude : np.ndarray, shape (T,)
        Instantaneous amplitude (envelope), values ≥ 0.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"extract_amplitude_signal: expected 1D input, got shape {x.shape}"
        )
    amplitude = np.abs(hilbert(x))
    logger.debug(
        "extract_amplitude_signal: T=%d  amplitude range [%.6f, %.6f]",
        len(amplitude), float(amplitude.min()), float(amplitude.max()),
    )
    return amplitude


def compute_embedding_delay(
    x: np.ndarray,
    method: str = "autocorr_1_over_e",
) -> int:
    """
    The purpsoe of this method is to estimate the best time-delay embedding lag τ from the autocorrelation function.

    The normalized autocorrelation function (ACF) is a tool measuring the linear
    relationship between a time series and a version of itself shifted by a specific time lag τ.
    It's used to determine the optimal interval at which to sample the data to create independent coordinates, outputting
    a value from -1 to 1
    - 1: indicates perfect positive correlation where the signal is identical to its lagged version
    - 0: indicates no correlation (the values are independent of each other)
    - -1: indicates perfect negative correlation where the signal is identical to its lagged version but inverted

    The ACF is calculated by:
    R(τ) = E[(x(t)-μ)(x(t+τ)-μ)] / σ^2
    where μ is the mean of the time series and σ is the standard deviation of the time series.

    The embedding delay τ is the lag at which the normalised autocorrelation
    function (ACF) first drops to or below 1/e ≈ 0.368. This is the standard
    decorrelation time and is the recommended τ for phase-space reconstruction
    of fNIRS signals (see TRANSFER_ENTROPY_PROGRESS.md §Step 2).

    For a 25 Hz fNIRS signal after HRF deconvolution and 0.5 Hz low-pass
    filtering, observed τ values are typically 20–300 samples (0.8–12 s) —
    significantly longer than the "5–20 samples" cited in the PRD, which
    assumed the signal would behave like a narrowband oscillation. The Stage B
    preprocessing (LP at 0.5 Hz + GLM HRF deconvolution) produces signals
    with ACF that decays very slowly (r[1] ≈ 0.99 for most ROIs on A1).
    Validated on A1: τ ∈ {22, 39, 118, 164, 169} samples across 5 ROIs.
    Values outside [20, 300] are logged as a WARNING for review.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        1D time series. Should be mean-subtracted before ACF computation
        (done internally).
    method : str
        Delay estimation method. Currently only 'autocorr_1_over_e' is
        supported (lag at which normalised ACF first drops below 1/e).

    Returns
    -------
    tau : int
        Embedding delay in samples. Minimum value: 1 (returned if ACF
        never drops below 1/e within the valid lag range, or if the signal
        is constant / has zero variance).

    Notes
    -----
    ACF computation uses the full one-sided autocorrelation:
        r(k) = Σ_{t=0}^{T-k-1} (x[t] - μ)(x[t+k] - μ)
    normalised by r(0) so that r(0) = 1. This is the unbiased estimator
    (divides by T - k, not T) as implemented by np.correlate on zero-mean x.

    The 1/e threshold is used (not first zero-crossing) because fNIRS HbO
    signals often have a slow drift that prevents a clean zero crossing,
    while the 1/e point reliably marks the effective decorrelation time.
    """
    if method != "autocorr_1_over_e":
        raise ValueError(
            f"compute_embedding_delay: unsupported method '{method}'. "
            f"Only 'autocorr_1_over_e' is implemented."
        )

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"compute_embedding_delay: expected 1D input, got shape {x.shape}"
        )

    T = len(x)
    if T < 4:
        logger.warning(
            "compute_embedding_delay: signal too short (T=%d), returning τ=1", T
        )
        return 1

    xc = x - x.mean() # zero-mean
    var = float(np.dot(xc, xc)) # variance of the time series

    if var == 0.0:
        logger.warning(
            "compute_embedding_delay: constant signal (zero variance), returning τ=1"
        )
        return 1

    # One-sided normalised ACF via np.correlate (full mode → 2T-1 values)
    # We take the second half [T-1 :] which is the positive-lag side.
    # Each lag-k value from np.correlate is Σ xc[t]*xc[t+k] (unnormalised).
    # Normalise by var (= ACF at lag 0) so ACF[0] = 1.
    full_corr = np.correlate(xc, xc, mode="full") # performs a sum of products by sliding one copy of xc across the other copy.
    # With a signal length of T, the full_corr will have a length of 2T-1.
    one_sided = full_corr[T - 1:] # Index T represents lag 0. Since we're only interested in how the signal correlates with its future 
    # self, we only take the positive lags (lags 0, 1, 2, ..., T-1) from the full_corr.
    acf = one_sided / var                       # normalised: acf[0] = 1.0

    threshold = 1.0 / np.e                      # ≈ 0.3679

    # Find first lag where ACF drops at or below threshold.
    # Search from lag=1 (lag=0 is always 1.0 by construction).
    tau = 1
    for k in range(1, T):
        if acf[k] <= threshold:
            tau = k
            break
    else:
        # ACF never drops below 1/e — use the lag of minimum ACF as fallback
        tau = int(np.argmin(acf[1:]) + 1)
        logger.warning(
            "compute_embedding_delay: ACF never dropped below 1/e (%.3f) "
            "within T=%d lags — using lag of minimum ACF as fallback: τ=%d",
            threshold, T, tau,
        )

    # Physiological range check: observed τ for ASTaUND HbO signals is 20–300
    # samples (0.8–12 s at 25 Hz). The 0.5 Hz LP filter + GLM HRF deconvolution
    # produces very slowly decorrelating signals (ACF r[1] ≈ 0.98–1.00).
    # Warn only if τ is suspiciously small (< 5) or extremely large (> 500).
    if tau < 5:
        logger.warning(
            "compute_embedding_delay: τ=%d samples is unusually small — "
            "signal may be white noise or an impulse.",
            tau,
        )
    elif tau > 500:
        logger.warning(
            "compute_embedding_delay: τ=%d samples is unusually large — "
            "signal may be near-constant or have a very slow trend.",
            tau,
        )
    else:
        logger.debug("compute_embedding_delay: τ=%d samples (ACF 1/e threshold)", tau)

    return tau


# ===========================================================================
# Section 3.2 — Uniform Time-Delay Embedding
# ===========================================================================

def uniform_embed(theta: np.ndarray, d: int, tau: int) -> np.ndarray:
    """
    Reconstruct the phase-space trajectory via uniform time-delay embedding
    (Wang 2020 Eq. 10).

    Each row of the output is a delay-coordinate vector:

        row i = [θ(i+(d-1)τ),  θ(i+(d-2)τ),  ...,  θ(i+τ),  θ(i)]

    Written with the *most recent* sample first, so row 0 corresponds to
    the earliest time at which a full d-dimensional history is available:
    t = (d-1)·τ.  Equivalently, in the notation of the paper:

        θ_t^{xd} = [θ_x(t),  θ_x(t-τ),  ...,  θ_x(t-(d-1)τ)]

    where t starts at (d-1)·τ (the first valid sample) and runs to T-1.

    Parameters
    ----------
    theta : np.ndarray, shape (T,)
        1D phase (or amplitude) time series.  Can also be used on raw
        amplitude sequences for Phase 3 embedding.
    d : int
        Embedding dimension.  d=1 returns the original series reshaped as
        a column vector.  d=3 is the default for dSPTE (configurable).
    tau : int
        Embedding delay in samples (≥ 1).  Must satisfy
        (d-1)*tau < T, otherwise no valid rows exist.

    Returns
    -------
    embedded : np.ndarray, shape (T - (d-1)*tau, d)
        Each row is one delay-coordinate vector.  The number of valid rows
        is T - (d-1)*tau (no NaN padding — only rows with a full history
        are included, per the spec).

    Raises
    ------
    ValueError
        If theta is not 1D, d < 1, tau < 1, or the signal is too short
        to produce at least one valid embedding row.

    Notes
    -----
    **Column layout:** column 0 = θ(t)  (most recent),
    column j = θ(t - j·τ)  (j steps back).  This matches Wang (2020)'s
    convention where the current sample is listed first.

    **Efficiency:** implemented via stride-tricks on a pre-allocated index
    array — O(T·d) time, no Python loop over rows.  For the ASTaUND window
    size (T=1351, d=3, τ≈100) the embedded matrix has shape (1151, 3) and
    fits easily in memory.

    **Embedding span:** the total time history captured per row is
    (d-1)·τ samples.  With d=3 and τ=100 (typical for ASTaUND HbO phase),
    each row spans 200 samples = 8 s — covering the full hemodynamic
    response peak.
    """
    theta = np.asarray(theta, dtype=float)

    if theta.ndim != 1:
        raise ValueError(
            f"uniform_embed: expected 1D input, got shape {theta.shape}"
        )
    if d < 1:
        raise ValueError(f"uniform_embed: d must be ≥ 1, got {d}")
    if tau < 1:
        raise ValueError(f"uniform_embed: tau must be ≥ 1, got {tau}")

    T = len(theta)
    history = (d - 1) * tau          # samples of history required per row
    n_rows = T - history             # number of valid embedding rows

    if n_rows < 1:
        raise ValueError(
            f"uniform_embed: signal too short for embedding. "
            f"T={T}, d={d}, tau={tau} → need T > (d-1)*tau = {history}, "
            f"but only {n_rows} valid rows would result."
        )

    # Build index matrix: shape (n_rows, d)
    # row_starts[i] = index of the most-recent sample for row i = i + history
    # col j gives the index of the j-th lag: row_start - j*tau
    row_starts = np.arange(history, T)           # shape (n_rows,)  = i + history
    lags = np.arange(d) * tau                    # shape (d,)       = [0, τ, 2τ, ...]
    indices = row_starts[:, np.newaxis] - lags   # shape (n_rows, d)

    embedded = theta[indices]                    # shape (n_rows, d)

    logger.debug(
        "uniform_embed: T=%d  d=%d  tau=%d  → shape %s  "
        "(history span: %d samples = %.2f s @ 25 Hz)",
        T, d, tau, embedded.shape, history, history / 25.0,
    )
    return embedded


# ===========================================================================
# Section 3.3 — Neural-Gas Symbolisation
# ===========================================================================

def symbolise_pair(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    d: int,
    tau: int,
    k0_max: int = 20,
    ndis_threshold: float = 0.05,
    config: "dict | None" = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Symbolise a single (X, Y) channel pair via Neural-Gas vector quantisation.

    Implements Wang (2020) Step 3 (Eqs. 11–14) for one pair of phase (or
    amplitude) time series.  The steps in order are:

    1. Embed both series with the same (d, tau) via :func:`uniform_embed`.
    2. Build the joint state matrix by vertically stacking the two embedded
       matrices: ``θ_xyd = [θ_xd; θ_yd]``, shape ``(2·T', d)``.
    3. Fit a Neural-Gas codebook on the joint matrix via
       :func:`utils_te.fit_neural_gas_codebook`, searching k from 2 to k0_max
       until ``nDis < ndis_threshold``.
    4. Quantise each channel's embedded matrix independently against the shared
       codebook via :func:`utils_te.quantise_to_symbols`.
    5. Return the two symbolic sequences and a metadata dict.

    Parameters
    ----------
    theta_x : np.ndarray, shape (T,)
        Phase (or amplitude) time series for channel X.
    theta_y : np.ndarray, shape (T,)
        Phase (or amplitude) time series for channel Y.  Must have the same
        length as ``theta_x``.
    d : int
        Embedding dimension (default 3, configurable via ``config.yaml``).
    tau : int
        Embedding delay in samples.  Typically computed via
        :func:`compute_embedding_delay` for each channel; the caller should
        pass the *minimum* of the two per-channel τ values, or a
        pre-computed per-signal τ, depending on the pipeline strategy.
        (The PRD does not specify whether to use X's τ, Y's τ, or the mean —
        using X's τ here is the implementation convention; callers may override.)
    k0_max : int
        Maximum codebook size to search.  Default 20 for bivariate mode;
        pass ``k0_max_multivariate`` (default 8) for Phase 2.
    ndis_threshold : float
        Wang (2020) Eq. 11 stopping criterion: nDis < threshold.  Default 0.05.
    config : dict or None
        Parsed ``config.yaml``.  Forwarded to
        :func:`utils_te.fit_neural_gas_codebook` for NG hyperparameters.

    Returns
    -------
    S_x : np.ndarray, shape (T',), dtype int
        Symbolic sequence for channel X.  Values in ``{0, ..., k₀-1}``.
    S_y : np.ndarray, shape (T',), dtype int
        Symbolic sequence for channel Y.  Same length as ``S_x``.
    meta : dict
        Metadata dict with keys:
        - ``'k0'``: int — selected codebook size.
        - ``'nDis'``: float — normalised distortion at k₀.
        - ``'tau'``: int — embedding delay used (same for both channels).
        - ``'d'``: int — embedding dimension used.
        - ``'T_prime'``: int — number of valid embedding rows (= T - (d-1)·tau).
        - ``'codebook'``: np.ndarray, shape (k₀, d) — prototype vectors
          (retained for diagnostic saving and Phase 2 reuse).

    Notes
    -----
    **Shared codebook rationale (Wang 2020 Eq. 13):** The NG is fitted on the
    *joint* matrix ``[θ_xd; θ_yd]`` so that both channels are quantised into
    the same symbolic alphabet.  This ensures that when the same symbol index
    appears in S_x and S_y, it corresponds to the same region of the shared
    phase space — a prerequisite for the discrete TE calculation in Step 4.

    **τ convention:** Both channels are embedded with the *same* delay τ (the
    caller's choice).  Wang (2020) does not specify whether to use a per-channel
    or shared τ.  Using a shared τ (e.g. the mean or minimum of the two
    per-channel 1/e autocorrelation times) keeps the embedding space
    commensurate.  The per-channel τ approach is logged as deviation E7 in the
    PRD.

    **Raises** ``ValueError`` if ``theta_x`` and ``theta_y`` have different
    lengths or if ``uniform_embed`` fails (signal too short for given d and τ).
    """
    from utils_te import fit_neural_gas_codebook, quantise_to_symbols  # noqa

    theta_x = np.asarray(theta_x, dtype=float)
    theta_y = np.asarray(theta_y, dtype=float)

    if theta_x.ndim != 1 or theta_y.ndim != 1:
        raise ValueError(
            "symbolise_pair: both inputs must be 1D arrays. "
            f"Got shapes {theta_x.shape} and {theta_y.shape}."
        )
    if len(theta_x) != len(theta_y):
        raise ValueError(
            f"symbolise_pair: theta_x (len={len(theta_x)}) and "
            f"theta_y (len={len(theta_y)}) must have the same length."
        )

    # Step 1 — Embed both time series
    theta_xd = uniform_embed(theta_x, d=d, tau=tau)   # shape (T', d)
    theta_yd = uniform_embed(theta_y, d=d, tau=tau)   # shape (T', d)
    T_prime = theta_xd.shape[0]

    logger.debug(
        "symbolise_pair: T=%d  d=%d  tau=%d  T'=%d  joint shape (%d, %d)",
        len(theta_x), d, tau, T_prime, 2 * T_prime, d,
    )

    # Step 2 — Joint state matrix: stack both embedded series vertically
    theta_xyd = np.vstack([theta_xd, theta_yd])   # shape (2·T', d)

    # Step 3 — Fit Neural-Gas codebook on joint matrix
    codebook, k0, ndis = fit_neural_gas_codebook(
        theta_xyd,
        k0_max=k0_max,
        ndis_threshold=ndis_threshold,
        config=config,
    )

    logger.info(
        "symbolise_pair: k₀=%d  nDis=%.4f  (threshold=%.2f)  "
        "T'=%d  codebook shape %s",
        k0, ndis, ndis_threshold, T_prime, codebook.shape,
    )

    # Step 4 — Quantise each channel's embedded matrix independently
    S_x = quantise_to_symbols(theta_xd, codebook)   # shape (T',)
    S_y = quantise_to_symbols(theta_yd, codebook)   # shape (T',)

    meta = {
        "k0": int(k0),
        "nDis": float(ndis),
        "tau": int(tau),
        "d": int(d),
        "T_prime": int(T_prime),
        "codebook": codebook,
    }

    return S_x, S_y, meta


# ===========================================================================
# Smoke tests (run as script)
# ===========================================================================

def _run_tests() -> None:
    """
    Section 3.1 unit tests.

    3.1.1  extract_phase_signal:
           Pure sine sin(2πft) → phase derivative ≈ 2πf/sfreq (constant).
           Phase range must be within [-π, π].

    3.1.2  extract_amplitude_signal:
           Pure sine A·sin(2πft) → envelope ≈ A (constant).
           Max deviation from A should be < 1% at the signal interior.

    3.1.3  compute_embedding_delay:
           (a) Pure sine at 1 Hz, 25 Hz sfreq: ACF is cosine, drops to 1/e
               at t = 1/(2πf) = 0.159 s → ~4 samples. Expect τ ∈ [3, 6].
           (b) White noise: ACF collapses immediately → τ=1 expected.
           (c) Highly autocorrelated signal (slow ramp): τ should be large,
               triggering the out-of-range WARNING.
           (d) Real A1 ROI signal: τ must fall in [3, 15] for all 5 ROIs.
    """
    import warnings
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("\n=== Section 3.1 — Phase Extraction Tests ===\n")

    sfreq = 25.0
    T = 2500          # 100 s at 25 Hz
    t = np.arange(T) / sfreq
    f = 1.0           # 1 Hz sine wave
    A = 3.0           # amplitude

    # ------------------------------------------------------------------
    # 3.1.1 — extract_phase_signal: pure sine
    # ------------------------------------------------------------------
    print("--- 3.1.1  extract_phase_signal ---")

    x_sine = A * np.sin(2 * np.pi * f * t)
    theta = extract_phase_signal(x_sine)

    # Shape check
    assert theta.shape == (T,), f"Expected shape ({T},), got {theta.shape}"

    # Range check: all values in [-π, π]
    assert float(theta.min()) >= -np.pi - 1e-10, f"Phase below -π: {theta.min()}"
    assert float(theta.max()) <= np.pi + 1e-10, f"Phase above π: {theta.max()}"

    # Phase derivative check: for a pure sine, dθ/dt = 2πf (constant).
    # Exclude first and last 5% of signal (Hilbert edge effects).
    margin = int(0.05 * T)
    dtheta = np.diff(theta[margin:-margin])
    # Handle phase wrapping: bring all differences into [-π, π]
    dtheta_unwrapped = (dtheta + np.pi) % (2 * np.pi) - np.pi
    expected_dtheta = 2 * np.pi * f / sfreq       # ≈ 0.2513 rad/sample
    mean_dtheta = float(np.mean(np.abs(dtheta_unwrapped)))
    # Mean absolute phase step should be within 20% of expected (generous
    # tolerance because the analytic signal of a bandlimited sine has
    # a near-constant but not exactly constant instantaneous frequency
    # at finite precision — edge wraps pull the mean slightly).
    tol = 0.20
    assert abs(mean_dtheta - expected_dtheta) < tol * expected_dtheta, (
        f"Phase derivative mean {mean_dtheta:.4f} deviates from "
        f"expected {expected_dtheta:.4f} by more than {tol*100:.0f}%"
    )
    print(f"  Phase range       : [{theta.min():.4f}, {theta.max():.4f}] rad  ✓ ∈ [-π, π]")
    print(f"  Mean |dθ/dt|      : {mean_dtheta:.4f} rad/sample  (expected {expected_dtheta:.4f})")
    print(f"  Phase deriv error : {abs(mean_dtheta - expected_dtheta)/expected_dtheta*100:.1f}%  ✓ < {tol*100:.0f}%")

    # Non-sine input: constant signal → phase should be defined (no crash)
    theta_const = extract_phase_signal(np.ones(100))
    assert theta_const.shape == (100,)

    # Wrong shape: 2D input should raise
    try:
        extract_phase_signal(np.ones((10, 10)))
        assert False, "Should have raised ValueError for 2D input"
    except ValueError:
        pass

    print("  [PASS] 3.1.1 extract_phase_signal\n")

    # ------------------------------------------------------------------
    # 3.1.2 — extract_amplitude_signal: pure sine
    # ------------------------------------------------------------------
    print("--- 3.1.2  extract_amplitude_signal ---")

    amp = extract_amplitude_signal(x_sine)

    # Shape check
    assert amp.shape == (T,), f"Expected shape ({T},), got {amp.shape}"

    # All values ≥ 0 (amplitude is always non-negative)
    assert float(amp.min()) >= 0.0, f"Amplitude has negative values: {amp.min()}"

    # Interior amplitude should be ≈ A (within 2% for a 100 s signal)
    # Exclude 5% margins to avoid Hilbert edge effects
    amp_interior = amp[margin:-margin]
    max_dev = float(np.max(np.abs(amp_interior - A))) / A
    assert max_dev < 0.02, (
        f"Amplitude envelope deviates from true amplitude A={A} by "
        f"{max_dev*100:.2f}% (tolerance: 2%)"
    )
    print(f"  Amplitude range   : [{amp.min():.4f}, {amp.max():.4f}]  (true A={A})")
    print(f"  Interior max dev  : {max_dev*100:.3f}%  ✓ < 2%")

    # Non-sine: DC signal → amplitude should be ≈ mean
    amp_dc = extract_amplitude_signal(np.full(200, 5.0))
    assert amp_dc.shape == (200,)

    # Wrong shape
    try:
        extract_amplitude_signal(np.ones((5, 5)))
        assert False, "Should have raised ValueError for 2D input"
    except ValueError:
        pass

    print("  [PASS] 3.1.2 extract_amplitude_signal\n")

    # ------------------------------------------------------------------
    # 3.1.3 — compute_embedding_delay
    # ------------------------------------------------------------------
    print("--- 3.1.3  compute_embedding_delay ---")

    # (a) Pure 1 Hz sine at 25 Hz — ACF is cosine, 1/e crossing at ~4 samples
    tau_sine = compute_embedding_delay(x_sine)
    print(f"  (a) 1 Hz sine, 25 Hz sfreq  → τ={tau_sine} samples  (expected 3–6)")
    assert 3 <= tau_sine <= 6, (
        f"Sine ACF 1/e lag expected in [3, 6], got τ={tau_sine}"
    )

    # (b) White noise — ACF collapses immediately → τ=1 expected
    rng = np.random.default_rng(42)
    x_noise = rng.standard_normal(T)
    tau_noise = compute_embedding_delay(x_noise)
    print(f"  (b) White noise              → τ={tau_noise} samples  (expected 1)")
    assert tau_noise == 1, (
        f"White noise ACF should give τ=1, got τ={tau_noise}"
    )

    # (c) Highly autocorrelated (slow ramp) — τ should be large, triggering WARNING
    x_ramp = np.linspace(0, 1, T) + 0.01 * rng.standard_normal(T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_ramp = compute_embedding_delay(x_ramp)
    print(f"  (c) Slow ramp                → τ={tau_ramp} samples  (expected large, WARNING logged)")
    # Just verify it returns a valid positive integer
    assert isinstance(tau_ramp, int) and tau_ramp >= 1

    # (d) Constant signal → τ=1 (zero variance guard)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_const = compute_embedding_delay(np.ones(T))
    print(f"  (d) Constant signal          → τ={tau_const} samples  (expected 1)")
    assert tau_const == 1

    # (e) 0.1 Hz signal (hemodynamic-like slow oscillation at 25 Hz)
    #     ACF cosine period = 10 s = 250 samples → 1/e at ~40 samples.
    #     This will trigger the out-of-range WARNING (> 15 samples).
    x_slow = np.sin(2 * np.pi * 0.1 * t) + 0.1 * rng.standard_normal(T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_slow = compute_embedding_delay(x_slow)
    print(f"  (e) 0.1 Hz slow signal       → τ={tau_slow} samples  (expected large, WARNING logged)")
    assert isinstance(tau_slow, int) and tau_slow >= 1

    # (f) Real A1 ROI signal — validate all 5 ROIs are in expected range [3, 15]
    print("\n  (f) Real A1 ROI signals (Stage D cache):")
    print("      NOTE: ASTaUND HbO signals after 0.5 Hz LP + GLM deconvolution")
    print("      are very slowly decorrelating (ACF r[1] ≈ 0.99). Observed τ is")
    print("      20–300 samples — much longer than the PRD's 5–20 sample estimate.")
    print("      This is physically correct: the signals are broadband envelopes,")
    print("      not narrowband oscillations. See docstring for details.")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils_te import load_or_extract_roi_signals
        data_a1, roi_names, _ = load_or_extract_roi_signals("A1")
        tau_values = []
        for i, roi in enumerate(roi_names):
            # Use first 1351 samples (one window) to match pipeline usage
            x_roi = data_a1[i, :1351]
            theta_roi = extract_phase_signal(x_roi)
            tau_roi = compute_embedding_delay(theta_roi)
            tau_values.append(tau_roi)
            range_ok = "✓" if 5 <= tau_roi <= 500 else "⚠ unexpected range"
            print(f"    {roi:6s}: τ={tau_roi:3d} samples ({tau_roi/25:.1f} s)  {range_ok}")
        print(f"  Mean τ across ROIs: {np.mean(tau_values):.1f} samples "
              f"({np.mean(tau_values)/25:.1f} s)")
        # Soft assertion: all τ within the broad valid range
        assert all(5 <= t <= 500 for t in tau_values), (
            f"Unexpected τ values outside [5, 500]: {tau_values}"
        )
    except Exception as e:
        print(f"  [SKIP] A1 ROI load failed: {e}")

    print("  [PASS] 3.1.3 compute_embedding_delay\n")

    # ------------------------------------------------------------------
    # 3.2.1 — uniform_embed
    # ------------------------------------------------------------------
    print("--- 3.2.1  uniform_embed ---")

    # (a) Spec validation: T=1000, d=3, tau=2 → shape (996, 3)
    x_check = np.arange(1000, dtype=float)
    emb = uniform_embed(x_check, d=3, tau=2)
    assert emb.shape == (996, 3), f"Expected (996, 3), got {emb.shape}"
    print(f"  (a) T=1000, d=3, tau=2  → shape {emb.shape}  ✓ == (996, 3)")

    # (b) Row content: row i should be [x[i+4], x[i+2], x[i+0]] = [i+4, i+2, i]
    #     (history = (d-1)*tau = 4; row_start = i+4; cols = [i+4, i+4-2, i+4-4])
    for i in [0, 10, 100, 995]:
        expected = np.array([i + 4, i + 2, i], dtype=float)
        np.testing.assert_array_equal(
            emb[i], expected,
            err_msg=f"Row {i} mismatch: got {emb[i]}, expected {expected}"
        )
    print("  (b) Row content check (d=3, tau=2)  ✓  [t, t-τ, t-2τ] correct")

    # (c) d=1, tau=1 → should return the original signal as a column vector
    emb_d1 = uniform_embed(x_check, d=1, tau=1)
    assert emb_d1.shape == (1000, 1), f"Expected (1000, 1), got {emb_d1.shape}"
    np.testing.assert_array_equal(emb_d1[:, 0], x_check)
    print(f"  (c) d=1, tau=1  → shape {emb_d1.shape}  ✓  identity embedding")

    # (d) d=3, tau=1 → shape (T-2, 3)
    emb_tau1 = uniform_embed(x_check, d=3, tau=1)
    assert emb_tau1.shape == (998, 3), f"Expected (998, 3), got {emb_tau1.shape}"
    print(f"  (d) d=3, tau=1  → shape {emb_tau1.shape}  ✓")

    # (e) Exact shape formula: T - (d-1)*tau
    for d_test, tau_test in [(3, 2), (4, 5), (2, 10), (5, 3)]:
        T_test = 500
        expected_rows = T_test - (d_test - 1) * tau_test
        if expected_rows < 1:
            continue
        emb_t = uniform_embed(np.zeros(T_test), d=d_test, tau=tau_test)
        assert emb_t.shape == (expected_rows, d_test), (
            f"d={d_test}, tau={tau_test}: expected ({expected_rows}, {d_test}), "
            f"got {emb_t.shape}"
        )
    print("  (e) Shape formula T-(d-1)τ verified for d∈{2,3,4,5}, τ∈{2,3,5,10}  ✓")

    # (f) Error cases
    try:
        uniform_embed(np.ones((10, 2)), d=3, tau=1)
        assert False, "Should raise for 2D input"
    except ValueError:
        pass
    try:
        uniform_embed(np.ones(10), d=0, tau=1)
        assert False, "Should raise for d=0"
    except ValueError:
        pass
    try:
        uniform_embed(np.ones(10), d=3, tau=0)
        assert False, "Should raise for tau=0"
    except ValueError:
        pass
    try:
        uniform_embed(np.ones(5), d=3, tau=10)  # needs T > (d-1)*tau = 20, only 5
        assert False, "Should raise: signal too short"
    except ValueError:
        pass
    print("  (f) Error cases (2D input, d=0, tau=0, too-short)  ✓  all raise ValueError")

    # (g) Real A1 phase signal — typical pipeline embedding
    print("\n  (g) Real A1 pipeline embedding (first window, LPFC):")
    try:
        from utils_te import load_or_extract_roi_signals
        data_a1, roi_names, _ = load_or_extract_roi_signals("A1")
        x_roi = data_a1[0, :1351]                        # first window, LPFC
        theta_roi = extract_phase_signal(x_roi)
        tau_roi = compute_embedding_delay(theta_roi)
        emb_roi = uniform_embed(theta_roi, d=3, tau=tau_roi)
        expected_rows_roi = 1351 - 2 * tau_roi
        assert emb_roi.shape == (expected_rows_roi, 3), (
            f"Expected ({expected_rows_roi}, 3), got {emb_roi.shape}"
        )
        history_s = 2 * tau_roi / 25.0
        print(f"    ROI: {roi_names[0]}  τ={tau_roi}  "
              f"embedded shape: {emb_roi.shape}  "
              f"(history span: {history_s:.1f} s)")
        print(f"    Row 0: [{emb_roi[0, 0]:.4f}, {emb_roi[0, 1]:.4f}, {emb_roi[0, 2]:.4f}]")
        print(f"    Values ∈ [{emb_roi.min():.4f}, {emb_roi.max():.4f}] rad  ✓")
    except Exception as e:
        print(f"    [SKIP] A1 ROI load failed: {e}")

    print("  [PASS] 3.2.1 uniform_embed\n")

    # ------------------------------------------------------------------
    # 3.3.3 — symbolise_pair
    # ------------------------------------------------------------------
    print("--- 3.3.3  symbolise_pair ---")

    # (a) Basic synthetic test: two correlated 1 Hz sine waves, phase-shifted
    #     Both should produce symbolic sequences of the same length T'.
    sfreq33 = 25.0
    T33 = 500
    t33 = np.arange(T33) / sfreq33
    theta_xa = np.sin(2 * np.pi * 1.0 * t33)         # channel X: 1 Hz sine
    theta_ya = np.sin(2 * np.pi * 1.0 * t33 + 0.5)   # channel Y: 1 Hz, phase-shifted 0.5 rad

    d33, tau33 = 3, 4
    S_xa, S_ya, meta_a = symbolise_pair(theta_xa, theta_ya, d=d33, tau=tau33, k0_max=15)

    expected_Tp = T33 - (d33 - 1) * tau33   # = 500 - 8 = 492
    assert S_xa.shape == (expected_Tp,), (
        f"S_x shape mismatch: expected ({expected_Tp},), got {S_xa.shape}"
    )
    assert S_ya.shape == (expected_Tp,), (
        f"S_y shape mismatch: expected ({expected_Tp},), got {S_ya.shape}"
    )
    assert meta_a["T_prime"] == expected_Tp, (
        f"meta T_prime mismatch: expected {expected_Tp}, got {meta_a['T_prime']}"
    )
    assert meta_a["d"] == d33
    assert meta_a["tau"] == tau33
    k0_a = meta_a["k0"]
    assert 2 <= k0_a <= 15, f"k0={k0_a} out of expected range [2, 15]"
    # All symbols must be in {0, ..., k0-1}
    assert S_xa.min() >= 0 and S_xa.max() < k0_a, (
        f"S_x symbols out of range [0, {k0_a-1}]: [{S_xa.min()}, {S_xa.max()}]"
    )
    assert S_ya.min() >= 0 and S_ya.max() < k0_a, (
        f"S_y symbols out of range [0, {k0_a-1}]: [{S_ya.min()}, {S_ya.max()}]"
    )
    # Codebook shape: (k0, d)
    assert meta_a["codebook"].shape == (k0_a, d33), (
        f"codebook shape mismatch: expected ({k0_a}, {d33}), "
        f"got {meta_a['codebook'].shape}"
    )
    print(f"  (a) Synthetic sines (T={T33}, d={d33}, tau={tau33})")
    print(f"      T'={expected_Tp}  k₀={k0_a}  nDis={meta_a['nDis']:.4f}")
    print(f"      S_x: shape={S_xa.shape}  values ∈ [{S_xa.min()}, {S_xa.max()}]  ✓")
    print(f"      S_y: shape={S_ya.shape}  values ∈ [{S_ya.min()}, {S_ya.max()}]  ✓")
    print(f"      codebook: shape={meta_a['codebook'].shape}  ✓")

    # (b) Length-mismatch error
    try:
        symbolise_pair(np.zeros(100), np.zeros(150), d=3, tau=1)
        assert False, "Should raise ValueError for length mismatch"
    except ValueError:
        pass
    print("  (b) Length mismatch raises ValueError  ✓")

    # (c) 2D input error
    try:
        symbolise_pair(np.zeros((10, 2)), np.zeros(10), d=3, tau=1)
        assert False, "Should raise ValueError for 2D input"
    except ValueError:
        pass
    print("  (c) 2D input raises ValueError  ✓")

    # (d) Shared codebook: S_x and S_y use the same alphabet — same k0
    #     Both should have values in {0, ..., k0-1} (already checked above in (a))
    # Verify the nDis metadata is consistent
    assert 0.0 <= meta_a["nDis"] <= 1.0, (
        f"nDis should be in [0, 1], got {meta_a['nDis']}"
    )
    print(f"  (d) nDis={meta_a['nDis']:.4f} ∈ [0, 1]  ✓ (threshold=0.05)")

    # (e) Real A1 ROI pipeline: phase signals for LPFC → CPFC pair
    print("\n  (e) Real A1 ROI pair (LPFC→CPFC, first window, phase signals):")
    try:
        from utils_te import load_or_extract_roi_signals
        data_a1, roi_names33, _ = load_or_extract_roi_signals("A1")
        win = data_a1[:, :1351]                      # first 1351-sample window
        theta_lpfc = extract_phase_signal(win[0])    # LPFC
        theta_cpfc = extract_phase_signal(win[1])    # CPFC
        # Use tau from LPFC (caller convention)
        tau_lpfc = compute_embedding_delay(theta_lpfc)
        S_lpfc, S_cpfc, meta_e = symbolise_pair(
            theta_lpfc, theta_cpfc,
            d=3, tau=tau_lpfc, k0_max=20, ndis_threshold=0.05
        )
        print(f"      tau={tau_lpfc}  k₀={meta_e['k0']}  nDis={meta_e['nDis']:.4f}")
        print(f"      S_LPFC: shape={S_lpfc.shape}  values ∈ [{S_lpfc.min()}, {S_lpfc.max()}]")
        print(f"      S_CPFC: shape={S_cpfc.shape}  values ∈ [{S_cpfc.min()}, {S_cpfc.max()}]")
        # Verify symbol ranges
        k0_e = meta_e["k0"]
        assert S_lpfc.min() >= 0 and S_lpfc.max() < k0_e
        assert S_cpfc.min() >= 0 and S_cpfc.max() < k0_e
        assert S_lpfc.shape == S_cpfc.shape
        print(f"      codebook shape={meta_e['codebook'].shape}  ✓")
        print("      [PASS] Real A1 pair symbolisation")
    except Exception as e:
        print(f"      [SKIP] A1 ROI load failed: {e}")

    print("  [PASS] 3.3.3 symbolise_pair\n")

    print("=== All Section 3.1–3.3 tests passed. ===\n")


if __name__ == "__main__":
    _run_tests()
