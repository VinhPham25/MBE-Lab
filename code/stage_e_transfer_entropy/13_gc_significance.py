"""
Stage E — Section 10.7: GC-Native Analytic Significance (Geweke 1982).

**STATUS: WIRE ONLY (deferred).**  This module is implemented and
smoke-tested (Section 10.8-style synthetic AR check) in this pass, but is
**not batch-run** against the 13 real ASTaUND subjects.  See the "Deferred
Work Runbook" at the end of ``docs/TRANSFER_ENTROPY_PROGRESS.md`` Section 10
for the exact command to activate it later.

Method rationale
-----------------
Section 10's registry treats every method (GC, Phase 2, Phase 4) as producing
a per-connection significance annotation, but the *mechanism* for generating
that annotation is method-specific:

- Phase 2 / Phase 4 (NG-symbolic TE): significance comes from
  ``11_symbolic_ete_core.py``'s circular-shift **surrogate** test —
  ``ETE = TE_raw - TE_surrogate_floor`` (Marschinski & Kantz 2002).
- Stage D linear GC: continuous VAR / frequency-domain — there is no
  discrete symbol sequence to circularly shift, so the surrogate mechanism
  above does not apply.  GC has its own **analytic** significance test that
  plays the same architectural role: **Geweke's (1982) asymptotic
  chi-squared / F-test on the VAR exclusion restriction.**

Geweke (1982) test definition
------------------------------
For a bivariate VAR(p) fitted jointly on (X, Y), Granger causality
X -> Y corresponds to testing whether X's lagged coefficients in Y's
equation are jointly zero:

    H0:  a_{Y,X}(1) = a_{Y,X}(2) = ... = a_{Y,X}(p) = 0

This is tested by comparing the residual variance of a **restricted** VAR
(Y regressed on its own lags only) against an **unrestricted** VAR (Y
regressed on its own lags AND X's lags):

    GC_{X->Y} = ln( sigma^2_restricted / sigma^2_unrestricted )

Under H0, the corresponding likelihood-ratio statistic

    LR = T * ln( sigma^2_restricted / sigma^2_unrestricted )

is asymptotically chi-squared distributed with `p` degrees of freedom
(Geweke, 1982, "Measurement of Linear Dependence and Feedback Between
Multiple Time Series", JASA 77(378), 304-313). An equivalent finite-sample
F-test (Granger's original 1969 formulation) is also reported since T is
finite (~1350 samples/window) and the asymptotic chi-squared approximation
is known to over-reject in finite samples relative to the F-test.

This is the exact GC-native counterpart of a surrogate-based p-value: both
answer "does the observed coupling exceed what a null model with no true
directed influence would produce," just via an asymptotic distribution
instead of resampling.

Investigation (Section 10.7 first sub-task)
--------------------------------------------
**Question:** Does ``stage_d_granger/03_linear_gc.py``'s current VAR fit
already expose the restricted-vs-unrestricted residual covariance matrices
needed for the test statistic?

**Finding (2026-08-29):** No, not directly, but a restricted refit is cheap.
- ``compute_gc_spectrum_bivariate()`` (Stage D) fits a bivariate
  ``statsmodels.tsa.vector_ar.var_model.VAR`` per (src, tgt) pair via
  ``VAR(Y_pair.T).fit(maxlags=order, ...)``.  The resulting
  ``VARResultsWrapper`` exposes ``.sigma_u`` (2x2 unrestricted residual
  covariance matrix) and ``.coefs`` (VAR coefficients) — this *is* the
  unrestricted model's residual variance, so no re-fit is needed for that
  half of the ratio.
- However, Stage D's fitted VAR is **only ever the unrestricted (bivariate)
  model** — there is no restricted (Y-on-its-own-lags-only, i.e. univariate
  AR(order)) fit anywhere in the pipeline.  The restricted residual variance
  is therefore **not exposed** and must be computed by this module.
- **Decision: option (a) — re-derive from a cheap additional fit, not a
  full VAR refit.**  Fitting a univariate ``AR(order)`` model on Y alone
  (``statsmodels.tsa.ar_model.AutoReg`` or an OLS lag regression) is O(T*p)
  and negligible compared to fitting 20 bivariate VARs per window, let alone
  500 ETE surrogates for the symbolic methods.  This module fits that
  restricted AR(order) once per (src, tgt, window) pair.

Output schema (once activated)
-------------------------------
Per-subject/trial/window: ``p_value`` and ``significant`` (bool, alpha=0.05
default) for all 20 directed connections — schema-compatible with the other
methods' significance annotations consumed by
``16_cross_method_significance.py``.

References
----------
Geweke, J. (1982). Measurement of Linear Dependence and Feedback Between
Multiple Time Series. Journal of the American Statistical Association,
77(378), 304-313. doi:10.2307/2287238

Granger, C. W. J. (1969). Investigating Causal Relations by Econometric
Models and Cross-spectral Methods. Econometrica, 37(3), 424-438.

Implementation refinement over the investigation note above
-------------------------------------------------------------
The investigation above suggested reusing Stage D's already-fitted bivariate
``VARResultsWrapper.sigma_u`` directly for the unrestricted half of the
ratio, refitting only the restricted (univariate) half separately (e.g. via
``statsmodels.tsa.ar_model.AutoReg``). In practice this has two problems:
(1) ``statsmodels.tsa.vector_ar.var_model.VAR`` raises ``ValueError: Only
gave one variable to VAR`` for single-column input — it cannot fit a
univariate model at all, so ``AutoReg`` (a different estimator class) would
have to be used for the restricted half; and (2) different statsmodels
estimator classes are not guaranteed to use identical residual-variance
normalisation conventions (e.g. ``nobs`` vs. a bias-corrected ``nobs - k``
denominator) — an inconsistency here directly corrupts the LR ratio, since
Geweke's test statistic is only valid when both variances are estimated
under the same normalisation. This module therefore implements **both**
the restricted and unrestricted regressions with the same manual OLS
lag-regression helper (``_fit_lag_ols``), guaranteeing identical conventions
(both use ``RSS / nobs``, no bias correction) — a first-principles
implementation choice that trades a few lines of manual OLS for guaranteed
internal consistency, rather than depending on cross-estimator-class parity.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_STAGE_E_ROOT = Path(__file__).parent            # stage_e_transfer_entropy/
_CODE_ROOT = _STAGE_E_ROOT.parent                # code/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

logger = logging.getLogger(__name__)


def _load_config_if_none(config: Optional[dict]) -> dict:
    """Load config.yaml from disk if config dict not yet provided."""
    if config is not None:
        return config
    try:
        from utils import load_config  # noqa

        return load_config()
    except ImportError:
        return {}


# ===========================================================================
# Core manual-OLS lag regression (shared normalisation for restricted /
# unrestricted fits — see module docstring's "Implementation refinement").
# ===========================================================================

def _fit_lag_ols(target: np.ndarray, predictors: "list[np.ndarray]", order: int) -> dict:
    """
    Fit ``target_t = c + sum_i sum_{lag=1..order} beta_{i,lag} * predictor_i[t-lag] + eps_t``
    via ordinary least squares.

    Parameters
    ----------
    target : np.ndarray, shape (T,)
        Dependent series.
    predictors : list of np.ndarray, each shape (T,)
        Predictor series (each contributes ``order`` lagged columns). For
        the restricted model this is ``[target]`` (AR-only); for the
        unrestricted model this is ``[target, source]``.
    order : int
        Number of lags per predictor.

    Returns
    -------
    dict
        ``{RSS, sigma2, nobs, k_params}`` — ``sigma2 = RSS / nobs`` (no
        bias correction, so restricted/unrestricted fits at the same
        ``order`` and ``T`` always share an identical ``nobs``, which is
        what the LR/F ratio requires).
    """
    target = np.asarray(target, dtype=float)
    T = len(target)
    nobs = T - order
    if nobs <= 0:
        raise ValueError(f"_fit_lag_ols: T={T} too short for order={order}.")

    cols = [np.ones((nobs, 1))]
    for pred in predictors:
        pred = np.asarray(pred, dtype=float)
        lag_block = np.column_stack(
            [pred[order - lag: T - lag] for lag in range(1, order + 1)]
        )
        cols.append(lag_block)
    Z = np.hstack(cols)
    y_target = target[order:]

    beta, _, _, _ = np.linalg.lstsq(Z, y_target, rcond=None)
    resid = y_target - Z @ beta
    RSS = float(np.sum(resid ** 2))

    return {"RSS": RSS, "sigma2": RSS / nobs, "nobs": nobs, "k_params": Z.shape[1]}


# ===========================================================================
# Section 10.7 — Core pairwise Geweke test
# ===========================================================================

def compute_geweke_pair(x: np.ndarray, y: np.ndarray, order: int = 20) -> dict:
    """
    Compute Geweke's (1982) analytic GC significance test for both
    directions of one (x, y) pair, at a fixed VAR order.

    For X -> Y: compares the restricted model (Y on its own ``order`` lags)
    against the unrestricted model (Y on its own lags AND X's lags). The
    likelihood-ratio statistic ``LR = nobs * ln(sigma2_r / sigma2_u)`` is
    asymptotically chi-squared with ``order`` degrees of freedom under H0
    (Geweke, 1982); the equivalent finite-sample F-test
    (Granger, 1969) is also reported since T is finite.

    Parameters
    ----------
    x, y : np.ndarray, shape (T,)
        The two time series (order doesn't matter — both directions are
        always computed).
    order : int
        VAR/AR lag order. Should match the order used elsewhere in the
        pipeline for comparability (Stage D default: 20).

    Returns
    -------
    dict
        ``{'x_to_y': {...}, 'y_to_x': {...}, 'nobs': int, 'order': int}``.
        Each direction's sub-dict has ``LR_stat, chi2_p, F_stat, F_p``.
    """
    from scipy.stats import chi2 as _chi2, f as _f

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("compute_geweke_pair: x and y must have the same length.")

    fit_r_y = _fit_lag_ols(y, [y], order)
    fit_u_y = _fit_lag_ols(y, [y, x], order)
    fit_r_x = _fit_lag_ols(x, [x], order)
    fit_u_x = _fit_lag_ols(x, [x, y], order)

    def _direction_test(fit_r: dict, fit_u: dict) -> dict:
        nobs = fit_u["nobs"]
        sigma2_r, sigma2_u = fit_r["sigma2"], fit_u["sigma2"]
        if sigma2_u <= 0 or sigma2_r <= 0:
            return {"LR_stat": np.nan, "chi2_p": np.nan, "F_stat": np.nan, "F_p": np.nan}

        LR = max(float(nobs * np.log(sigma2_r / sigma2_u)), 0.0)
        chi2_p = float(1.0 - _chi2.cdf(LR, df=order))

        df1 = order
        df2 = nobs - fit_u["k_params"]
        if df2 > 0 and fit_u["RSS"] > 0:
            F_stat = max(((fit_r["RSS"] - fit_u["RSS"]) / df1) / (fit_u["RSS"] / df2), 0.0)
            F_p = float(1.0 - _f.cdf(F_stat, df1, df2))
        else:
            F_stat, F_p = np.nan, np.nan

        return {"LR_stat": LR, "chi2_p": chi2_p, "F_stat": float(F_stat), "F_p": F_p}

    return {
        "x_to_y": _direction_test(fit_r_y, fit_u_y),
        "y_to_x": _direction_test(fit_r_x, fit_u_x),
        "nobs": fit_u_y["nobs"],
        "order": order,
    }


# ===========================================================================
# Section 10.7 — Full 5x5 significance matrix (bivariate, per pair)
# ===========================================================================

def compute_gc_significance_matrix(
    Y_window: np.ndarray,
    order: int = 20,
    alpha: float = 0.05,
    test: str = "F",
) -> tuple[dict, dict]:
    """
    Compute Geweke significance for every directed pair in a window.

    Bivariate (per-pair), matching Stage D's ``compute_gc_spectrum_bivariate``
    convention — no multivariate conditioning.

    Parameters
    ----------
    Y_window : np.ndarray, shape (n_rois, T)
        Window signal matrix. All-NaN rows (bad ROI channels) are skipped;
        their matrix entries remain NaN/False.
    order : int
        VAR/AR lag order.
    alpha : float
        Significance threshold.
    test : str
        ``'F'`` (finite-sample, recommended — Geweke's asymptotic chi-squared
        is known to over-reject at finite T) or ``'chi2'``.

    Returns
    -------
    tuple[dict, dict]
        ``(matrices, meta)`` where ``matrices`` has keys
        ``'p_value', 'significant', 'LR_stat', 'F_stat'`` (each
        ``(n_rois, n_rois)``), and ``meta`` has ``{'pair_meta', 'test',
        'alpha', 'order', 'n_rois'}``.
    """
    n_rois, _T = Y_window.shape
    p_value_matrix = np.full((n_rois, n_rois), np.nan)
    significant_matrix = np.zeros((n_rois, n_rois), dtype=bool)
    LR_matrix = np.full((n_rois, n_rois), np.nan)
    F_matrix = np.full((n_rois, n_rois), np.nan)
    pair_meta: dict = {}

    p_field = "chi2_p" if test == "chi2" else "F_p"

    for src in range(n_rois):
        for tgt in range(src + 1, n_rois):
            x = Y_window[src]
            y = Y_window[tgt]
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
            try:
                result = compute_geweke_pair(x, y, order=order)
            except Exception as exc:
                logger.warning(
                    "compute_gc_significance_matrix: pair (%d,%d) failed: %s",
                    src, tgt, exc,
                )
                continue

            p_xy = result["x_to_y"][p_field]
            p_yx = result["y_to_x"][p_field]

            p_value_matrix[src, tgt] = p_xy
            p_value_matrix[tgt, src] = p_yx
            significant_matrix[src, tgt] = bool(not np.isnan(p_xy) and p_xy < alpha)
            significant_matrix[tgt, src] = bool(not np.isnan(p_yx) and p_yx < alpha)
            LR_matrix[src, tgt] = result["x_to_y"]["LR_stat"]
            LR_matrix[tgt, src] = result["y_to_x"]["LR_stat"]
            F_matrix[src, tgt] = result["x_to_y"]["F_stat"]
            F_matrix[tgt, src] = result["y_to_x"]["F_stat"]
            pair_meta[(src, tgt)] = result

    matrices = {
        "p_value": p_value_matrix, "significant": significant_matrix,
        "LR_stat": LR_matrix, "F_stat": F_matrix,
    }
    meta = {"pair_meta": pair_meta, "test": test, "alpha": alpha, "order": order, "n_rois": n_rois}
    return matrices, meta


# ===========================================================================
# Section 10.7 — Real per-subject/trial/window wrapper (WIRE ONLY —
# fully implemented and functional, but not invoked against real ASTaUND
# data in this pass; see the Deferred Work Runbook, Section 10.13).
# ===========================================================================

def compute_gc_significance(
    subject_id: str,
    trial_id: str,
    window_id: str,
    config: Optional[dict] = None,
) -> dict:
    """
    Compute Geweke GC significance for one real (subject, trial, window).

    Loads data via Stage D's exact conventions (``load_preprocessed_fif``,
    ``01_extract_roi_signals.extract_roi_signals``,
    ``02_trial_segmentation.get_trial_segments``, and the same window-slice
    convention as ``03_linear_gc.compute_windowed_gc`` —
    ``Y_trial[:, w-window_size : w+1]``), then calls
    ``compute_gc_significance_matrix()`` and expands the result to the
    canonical 20-element directed-connection schema shared by every other
    method's significance annotation (Section 10's registry contract).

    **NOT invoked this pass** — see module docstring / Section 10.13.

    Parameters
    ----------
    subject_id : str
        e.g. ``'A1'``.
    trial_id : str
        e.g. ``'Tr_1'``.
    window_id : str
        e.g. ``'Win_1'``.
    config : dict, optional
        Parsed config.yaml.

    Returns
    -------
    dict
        ``{'p_value': np.ndarray(20,), 'significant': np.ndarray(20,) bool,
        'connection_labels': list[str], 'order': int, 'test': str,
        'subject_id', 'trial_id', 'window_id'}``.
    """
    import importlib.util as _ilu

    config = _load_config_if_none(config)

    from stage_d_granger.utils_gc import (
        ROI_NAMES, get_connection_labels, load_preprocessed_fif,
    )

    def _load_stage_d_module(name: str):
        spec = _ilu.spec_from_file_location(
            f"gc_sig_{name}", _CODE_ROOT / "stage_d_granger" / f"{name}.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    roi_mod = _load_stage_d_module("01_extract_roi_signals")
    seg_mod = _load_stage_d_module("02_trial_segmentation")

    raw = load_preprocessed_fif(subject_id, study="expert_novice", config=config)
    Data_Reg, valid_roi_names = roi_mod.extract_roi_signals(raw)
    segments = seg_mod.get_trial_segments(raw, config=config)

    trial_idx = int(trial_id.split("_")[1]) - 1
    if trial_idx < 0 or trial_idx >= len(segments):
        raise ValueError(
            f"compute_gc_significance [{subject_id} {trial_id}]: trial index "
            f"{trial_idx} out of range (found {len(segments)} trials)."
        )
    onset, offset = segments[trial_idx]
    Y_trial = Data_Reg[:, onset: offset + 1]

    gc_cfg = config["stage_d_granger"]["linear_gc"]
    window_size = int(gc_cfg["window"]["size_samples"])
    order = int(gc_cfg["var"]["order"])
    alpha = float(config.get("stage_e_transfer_entropy", {}).get("statistics", {}).get("alpha", 0.05))

    win_num = int(window_id.split("_")[1])
    w = window_size * win_num  # matches compute_windowed_gc's `for w in range(window_size, T, stride)`
    ws, we = w - window_size, w
    if we + 1 > Y_trial.shape[1]:
        raise ValueError(
            f"compute_gc_significance [{subject_id} {trial_id} {window_id}]: "
            f"window end {we + 1} exceeds trial length {Y_trial.shape[1]}."
        )
    Y_win = Y_trial[:, ws: we + 1]

    matrices, _meta = compute_gc_significance_matrix(Y_win, order=order, alpha=alpha, test="F")

    n_all = len(ROI_NAMES)
    p_full = np.full((n_all, n_all), np.nan)
    sig_full = np.zeros((n_all, n_all), dtype=bool)
    for i_v, roi_i in enumerate(valid_roi_names):
        for j_v, roi_j in enumerate(valid_roi_names):
            if roi_i == roi_j:
                continue
            i_c, j_c = ROI_NAMES.index(roi_i), ROI_NAMES.index(roi_j)
            p_full[i_c, j_c] = matrices["p_value"][i_v, j_v]
            sig_full[i_c, j_c] = matrices["significant"][i_v, j_v]

    diag_indices = [i * (n_all + 1) for i in range(n_all)]
    p_vec_20 = np.delete(p_full.flatten(), diag_indices)
    sig_vec_20 = np.delete(sig_full.flatten(), diag_indices)

    return {
        "p_value": p_vec_20,
        "significant": sig_vec_20,
        "connection_labels": get_connection_labels(),
        "order": order,
        "alpha": alpha,
        "test": "F",
        "subject_id": subject_id,
        "trial_id": trial_id,
        "window_id": window_id,
    }


# ===========================================================================
# Section 10.7 — Smoke tests (synthetic AR model; NOT run against real data)
# ===========================================================================

def _run_smoke_tests() -> None:
    """
    Smoke-test ``compute_geweke_pair`` on synthetic data with known
    ground-truth (a)directionality and (b) absence of coupling.

    (a) Linear AR model with a known true causal link X(t-3) -> Y(t) (pure
        linear, unlike the ETE core's AR-with-abs() model, since Geweke's
        test is inherently a *linear* VAR test): verify p(X->Y) < 0.05 and
        p(Y->X) is not spuriously significant.
    (b) Independent noise pair: verify neither direction is spuriously
        significant.
    """
    import logging as _lg

    _lg.basicConfig(level=_lg.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    log = logging.getLogger("smoke_gc_significance")
    log.setLevel(_lg.INFO)

    n_fail = 0
    order = 20
    T = 1350

    log.info("=" * 60)
    log.info("Test (a): linear AR model — X(t-3) -> Y(t), no reverse coupling")
    try:
        rng = np.random.default_rng(seed=42)
        X = np.zeros(T + 100)
        eps_x = rng.standard_normal(T + 100) * 0.1
        eps_y = rng.standard_normal(T + 100) * 0.1
        for t in range(2, T + 100):
            X[t] = 0.4 * X[t - 1] - 0.6 * X[t - 2] + eps_x[t]
        Y = np.zeros(T + 100)
        for t in range(max(3, 1), T + 100):
            Y[t] = 0.5 * X[t - 3] + 0.3 * Y[t - 1] + eps_y[t]
        X, Y = X[100:], Y[100:]

        result = compute_geweke_pair(X, Y, order=order)
        p_xy_F = result["x_to_y"]["F_p"]
        p_yx_F = result["y_to_x"]["F_p"]
        p_xy_chi2 = result["x_to_y"]["chi2_p"]
        log.info(
            "Test (a): F-test p(X->Y)=%.4g  p(Y->X)=%.4g  |  chi2 p(X->Y)=%.4g  nobs=%d",
            p_xy_F, p_yx_F, p_xy_chi2, result["nobs"],
        )
        if p_xy_F < 0.05 and p_yx_F >= 0.05:
            log.info("Test (a) PASS \u2713 \u2014 true X->Y link detected, no spurious Y->X")
        else:
            log.error(
                "Test (a) FAIL \u2717 \u2014 expected p(X->Y)<0.05 and p(Y->X)>=0.05, got %.4g / %.4g",
                p_xy_F, p_yx_F,
            )
            n_fail += 1
    except Exception as exc:
        log.error("Test (a) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    log.info("=" * 60)
    log.info("Test (b): independent noise pair — neither direction should be significant")
    try:
        rng2 = np.random.default_rng(seed=7)
        X_n = rng2.standard_normal(T)
        Y_n = rng2.standard_normal(T)
        result_n = compute_geweke_pair(X_n, Y_n, order=order)
        p_xy_n = result_n["x_to_y"]["F_p"]
        p_yx_n = result_n["y_to_x"]["F_p"]
        log.info("Test (b): F-test p(X->Y)=%.4g  p(Y->X)=%.4g", p_xy_n, p_yx_n)
        if p_xy_n >= 0.05 and p_yx_n >= 0.05:
            log.info("Test (b) PASS \u2713 \u2014 no spurious significance on independent noise")
        else:
            log.warning(
                "Test (b) WARN \u2014 p(X->Y)=%.4g p(Y->X)=%.4g (one below 0.05 on a single "
                "realisation has ~5%% base-rate probability by chance; not a hard failure).",
                p_xy_n, p_yx_n,
            )
    except Exception as exc:
        log.error("Test (b) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    log.info("=" * 60)
    if n_fail == 0:
        log.info("ALL GC SIGNIFICANCE SMOKE TESTS PASSED \u2713  (0 hard failures)")
    else:
        log.error("%d GC SIGNIFICANCE SMOKE TEST(S) FAILED \u2717", n_fail)
        sys.exit(1)


if __name__ == "__main__":
    _run_smoke_tests()
