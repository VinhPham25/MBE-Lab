"""
Stage E — Section 10: Effective Transfer Entropy (ETE) Surrogate Core.

**Method-agnostic.**  This module is shared, unmodified, by Phase 2
(``phase_2/12_multivariate_ete.py``, WIRE ONLY this pass) and Phase 4
(``phase_4/19_amplitude_ete.py``, RUN NOW this pass).  Adding a 3rd symbolic
method later should require zero changes here — only a new thin wiring file
that supplies a different ``raw_compute_fn`` callback.

Motivation
----------
Raw amplitude/phase dSTE/dSPTE (Sections 7-9) has a non-zero floor even in
the complete absence of true directed coupling: any finite-sample
frequency-count TE estimator is upward-biased (Marschinski & Kantz, 2002),
and residual shared/background structure inflates the apparent value
further.  The standard correction in the EEG/fNIRS connectivity literature
(e.g. TRENTOOL — Lindner et al., 2011) is a **circular-shift surrogate
test**:

    ETE = TE_raw - TE_surrogate_floor

where ``TE_surrogate_floor`` is the mean (or median) TE computed on many
realisations of the source symbol sequence with its temporal structure
destroyed via a circular (wrap-around) shift, while preserving:
  - the exact marginal symbol distribution of the source (a shuffle would
    not; a shift does, since it is just a re-indexing), and
  - the full temporal structure of the target and conditioning channels
    (only the source-target *relationship* is what should be nulled).

A circular shift (rather than a full random permutation) is the standard
choice for this exact reason: it destroys the *specific temporal
alignment* between X and Y that the TE statistic is sensitive to, while
leaving X's own autocorrelation structure intact — a full shuffle would
additionally destroy X's autocorrelation, which is not what is being
tested (Lindner et al., 2011, TRENTOOL).

**Fixed-lag convention.**  ``compute_pair_ete`` takes the already-known
optimal lag ``u_opt`` (from the real-data multidelay search) and does
**not** re-search ``argmax`` over lags on every surrogate.  This is the
standard surrogate-testing convention: it tests whether the data-selected
statistic exceeds chance at that *specific* lag, avoiding the upward bias
that maximising over ``u_max`` independent lags on each of 500 null
realisations would otherwise introduce (the null distribution of a
max-over-lags statistic is stochastically larger than the null distribution
at a single fixed lag).

**PCA-conditioning design (2026-08-30 fix; supersedes the 2026-08-29 design
below the researcher originally confirmed).**
The raw pipelines (``compute_multivariate_dspte_matrix`` /
``compute_amplitude_conditional_dste_matrix``) may PCA-compress the
``n_z`` conditioning ROIs into a single composite symbol sequence before
calling ``compute_conditional_dspte_at_lag`` (``phase_*.conditional_pca.
enabled``). A Stage 1 diagnostic (2026-08-30, at the final selected Phase 2
configuration) found that at the configuration actually used for results,
100% of the 20 directed pairs across every subject/trial/window used this
genuine multivariate PCA conditioning (0% bivariate sparsity fallback) —
so conditioning every surrogate on anything other than that exact
composite would silently invalidate the ETE null for essentially every
pair, not just an edge case.

The Section 10.2 additive metadata contract therefore now also exposes the
PCA-compressed composite itself, per pair, per window
(``z_composite_per_pair``), plus a ``conditioning_config_hash`` tag
identifying the exact parameter set used to build it (see
``utils_te.compute_conditioning_config_hash``). ``compute_pair_ete``
conditions every surrogate recomputation on this **exact stored composite**
for pairs where it is available — never reconstructing it from raw
per-ROI symbols and never recomputing PCA — reusing the identical,
unchanged composite across the observed (non-surrogate) recompute and all
``n_surrogates`` circular-shift realisations (only ``S_x`` is shifted; the
composite, like ``S_y``, is untouched by the surrogate procedure and would
be identical if recomputed anyway).

For a pair/window where ``z_composite_per_pair`` has no entry, there are
exactly two possible reasons, and this module distinguishes them via the
per-pair ``z_composite_source`` diagnostic (see ``compute_pair_ete``'s
return value and ``compute_symbolic_ete_matrix``'s per-pair metadata) —
deliberately a *different* field from ``pca_consistency_ratio`` below, so
"the real pipeline used bivariate for this pair" is never conflated with
"no valid stored composite was found":
  1. ``'real_pipeline_bivariate'`` — the real pipeline itself
     majority-fell-back to bivariate TE for that specific pair (the
     existing sparsity-fallback safety net inside
     ``compute_conditional_dspte_at_lag`` fired for most lags). ETE then
     also conditions on nothing (``S_z_list=[]``, bivariate) — the same
     graceful-degradation path the raw pipeline uses, preserved unchanged.
  2. ``'raw_compute_fn_missing_key'`` — the ``raw_compute_fn`` supplied
     does not (yet) populate ``z_composite_per_pair``/
     ``conditioning_config_hash`` at all (e.g. a future 3rd symbolic method
     that hasn't added the Stage 2 additive keys). ETE falls back to
     bivariate for backward compatibility rather than erroring, since this
     is a purely additive contract.
A **hash mismatch** (a ``conditioning_config_hash`` present in the
metadata but different from what the current run's config/k0 would
produce) is treated differently from either of the above: it indicates a
stored composite computed under different parameters is being reused
across a config change, which is a correctness bug, not a graceful
degradation case — ``compute_symbolic_ete_matrix`` raises a ``ValueError``
immediately rather than silently reusing it.

With this fix, ``raw_te`` recomputed by ``compute_pair_ete`` for a pair
using the stored composite is expected to match the original
PCA-conditioned ``EC_matrix`` value for that pair almost exactly
(``pca_consistency_ratio`` ≈ 1.0 — see the per-pair metadata), because it
is now built from literally the same ``(S_x, S_y, [z_composite])`` inputs
at the same lag ``u_opt`` and the same ``k0`` that produced the original
value — not from a different, mismatched-dimensionality conditioning set.

References
----------
Marschinski, R., & Kantz, H. (2002). Analysing the information flow between
financial time series. European Physical Journal B, 30(2), 275-281.
doi:10.1140/epjb/e2002-00379-2

Lindner, M., Vicente, R., Priesemann, V., & Wibral, M. (2011). TRENTOOL: a
Matlab open source toolbox to analyse information flow in time series data
with transfer entropy. BMC Neuroscience, 12, 119.
doi:10.1186/1471-2202-12-119

See ``13_gc_significance.py`` for the GC-native analytic equivalent
(Geweke, 1982) — GC has no discrete symbol sequence to shift, so it uses a
chi-squared/F asymptotic test on the VAR exclusion restriction instead of a
surrogate distribution.  Both plug into the same higher-level significance
registry (``config.yaml: stage_e_transfer_entropy.significance_methods``)
via the shared contract: every method produces ``(B_Exp, B_Nov)``
mean-connectivity arrays plus a per-connection significance annotation.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping (mirrors every other Stage E module)
# ---------------------------------------------------------------------------
_STAGE_E_ROOT = Path(__file__).parent           # stage_e_transfer_entropy/
_CODE_ROOT = _STAGE_E_ROOT.parent               # code/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy import (load once, reuse across calls)
# ---------------------------------------------------------------------------
_MULTIVARIATE_MOD = None   # phase_2/06_multivariate_dspte.py — compute_conditional_dspte_at_lag


def _get_multivariate_mod() -> object:
    """Return cached phase_2/06_multivariate_dspte module (method-agnostic reuse)."""
    global _MULTIVARIATE_MOD
    if _MULTIVARIATE_MOD is None:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location(
            "multivariate_dspte_ete_core", _STAGE_E_ROOT / "phase_2" / "06_multivariate_dspte.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MULTIVARIATE_MOD = mod
    return _MULTIVARIATE_MOD


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
# Section 10.3 — Circular-shift surrogate generator (method-agnostic)
# ===========================================================================

def generate_circular_shift_surrogate(
    S_x: np.ndarray,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one circular-shift surrogate of a symbolic sequence.

    Draws a random integer offset ``r ~ Uniform(min_shift, T' - min_shift)``
    and returns ``np.roll(S_x, r)``.  Operates on any integer symbol array
    regardless of which phase (phase-domain or amplitude-domain) produced
    it — this function has no knowledge of the underlying signal domain.

    Parameters
    ----------
    S_x : np.ndarray, shape (T',), dtype int
        Source symbol sequence to be shifted.
    min_shift : int
        Minimum |shift| in samples (``config.<method>.ete.min_shift_fraction
        * T'``, enforced by the caller).  Guards against near-zero shifts
        that would leave ``S_x`` almost unchanged (an uninformative null
        realisation nearly identical to the real data).
    rng : np.random.Generator
        Seeded random generator — pass a freshly-seeded generator (or reuse
        one across calls within a single pair's surrogate loop) for
        reproducibility.

    Returns
    -------
    np.ndarray, shape (T',), dtype int
        Circularly-shifted copy of ``S_x``.  Same multiset of symbol values
        (marginal distribution exactly preserved by construction), but the
        specific time alignment with ``S_y`` is destroyed.

    Notes
    -----
    If ``T' - min_shift <= min_shift`` (i.e. the sequence is too short for
    the requested minimum shift fraction to leave a valid sampling range),
    falls back to a single fixed offset of ``T' // 2`` (still guaranteed to
    avoid the zero-offset identity shift) and logs a warning.
    """
    S_x = np.asarray(S_x)
    T_prime = len(S_x)
    min_shift = max(1, int(min_shift))
    upper = T_prime - min_shift

    if upper <= min_shift:
        r = T_prime // 2 if T_prime > 1 else 0
        logger.warning(
            "generate_circular_shift_surrogate: T'=%d too short for "
            "min_shift=%d (valid sampling range empty) — using fixed "
            "fallback shift r=%d.",
            T_prime, min_shift, r,
        )
    else:
        r = int(rng.integers(min_shift, upper))

    return np.roll(S_x, r)


# ===========================================================================
# Section 10.4 — Per-pair ETE + full-matrix ETE (method-agnostic core)
# ===========================================================================

def compute_pair_ete(
    S_x: np.ndarray,
    S_y: np.ndarray,
    S_z_list: "list[np.ndarray]",
    u_opt: int,
    k0: int,
    ete_params: dict,
    sparsity_threshold: float = 0.95,
    seed: Optional[int] = None,
) -> dict:
    """
    Compute Effective Transfer Entropy (ETE) for one directed pair at a
    fixed, already-known optimal lag.

    Steps (Section 10.4):
      (a) Recompute ``raw_te`` via ``compute_conditional_dspte_at_lag()`` as
          a consistency check (see module docstring). When the caller
          passes the exact stored PCA composite for this pair (Section
          10.2's ``z_composite_per_pair``), this recomputed value is
          expected to match the caller's original ``EC_matrix`` entry for
          that pair almost exactly (``pca_consistency_ratio`` ≈ 1.0,
          diagnosed one level up in ``compute_symbolic_ete_matrix``), since
          it is built from the identical ``(S_x, S_y, S_z_list)`` inputs at
          the same lag ``u_opt`` and ``k0``.
      (b) Generate ``n_surrogates`` circularly-shifted copies of ``S_x``.
      (c) Recompute ``compute_conditional_dspte_at_lag`` per surrogate
          (``S_y``, ``S_z_list``, ``u_opt``, ``k0`` unchanged — only ``S_x``
          varies).
      (d) ``ete = raw_te - null_mean``; ``p_value`` = one-sided fraction of
          the null distribution ``>= raw_te``; ``z_score`` standardises
          ``raw_te`` against the null distribution.

    Parameters
    ----------
    S_x, S_y : np.ndarray, shape (T',), dtype int
        Source / target symbol sequences (already lagged internally by
        ``compute_conditional_dspte_at_lag``, not pre-shifted here).
    S_z_list : list of np.ndarray, each shape (T',), dtype int
        Conditioning channel symbol sequence(s), already resolved by the
        caller (``compute_symbolic_ete_matrix``) per the Section 10.2 fix:
        either a single-element list containing the exact stored
        PCA-compressed composite for this pair, or ``[]`` for bivariate
        (no conditioning) — see module docstring.  This function itself is
        agnostic to which; it just passes whatever it is given straight
        through to ``compute_conditional_dspte_at_lag``.
    u_opt : int
        Fixed causal lag (samples), taken from the real-data multidelay
        search.  NOT re-searched per surrogate (see module docstring).
    k0 : int
        Shared codebook size (alphabet cardinality).
    ete_params : dict
        Resolved ``stage_e_transfer_entropy.<method>.ete`` config sub-block
        (e.g. ``config['stage_e_transfer_entropy']['phase_4']['ete']``).
        Keys used: ``n_surrogates``, ``min_shift_fraction``,
        ``floor_estimator`` ('mean' | 'median'), ``clip_negative`` (bool),
        ``p_value_alternative`` ('greater' | 'two-sided').
    sparsity_threshold : float
        Passed through to ``compute_conditional_dspte_at_lag`` (method's own
        ``sparsity_fallback_threshold``).
    seed : int, optional
        Seed for the surrogate RNG.  Passing the same
        ``(window, pair, seed)`` triple must reproduce an identical null
        distribution (Section 10.8 reproducibility smoke test).

    Returns
    -------
    dict
        ``raw_te`` (float), ``null_distribution`` (np.ndarray, shape
        ``(n_surrogates,)``), ``null_mean`` (float), ``null_std`` (float),
        ``ete`` (float, signed — not clipped unless ``clip_negative=True``),
        ``p_value`` (float, one-sided by default), ``z_score`` (float),
        ``n_surrogates`` (int), ``min_shift`` (int), ``fallback`` (bool —
        whether the *real* recompute at ``u_opt`` triggered the sparsity
        fallback), ``sparsity_pct`` (float), ``mean_count_occupied`` (float),
        ``n_surrogate_fallbacks`` (int — how many of the null realisations
        also fell back; a high count indicates chronic sparsity, not a
        surrogate-specific artefact), ``elapsed_s`` (float).
    """
    t0 = time.perf_counter()
    multivariate = _get_multivariate_mod()

    S_x = np.asarray(S_x, dtype=int)
    S_y = np.asarray(S_y, dtype=int)
    S_z_list = [np.asarray(z, dtype=int) for z in S_z_list]

    n_surrogates = int(ete_params.get("n_surrogates", 500))
    min_shift_fraction = float(ete_params.get("min_shift_fraction", 0.05))
    floor_estimator = str(ete_params.get("floor_estimator", "mean"))
    clip_negative = bool(ete_params.get("clip_negative", False))
    p_value_alternative = str(ete_params.get("p_value_alternative", "greater"))

    T_prime = len(S_x)
    min_shift = max(1, int(round(min_shift_fraction * T_prime)))

    # ------------------------------------------------------------------
    # (a) Consistency-check recompute of the real (non-surrogate) TE at u_opt
    # ------------------------------------------------------------------
    raw_te, lag_meta = multivariate.compute_conditional_dspte_at_lag(
        S_x, S_y, S_z_list, u=u_opt, k0=k0, sparsity_threshold=sparsity_threshold,
    )

    # ------------------------------------------------------------------
    # (b), (c) Surrogate loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    null_distribution = np.empty(n_surrogates, dtype=np.float64)
    n_surrogate_fallbacks = 0

    for i in range(n_surrogates):
        S_x_surr = generate_circular_shift_surrogate(S_x, min_shift=min_shift, rng=rng)
        te_surr, surr_meta = multivariate.compute_conditional_dspte_at_lag(
            S_x_surr, S_y, S_z_list, u=u_opt, k0=k0,
            sparsity_threshold=sparsity_threshold,
        )
        null_distribution[i] = te_surr
        if surr_meta.get("fallback"):
            n_surrogate_fallbacks += 1

    # ------------------------------------------------------------------
    # (d) ETE, p-value, z-score
    # ------------------------------------------------------------------
    null_mean = float(np.mean(null_distribution))
    null_std = float(np.std(null_distribution, ddof=1)) if n_surrogates > 1 else 0.0
    floor = null_mean if floor_estimator == "mean" else float(np.median(null_distribution))

    ete = raw_te - floor
    if clip_negative:
        ete = max(0.0, ete)

    if p_value_alternative == "greater":
        p_value = float(np.mean(null_distribution >= raw_te))
    else:
        centred_obs = abs(raw_te - null_mean)
        p_value = float(np.mean(np.abs(null_distribution - null_mean) >= centred_obs))

    z_score = float((raw_te - null_mean) / null_std) if null_std > 1e-12 else float("nan")

    elapsed_s = time.perf_counter() - t0

    return {
        "raw_te": float(raw_te),
        "null_distribution": null_distribution,
        "null_mean": null_mean,
        "null_std": null_std,
        "ete": float(ete),
        "p_value": p_value,
        "z_score": z_score,
        "n_surrogates": n_surrogates,
        "min_shift": min_shift,
        "fallback": bool(lag_meta.get("fallback", False)),
        "sparsity_pct": float(lag_meta.get("sparsity_pct", float("nan"))),
        "mean_count_occupied": float(lag_meta.get("mean_count_occupied", float("nan"))),
        "n_surrogate_fallbacks": n_surrogate_fallbacks,
        "elapsed_s": elapsed_s,
    }


def compute_symbolic_ete_matrix(
    raw_compute_fn: Callable[[np.ndarray, dict, "Optional[list[str]]"], tuple],
    window_data: np.ndarray,
    config: dict,
    roi_names: "Optional[list[str]]" = None,
    method_key: str = "phase_4",
) -> tuple[dict, dict]:
    """
    Compute the full ETE-corrected connectivity matrix for one window.

    Method-agnostic: takes any ``raw_compute_fn`` matching the Section
    9/10.2 metadata contract — i.e. callable as
    ``raw_compute_fn(window_data, config, roi_names) -> (EC_matrix,
    delay_matrix, metadata)`` where ``metadata`` includes the Section 10.2
    additive keys (``symbol_seqs``, ``valid_indices``, ``codebook``).

    Calls ``raw_compute_fn`` exactly **once** per window (the expensive
    symbolisation + multidelay search happens only once, not once per
    surrogate — this is what keeps the ``O(n_surrogates)`` scaling from
    Section 10.8(d) limited to the per-pair single-lag recompute, not the
    full window pipeline), then loops the ``n_valid * (n_valid - 1)`` pairs
    calling ``compute_pair_ete()`` at ``u_opt = delay_matrix[src, tgt]``.

    Parameters
    ----------
    raw_compute_fn : callable
        Either ``compute_multivariate_dspte_matrix`` (Phase 2, bound to a fixed
        ``taus`` via a closure/partial by the caller — see
        ``phase_2/12_multivariate_ete.py``) or
        ``compute_amplitude_conditional_dste_matrix`` (Phase 4, used
        directly — its signature already matches).
    window_data : np.ndarray, shape (n_rois, T_win)
        Raw window data in whatever domain ``raw_compute_fn`` expects
        (broadband HbO for Phase 4; pre-filtered Hilbert phase for Phase 2).
    config : dict
        Parsed config.yaml.
    roi_names : list of str, optional
        Human-readable ROI names, same order as ``window_data`` rows.
    method_key : str
        Key into ``config['stage_e_transfer_entropy']`` used to resolve
        ``<method_key>.ete`` (surrogate parameters) and
        ``<method_key>.sparsity_fallback_threshold``.  **No method-specific
        string literals appear anywhere else in this function** — adding a
        3rd symbolic method requires zero changes here (Section 10.6's
        completion check).

    Returns
    -------
    matrices : dict
        ``{'raw', 'ete', 'delay', 'null_mean', 'null_std', 'p_value',
        'z_score'}`` — each an ``(n_rois, n_rois)`` ``np.ndarray`` (diagonal
        and invalid-ROI entries are 0 / NaN as appropriate).
    meta : dict
        ``{'raw_metadata': <metadata dict from raw_compute_fn>,
        'pair_meta': {(src_v, tgt_v): <compute_pair_ete result dict>},
        'n_valid': int, 'k0': int, 'method_key': str,
        'elapsed_raw_s': float, 'elapsed_ete_s': float}``.
    """
    te_cfg = config.get("stage_e_transfer_entropy", {})
    method_cfg = te_cfg.get(method_key, {})
    ete_params = method_cfg.get("ete", {})
    sparsity_threshold = float(method_cfg.get("sparsity_fallback_threshold", 0.95))
    seed_base = int(ete_params.get("random_seed_base", 42))

    t0 = time.perf_counter()
    EC_matrix, delay_matrix, metadata = raw_compute_fn(window_data, config, roi_names)
    elapsed_raw_s = time.perf_counter() - t0

    symbol_seqs = metadata.get("symbol_seqs", [])
    valid_indices = metadata.get("valid_indices", [])
    k0 = int(metadata.get("k0", 4))
    n_valid = len(symbol_seqs)

    # ------------------------------------------------------------------
    # Section 10.2 (Stage 2 fix) — resolve the per-pair PCA composite
    # conditioning contract. See module docstring for the full rationale
    # and the three possible outcomes (stored composite / real-pipeline
    # bivariate / raw_compute_fn predates this contract).
    # ------------------------------------------------------------------
    has_z_composite_contract = "z_composite_per_pair" in metadata
    z_composite_per_pair = metadata.get("z_composite_per_pair", {}) or {}
    stored_conditioning_hash = metadata.get("conditioning_config_hash")

    if stored_conditioning_hash is not None:
        from utils_te import compute_conditioning_config_hash  # noqa

        current_conditioning_hash = compute_conditioning_config_hash(
            config, method_key, k0
        )
        if current_conditioning_hash != stored_conditioning_hash:
            raise ValueError(
                f"compute_symbolic_ete_matrix [{method_key}]: stored "
                f"conditioning_config_hash={stored_conditioning_hash!r} does "
                f"not match the current run's hash="
                f"{current_conditioning_hash!r}. The z_composite_per_pair "
                f"composites in this window's metadata were built under a "
                f"different parameter set (filter/embedding/k0/PCA "
                f"settings) than the current config — reusing them would "
                f"silently invalidate the ETE surrogate null. Refusing to "
                f"proceed; recompute raw_compute_fn's metadata under the "
                f"current config instead of reusing a stale one."
            )

    ETE_matrix = np.zeros_like(EC_matrix)
    null_mean_matrix = np.full_like(EC_matrix, np.nan)
    null_std_matrix = np.full_like(EC_matrix, np.nan)
    p_value_matrix = np.full_like(EC_matrix, np.nan)
    z_score_matrix = np.full_like(EC_matrix, np.nan)
    pair_meta: dict = {}

    matrices = {
        "raw": EC_matrix,
        "ete": ETE_matrix,
        "delay": delay_matrix,
        "null_mean": null_mean_matrix,
        "null_std": null_std_matrix,
        "p_value": p_value_matrix,
        "z_score": z_score_matrix,
    }

    if n_valid < 2:
        logger.warning(
            "compute_symbolic_ete_matrix [%s]: fewer than 2 valid ROIs (%d) "
            "— returning zero/NaN ETE matrices.", method_key, n_valid,
        )
        return matrices, {
            "raw_metadata": metadata, "pair_meta": pair_meta,
            "n_valid": n_valid, "k0": k0, "method_key": method_key,
            "elapsed_raw_s": elapsed_raw_s, "elapsed_ete_s": 0.0,
        }

    t1 = time.perf_counter()
    te_profiles = metadata.get("te_profiles", {})

    for src_v in range(n_valid):
        for tgt_v in range(n_valid):
            if src_v == tgt_v:
                continue

            src_orig = valid_indices[src_v]
            tgt_orig = valid_indices[tgt_v]

            u_opt = int(delay_matrix[src_orig, tgt_orig])
            if u_opt < 1:
                # Degenerate/zero-connectivity pair — skip surrogate cost.
                continue

            S_x = symbol_seqs[src_v]
            S_y = symbol_seqs[tgt_v]

            # Section 10.2 (Stage 2 fix) — condition on the exact stored
            # PCA composite the real pipeline used for this pair, not on
            # the full raw per-ROI symbol set (which builds a mismatched,
            # higher-dimensional table that would trigger its own sparsity
            # fallback — see module docstring).
            z_composite = z_composite_per_pair.get((src_v, tgt_v))
            if z_composite is not None:
                S_z_list = [z_composite]
                z_composite_source = "stored_pca_composite"
            elif not has_z_composite_contract:
                z_composite_source = "raw_compute_fn_missing_key"
                S_z_list = []
            else:
                z_composite_source = "real_pipeline_bivariate"
                S_z_list = []

            if z_composite_source != "stored_pca_composite":
                logger.info(
                    "compute_symbolic_ete_matrix [%s]: pair (%d,%d) using "
                    "bivariate conditioning for ETE (z_composite_source=%s).",
                    method_key, src_v, tgt_v, z_composite_source,
                )

            # Deterministic per-(window, pair) seed so identical
            # (window, pair, seed_base) always reproduces the same null
            # distribution (Section 10.8 reproducibility smoke test).
            seed = seed_base + src_v * 1009 + tgt_v * 101

            result = compute_pair_ete(
                S_x, S_y, S_z_list, u_opt=u_opt, k0=k0,
                ete_params=ete_params, sparsity_threshold=sparsity_threshold,
                seed=seed,
            )

            ETE_matrix[src_orig, tgt_orig] = result["ete"]
            null_mean_matrix[src_orig, tgt_orig] = result["null_mean"]
            null_std_matrix[src_orig, tgt_orig] = result["null_std"]
            p_value_matrix[src_orig, tgt_orig] = result["p_value"]
            z_score_matrix[src_orig, tgt_orig] = result["z_score"]

            # Diagnostic: compare against the original multidelay-search
            # value at this same lag (see module docstring's PCA note).
            profile = te_profiles.get((src_v, tgt_v))
            raw_orig = None
            if profile is not None and 1 <= u_opt <= len(profile):
                raw_orig = float(np.asarray(profile)[u_opt - 1])
            pca_consistency_ratio = (
                result["raw_te"] / raw_orig
                if (raw_orig is not None and raw_orig > 1e-12)
                else float("nan")
            )

            pair_meta[(src_v, tgt_v)] = {
                **result,
                "raw_te_original_pipeline": raw_orig,
                "pca_consistency_ratio": pca_consistency_ratio,
                # Section 10.2 (Stage 2 fix) — distinct from
                # pca_consistency_ratio: which conditioning source ETE's
                # surrogate loop actually used for this pair/window.
                "z_composite_source": z_composite_source,
            }

    elapsed_ete_s = time.perf_counter() - t1

    meta = {
        "raw_metadata": metadata,
        "pair_meta": pair_meta,
        "n_valid": n_valid,
        "k0": k0,
        "method_key": method_key,
        "elapsed_raw_s": elapsed_raw_s,
        "elapsed_ete_s": elapsed_ete_s,
    }
    return matrices, meta


# ===========================================================================
# Section 10.8 — Smoke tests (run via  python3 11_symbolic_ete_core.py)
# ===========================================================================

def _get_phase4_raw_fn():
    """Lazy-load compute_amplitude_conditional_dste_matrix for smoke tests."""
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "amplitude_dste_for_ete_smoke",
        _STAGE_E_ROOT / "phase_4" / "14_amplitude_dste.py",
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_amplitude_conditional_dste_matrix


def _smoke_config(n_surrogates: int = 100, u_max: int = 8) -> dict:
    """Minimal config for ETE core smoke tests (no config.yaml needed)."""
    return {
        "stage_e_transfer_entropy": {
            "signal": {"sfreq": 25.0, "embedding_dimension": 2, "u_max": u_max},
            "symbolisation": {
                "ndis_threshold": 0.05,
                "k0_min": 2,
                "k0_max_bivariate": 20,
                "k0_max_multivariate": 2,
                "ng_max_iterations": 300,
                "ng_learning_rate_i": 0.5,
                "ng_learning_rate_f": 0.01,
                "ng_neighbourhood_i": 10.0,
                "ng_neighbourhood_f": 0.1,
            },
            "phase_2": {
                "sparsity_fallback_threshold": 0.95,
                "tau_divergence_ratio": 10.0,
                "degenerate_entropy_threshold": 0.5,
                "conditional_pca": {
                    "enabled": False, "n_components": 1,
                    "min_explained_var_warning_pct": 60,
                },
            },
            "phase_4": {
                "k0_max": 6,
                "ndis_threshold": 0.05,
                "envelope_normalize": True,
                "conditional_pca": {
                    "enabled": False, "n_components": 1,
                    "min_explained_var_warning_pct": 60,
                },
                "min_samples_per_cell": 5,
                "sparsity_fallback_threshold": 0.95,
                "ete": {
                    "n_surrogates": n_surrogates,
                    "surrogate_method": "circular_shift",
                    "min_shift_fraction": 0.05,
                    "fixed_lag_from_raw": True,
                    "floor_estimator": "mean",
                    "clip_negative": False,
                    "p_value_alternative": "greater",
                    "random_seed_base": 42,
                },
            },
        }
    }


def _run_smoke_tests() -> None:
    """
    Section 10.8 smoke tests.

    (a) Synthetic AR model with known ground-truth directionality (reuses
        Phase 4 Section 9.4 Test 5's AR model) — verify ETE preserves/
        sharpens correct-direction dominance after floor subtraction.
    (b) Independent noise-vs-noise pair (no coupling by construction) —
        verify ETE is small and p_value is not systematically < 0.05.
    (c) Reproducibility — same (window, pair, seed) -> identical null
        distribution.
    (d) Profile per-window wall-clock overhead of the full ETE matrix vs.
        raw-only, confirming O(n_surrogates) scaling.
    """
    import logging as _lg

    _lg.basicConfig(level=_lg.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    log = logging.getLogger("smoke_ete_core")
    log.setLevel(_lg.INFO)

    raw_fn = _get_phase4_raw_fn()
    n_fail = 0

    # ------------------------------------------------------------------
    # Test (a): AR model directionality — X(t-3) amplitude drives Y
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Test (a): AR model directionality — ETE(X->Y) should exceed ETE(Y->X)")
    try:
        N_AR = 1351
        rng2 = np.random.default_rng(seed=42)
        X_ar = np.zeros(N_AR + 100)
        eps_x = rng2.standard_normal(N_AR + 100) * 0.1
        eps_y = rng2.standard_normal(N_AR + 100) * 0.1
        for t in range(2, N_AR + 100):
            X_ar[t] = 0.4 * X_ar[t - 1] - 0.6 * X_ar[t - 2] + eps_x[t]
        Y_ar = np.zeros(N_AR + 100)
        for t in range(3, N_AR + 100):
            Y_ar[t] = 0.9 * abs(X_ar[t - 3]) + eps_y[t]
        X_ar = X_ar[100:]
        Y_ar = Y_ar[100:]
        Y_2roi = np.stack([X_ar, Y_ar], axis=0)

        cfg = _smoke_config(n_surrogates=100, u_max=8)
        matrices, meta = compute_symbolic_ete_matrix(
            raw_fn, Y_2roi, cfg, roi_names=["X", "Y"], method_key="phase_4",
        )
        ete_xy = matrices["ete"][0, 1]
        ete_yx = matrices["ete"][1, 0]
        raw_xy = matrices["raw"][0, 1]
        raw_yx = matrices["raw"][1, 0]
        p_xy = matrices["p_value"][0, 1]
        p_yx = matrices["p_value"][1, 0]

        log.info(
            "Test (a): raw(X->Y)=%.4f raw(Y->X)=%.4f | ETE(X->Y)=%.4f "
            "ETE(Y->X)=%.4f | p(X->Y)=%.3f p(Y->X)=%.3f",
            raw_xy, raw_yx, ete_xy, ete_yx, p_xy, p_yx,
        )
        direction_ok = ete_xy > ete_yx
        if direction_ok:
            log.info("Test (a) PASS \u2713 \u2014 ETE preserves correct directionality X->Y > Y->X")
        else:
            log.error(
                "Test (a) FAIL \u2717 \u2014 ETE(X->Y)=%.4f <= ETE(Y->X)=%.4f", ete_xy, ete_yx,
            )
            n_fail += 1
    except Exception as exc:
        log.error("Test (a) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    # ------------------------------------------------------------------
    # Test (b): independent noise-vs-noise — no coupling by construction
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Test (b): independent noise pair — ETE should be small, p not systematically < 0.05")
    try:
        rng3 = np.random.default_rng(seed=7)
        N_NOISE = 1351
        X_noise = rng3.standard_normal(N_NOISE)
        Y_noise = rng3.standard_normal(N_NOISE)   # independent of X_noise
        Y_2roi_noise = np.stack([X_noise, Y_noise], axis=0)

        cfg = _smoke_config(n_surrogates=100, u_max=8)
        matrices_n, meta_n = compute_symbolic_ete_matrix(
            raw_fn, Y_2roi_noise, cfg, roi_names=["X", "Y"], method_key="phase_4",
        )
        ete_xy_n = matrices_n["ete"][0, 1]
        ete_yx_n = matrices_n["ete"][1, 0]
        p_xy_n = matrices_n["p_value"][0, 1]
        p_yx_n = matrices_n["p_value"][1, 0]

        log.info(
            "Test (b): ETE(X->Y)=%.4f (p=%.3f)  ETE(Y->X)=%.4f (p=%.3f)",
            ete_xy_n, p_xy_n, ete_yx_n, p_yx_n,
        )
        # Neither p-value should be deep in the significant tail for pure noise.
        not_systematically_significant = (p_xy_n > 0.01) and (p_yx_n > 0.01)
        if not_systematically_significant:
            log.info("Test (b) PASS \u2713 \u2014 no spurious significance on independent noise")
        else:
            log.warning(
                "Test (b) WARN \u2014 p(X->Y)=%.3f p(Y->X)=%.3f (one below 0.01 on a "
                "single realisation can happen by chance ~1%% of the time; not "
                "a hard failure for a single-draw smoke test).",
                p_xy_n, p_yx_n,
            )
    except Exception as exc:
        log.error("Test (b) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    # ------------------------------------------------------------------
    # Test (c): reproducibility — same seed -> identical null distribution
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Test (c): reproducibility — same (S_x, S_y, S_z, u_opt, seed) -> identical null")
    try:
        rng4 = np.random.default_rng(seed=1)
        S_x_test = rng4.integers(0, 5, size=500)
        S_y_test = rng4.integers(0, 5, size=500)
        ete_params_test = {"n_surrogates": 50, "min_shift_fraction": 0.05}

        result_1 = compute_pair_ete(
            S_x_test, S_y_test, [], u_opt=3, k0=5,
            ete_params=ete_params_test, sparsity_threshold=0.95, seed=123,
        )
        result_2 = compute_pair_ete(
            S_x_test, S_y_test, [], u_opt=3, k0=5,
            ete_params=ete_params_test, sparsity_threshold=0.95, seed=123,
        )
        identical = np.array_equal(result_1["null_distribution"], result_2["null_distribution"])
        same_ete = result_1["ete"] == result_2["ete"]
        if identical and same_ete:
            log.info("Test (c) PASS \u2713 \u2014 identical null distribution and ETE across repeated calls")
        else:
            log.error("Test (c) FAIL \u2717 \u2014 repeated calls with the same seed diverged")
            n_fail += 1
    except Exception as exc:
        log.error("Test (c) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    # ------------------------------------------------------------------
    # Test (d): wall-clock scaling — O(n_surrogates)
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Test (d): wall-clock overhead scaling vs. n_surrogates")
    try:
        Y_2roi_scale = np.stack([X_ar, Y_ar], axis=0)  # reuse Test (a) AR data

        timings = {}
        for n_surr in (20, 100):
            cfg_s = _smoke_config(n_surrogates=n_surr, u_max=8)
            _, meta_s = compute_symbolic_ete_matrix(
                raw_fn, Y_2roi_scale, cfg_s, roi_names=["X", "Y"], method_key="phase_4",
            )
            timings[n_surr] = meta_s["elapsed_ete_s"]
            log.info(
                "Test (d): n_surrogates=%d  elapsed_raw_s=%.3f  elapsed_ete_s=%.3f",
                n_surr, meta_s["elapsed_raw_s"], meta_s["elapsed_ete_s"],
            )

        ratio_expected = 100 / 20
        ratio_observed = timings[100] / max(timings[20], 1e-9)
        log.info(
            "Test (d): observed elapsed_ete_s ratio (100/20 surrogates) = %.2f "
            "(expected ~%.1f for linear O(n_surrogates) scaling)",
            ratio_observed, ratio_expected,
        )
        # Loose bound — confirms roughly linear scaling, not e.g. quadratic
        # or constant (which would indicate the raw pipeline is being
        # re-run per surrogate instead of once per window).
        scaling_ok = 1.5 <= ratio_observed <= 8.0
        if scaling_ok:
            log.info("Test (d) PASS \u2713 \u2014 scaling consistent with O(n_surrogates)")
        else:
            log.warning(
                "Test (d) WARN \u2014 observed ratio %.2f outside the loose [1.5, 8.0] "
                "sanity band (may be timing noise on a very fast smoke-test "
                "workload — inspect manually on real data during 10.5's benchmark).",
                ratio_observed,
            )
    except Exception as exc:
        log.error("Test (d) FAIL \u2717 \u2014 %s", exc, exc_info=True)
        n_fail += 1

    log.info("=" * 60)
    if n_fail == 0:
        log.info("ALL ETE CORE SMOKE TESTS PASSED \u2713  (0 hard failures)")
    else:
        log.error("%d ETE CORE SMOKE TEST(S) FAILED \u2717", n_fail)
        sys.exit(1)


if __name__ == "__main__":
    _run_smoke_tests()
