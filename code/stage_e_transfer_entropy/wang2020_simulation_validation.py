"""
Wang (2020) Simulation Validation — Sections 3.4 Equations 26, 27, 28

Tests all three simulation models from Wang (2020) §3.4 to benchmark the
current bivariate dSPTE implementation before applying it to real fNIRS data.
Serves as the reference baseline for comparing how Phases 2 (multivariate
dSPTE) and 3 (amplitude dSPTE) improve on Phase 1.

─────────────────────────────────────────────────────────────────────────────
Model 1 — Linear AR (Eq. 26)  [EXPECTED: PASS]
    X(t) = 0.4·X(t−1) − 0.6·X(t−2) + ε_x
    Y(t) = a·X(t−δ) + ε_y
    Bivariate, linear coupling. Baseline directionality check.

Model 2 — Nonlinear AR (Eq. 27)  [EXPECTED: OPTIONAL PASS]
    X(t) = 0.4·X(t−1) + ε_x
    Y(t) = a·[X(t−δ)]³ + ε_y
    Bivariate, cubic nonlinear coupling. Tests whether phase-based TE
    captures nonlinear coupling; may pass or fail since dSPTE was designed
    for phase information, not amplitude-nonlinearity.

Model 3 — Multivariate Complex Hybrid (Eq. 28)  [EXPECTED: BENCHMARK FAIL]
    X(t) = 0.4·X(t−1) + ε_x
    Y(t) = 0.9·[X(t−5)]³ + ε_y
    Z(t) = 0.9·Y(t−9) + ε_z
    P(t) = 0.9·Y(t−3) + ε_p
    Four-channel system with cascade and indirect connections. Bivariate
    dSPTE cannot condition on the intermediate nodes, so indirect paths
    (e.g. X→Z via Y) will appear as spurious direct connections. This is
    the documented limitation motivating Phase 2 (multivariate dSPTE).
    Results are recorded as a Phase 2/3 improvement baseline, not a pass/fail.

─────────────────────────────────────────────────────────────────────────────
Scenario interpretation (Models 1 and 2):
  A — Both PASS: TE computation correct; real-data issues are signal-domain.
  B — Peak at wrong lag: embedding/phase extraction distorting lag detection.
  C — X→Y ≈ Y→X: Likely bug in compute_dspte_at_lag contingency table.

Multivariate benchmark (Model 3):
  Bivariate dSPTE will detect X→Y (u=5, nonlinear) ✓ and Y→Z (u=9) ✓ and
  Y→P (u=3) ✓, but will also spuriously detect X→Z and X→P (indirect).
  Phase 2 should suppress the indirect connections via multivariate conditioning.

Reference
─────────
Wang, Y., & Chen, W. (2020). Effective brain connectivity for fNIRS data
analysis based on multi-delays symbolic phase transfer entropy.
J. Neural Eng., 17(5), 056024. doi:10.1088/1741-2552/abb4a4  §3.4 Eqs. 26–28
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent           # code/stage_e_transfer_entropy/
_CODE_ROOT = _HERE.parent               # code/
_PROJECT_ROOT = _CODE_ROOT.parent       # repo root

for _p in [str(_HERE), str(_CODE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from utils_te import fit_neural_gas_codebook, quantise_to_symbols  # noqa
import importlib.util as _ilu

_core_spec = _ilu.spec_from_file_location("dspte_core", _HERE / "phase_1" / "01_dspte_core.py")
_core_mod = _ilu.module_from_spec(_core_spec)
_core_spec.loader.exec_module(_core_mod)

extract_phase_signal    = _core_mod.extract_phase_signal
compute_embedding_delay = _core_mod.compute_embedding_delay
uniform_embed           = _core_mod.uniform_embed
compute_dspte_at_lag    = _core_mod.compute_dspte_at_lag


# ===========================================================================
# Shared helpers
# ===========================================================================

def _symbolise_pair(X: np.ndarray, Y: np.ndarray, d: int, tau: int,
                    k0_max: int = 30, ndis_threshold: float = 0.05
                    ) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Embed, fit shared Neural-Gas codebook, return (S_x, S_y, k0, nDis)."""
    emb_x = uniform_embed(X, d=d, tau=tau)
    emb_y = uniform_embed(Y, d=d, tau=tau)
    T_prime = min(len(emb_x), len(emb_y))
    emb_x, emb_y = emb_x[:T_prime], emb_y[:T_prime]
    joint = np.vstack([emb_x, emb_y])
    codebook, k0, nDis = fit_neural_gas_codebook(
        joint, k0_max=k0_max, ndis_threshold=ndis_threshold
    )
    S_x = quantise_to_symbols(emb_x, codebook)
    S_y = quantise_to_symbols(emb_y, codebook)
    return S_x, S_y, k0, nDis


def _compute_te_profile(S_src: np.ndarray, S_tgt: np.ndarray,
                        k0: int, u_max: int) -> np.ndarray:
    """Compute TE(src→tgt) for lags 1..u_max, return (u_max,) array."""
    profile = np.zeros(u_max)
    for u in range(1, u_max + 1):
        profile[u - 1] = compute_dspte_at_lag(S_src, S_tgt, u=u, k0=k0)
    return profile


def _tau_for_phase(signal: np.ndarray, tau_override: Optional[int]) -> int:
    """Return embedding delay: override if given, else ACF 1/e on phase."""
    if tau_override is not None:
        return tau_override
    theta = extract_phase_signal(signal)
    return compute_embedding_delay(theta)


def _print_pair_result(label: str, fwd: np.ndarray, rev: np.ndarray,
                       true_lag: Optional[int], u_max: int) -> dict:
    """Print directionality + peak check for one directed pair; return result dict."""
    opt_fwd = int(np.argmax(fwd)) + 1
    opt_rev = int(np.argmax(rev)) + 1
    pass_direction = fwd.max() > rev.max()
    pass_peak = (true_lag is not None) and (opt_fwd == true_lag)

    src, tgt = label.split("→")
    print(f"\n  Pair {src} → {tgt}  (true lag = {true_lag})")
    print(f"    TE({src}→{tgt}) max = {fwd.max():.5f}  at lag {opt_fwd}")
    print(f"    TE({tgt}→{src}) max = {rev.max():.5f}  at lag {opt_rev}")
    ratio = fwd.max() / rev.max() if rev.max() > 0 else float("inf")
    print(f"    Directionality ratio = {ratio:.2f}×  "
          f"({'PASS ✓' if pass_direction else 'FAIL ✗'})")
    if true_lag is not None:
        print(f"    Peak lag detection: opt={opt_fwd}, expected={true_lag}  "
              f"({'PASS ✓' if pass_peak else 'FAIL ✗' if abs(opt_fwd - true_lag) > 1 else 'NEAR (±1) ~'})")
    return {
        "fwd_profile": fwd, "rev_profile": rev,
        "opt_fwd": opt_fwd, "opt_rev": opt_rev,
        "pass_direction": pass_direction,
        "pass_peak": pass_peak,
        "ratio": ratio,
    }


# ===========================================================================
# Model 1 — Linear AR (Eq. 26)
# ===========================================================================

def run_linear_ar(
    N: int = 20_000,
    delta: int = 3,
    a: float = 0.9,
    u_max: int = 15,
    d: int = 3,
    tau_override: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Wang (2020) Eq. 26 — Linear AR model.

    X(t) = 0.4·X(t−1) − 0.6·X(t−2) + ε_x
    Y(t) = a·X(t−δ) + ε_y

    Parameters
    ----------
    N : int       Number of samples (paper uses 20,000).
    delta : int   True causal delay X→Y (paper default 3).
    a : float     Coupling strength (paper default 0.9).
    u_max : int   Max lag to search.
    d : int       Embedding dimension (default 3).
    tau_override : int or None
        Skip ACF computation and use this τ directly (sensitivity check).
    verbose : bool

    Returns
    -------
    dict with keys: te_xy_vals, te_yx_vals, opt_xy, opt_yx, tau_used, k0,
                    nDis, pass_peak, pass_direction, pass_max_direction.
    """
    rng = np.random.default_rng(42)

    X = np.zeros(N)
    Y = np.zeros(N)
    for t in range(2, N):
        X[t] = 0.4 * X[t - 1] - 0.6 * X[t - 2] + rng.standard_normal()
    for t in range(delta, N):
        Y[t] = a * X[t - delta] + rng.standard_normal()

    if verbose:
        print(f"\n{'='*64}")
        print("Model 1 — Linear AR (Wang 2020 Eq. 26)")
        print(f"{'='*64}")
        print(f"  X(t) = 0.4·X(t−1) − 0.6·X(t−2) + ε_x")
        print(f"  Y(t) = {a}·X(t−{delta}) + ε_y")
        print(f"  N={N:,}  δ={delta}  a={a}  d={d}  u_max={u_max}")

    theta_x = extract_phase_signal(X)
    theta_y = extract_phase_signal(Y)

    tau = tau_override if tau_override is not None else max(
        compute_embedding_delay(theta_x),
        compute_embedding_delay(theta_y),
        1,
    )
    if verbose:
        print(f"\n[Embedding] τ={'override=' + str(tau_override) if tau_override else tau}")

    S_x, S_y, k0, nDis = _symbolise_pair(theta_x, theta_y, d=d, tau=tau)
    if verbose:
        print(f"[Neural-Gas] k₀={k0}  nDis={nDis:.4f}  "
              f"({'CONVERGED' if nDis < 0.05 else 'AT k0_max'})")

    te_xy = _compute_te_profile(S_x, S_y, k0, u_max)
    te_yx = _compute_te_profile(S_y, S_x, k0, u_max)

    opt_xy = int(np.argmax(te_xy)) + 1
    opt_yx = int(np.argmax(te_yx)) + 1
    pass_peak = (opt_xy == delta)
    pass_direction = te_xy[delta - 1] > te_yx[delta - 1]
    pass_max_direction = te_xy.max() > te_yx.max()

    if verbose:
        r = _print_pair_result("X→Y", te_xy, te_yx, true_lag=delta, u_max=u_max)
        print(f"\n  TE(X→Y) profile: {np.round(te_xy, 5)}")
        print(f"  TE(Y→X) profile: {np.round(te_yx, 5)}")
        print(f"\n  At true lag δ={delta}:")
        print(f"    X→Y = {te_xy[delta-1]:.5f}  Y→X = {te_yx[delta-1]:.5f}  "
              f"→ direction {'PASS ✓' if pass_direction else 'FAIL ✗'}")
        print(f"\n  Scenario: ", end="")
        if pass_peak and pass_direction:
            print("A — BOTH PASS ✓")
        elif pass_direction and not pass_peak:
            print(f"B — Direction correct, peak at u={opt_xy} ≠ δ={delta}")
        else:
            print("C — X→Y ≈ Y→X — check compute_dspte_at_lag indexing")

    return {
        "model": "linear_ar",
        "te_xy_vals": te_xy, "te_yx_vals": te_yx,
        "opt_xy": opt_xy, "opt_yx": opt_yx,
        "tau_used": tau, "k0": k0, "nDis": nDis,
        "pass_peak": pass_peak,
        "pass_direction": pass_direction,
        "pass_max_direction": pass_max_direction,
    }


# ===========================================================================
# Model 2 — Nonlinear AR (Eq. 27)
# ===========================================================================

def run_nonlinear_ar(
    N: int = 20_000,
    delta: int = 3,
    a: float = 0.9,
    u_max: int = 15,
    d: int = 3,
    tau_override: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Wang (2020) Eq. 27 — Nonlinear AR model.

    X(t) = 0.4·X(t−1) + ε_x
    Y(t) = a·[X(t−δ)]³ + ε_y

    Cubic nonlinear coupling. Phase-based TE may partially capture this
    (phase of a cubic-transformed signal carries nonlinear timing info)
    but performance is expected to be weaker than the linear case.
    A directionality PASS is desirable but not required for Phase 1 validation.

    Parameters mirror run_linear_ar().
    """
    rng = np.random.default_rng(42)

    X = np.zeros(N)
    Y = np.zeros(N)
    for t in range(1, N):
        X[t] = 0.4 * X[t - 1] + rng.standard_normal()
    for t in range(delta, N):
        Y[t] = a * X[t - delta] ** 3 + rng.standard_normal()

    if verbose:
        print(f"\n{'='*64}")
        print("Model 2 — Nonlinear AR (Wang 2020 Eq. 27)")
        print(f"{'='*64}")
        print(f"  X(t) = 0.4·X(t−1) + ε_x")
        print(f"  Y(t) = {a}·[X(t−{delta})]³ + ε_y")
        print(f"  N={N:,}  δ={delta}  a={a}  d={d}  u_max={u_max}")

    theta_x = extract_phase_signal(X)
    theta_y = extract_phase_signal(Y)

    tau = tau_override if tau_override is not None else max(
        compute_embedding_delay(theta_x),
        compute_embedding_delay(theta_y),
        1,
    )
    if verbose:
        print(f"\n[Embedding] τ={'override=' + str(tau_override) if tau_override else tau}")

    S_x, S_y, k0, nDis = _symbolise_pair(theta_x, theta_y, d=d, tau=tau)
    if verbose:
        print(f"[Neural-Gas] k₀={k0}  nDis={nDis:.4f}  "
              f"({'CONVERGED' if nDis < 0.05 else 'AT k0_max'})")

    te_xy = _compute_te_profile(S_x, S_y, k0, u_max)
    te_yx = _compute_te_profile(S_y, S_x, k0, u_max)

    opt_xy = int(np.argmax(te_xy)) + 1
    opt_yx = int(np.argmax(te_yx)) + 1
    pass_direction = te_xy.max() > te_yx.max()
    pass_peak = (opt_xy == delta)

    if verbose:
        _print_pair_result("X→Y", te_xy, te_yx, true_lag=delta, u_max=u_max)
        print(f"\n  TE(X→Y) profile: {np.round(te_xy, 5)}")
        print(f"  TE(Y→X) profile: {np.round(te_yx, 5)}")
        print(f"\n  Note: PASS is desirable but OPTIONAL for Phase 1 (bivariate phase TE).")
        print(f"  Phase 3 (amplitude dSPTE) is expected to perform better on cubic coupling.")

    return {
        "model": "nonlinear_ar",
        "te_xy_vals": te_xy, "te_yx_vals": te_yx,
        "opt_xy": opt_xy, "opt_yx": opt_yx,
        "tau_used": tau, "k0": k0, "nDis": nDis,
        "pass_direction": pass_direction,
        "pass_peak": pass_peak,
        "pass_max_direction": pass_direction,
    }


# ===========================================================================
# Model 3 — Multivariate Complex Hybrid (Eq. 28)
# ===========================================================================

# True directed connectivity for Model 3 (from Eq. 28):
#   X → Y  (delay 5, nonlinear cubic)
#   Y → Z  (delay 9, linear)
#   Y → P  (delay 3, linear)
#   X, Z, P have no direct connections to each other
#   Indirect: X → Z via Y (delay 5+9=14), X → P via Y (delay 5+3=8)
#
# Expected bivariate dSPTE behaviour:
#   X→Y: DETECT (direct, delay 5)     Y→Z: DETECT (direct, delay 9)
#   Y→P: DETECT (direct, delay 3)     X→Z: SPURIOUS (indirect via Y)
#   X→P: SPURIOUS (indirect via Y)    Z→P / P→Z: near-zero (no coupling)
#   All reverse directions: near-zero

_MODEL3_TRUE_CONNECTIONS = {
    ("X", "Y"): {"delay": 5,  "type": "direct_nonlinear", "expected": "DETECT"},
    ("Y", "Z"): {"delay": 9,  "type": "direct_linear",    "expected": "DETECT"},
    ("Y", "P"): {"delay": 3,  "type": "direct_linear",    "expected": "DETECT"},
    ("X", "Z"): {"delay": 14, "type": "indirect",         "expected": "SPURIOUS"},
    ("X", "P"): {"delay": 8,  "type": "indirect",         "expected": "SPURIOUS"},
    ("Z", "P"): {"delay": None, "type": "none",           "expected": "ZERO"},
}


def run_multivariate_hybrid(
    N: int = 20_000,
    u_max: int = 15,
    d: int = 3,
    tau_override: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Wang (2020) Eq. 28 — Multivariate complex hybrid model.

    X(t) = 0.4·X(t−1) + ε_x
    Y(t) = 0.9·[X(t−5)]³ + ε_y
    Z(t) = 0.9·Y(t−9) + ε_z
    P(t) = 0.9·Y(t−3) + ε_p

    This is a BENCHMARK test — bivariate dSPTE is expected to detect the
    direct connections (X→Y, Y→Z, Y→P) but will also produce spurious
    indirect connections (X→Z, X→P) because it cannot condition on Y.
    Results are stored for Phase 2 comparison, not treated as pass/fail.

    True connectivity graph:
        X --[5, cubic]--> Y --[9]--> Z
                          |
                          +-[3]--> P
    Indirect (spurious for bivariate TE):
        X --[14]--> Z  (via Y)
        X --[8] --> P  (via Y)

    Parameters
    ----------
    N : int         Number of samples (paper uses 20,000).
    u_max : int     Max lag to search (should be ≥ 14 to cover all indirect delays).
    d : int         Embedding dimension.
    tau_override : int or None
    verbose : bool

    Returns
    -------
    dict with per-pair TE profiles and the benchmark summary table.
    """
    rng = np.random.default_rng(42)

    Xv = np.zeros(N)
    Yv = np.zeros(N)
    Zv = np.zeros(N)
    Pv = np.zeros(N)

    for t in range(1, N):
        Xv[t] = 0.4 * Xv[t - 1] + rng.standard_normal()
    for t in range(5, N):
        Yv[t] = 0.9 * Xv[t - 5] ** 3 + rng.standard_normal()
    for t in range(9, N):
        Zv[t] = 0.9 * Yv[t - 9] + rng.standard_normal()
    for t in range(3, N):
        Pv[t] = 0.9 * Yv[t - 3] + rng.standard_normal()

    signals = {"X": Xv, "Y": Yv, "Z": Zv, "P": Pv}
    channel_names = list(signals.keys())

    if verbose:
        print(f"\n{'='*64}")
        print("Model 3 — Multivariate Complex Hybrid (Wang 2020 Eq. 28)")
        print(f"{'='*64}")
        print("  X(t) = 0.4·X(t−1) + ε_x")
        print("  Y(t) = 0.9·[X(t−5)]³ + ε_y")
        print("  Z(t) = 0.9·Y(t−9) + ε_z")
        print("  P(t) = 0.9·Y(t−3) + ε_p")
        print(f"  N={N:,}  u_max={u_max}  d={d}")
        print()
        print("  True graph:  X→Y(5,cubic)  Y→Z(9)  Y→P(3)")
        print("  Expected bivariate failures:")
        print("    X→Z spurious (indirect via Y, apparent delay ~14)")
        print("    X→P spurious (indirect via Y, apparent delay ~8)")
        print("  [This is a Phase 2 benchmark, not a Phase 1 pass/fail]")

    # Extract phases and compute τ per channel
    phases = {ch: extract_phase_signal(sig) for ch, sig in signals.items()}

    if tau_override is not None:
        tau = tau_override
    else:
        taus = {ch: compute_embedding_delay(ph) for ch, ph in phases.items()}
        tau = max(taus.values(), default=1)
        if verbose:
            print(f"\n[Embedding] τ per channel: {taus}  → shared τ={tau}")

    # Compute bivariate TE for all ordered pairs
    pair_results: dict = {}
    all_pairs = [
        (src, tgt)
        for src in channel_names
        for tgt in channel_names
        if src != tgt
    ]

    if verbose:
        print(f"\n[Computing bivariate dSPTE for all {len(all_pairs)} directed pairs]")

    for src, tgt in all_pairs:
        S_src, S_tgt, k0, nDis = _symbolise_pair(
            phases[src], phases[tgt], d=d, tau=tau
        )
        fwd = _compute_te_profile(S_src, S_tgt, k0, u_max)
        rev = _compute_te_profile(S_tgt, S_src, k0, u_max)
        pair_results[(src, tgt)] = {
            "fwd_profile": fwd,  # TE(src→tgt)
            "rev_profile": rev,  # TE(tgt→src)
            "max_fwd": float(fwd.max()),
            "max_rev": float(rev.max()),
            "opt_fwd": int(np.argmax(fwd)) + 1,
            "opt_rev": int(np.argmax(rev)) + 1,
            "k0": k0, "nDis": float(nDis),
        }

    if verbose:
        print(f"\n{'─'*64}")
        print("  Benchmark Results (bivariate Phase 1)")
        print(f"  {'Pair':<8} {'max TE':>9} {'det.lag':>8} {'TrueConn':>12}  {'ExpectedBivar':>14}  Outcome")
        print(f"  {'─'*7} {'─'*9} {'─'*8} {'─'*12}  {'─'*14}  {'─'*12}")

        for src, tgt in all_pairs:
            r = pair_results[(src, tgt)]
            conn_info = _MODEL3_TRUE_CONNECTIONS.get((src, tgt),
                         _MODEL3_TRUE_CONNECTIONS.get((tgt, src), {}))
            true_delay = conn_info.get("delay", "—")
            conn_type  = conn_info.get("type", "none") if (src, tgt) in _MODEL3_TRUE_CONNECTIONS else "none"
            expected   = conn_info.get("expected", "ZERO") if (src, tgt) in _MODEL3_TRUE_CONNECTIONS else "ZERO"

            # Simple outcome assessment
            detected = r["max_fwd"] > 0.001  # rough threshold
            if expected == "DETECT":
                outcome = "DETECT ✓" if detected else "MISS ✗"
            elif expected == "SPURIOUS":
                outcome = "SPURIOUS (expected)" if detected else "SUPPRESSED ✓"
            else:
                outcome = "ZERO ✓" if not detected else "FALSE POS ✗"

            print(f"  {src}→{tgt:<4}  {r['max_fwd']:>9.5f} {r['opt_fwd']:>8d} "
                  f"{str(true_delay):>12}  {expected:>14}  {outcome}")

    # Summary table for Phase 2 comparison
    benchmark_summary = {}
    for src, tgt in all_pairs:
        r = pair_results[(src, tgt)]
        conn_key = f"{src}→{tgt}"
        expected = (_MODEL3_TRUE_CONNECTIONS.get((src, tgt), {}).get("expected", "ZERO")
                    if (src, tgt) in _MODEL3_TRUE_CONNECTIONS else "ZERO")
        benchmark_summary[conn_key] = {
            "max_te": round(r["max_fwd"], 6),
            "detected_lag": r["opt_fwd"],
            "true_delay": _MODEL3_TRUE_CONNECTIONS.get((src, tgt), {}).get("delay"),
            "expected_bivariate": expected,
            "connection_type": _MODEL3_TRUE_CONNECTIONS.get((src, tgt), {}).get("type", "none"),
        }

    if verbose:
        print(f"\n  Note: Spurious detections for X→Z and X→P are expected and")
        print(f"  document the limitation of bivariate TE on cascade networks.")
        print(f"  Phase 2 (multivariate dSPTE) should suppress these by conditioning")
        print(f"  on Y when computing X→Z and X→P.")

    return {
        "model": "multivariate_hybrid",
        "pair_results": pair_results,
        "benchmark_summary": benchmark_summary,
        "tau_used": tau,
    }


# ===========================================================================
# Top-level runner
# ===========================================================================

def run_all_simulations(
    N: int = 20_000,
    u_max_bivariate: int = 15,
    u_max_multivariate: int = 20,
    d: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Run all three Wang (2020) simulations and print a consolidated summary.

    Parameters
    ----------
    N : int
        Samples per simulation (paper uses 20,000).
    u_max_bivariate : int
        Max lag for Models 1 and 2 (default 15; covers true delay δ=3 with margin).
    u_max_multivariate : int
        Max lag for Model 3 (default 20; covers indirect delay ~14 with margin).
    d : int
        Embedding dimension (default 3, matching pipeline).
    verbose : bool

    Returns
    -------
    dict with keys 'linear', 'nonlinear', 'multivariate'.
    """
    if verbose:
        print("\n" + "=" * 64)
        print("Wang (2020) Simulation Validation — All Three Models")
        print("=" * 64)

    r1 = run_linear_ar(N=N, u_max=u_max_bivariate, d=d, verbose=verbose)
    r2 = run_nonlinear_ar(N=N, u_max=u_max_bivariate, d=d, verbose=verbose)
    r3 = run_multivariate_hybrid(N=N, u_max=u_max_multivariate, d=d, verbose=verbose)

    if verbose:
        print(f"\n{'='*64}")
        print("CONSOLIDATED SUMMARY")
        print(f"{'='*64}")
        print(f"{'Model':<36} {'Direction':>12} {'Peak lag':>10} {'Status'}")
        print(f"{'─'*35} {'─'*12} {'─'*10} {'─'*20}")

        # Model 1
        s1 = "PASS ✓" if r1["pass_direction"] and r1["pass_peak"] else (
             "DIR PASS / PEAK FAIL" if r1["pass_direction"] else "FAIL ✗")
        print(f"{'1. Linear AR (Eq. 26)':<36} "
              f"{'PASS ✓' if r1['pass_direction'] else 'FAIL ✗':>12} "
              f"{'PASS ✓' if r1['pass_peak'] else 'FAIL ✗':>10}  {s1}")

        # Model 2
        s2_dir = "PASS ✓" if r2["pass_direction"] else "FAIL (optional)"
        s2_peak = "PASS ✓" if r2["pass_peak"] else "FAIL (optional)"
        s2 = "PASS ✓" if r2["pass_direction"] and r2["pass_peak"] else "PARTIAL (optional)"
        print(f"{'2. Nonlinear AR (Eq. 27) [optional]':<36} "
              f"{s2_dir:>12} {s2_peak:>10}  {s2}")

        # Model 3
        direct_detected = sum(
            1 for (s, t), info in _MODEL3_TRUE_CONNECTIONS.items()
            if info["expected"] == "DETECT"
            and r3["pair_results"].get((s, t), {}).get("max_fwd", 0) > 0.001
        )
        spurious_present = sum(
            1 for (s, t), info in _MODEL3_TRUE_CONNECTIONS.items()
            if info["expected"] == "SPURIOUS"
            and r3["pair_results"].get((s, t), {}).get("max_fwd", 0) > 0.001
        )
        print(f"{'3. Multivariate Hybrid (Eq. 28) [benchmark]':<36} "
              f"{'—':>12} {'—':>10}  "
              f"Direct: {direct_detected}/3 DETECT, "
              f"Indirect: {spurious_present}/2 spurious")

        print(f"\nPhase 2 target: suppress the {spurious_present} spurious indirect connection(s)")
        print(f"Phase 3 target: improve nonlinear AR detection (Model 2) via amplitude branch")

    return {"linear": r1, "nonlinear": r2, "multivariate": r3}


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Wang (2020) dSPTE simulation validation (Eqs. 26–28)."
    )
    p.add_argument("--model", choices=["1", "2", "3", "all"], default="all",
                   help="Which model to run (default: all).")
    p.add_argument("--N", type=int, default=20_000,
                   help="Number of samples (default 20,000).")
    p.add_argument("--u-max", type=int, default=15,
                   help="Max lag for Models 1/2 (default 15).")
    p.add_argument("--u-max-mv", type=int, default=20,
                   help="Max lag for Model 3 (default 20).")
    p.add_argument("--d", type=int, default=3,
                   help="Embedding dimension (default 3).")
    p.add_argument("--tau", type=int, default=None,
                   help="Override embedding delay τ for all models.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-model verbose output.")
    args = p.parse_args()

    verbose = not args.quiet

    if args.model == "all":
        run_all_simulations(
            N=args.N,
            u_max_bivariate=args.u_max,
            u_max_multivariate=args.u_max_mv,
            d=args.d,
            verbose=verbose,
        )
    elif args.model == "1":
        r = run_linear_ar(N=args.N, u_max=args.u_max, d=args.d,
                          tau_override=args.tau, verbose=verbose)
        if not r["pass_peak"] and args.tau is None:
            print("\n[Sensitivity check: τ=1]")
            run_linear_ar(N=args.N, u_max=args.u_max, d=args.d,
                          tau_override=1, verbose=verbose)
    elif args.model == "2":
        run_nonlinear_ar(N=args.N, u_max=args.u_max, d=args.d,
                         tau_override=args.tau, verbose=verbose)
    elif args.model == "3":
        run_multivariate_hybrid(N=args.N, u_max=args.u_max_mv, d=args.d,
                                tau_override=args.tau, verbose=verbose)
