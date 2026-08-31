"""
Phase 4 Diagnostic Script — NG Convergence & Amplitude Embedding Space

DO NOT DELETE — Purpose / justification
----------------------------------------
Standalone diagnostic, not part of the numbered pipeline (14-18) and not
required by any downstream script. It exists to empirically verify, on real
subject data, the load-bearing claim behind the entire Phase 4 design
(and the reason the project pivoted off phase-based TE in Phases 1-3):
that the Hilbert AMPLITUDE envelope clusters non-uniformly in delay-embedding
space (so Neural-Gas symbolisation captures genuine structure), unlike
Hilbert PHASE, which is quasi-uniform and made the NG codebook fail to
converge (see CLAUDE.md Section 3 and the k0_max derivation comments in
config.yaml's `stage_e_transfer_entropy.phase_4` block).

Without this script, that claim exists only as a comment/assertion in
config.yaml and CLAUDE.md, with no regenerable evidence. Keep this file even
though it's not "production" — re-run it (`python3 phase_4_debug.py`) any
time k0_max, ng_max_iterations, or the amplitude extraction method change,
to confirm the convergence assumptions still hold.

Two tests:

  Test 1 — NG Convergence Curve
  -------
  Runs the Neural-Gas codebook search from k=2 to k=20 at four iteration
  counts (500, 1000, 2000, 5000) on one real subject window's amplitude
  embedding matrix.  Plots nDis vs k for each iteration count alongside
  the theoretical floor nDis_floor(k) = 0.5/k (d=2 uniform upper bound).

  Answers: "Is nDis failing because 500 iterations is too few (Reason 1),
  or because the data is genuinely near-uniform (Reason 2)?"

  - If the 5000-iteration curve dips below 0.05 somewhere between k=8-15
    → Reason 1 confirmed (NG not converged at 500 iters).
  - If even the 5000-iteration curve tracks near the theoretical floor and
    never crosses 0.05 for k ≤ 20  → Reason 2 (near-uniform amplitude data).

  Test 2 — Amplitude Embedding Space Visualisation
  -------
  Embeds each ROI's z-scored amplitude envelope into 2D delay space
  [A(t), A(t-τ)], stacks all ROIs into one global matrix (n_rois×T', 2),
  and scatter-plots the point cloud.  Overlays the k₀=4 NG codebook
  vectors as red crosses.

  Answers: "Does the amplitude embedding space cluster near the origin
  (non-uniform → NG should converge at low k), or spread uniformly
  (uniform → nDis floor governs)?"

Usage:
  python phase_4_debug.py              # uses default subject A1, trial Tr_1
  python phase_4_debug.py --sub R3     # different subject
  python phase_4_debug.py --sub A1 --trial Tr_2 --win 2
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors every other Stage E module
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent               # phase_4/
_STAGE_E = _HERE.parent                     # stage_e_transfer_entropy/
_CODE = _STAGE_E.parent                     # code/
_REPO = _CODE.parent                        # repo root

for _p in [str(_CODE), str(_STAGE_E)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Logging — enable DEBUG so fit_neural_gas_codebook prints every k
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("phase_4_debug")

# ---------------------------------------------------------------------------
# Lazy imports from Stage E infrastructure
# ---------------------------------------------------------------------------
import importlib.util as _ilu

def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_utils_te   = _load_module("utils_te",    _STAGE_E / "utils_te.py")
_amp_mod    = _load_module("amp_dste",   _HERE / "14_amplitude_dste.py")
_core_mod   = _load_module("bivariate_dspte_core",  _STAGE_E / "phase_1" / "01_bivariate_dspte_core.py")
_seg_mod    = _load_module("trial_seg",   _CODE / "stage_d_granger" / "02_trial_segmentation.py")

NeuralGas              = _utils_te.NeuralGas
fit_neural_gas_codebook = _utils_te.fit_neural_gas_codebook
extract_amplitude_matrix = _amp_mod.extract_amplitude_matrix
uniform_embed            = _core_mod.uniform_embed

from utils import load_config  # noqa: E402


# ===========================================================================
# Helper — load one window from a given subject / trial
# ===========================================================================

def _load_window(subject_id: str, trial_id: str, win_idx: int, config: dict):
    """
    Returns A_win (n_rois, T, z-scored amplitude) and taus for one window.

    Internally calls extract_amplitude_matrix on the raw Y_win slice so the
    returned A_win is exactly what Phase 4 would feed to symbolise_multivariate_amplitude.
    """
    # Load ROI signals from Stage D cache
    Y_roi, roi_names, sfreq = _utils_te.load_or_extract_roi_signals(
        subject_id, study="expert_novice", config=config
    )

    # Load trial segments
    raw_fif = _utils_te.load_preprocessed_fif(
        subject_id, study="expert_novice", config=config
    )
    segments = _seg_mod.get_trial_segments(raw_fif, config=config)

    win_cfg  = config.get("stage_e_transfer_entropy", {}).get("window", {})
    win_size = int(win_cfg.get("size_samples", 1350))
    stride   = int(win_cfg.get("stride_samples", 1350))

    trial_num = int(trial_id.replace("Tr_", "")) - 1
    if trial_num >= len(segments):
        raise ValueError(f"Trial {trial_id} not found (only {len(segments)} trials)")

    onset, offset = segments[trial_num]
    Y_trial = Y_roi[:, onset: offset + 1]

    windows = list(range(win_size, Y_trial.shape[1], stride))
    if win_idx >= len(windows):
        raise ValueError(
            f"Window index {win_idx} out of range ({len(windows)} windows in {trial_id})"
        )

    w  = windows[win_idx]
    ws = w - win_size
    Y_win = Y_trial[:, ws: w + 1]   # shape (n_rois, win_size+1)

    logger.info(
        "Loaded window: sub=%s  trial=%s  win_idx=%d  Y_win.shape=%s  "
        "roi_names=%s  sfreq=%.1f",
        subject_id, trial_id, win_idx, Y_win.shape, roi_names, sfreq,
    )

    # Extract amplitude (z-scored Hilbert envelope) — exactly as Phase 4 does
    A_win, taus = extract_amplitude_matrix(Y_win, config=config)
    logger.info("Amplitude extracted.  taus=%s", taus)

    return A_win, taus, roi_names


# ===========================================================================
# TEST 1 — Neural-Gas nDis convergence curve
# ===========================================================================

def test1_ng_convergence(
    A_win: np.ndarray,
    taus: list[int],
    config: dict,
    iteration_counts: tuple[int, ...] = (500, 1000, 2000, 5000),
    k_max: int = 20,
    out_path: Path | None = None,
):
    """
    Fit Neural-Gas from k=2 to k_max at each iteration count.
    Plots nDis vs k with theoretical floor nDis_floor(k) = 0.5/k.

    Parameters
    ----------
    A_win : (n_rois, T) z-scored amplitude matrix
    taus  : per-ROI embedding delays (tau_min is used for all)
    iteration_counts : list of max_iterations values to test
    k_max : maximum codebook size (x-axis limit)
    out_path : save figure here; shows interactively if None
    """
    te_cfg  = config.get("stage_e_transfer_entropy", {})
    sym_cfg = te_cfg.get("symbolisation", {})
    sig_cfg = te_cfg.get("signal", {})
    d       = int(sig_cfg.get("embedding_dimension", 2))

    tau_min = int(min(taus))
    n_rois, T = A_win.shape
    history   = (d - 1) * tau_min
    T_prime   = T - history

    logger.info(
        "TEST 1 — NG convergence.  n_rois=%d  d=%d  tau_min=%d  T'=%d  "
        "Global matrix rows=%d  k_max=%d",
        n_rois, d, tau_min, T_prime, n_rois * T_prime, k_max,
    )

    # Build global joint amplitude embedding matrix (exactly as Phase 4 does)
    embedded_list = []
    for i in range(n_rois):
        emb = uniform_embed(A_win[i], d=d, tau=tau_min)
        embedded_list.append(emb)
    global_matrix = np.vstack(embedded_list)   # (n_rois * T', d)
    logger.info("global_matrix shape: %s", global_matrix.shape)

    ng_lr_i = float(sym_cfg.get("ng_learning_rate_i", 0.5))
    ng_lr_f = float(sym_cfg.get("ng_learning_rate_f", 0.01))
    ng_nb_i = float(sym_cfg.get("ng_neighbourhood_i", 10.0))
    ng_nb_f = float(sym_cfg.get("ng_neighbourhood_f", 0.1))

    k_range   = list(range(2, k_max + 1))
    threshold = 0.05
    floor_vals = [0.5 / k for k in k_range]   # theoretical d=2 uniform floor

    results: dict[int, list[float]] = {}

    for n_iter in iteration_counts:
        ndis_vals = []
        t0 = time.perf_counter()
        for k in k_range:
            ng = NeuralGas(
                n_prototypes=k,
                max_iterations=n_iter,
                learning_rate_i=ng_lr_i,
                learning_rate_f=ng_lr_f,
                neighbourhood_i=ng_nb_i,
                neighbourhood_f=ng_nb_f,
                random_state=42,
            )
            ng.fit(global_matrix)
            ndis_vals.append(float(ng.distortion))
            logger.debug("iter=%d  k=%d  nDis=%.4f", n_iter, k, ng.distortion)
        elapsed = time.perf_counter() - t0
        logger.info(
            "iter=%d done in %.1f s  |  k where nDis<0.05: %s",
            n_iter, elapsed,
            [k_range[i] for i, v in enumerate(ndis_vals) if v < threshold] or "none",
        )
        results[n_iter] = ndis_vals

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    colors = ["#aaaaaa", "#6baed6", "#2171b5", "#08306b"]
    linewidths = [1.0, 1.2, 1.5, 2.0]
    alphas = [0.5, 0.7, 0.85, 1.0]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Theoretical floor
    ax.plot(k_range, floor_vals, "k--", lw=1.2, label="Theoretical floor (0.5/k, d=2 uniform)", zorder=1)

    # nDis per iteration count
    for col, lw, alpha, n_iter in zip(colors, linewidths, alphas, iteration_counts):
        ax.plot(
            k_range, results[n_iter],
            color=col, lw=lw, alpha=alpha,
            marker="o", markersize=4,
            label=f"NG {n_iter:,} iter",
            zorder=3,
        )

    # Threshold line
    ax.axhline(threshold, color="red", lw=1.2, ls=":", label="nDis threshold (0.05)", zorder=2)

    # Mark current k0_max in production
    ax.axvline(4, color="orange", lw=1.5, ls="--", alpha=0.7, label="k₀_max=4 (production)", zorder=2)

    ax.set_xlabel("Codebook size k", fontsize=12)
    ax.set_ylabel("Normalised Distortion (nDis)", fontsize=12)
    ax.set_title(
        f"Test 1 — NG nDis vs k at different iteration counts\n"
        f"(global_matrix shape: {global_matrix.shape[0]}×{global_matrix.shape[1]}, d={d}, τ_min={tau_min})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(k_range)
    ax.grid(True, alpha=0.3)

    # Annotate floor at 0.05 crossing
    floor_arr = np.array(floor_vals)
    cross_idx = np.where(floor_arr < threshold)[0]
    if len(cross_idx) > 0:
        k_cross = k_range[cross_idx[0]]
        ax.annotate(
            f"Floor crosses 0.05\nat k={k_cross}",
            xy=(k_cross, floor_arr[cross_idx[0]]),
            xytext=(k_cross + 1, threshold + 0.02),
            fontsize=8, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Test 1 figure saved → %s", out_path)
    else:
        plt.show()

    plt.close(fig)
    return results


# ===========================================================================
# TEST 2 — Amplitude embedding space scatter
# ===========================================================================

def test2_embedding_space(
    A_win: np.ndarray,
    taus: list[int],
    roi_names: list[str],
    config: dict,
    out_path: Path | None = None,
):
    """
    Scatter-plot the 2D delay embedding space for the amplitude data.

    Each ROI's embedded vectors [A(t), A(t-τ)] are plotted in a separate
    colour.  Overlay: the k₀=4 NG codebook prototypes (red ×) and the
    circle at radius √2 (the theoretical extent of a uniform unit-variance
    distribution in 2D).

    Interpretation:
      - Compact oval cluster near origin → non-uniform, NG should converge → Reason 1
      - Diffuse cloud filling the unit circle → near-uniform → Reason 2
    """
    te_cfg  = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    sym_cfg = te_cfg.get("symbolisation", {})
    d       = int(sig_cfg.get("embedding_dimension", 2))
    tau_min = int(min(taus))

    n_rois = A_win.shape[0]
    history = (d - 1) * tau_min

    embedded_list = []
    for i in range(n_rois):
        emb = uniform_embed(A_win[i], d=d, tau=tau_min)
        embedded_list.append(emb)

    global_matrix = np.vstack(embedded_list)

    logger.info(
        "TEST 2 — Embedding space.  n_rois=%d  d=%d  τ_min=%d  T'=%d  "
        "global shape=%s",
        n_rois, d, tau_min, A_win.shape[1] - history, global_matrix.shape,
    )

    # Compute nDis at k₀=4 (production setting) — for annotation
    ng4 = NeuralGas(
        n_prototypes=4,
        max_iterations=int(sym_cfg.get("ng_max_iterations", 500)),
        learning_rate_i=float(sym_cfg.get("ng_learning_rate_i", 0.5)),
        learning_rate_f=float(sym_cfg.get("ng_learning_rate_f", 0.01)),
        neighbourhood_i=float(sym_cfg.get("ng_neighbourhood_i", 10.0)),
        neighbourhood_f=float(sym_cfg.get("ng_neighbourhood_f", 0.1)),
        random_state=42,
    )
    ng4.fit(global_matrix)
    codebook4 = ng4.prototypes_          # shape (4, 2)
    ndis4 = float(ng4.distortion)
    logger.info("k=4 NG fitted: nDis=%.4f", ndis4)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    roi_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: all ROIs together (full global matrix)
    ax = axes[0]
    T_prime = A_win.shape[1] - history
    for i, (roi_name, emb) in enumerate(zip(roi_names, embedded_list)):
        ax.scatter(
            emb[:, 0], emb[:, 1],
            s=2, alpha=0.15, c=roi_colors[i % len(roi_colors)],
            label=roi_name,
            rasterized=True,
        )
    ax.scatter(
        codebook4[:, 0], codebook4[:, 1],
        s=120, c="black", marker="X", zorder=5, label=f"k₀=4 codebook\n(nDis={ndis4:.3f})",
    )

    # Theoretical extent circle for unit-variance uniform data (radius ≈ √2)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta),
            "k:", lw=0.8, alpha=0.4, label="Uniform 2D boundary (r=√2σ)")

    ax.set_xlabel("A(t)  [z-scored]", fontsize=10)
    ax.set_ylabel("A(t − τ)  [z-scored]", fontsize=10)
    ax.set_title(f"All ROIs overlaid (T'={T_prime}  τ_min={tau_min})", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Right panel: separate subplots per ROI (small multiples)
    ax2 = axes[1]
    ax2.remove()
    inner_axes = fig.add_axes([0.57, 0.08, 0.41, 0.84])
    inner_axes.set_visible(False)

    n_cols = 3
    n_rows = int(np.ceil(n_rois / n_cols))
    sub_fig_w = 0.41 / n_cols
    sub_fig_h = 0.84 / n_rows
    x0 = 0.57

    for i, (roi_name, emb) in enumerate(zip(roi_names, embedded_list)):
        row = i // n_cols
        col = i % n_cols
        left   = x0 + col * sub_fig_w + 0.005
        bottom = 0.84 - (row + 1) * sub_fig_h + 0.06
        sub_ax = fig.add_axes([left, bottom, sub_fig_w - 0.01, sub_fig_h - 0.04])
        sub_ax.scatter(emb[:, 0], emb[:, 1], s=1, alpha=0.1,
                       c=roi_colors[i % len(roi_colors)], rasterized=True)
        sub_ax.scatter(codebook4[:, 0], codebook4[:, 1], s=40,
                       c="black", marker="X", zorder=5)
        sub_ax.set_title(roi_name, fontsize=7, pad=2)
        sub_ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
        sub_ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
        sub_ax.set_xlim(-3.5, 3.5)
        sub_ax.set_ylim(-3.5, 3.5)
        sub_ax.set_aspect("equal")
        sub_ax.tick_params(labelsize=5)

    fig.suptitle(
        "Test 2 — Phase 4 Amplitude Embedding Space  [A(t) vs A(t−τ), z-scored]\n"
        "Compact cluster → non-uniform (NG should converge at low k)  |  "
        "Diffuse cloud → near-uniform (nDis floor governs)",
        fontsize=9, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Test 2 figure saved → %s", out_path)
    else:
        plt.show()

    plt.close(fig)

    # Summary statistics
    var_ratio = float(np.var(global_matrix) / 1.0)   # expected =1 for z-scored
    cov = np.cov(global_matrix.T)
    off_diag = float(cov[0, 1])
    logger.info(
        "Embedding summary: global var=%.3f (expect ≈1.0 for z-scored)  "
        "Cov(A(t), A(t-τ))=%.3f  (positive → temporal autocorrelation present)",
        var_ratio, off_diag,
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Diagnostic Tests")
    parser.add_argument("--sub",   default="A1",  help="Subject ID (default: A1)")
    parser.add_argument("--trial", default="Tr_1", help="Trial ID (default: Tr_1)")
    parser.add_argument("--win",   default=0, type=int, help="Window index 0-based (default: 0)")
    parser.add_argument("--k_max", default=20, type=int, help="Max codebook size (default: 20)")
    args = parser.parse_args()

    config = load_config()

    out_dir = (
        _REPO / "outputs" / "stage_e_transfer_entropy" / "expert-novice"
        / "phase_4_amplitude_dste" / "visualizations"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== Phase 4 Debug  sub=%s  trial=%s  win=%d  k_max=%d ===",
        args.sub, args.trial, args.win, args.k_max,
    )

    # Load one window's amplitude matrix
    A_win, taus, roi_names = _load_window(
        args.sub, args.trial, args.win, config
    )

    # -----------------------------------------------------------------------
    # Test 1 — NG convergence curve
    # -----------------------------------------------------------------------
    logger.info("--- Running Test 1: NG convergence curve ---")
    t1_out = out_dir / f"p4_debug_test1_ng_convergence_{args.sub}_{args.trial}_w{args.win}.png"
    test1_ng_convergence(
        A_win, taus, config,
        iteration_counts=(500, 1000, 2000, 5000),
        k_max=args.k_max,
        out_path=t1_out,
    )

    # -----------------------------------------------------------------------
    # Test 2 — Embedding space scatter
    # -----------------------------------------------------------------------
    logger.info("--- Running Test 2: Embedding space ---")
    t2_out = out_dir / f"p4_debug_test2_embedding_{args.sub}_{args.trial}_w{args.win}.png"
    test2_embedding_space(
        A_win, taus, list(roi_names), config,
        out_path=t2_out,
    )

    logger.info("=== Done.  Figures written to %s ===", out_dir)


if __name__ == "__main__":
    main()
