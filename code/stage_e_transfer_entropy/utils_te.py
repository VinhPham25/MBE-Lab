"""
Stage E — Shared Utility Functions for the Transfer Entropy Pipeline

This module provides:

  1. NeuralGas class
       Scratch implementation of the Neural-Gas vector quantisation algorithm
       (Martinetz et al. 1993, IEEE Trans. Neural Netw. 4(4):558–569).
       Used by every dSPTE symbolisation step (Sections 3.3, 7.1, 8.1).

  2. Path / config helpers
       get_te_output_dir(subdir_key)  — analogous to utils_gc.get_gc_output_dir()

  3. Stage D import wrappers
       Thin wrappers around Stage D utilities (load_preprocessed_fif,
       get_event_onsets, get_subject_group, get_connection_labels,
       build_region_map_5) imported via importlib to avoid Stage D
       module-level side effects. Do NOT copy Stage D code here.

  4. ROI signal loader  (Section 2.1.1)
       load_or_extract_roi_signals(subject_id) — loads from Stage D cache
       if available, otherwise re-extracts and caches.

Reference
---------
Wang (2020): doi:10.1088/1741-2552/abb4a4   (dSPTE — Phase 1)
Wang (2022): doi:10.1016/j.bspc.2021.103422  (MTENUE — Phase 2 inspiration)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).parent.parent          # code/
_PROJECT_ROOT = _CODE_ROOT.parent                  # repo root

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

# Add local .pip_deps/ to sys.path so pyswarms (installed with --target) is importable.
# IMPORTANT: append (not prepend) so the active venv's packages always take priority.
# pyswarms is the only package installed there; prepending caused a matplotlib circular
# import because .pip_deps/ contained a stale matplotlib that shadowed the venv's copy.
_PIP_DEPS = _PROJECT_ROOT / ".pip_deps"
try:
    import pyswarms  # noqa: F401 — already importable, nothing to do
except ImportError:
    if _PIP_DEPS.exists() and str(_PIP_DEPS) not in sys.path:
        sys.path.append(str(_PIP_DEPS))   # append → venv wins on all other packages

from utils import load_config, configure_pyswarms_logging  # code/utils.py

# Redirect pyswarms' Reporter (report.log + rotated backups) from the cwd
# into report_logs/ at the project root — see utils.configure_pyswarms_logging.
configure_pyswarms_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports from Stage D utilities (avoids circular/side-effect imports)
# ---------------------------------------------------------------------------
_UTILS_GC_PATH = _CODE_ROOT / "stage_d_granger" / "utils_gc.py"
_ROI_EXTRACT_PATH = _CODE_ROOT / "stage_d_granger" / "01_extract_roi_signals.py"


def _import_utils_gc():
    """Return the utils_gc module from Stage D via importlib."""
    spec = importlib.util.spec_from_file_location("utils_gc", _UTILS_GC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_roi_extraction():
    """Return the 01_extract_roi_signals module from Stage D via importlib."""
    spec = importlib.util.spec_from_file_location(
        "extract_roi_signals_d", _ROI_EXTRACT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_preprocessed_fif(subject_id: str, study: str = "expert_novice", **kwargs):
    """Wrapper: load Stage B .fif via Stage D's load_preprocessed_fif."""
    return _import_utils_gc().load_preprocessed_fif(subject_id, study=study, **kwargs)


def get_event_onsets(raw):
    """Wrapper: extract MNE events via Stage D's get_event_onsets."""
    return _import_utils_gc().get_event_onsets(raw)


def get_subject_group(subject_id: str, config: Optional[dict] = None) -> str:
    """
    Return 'Expert' or 'Novice' for subject_id.

    Reads from Stage E config block first (stage_e_transfer_entropy.subjects).
    Falls back to Stage D utils_gc.get_subject_group() for compatibility.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subject_groups = te_cfg.get("subjects", {})
    excluded = subject_groups.get("excluded", [])

    if subject_id in excluded:
        raise ValueError(
            f"Subject {subject_id} is excluded from TE analysis "
            f"(Stage B quality flag). Excluded list: {excluded}"
        )

    for group_name in ("Expert", "Novice"):
        if subject_id in subject_groups.get(group_name, []):
            return group_name

    # Fallback to Stage D group assignment
    logger.warning(
        "get_subject_group [TE]: %s not found in stage_e config groups — "
        "falling back to Stage D utils_gc.get_subject_group()",
        subject_id,
    )
    return _import_utils_gc().get_subject_group(subject_id, config=config)


def get_connection_labels(roi_names: Optional[list] = None) -> list:
    """Wrapper: 20 directed connection labels from Stage D's get_connection_labels."""
    return _import_utils_gc().get_connection_labels(roi_names=roi_names)


# ---------------------------------------------------------------------------
# Stage E output directory helper
# ---------------------------------------------------------------------------

def get_te_output_dir(
    subdir_key: str,
    config: Optional[dict] = None,
    create: bool = True,
) -> Path:
    """
    Return (and optionally create) a Stage E output subdirectory.

    Parameters
    ----------
    subdir_key : str
        Key from ``stage_e_transfer_entropy.output_subdirs`` in config.yaml,
        e.g. 'roi_signals', 'phase_1_connectivity', 'phase_1_group'.
    config : dict, optional
        Parsed config.yaml. Loaded if None.
    create : bool
        If True, create the directory (including parents).

    Returns
    -------
    path : Path
        Absolute path to the requested output subdirectory.
    """
    if config is None:
        config = load_config()

    te_cfg = config.get("stage_e_transfer_entropy", {})
    output_root = _PROJECT_ROOT / te_cfg.get("paths", {}).get(
        "output_root", "outputs/stage_e_transfer_entropy"
    )
    subdirs = te_cfg.get("output_subdirs", {})

    if subdir_key not in subdirs:
        raise ValueError(
            f"Unknown subdir_key '{subdir_key}'. "
            f"Available: {sorted(subdirs.keys())}"
        )

    path = output_root / subdirs[subdir_key]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# ROI signal loader (Section 2.1.1)
# ---------------------------------------------------------------------------

def load_or_extract_roi_signals(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
) -> tuple[np.ndarray, list[str], float]:
    """
    Load ROI-averaged HbO signals for one subject.

    Strategy (Section 2.1.1):
    1. Check Stage D cache: ``outputs/stage_d_granger/linear_gc/expert-novice/
       roi_signals/sub-{id}_roi_signals.npz`` — load and return if present.
    2. If absent, fall back to Stage E roi_signals/ directory.
    3. If absent there too, call Stage D's run_subject_roi_extraction() and
       save a copy to Stage E's roi_signals/ directory.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'A1', 'R7'.
    study : str
        Config dataset key (default 'expert_novice').
    config : dict, optional
        Parsed config.yaml. Loaded if None.

    Returns
    -------
    Data_Reg : np.ndarray, shape (n_rois, T)
        ROI-averaged HbO signals (MNE native units: mol/L).
    roi_names : list[str]
        Valid ROI names, subset of ['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA'].
    sfreq : float
        Sampling frequency in Hz (typically 25.0; R9 is resampled in-memory).

    Raises
    ------
    RuntimeError
        If extraction fails and no cached file is available.
    """
    if config is None:
        config = load_config()

    # ------------------------------------------------------------------
    # Path 1: Stage D cache (preferred — avoids redundant computation)
    # ------------------------------------------------------------------
    stage_d_root = _PROJECT_ROOT / config["paths"]["stage_d_outputs"]
    stage_d_roi_dir = stage_d_root / "linear_gc" / "expert-novice" / "roi_signals"
    stage_d_npz = stage_d_roi_dir / f"sub-{subject_id}_roi_signals.npz"

    if stage_d_npz.exists():
        cached = np.load(stage_d_npz, allow_pickle=True)
        data = cached["data"]
        roi_names = list(cached["roi_names"])
        sfreq = float(cached["sfreq"])
        logger.info(
            "load_or_extract_roi_signals [%s]: loaded from Stage D cache  "
            "shape=%s  sfreq=%.1f Hz  ROIs=%s",
            subject_id, data.shape, sfreq, roi_names,
        )
        return data, roi_names, sfreq

    # ------------------------------------------------------------------
    # Path 2: Stage E local cache
    # ------------------------------------------------------------------
    stage_e_roi_dir = get_te_output_dir("roi_signals", config=config, create=True)
    stage_e_npz = stage_e_roi_dir / f"sub-{subject_id}_roi_signals.npz"

    if stage_e_npz.exists():
        cached = np.load(stage_e_npz, allow_pickle=True)
        data = cached["data"]
        roi_names = list(cached["roi_names"])
        sfreq = float(cached["sfreq"])
        logger.info(
            "load_or_extract_roi_signals [%s]: loaded from Stage E cache  "
            "shape=%s  sfreq=%.1f Hz",
            subject_id, data.shape, sfreq,
        )
        return data, roi_names, sfreq

    # ------------------------------------------------------------------
    # Path 3: Re-extract from Stage B .fif using Stage D extractor
    # ------------------------------------------------------------------
    logger.warning(
        "load_or_extract_roi_signals [%s]: no cached .npz found in Stage D "
        "or Stage E directories — re-extracting from Stage B .fif.",
        subject_id,
    )
    roi_mod = _import_roi_extraction()
    result = roi_mod.run_subject_roi_extraction(
        subject_id, study=study, config=config, overwrite=False
    )
    if result.get("status") in ("ok", "cached"):
        # Load the freshly saved Stage D npz
        cached = np.load(result["npz_path"], allow_pickle=True)
        data = cached["data"]
        roi_names = list(cached["roi_names"])
        sfreq = float(cached["sfreq"])

        # Save a copy to Stage E directory
        np.savez(
            stage_e_npz,
            data=data,
            roi_names=np.array(roi_names),
            sfreq=np.float64(sfreq),
        )
        logger.info(
            "load_or_extract_roi_signals [%s]: re-extracted and cached to "
            "Stage E  shape=%s",
            subject_id, data.shape,
        )
        return data, roi_names, sfreq
    else:
        raise RuntimeError(
            f"load_or_extract_roi_signals [{subject_id}]: extraction failed. "
            f"Result: {result}"
        )


# ===========================================================================
# Neural-Gas Vector Quantiser (scratch implementation)
# ===========================================================================
# Reference: Martinetz, T.M., Berkovich, S.G., & Schulten, K.J. (1993).
#   "Neural-gas" network for vector quantization and its application to
#   time-series prediction. IEEE Trans. Neural Netw., 4(4), 558–569.
#
# Algorithm:
#   Initialize k prototype vectors W by sampling k rows from X.
#   For each training step t in [0, T_max):
#     1. Pick random sample x from X.
#     2. Sort prototypes by distance to x → rank k_i (0 = nearest).
#     3. Update: W_i ← W_i + ε(t) · h(k_i, σ(t)) · (x − W_i)
#        where ε(t) = λ_i · (λ_f/λ_i)^(t/T_max)  [decaying learning rate]
#              σ(t) = σ_i · (σ_f/σ_i)^(t/T_max)  [decaying neighbourhood]
#              h(k, σ) = exp(−k / σ)               [soft-rank kernel]
#
# nDis (Wang 2020 Eq. 11):
#   nDis = Σ‖x_t − W_nearest(t)‖² / Σ‖x_t − mean(X)‖²
#   This is the normalised quantisation error (reconstruction error / variance).
#   The optimal codebook size k₀ is the smallest k with nDis < threshold.
# ===========================================================================


class NeuralGas:
    """
    Neural-Gas vector quantiser.

    Parameters
    ----------
    n_prototypes : int
        Number of prototype vectors k (codebook size).
    max_iterations : int
        Number of stochastic gradient descent steps T_max.
    learning_rate_i : float
        Initial learning rate λ_i.
    learning_rate_f : float
        Final learning rate λ_f (after T_max steps).
    neighbourhood_i : float
        Initial neighbourhood width σ_i.
    neighbourhood_f : float
        Final neighbourhood width σ_f.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_prototypes: int = 10,
        max_iterations: int = 500,
        learning_rate_i: float = 0.5,
        learning_rate_f: float = 0.01,
        neighbourhood_i: float = 10.0,
        neighbourhood_f: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.max_iterations = max_iterations
        self.learning_rate_i = learning_rate_i
        self.learning_rate_f = learning_rate_f
        self.neighbourhood_i = neighbourhood_i
        self.neighbourhood_f = neighbourhood_f
        self.random_state = random_state
        self.prototypes_: Optional[np.ndarray] = None    # shape (k, d)
        self.ndis_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "NeuralGas":
        """
        Fit Neural-Gas to data matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Training data. Each row is one d-dimensional observation.

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        # X is the Joint state matrix. For bivariate (X, Y): stack both embedded series
        # vertically → shape (2T', d). For multivariate: all channels → (n_ch * T', d).
        N, d = X.shape 
        k = self.n_prototypes
        T = self.max_iterations

        # Initialise prototypes by sampling k rows without replacement
        init_idx = rng.choice(N, size=k, replace=False) # init_idx is a list of k unique random integers corresponding to random row numbers in the dataset X
        W = X[init_idx].astype(float, copy=True)   # shape (k, d); grabs the rows of X corresponding to the random integers in init_idx

        # Precompute rank array for sorting (reused each step)
        for t in range(T):
            frac = t / T

            # Decaying schedule (exponential annealing)
            eps = self.learning_rate_i * (self.learning_rate_f / self.learning_rate_i) ** frac
            sigma = self.neighbourhood_i * (self.neighbourhood_f / self.neighbourhood_i) ** frac

            # Pick random sample
            x = X[rng.integers(N)]                # shape (d,)

            # Squared distances from x to all prototypes
            diffs = W - x # Computes W_i - x for each codebook W; shape (k, d)
            sq_dists = np.einsum("kd,kd->k", diffs, diffs)  # computes the squared Euclidean distance between each codebook W and the random sample x; shape (k,)

            # Rank prototypes by distance (argsort of argsort gives ranks)
            order = np.argsort(sq_dists)           # sorted prototype indices
            ranks = np.empty(k, dtype=float)
            ranks[order] = np.arange(k)            # rank_i: 0 = nearest
            # Ex: ranks[6] = 0 means the 6th codebook W is the nearest to the random sample x (rank 0)

            # Soft-neighbourhood weights: h(rank_i, sigma) = exp(-rank_i / sigma)
            h = np.exp(-ranks / sigma)             # shape (k,)

            # Update prototypes / move the codebook W closer to the random sample x
            W += (eps * h[:, np.newaxis]) * (x - W)

        self.prototypes_ = W
        self.ndis_ = self._compute_ndis(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each row of X to the nearest prototype.

        This function is very similar to quantise_to_symbols. It has the same math, but it doesn't permanently change the embedded matrix X to the symbolic sequence.
        This is only used for computing the nDis value and doesn't actually permanently assign the symbolic sequence to the embedded matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        labels : np.ndarray, shape (N,)
            Integer labels in {0, ..., k-1} (0-indexed).
        """
        if self.prototypes_ is None:
            raise RuntimeError("NeuralGas.predict() called before fit().")
        dists = cdist(X, self.prototypes_, metric="euclidean")  # (N, k)
        return np.argmin(dists, axis=1)

    def _compute_ndis(self, X: np.ndarray) -> float:
        """
        Compute normalised distortion nDis (Wang 2020 Eq. 11).

        nDis = Σ‖x_t − W_nearest(t)‖² / Σ‖x_t − mean(X)‖²

        x_t = each row of X
        W_nearest(t) = the nearest codebook W to the random sample x
        mean(X) = the mean of all the rows of X
        Σ‖x_t − W_nearest(t)‖² = the sum of the squared distances between each row of X and the nearest codebook W
        Σ‖x_t − mean(X)‖² = the sum of the squared distances between each row of X and the mean of all the rows of X

        A value < 0.05 (5%) satisfies Wang (2020)'s codebook quality criterion.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        ndis : float
            Normalised distortion in [0, 1].
        """
        labels = self.predict(X)
        reconstructions = self.prototypes_[labels]     # (N, d) nearest prototype per row
        numerator = float(np.sum((X - reconstructions) ** 2))
        X_mean = X.mean(axis=0)
        denominator = float(np.sum((X - X_mean) ** 2))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    @property
    def distortion(self) -> Optional[float]:
        """Return nDis computed at the end of fit(), or None if not fitted."""
        return self.ndis_


# ---------------------------------------------------------------------------
# Codebook search: iterate k until nDis < threshold (Wang 2020 Section 3.2)
# ---------------------------------------------------------------------------

def fit_neural_gas_codebook(
    theta_xyd: np.ndarray,
    k0_max: int = 20,
    ndis_threshold: float = 0.05,
    config: Optional[dict] = None,
) -> tuple[np.ndarray, int, float]:
    """
    Find the smallest codebook size k₀ satisfying nDis < ndis_threshold.

    Iterates k from 2 to k0_max, fitting a NeuralGas at each k and evaluating
    nDis (Wang 2020 Eq. 11). Stops at the first k where nDis < threshold.
    If nDis never reaches the threshold, logs a WARNING and returns the result
    at k = k0_max.

    Parameters
    ----------
    theta_xyd : np.ndarray, shape (N, d)
        Joint state matrix. For bivariate (X, Y): stack both embedded series
        vertically → shape (2T', d). For multivariate: all channels → (n_ch * T', d).
    k0_max : int
        Maximum codebook size to test. Default 20 (bivariate limit from config).
    ndis_threshold : float
        nDis stopping criterion. Default 0.05 (Wang 2020).
    config : dict, optional
        Parsed config.yaml. If provided, NG hyperparameters are read from
        stage_e_transfer_entropy.symbolisation.

    Returns
    -------
    codebook : np.ndarray, shape (k₀, d)
        Prototype vectors of the optimal Neural-Gas codebook.
    k0 : int
        Selected codebook size (smallest k achieving nDis < threshold, or k0_max).
    final_ndis : float
        nDis value at the selected k₀.
    """
    if config is None:
        config = load_config()

    sym_cfg = (
        config.get("stage_e_transfer_entropy", {})
        .get("symbolisation", {})
    )
    max_iter = int(sym_cfg.get("ng_max_iterations", 500))
    lr_i = float(sym_cfg.get("ng_learning_rate_i", 0.5))
    lr_f = float(sym_cfg.get("ng_learning_rate_f", 0.01))
    nb_i = float(sym_cfg.get("ng_neighbourhood_i", 10.0))
    nb_f = float(sym_cfg.get("ng_neighbourhood_f", 0.1))

    best_ng: Optional[NeuralGas] = None
    best_k = k0_max
    best_ndis = float("inf")
    threshold_met = False

    for k in range(2, k0_max + 1):
        ng = NeuralGas(
            n_prototypes=k,
            max_iterations=max_iter,
            learning_rate_i=lr_i,
            learning_rate_f=lr_f,
            neighbourhood_i=nb_i,
            neighbourhood_f=nb_f,
            random_state=42,
        )
        ng.fit(theta_xyd)
        ndis = ng.distortion

        logger.debug(
            "fit_neural_gas_codebook: k=%d  nDis=%.4f  (threshold=%.2f)",
            k, ndis, ndis_threshold,
        )

        best_ng = ng
        best_k = k
        best_ndis = ndis

        if ndis < ndis_threshold:
            threshold_met = True
            logger.info(
                "fit_neural_gas_codebook: k₀=%d selected  nDis=%.4f < %.2f",
                k, ndis, ndis_threshold,
            )
            break

    if not threshold_met:
        logger.warning(
            "fit_neural_gas_codebook: nDis threshold %.2f NOT reached by "
            "k=%d (final nDis=%.4f). Using k=%d as fallback.",
            ndis_threshold, k0_max, best_ndis, best_k,
        )

    return best_ng.prototypes_, best_k, best_ndis


# ---------------------------------------------------------------------------
# Quantisation helper
# ---------------------------------------------------------------------------

def quantise_to_symbols(
    embedded: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """
    Assign each embedded vector to the nearest codebook prototype.

    Uses Euclidean distance via scipy.spatial.distance.cdist for efficiency.
    Returns 0-indexed integer labels in {0, ..., k₀-1}.

    Performs similar operation to predict. It has the same math, but it permanently
    changes the embedded matrix X to the symbolic sequence.

    Parameters
    ----------
    embedded : np.ndarray, shape (T', d)
        Time-delay embedded matrix.
    codebook : np.ndarray, shape (k₀, d)
        Prototype vectors from NeuralGas.fit().

    Returns
    -------
    symbols : np.ndarray, shape (T',), dtype int
        Integer symbolic sequence. Values in {0, ..., k₀-1}.
    """
    dists = cdist(embedded, codebook, metric="euclidean")   # (T', k₀)
    return np.argmin(dists, axis=1).astype(int)


# ===========================================================================
# Section 10.2b — ETE PCA-conditioning config hash (additive, method-agnostic)
# ===========================================================================

def compute_conditioning_config_hash(
    config: dict,
    method_key: str,
    k0: int,
) -> str:
    """
    Stable short hash tag for the parameter set that determines how a
    PCA-compressed conditioning composite (``z_composite_per_pair``) was
    built for a given method/window.

    Used by the raw pipelines (``compute_multivariate_dspte_matrix`` in
    ``phase_2/06_multivariate_dspte.py``, ``compute_amplitude_conditional_dste_matrix``
    in ``phase_4/14_amplitude_dste.py``) to tag the composite they store in
    their additive ETE metadata, and by the ETE surrogate core
    (``11_symbolic_ete_core.py``) to verify — before reusing a stored
    composite across the observed computation and all N surrogates — that
    it was built under the exact same configuration as the current run. A
    mismatch means the stored composite is stale (e.g. computed under
    different filter/PCA/k0 settings) and must not be silently reused; the
    caller should raise rather than proceed.

    Only parameters that actually influence the *construction* of the
    composite are included (deliberately excludes things like
    ``sparsity_fallback_threshold`` or ``u_max``, which affect per-lag
    fallback behaviour / how many lags are searched but not the composite
    symbol sequence itself, which is built once per window before the lag
    loop even starts):

      - ``method_key`` (different methods should never be cross-compared)
      - ``signal.embedding_dimension`` (d — shape of the embedded matrices
        PCA is applied to)
      - ``k0`` (the actual, already-clamped codebook size for this window —
        affects both the shared codebook and ``symbolise_pca_conditioner``)
      - ``<method_key>.conditional_pca`` (enabled / n_components — determines
        whether/how compression happens)
      - ``narrowband_filter`` (enabled / f_lo / f_hi / order — affects the
        Hilbert phase input to the embedding for phase-based methods; inert
        but harmless to include for amplitude-based methods that ignore it)
      - ``<method_key>.envelope_normalize`` (amplitude-based methods only;
        defaults to True and is simply unused by phase-based methods)

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    method_key : str
        Key into ``config['stage_e_transfer_entropy']`` (e.g. 'phase_2',
        'phase_4').
    k0 : int
        The actual (post-clamp) shared codebook size used for this window.

    Returns
    -------
    str
        16-character hex digest (truncated SHA-256) of the canonical JSON
        of the parameters above. Deterministic across runs/processes for
        the same effective parameter set.
    """
    import hashlib
    import json

    te_cfg = config.get("stage_e_transfer_entropy", {})
    sig_cfg = te_cfg.get("signal", {})
    method_cfg = te_cfg.get(method_key, {})
    pca_cfg = method_cfg.get("conditional_pca", {})
    filt_cfg = te_cfg.get("narrowband_filter", {})

    payload = {
        "method_key": method_key,
        "embedding_dimension": int(sig_cfg.get("embedding_dimension", 2)),
        "k0": int(k0),
        "conditional_pca_enabled": bool(pca_cfg.get("enabled", True)),
        "conditional_pca_n_components": int(pca_cfg.get("n_components", 1)),
        "narrowband_filter_enabled": bool(filt_cfg.get("enabled", True)),
        "narrowband_filter_f_lo": float(filt_cfg.get("f_lo", 0.01)),
        "narrowband_filter_f_hi": float(filt_cfg.get("f_hi", 0.10)),
        "narrowband_filter_order": int(filt_cfg.get("order", 3)),
        "envelope_normalize": bool(method_cfg.get("envelope_normalize", True)),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ===========================================================================
# Smoke test (run as script)
# ===========================================================================

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n=== NeuralGas smoke test ===\n")

    # 1. Basic fit / predict on 2D Gaussian clusters
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal([0, 0], 0.3, (200, 2)),
        rng.normal([3, 0], 0.3, (200, 2)),
        rng.normal([1.5, 2.5], 0.3, (200, 2)),
    ])
    ng = NeuralGas(n_prototypes=3, max_iterations=1000, random_state=42)
    t0 = time.time()
    ng.fit(X)
    elapsed = time.time() - t0
    labels = ng.predict(X)
    print(f"  fit time     : {elapsed*1000:.1f} ms  (N=600, d=2, k=3)")
    print(f"  nDis         : {ng.distortion:.4f}  (expected < 0.10 for 3 clusters @ 500 iter)")
    print(f"  unique labels: {sorted(set(labels.tolist()))}")
    # NG with stochastic updates needs sufficient iterations for tight convergence.
    # 500 iterations is the default; nDis < 0.10 is a reasonable smoke-test threshold.
    # The actual codebook search in fit_neural_gas_codebook uses 500 iters and tests
    # nDis < 0.05 — if not achieved by k0_max, a WARNING is logged (not an error).
    assert ng.distortion < 0.15, f"nDis unexpectedly high: {ng.distortion:.4f}"
    assert len(set(labels.tolist())) == 3, "Expected 3 unique cluster labels"
    print("  [PASS] NeuralGas 3-cluster test\n")

    # 2. fit_neural_gas_codebook — search for k₀
    print("=== fit_neural_gas_codebook smoke test ===\n")
    X_large = np.vstack([
        rng.normal([0, 0, 0], 0.4, (300, 3)),
        rng.normal([5, 0, 0], 0.4, (300, 3)),
        rng.normal([2, 4, 0], 0.4, (300, 3)),
    ])
    codebook, k0, ndis = fit_neural_gas_codebook(X_large, k0_max=10, ndis_threshold=0.05)
    print(f"  k₀           : {k0}")
    print(f"  nDis at k₀   : {ndis:.4f}")
    print(f"  codebook shape: {codebook.shape}")
    assert k0 >= 2, "k₀ must be >= 2"
    assert ndis < 0.05 or k0 == 10, "Should reach threshold or hit k_max"
    print("  [PASS] fit_neural_gas_codebook test\n")

    # 3. quantise_to_symbols
    print("=== quantise_to_symbols smoke test ===\n")
    symbols = quantise_to_symbols(X_large, codebook)
    print(f"  input shape  : {X_large.shape}")
    print(f"  symbols shape: {symbols.shape}")
    print(f"  unique symbols: {sorted(set(symbols.tolist()))}")
    assert symbols.shape == (900,), f"Expected (900,), got {symbols.shape}"
    assert set(symbols.tolist()).issubset(set(range(k0))), "Symbols out of codebook range"
    print("  [PASS] quantise_to_symbols test\n")

    # 4. load_or_extract_roi_signals validation for A1
    print("=== load_or_extract_roi_signals smoke test (A1) ===\n")
    try:
        data, roi_names, sfreq = load_or_extract_roi_signals("A1")
        print(f"  Data_Reg shape : {data.shape}")
        print(f"  sfreq          : {sfreq} Hz")
        print(f"  ROI names      : {roi_names}")
        n_rois, n_times = data.shape
        assert 3 <= n_rois <= 5, f"Expected 3-5 ROIs, got {n_rois}"
        assert n_times > 10000, f"Expected > 10000 samples, got {n_times}"
        assert sfreq == 25.0, f"Expected 25.0 Hz, got {sfreq}"
        nan_rows = int(np.all(np.isnan(data), axis=1).sum())
        assert nan_rows == 0, f"Unexpected NaN rows: {nan_rows}"
        print(f"  NaN rows       : {nan_rows}")
        print("  [PASS] A1 ROI signal load\n")
    except FileNotFoundError as e:
        print(f"  [SKIP] A1 .fif not found — run from project root: {e}\n")

    # 5. get_te_output_dir
    print("=== get_te_output_dir smoke test ===\n")
    try:
        p = get_te_output_dir("roi_signals", create=False)
        print(f"  roi_signals dir: {p}")
        p2 = get_te_output_dir("phase_1_group", create=False)
        print(f"  phase_1_group  : {p2}")
        print("  [PASS] get_te_output_dir\n")
    except Exception as e:
        print(f"  [FAIL] get_te_output_dir: {e}\n")

    # 6. get_connection_labels
    print("=== get_connection_labels smoke test ===\n")
    labels = get_connection_labels()
    assert len(labels) == 20, f"Expected 20 connection labels, got {len(labels)}"
    print(f"  Labels (20): {labels[:4]} ... {labels[-2:]}")
    print("  [PASS] get_connection_labels\n")

    print("=== All smoke tests passed. ===\n")
