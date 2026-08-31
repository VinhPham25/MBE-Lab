"""
Stage E — Section 10.9-10.10: Method-Agnostic Asymmetry Index (AI) Core.

Refactors the statistical core of ``phase_4/18_asymmetry_index_statistics.py``
(``_compute_subject_ai``, ``_rank_biserial``, ``_cohens_d``, and the
H_AI1/H_AI2/H_AI3 test functions) into a **method-agnostic** form: every
statistical function takes a generic ``method_key`` string plus the
``(B_Exp, B_Nov)`` connectivity arrays it resolves through
``stage_e_transfer_entropy.significance_methods`` (``config.yaml``) — no
function body hardcodes ``te``/``gc`` column names or file paths, so an
arbitrary number of methods can be registered and analysed without new
statistical code (Section 10.12's extensibility proof exercises exactly
this).

Design change vs. the original ``18_asymmetry_index_statistics.py``
---------------------------------------------------------------------
The original script computed TE and GC asymmetry indices *side by side* in
a single set of CSVs (``te_mwu_p_raw``, ``gc_mwu_p_raw``, ...) so a
researcher could eyeball TE-vs-GC agreement directly. That dual-column
design does not generalise past 2 methods. Here, each call analyses **one**
method and writes to that method's own output directory (resolved via the
registry's ``ai_stats_dir``/``ai_viz_dir`` keys) — cross-method comparison
(e.g. "do GC, Phase 2, and Phase 4 agree on AI direction?") is deliberately
deferred to ``16_cross_method_significance.py`` (Section 10.11), which reads
these per-method CSVs back in. ``phase_4/18_asymmetry_index_statistics.py``
remains as a thin backward-compatible CLI shim that calls
``run_ai_for_method("phase_4", ...)`` from here.

Registry dispatch (``load_method_arrays``)
-------------------------------------------
Two ``ai_loader.kind`` values cover every registered track without new code:

- ``"symbolic_mean_npz"``: read a pre-built ``B_{Exp,Nov}_*.npz`` subject-mean
  pair straight off disk. This covers **both** raw-magnitude tracks (Phase 2,
  Phase 4 raw) **and** the floor-corrected ETE track (Phase 4 ETE) — from the
  AI core's point of view an ETE-corrected mean array and a raw mean array
  are structurally identical (n_subs, 20, n_trials) float arrays; only the
  registry's ``group_dir``/``exp_filename``/``nov_filename`` differ. Adding a
  5th symbolic method (10.12's dummy-method proof) needs *zero* new Python
  here, only one new registry entry with these three fields.
- ``"gc_windowed"``: GC group arrays are per-window
  (n_subs, 20, n_windows per trial key) rather than pre-averaged, so this
  loader trial-averages them into the same (n_subs, 20, n_trials) shape the
  rest of the pipeline expects. This is the one place a method's on-disk
  schema genuinely differs and needs bespoke (but still generic — not
  hardcoded to "gc" by name, just to the *shape convention*) handling.

Statistical design (unchanged from the original, per-method now)
------------------------------------------------------------------
H_AI1  Per-connection Mann-Whitney U (+ Welch's t) on the 10 unique
       undirected pairs, BH-FDR corrected across the 10 pairs.
H_AI2  Cross-pair Wilcoxon signed-rank (+ one-sample t, + binomial sign test
       for small n) on the 10 group-mean AI differences (Expert − Novice).
H_AI3  LM1-focused pre-specified analysis: the 3 PFC↔LM1 pairs individually
       (MWU + Welch's t, uncorrected, pre-registered) plus a pooled
       permutation test (21 Expert vs 18 Novice pooled observations).

CLI
---
python3 12_asymmetry_index_core.py --method phase_4_raw
python3 12_asymmetry_index_core.py --method phase_4_ete
python3 12_asymmetry_index_core.py --all
python3 12_asymmetry_index_core.py --all --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).parent.parent        # code/
_STAGE_E_ROOT = Path(__file__).parent            # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROI / connection constants (generic — not method-specific)
# ---------------------------------------------------------------------------

ROI_NAMES: list[str] = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
N_ROIS: int = 5

# 20-vector index -> (src_roi_idx, tgt_roi_idx)
CONN_TO_PAIR: list[tuple[int, int]] = [
    (src, tgt)
    for src in range(N_ROIS)
    for tgt in range(N_ROIS)
    if src != tgt
]
PAIR_TO_CONN: dict[tuple[int, int], int] = {p: i for i, p in enumerate(CONN_TO_PAIR)}

# 10 unique undirected pairs (canonical order: lower ROI index first)
UNIQUE_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(N_ROIS) for j in range(i + 1, N_ROIS)
]
UNIQUE_PAIR_LABELS: list[str] = [
    f"{ROI_NAMES[i]}\u2194{ROI_NAMES[j]}" for i, j in UNIQUE_PAIRS
]

LM1_IDX: int = ROI_NAMES.index("LM1")
LM1_PAIR_LABELS: list[str] = [lbl for lbl in UNIQUE_PAIR_LABELS if "LM1" in lbl]
LM1_PAIR_INDICES: list[int] = [pi for pi, lbl in enumerate(UNIQUE_PAIR_LABELS) if "LM1" in lbl]

# Pre-specified primary analysis: the 3 PFC<->LM1 connections (excludes LM1<->SMA).
PFC_LM1_PAIR_LABELS: list[str] = ["LPFC\u2194LM1", "CPFC\u2194LM1", "RPFC\u2194LM1"]
PFC_LM1_PAIR_INDICES: list[int] = [
    pi for pi, lbl in enumerate(UNIQUE_PAIR_LABELS) if lbl in PFC_LM1_PAIR_LABELS
]

_N_TRIALS_CANONICAL = 5
_N_PERMUTATIONS = 10_000


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Registry dispatch (Section 10.9)
# ---------------------------------------------------------------------------

def _get_registry_entry(method_key: str, config: dict) -> dict:
    te_cfg = config.get("stage_e_transfer_entropy", {})
    sm = te_cfg.get("significance_methods", {})
    if method_key not in sm:
        raise KeyError(
            f"_get_registry_entry: method_key={method_key!r} not found in "
            f"stage_e_transfer_entropy.significance_methods "
            f"(available: {list(sm.keys())})"
        )
    return sm[method_key]


def load_method_arrays(
    method_key: str,
    config: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a method's (B_Exp, B_Nov) subject-mean connectivity arrays,
    dispatching via the ``significance_methods`` registry's ``ai_loader``.

    Returns
    -------
    tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Each of shape ``(n_subs, 20, n_trials)``, or ``(None, None)`` if the
        method's ``ai_loader`` is missing/unrecognised or its output files
        are not present on disk (e.g. a WIRE ONLY track that hasn't been
        batch-run yet).
    """
    if config is None:
        config = load_config()

    entry = _get_registry_entry(method_key, config)
    loader_cfg = entry.get("ai_loader")
    if loader_cfg is None:
        logger.warning(
            "load_method_arrays: method_key=%r has no ai_loader configured — skipping.",
            method_key,
        )
        return None, None

    kind = loader_cfg.get("kind")

    if kind == "symbolic_mean_npz":
        from utils_te import get_te_output_dir

        group_dir = get_te_output_dir(loader_cfg["group_dir"], config=config)
        exp_path = group_dir / loader_cfg["exp_filename"]
        nov_path = group_dir / loader_cfg["nov_filename"]
        if not exp_path.exists() or not nov_path.exists():
            logger.warning(
                "load_method_arrays [%s]: files not found (%s, %s) — "
                "has this track's raw/ETE batch been run yet?",
                method_key, exp_path.name, nov_path.name,
            )
            return None, None
        B_Exp = np.load(exp_path, allow_pickle=True)["data"]
        B_Nov = np.load(nov_path, allow_pickle=True)["data"]
        return B_Exp, B_Nov

    if kind == "gc_windowed":
        try:
            from stage_d_granger.utils_gc import get_gc_output_dir
        except ImportError:
            logger.warning("load_method_arrays [%s]: stage_d_granger.utils_gc not importable.", method_key)
            return None, None

        gc_dir = get_gc_output_dir("group", config=config)
        exp_path = gc_dir / "B_Exp.npz"
        nov_path = gc_dir / "B_Nov.npz"
        if not exp_path.exists() or not nov_path.exists():
            logger.warning("load_method_arrays [%s]: GC group files not found in %s.", method_key, gc_dir)
            return None, None

        exp_npz = np.load(exp_path, allow_pickle=True)
        nov_npz = np.load(nov_path, allow_pickle=True)
        trial_keys = sorted(k for k in exp_npz.files if k.startswith("Tr_"))

        def _trial_means(npz) -> np.ndarray:
            arrays = [np.nanmean(npz[k], axis=2) for k in trial_keys]
            return np.stack(arrays, axis=2)   # (n_subs, 20, n_trials)

        return _trial_means(exp_npz), _trial_means(nov_npz)

    logger.warning("load_method_arrays [%s]: unknown ai_loader.kind=%r.", method_key, kind)
    return None, None


def _ai_stats_dir(method_key: str, config: dict) -> Path:
    from utils_te import get_te_output_dir

    entry = _get_registry_entry(method_key, config)
    dir_key = entry.get("ai_stats_dir")
    if dir_key is None:
        raise ValueError(f"_ai_stats_dir: method_key={method_key!r} has no ai_stats_dir configured.")
    d = get_te_output_dir(dir_key, config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ai_viz_dir(method_key: str, config: dict) -> Path:
    from utils_te import get_te_output_dir

    entry = _get_registry_entry(method_key, config)
    dir_key = entry.get("ai_viz_dir")
    if dir_key is None:
        raise ValueError(f"_ai_viz_dir: method_key={method_key!r} has no ai_viz_dir configured.")
    d = get_te_output_dir(dir_key, config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stats_cfg(config: dict) -> dict:
    return config["stage_e_transfer_entropy"]["statistics"]


def _subject_ids(config: Optional[dict] = None) -> tuple[list[str], list[str]]:
    if config is None:
        config = load_config()
    te_cfg = config["stage_e_transfer_entropy"]
    return list(te_cfg["subjects"]["Expert"]), list(te_cfg["subjects"]["Novice"])


def registered_ai_tracks(config: Optional[dict] = None) -> list[str]:
    """Return all method_keys with ``ai_track: true`` in the registry."""
    if config is None:
        config = load_config()
    sm = config["stage_e_transfer_entropy"]["significance_methods"]
    return [k for k, v in sm.items() if v.get("ai_track", False)]


# ---------------------------------------------------------------------------
# Core AI computation (method-agnostic)
# ---------------------------------------------------------------------------

def compute_subject_ai(B: np.ndarray) -> np.ndarray:
    """
    Compute per-subject Asymmetry Index for each of the 10 unique pairs.

    Parameters
    ----------
    B : np.ndarray, shape (n_subs, 20, n_trials)
        Group connectivity array for ANY method (raw magnitude or ETE).

    Returns
    -------
    np.ndarray, shape (n_subs, 10)
        AI[sub, pair] = (fwd_mean - rev_mean) / (fwd_mean + rev_mean),
        trial-averaged, NaN where the denominator is zero/NaN.
    """
    sub_mean = np.nanmean(B, axis=2)   # (n_subs, 20)
    n_subs = B.shape[0]
    ai = np.full((n_subs, len(UNIQUE_PAIRS)), np.nan)

    for pi, (i, j) in enumerate(UNIQUE_PAIRS):
        fwd_idx = PAIR_TO_CONN[(i, j)]
        rev_idx = PAIR_TO_CONN[(j, i)]
        fwd = sub_mean[:, fwd_idx]
        rev = sub_mean[:, rev_idx]
        denom = fwd + rev
        valid = (denom > 0) & np.isfinite(fwd) & np.isfinite(rev)
        ai[valid, pi] = (fwd[valid] - rev[valid]) / denom[valid]

    return ai


def _rank_biserial(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from a Mann-Whitney U statistic."""
    return 1.0 - (2.0 * U) / (n1 * n2)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_std


# ---------------------------------------------------------------------------
# Per-subject AI table builder
# ---------------------------------------------------------------------------

def build_ai_per_subject(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Build the per-subject AI table for one method and save to CSV.

    Returns
    -------
    Optional[pd.DataFrame]
        One row per subject x pair: ``subject_id, group, pair, pair_idx, ai``.
        ``None`` if the method's arrays are unavailable.
    """
    if config is None:
        config = load_config()

    out_path = _ai_stats_dir(method_key, config) / "ai_per_subject.csv"
    if not overwrite and out_path.exists():
        logger.info("build_ai_per_subject [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None

    expert_ids, novice_ids = _subject_ids(config)
    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)

    rows = []
    for sub_idx, sid in enumerate(expert_ids):
        for pi, pair_lbl in enumerate(UNIQUE_PAIR_LABELS):
            rows.append({
                "subject_id": sid, "group": "Expert", "pair": pair_lbl, "pair_idx": pi,
                "ai": float(ai_exp[sub_idx, pi]) if np.isfinite(ai_exp[sub_idx, pi]) else np.nan,
            })
    for sub_idx, sid in enumerate(novice_ids):
        for pi, pair_lbl in enumerate(UNIQUE_PAIR_LABELS):
            rows.append({
                "subject_id": sid, "group": "Novice", "pair": pair_lbl, "pair_idx": pi,
                "ai": float(ai_nov[sub_idx, pi]) if np.isfinite(ai_nov[sub_idx, pi]) else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("build_ai_per_subject [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# H_AI1: Per-connection Mann-Whitney U + Welch's t
# ---------------------------------------------------------------------------

def test_per_connection_ai(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[pd.DataFrame]:
    """
    H_AI1: Expert vs Novice AI compared per undirected pair (one method).

    Returns
    -------
    Optional[pd.DataFrame]
        Columns: ``pair, is_lm1, mwu_U, mwu_p_raw, mwu_p_fdr,
        mwu_significant, mwu_r, t_stat, t_p_raw, t_p_fdr, t_significant,
        cohens_d, exp_mean_ai, nov_mean_ai, ai_diff``.
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from statsmodels.stats.multitest import fdrcorrection

    if config is None:
        config = load_config()

    out_path = _ai_stats_dir(method_key, config) / "ai_group_comparison.csv"
    if not overwrite and out_path.exists():
        logger.info("test_per_connection_ai [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None

    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)
    n_pairs = len(UNIQUE_PAIRS)

    U_arr = np.full(n_pairs, np.nan)
    mwu_p = np.full(n_pairs, np.nan)
    mwu_r = np.full(n_pairs, np.nan)
    t_stat = np.full(n_pairs, np.nan)
    t_p = np.full(n_pairs, np.nan)
    d_arr = np.full(n_pairs, np.nan)

    for pi in range(n_pairs):
        exp_clean = ai_exp[:, pi][np.isfinite(ai_exp[:, pi])]
        nov_clean = ai_nov[:, pi][np.isfinite(ai_nov[:, pi])]
        if len(exp_clean) < 2 or len(nov_clean) < 2:
            continue
        try:
            U, p = mannwhitneyu(exp_clean, nov_clean, alternative="two-sided")
            U_arr[pi] = float(U)
            mwu_p[pi] = float(p)
            mwu_r[pi] = _rank_biserial(float(U), len(exp_clean), len(nov_clean))
        except Exception:
            pass
        try:
            t, tp = ttest_ind(exp_clean, nov_clean, equal_var=False)
            t_stat[pi] = float(t)
            t_p[pi] = float(tp)
        except Exception:
            pass
        d_arr[pi] = _cohens_d(exp_clean, nov_clean)

    def _fdr(p_raw: np.ndarray) -> np.ndarray:
        p_fdr = np.full(n_pairs, np.nan)
        valid = np.isfinite(p_raw)
        if valid.sum() > 0:
            _, p_corr = fdrcorrection(p_raw[valid], alpha=alpha, method=fdr_method)
            p_fdr[valid] = p_corr
        return p_fdr

    mwu_fdr = _fdr(mwu_p)
    t_fdr = _fdr(t_p)

    gm_exp = np.nanmean(ai_exp, axis=0)
    gm_nov = np.nanmean(ai_nov, axis=0)

    def _f(v):
        return round(float(v), 6) if np.isfinite(v) else np.nan

    rows = []
    for pi, pair_lbl in enumerate(UNIQUE_PAIR_LABELS):
        rows.append({
            "pair": pair_lbl,
            "is_lm1": "LM1" in pair_lbl,
            "mwu_U": _f(U_arr[pi]),
            "mwu_p_raw": _f(mwu_p[pi]),
            "mwu_p_fdr": _f(mwu_fdr[pi]),
            "mwu_significant": bool(np.isfinite(mwu_fdr[pi]) and mwu_fdr[pi] < alpha),
            "mwu_r": _f(mwu_r[pi]),
            "t_stat": _f(t_stat[pi]),
            "t_p_raw": _f(t_p[pi]),
            "t_p_fdr": _f(t_fdr[pi]),
            "t_significant": bool(np.isfinite(t_fdr[pi]) and t_fdr[pi] < alpha),
            "cohens_d": _f(d_arr[pi]),
            "exp_mean_ai": _f(gm_exp[pi]),
            "nov_mean_ai": _f(gm_nov[pi]),
            "ai_diff": _f(gm_exp[pi] - gm_nov[pi]) if (np.isfinite(gm_exp[pi]) and np.isfinite(gm_nov[pi])) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("test_per_connection_ai [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# H_AI2: Cross-pair Wilcoxon signed-rank on 10 group-mean differences
# ---------------------------------------------------------------------------

def test_cross_pair_ai(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[pd.DataFrame]:
    """
    H_AI2: Wilcoxon signed-rank + one-sample t-test on group-mean AI
    differences (Expert_AI - Novice_AI per undirected pair), one method.

    Returns
    -------
    Optional[pd.DataFrame]
        One row per test variant (all 10 pairs / PFC-LM1 / all-LM1 /
        non-LM1), plus one row per individual pair for reference.
    """
    from scipy.stats import wilcoxon, ttest_1samp, binomtest

    if config is None:
        config = load_config()

    out_path = _ai_stats_dir(method_key, config) / "ai_cross_pair_test.csv"
    if not overwrite and out_path.exists():
        logger.info("test_cross_pair_ai [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    alpha = float(_stats_cfg(config)["alpha"])

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None

    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)
    gm_exp = np.nanmean(ai_exp, axis=0)
    gm_nov = np.nanmean(ai_nov, axis=0)
    diff = gm_exp - gm_nov   # (10,)

    def _run_tests(d_full: np.ndarray, label: str, pair_mask: np.ndarray) -> dict:
        d = d_full[pair_mask & np.isfinite(d_full)]
        n = len(d)
        n_pos = int((d > 0).sum())
        result: dict = {
            "label": label,
            "n_pairs": n,
            "mean_diff": round(float(np.nanmean(d)), 6) if n > 0 else np.nan,
            "median_diff": round(float(np.nanmedian(d)), 6) if n > 0 else np.nan,
            "n_positive_diff": n_pos,
            "n_negative_diff": int((d < 0).sum()),
        }

        if n >= 5:
            try:
                stat_2s, p_2s = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
                result["wilcoxon_stat"] = round(float(stat_2s), 4)
                result["wilcoxon_p_twosided"] = round(float(p_2s), 6)
                result["wilcoxon_significant_twosided"] = bool(p_2s < alpha)
            except Exception as exc:
                logger.warning("Wilcoxon two-sided failed for %s: %s", label, exc)
                result.update({"wilcoxon_stat": np.nan, "wilcoxon_p_twosided": np.nan,
                                "wilcoxon_significant_twosided": False})
            try:
                _, p_1s = wilcoxon(d, alternative="greater", zero_method="wilcox")
                result["wilcoxon_p_onesided_greater"] = round(float(p_1s), 6)
                result["wilcoxon_significant_onesided"] = bool(p_1s < alpha)
            except Exception as exc:
                logger.warning("Wilcoxon one-sided failed for %s: %s", label, exc)
                result["wilcoxon_p_onesided_greater"] = np.nan
                result["wilcoxon_significant_onesided"] = False
        else:
            for key in ("wilcoxon_stat", "wilcoxon_p_twosided", "wilcoxon_p_onesided_greater"):
                result[key] = np.nan
            result["wilcoxon_significant_twosided"] = False
            result["wilcoxon_significant_onesided"] = False

        if n >= 1:
            try:
                sign_result = binomtest(n_pos, n=n, p=0.5, alternative="greater")
                result["sign_p_onesided_greater"] = round(float(sign_result.pvalue), 6)
                result["sign_significant_onesided"] = bool(sign_result.pvalue < alpha)
            except Exception:
                result["sign_p_onesided_greater"] = np.nan
                result["sign_significant_onesided"] = False
        else:
            result["sign_p_onesided_greater"] = np.nan
            result["sign_significant_onesided"] = False

        if n >= 2:
            try:
                t_stat, t_p = ttest_1samp(d, popmean=0.0)
                result["t_stat"] = round(float(t_stat), 4)
                result["t_p_twosided"] = round(float(t_p), 6)
                result["t_significant"] = bool(t_p < alpha)
            except Exception:
                result.update({"t_stat": np.nan, "t_p_twosided": np.nan, "t_significant": False})
        else:
            result.update({"t_stat": np.nan, "t_p_twosided": np.nan, "t_significant": False})

        return result

    all_mask = np.ones(len(UNIQUE_PAIRS), dtype=bool)
    pfc_lm1_mask = np.array([pi in PFC_LM1_PAIR_INDICES for pi in range(len(UNIQUE_PAIRS))])
    lm1_mask = np.array([pi in LM1_PAIR_INDICES for pi in range(len(UNIQUE_PAIRS))])
    non_lm1_mask = ~lm1_mask

    rows = [
        _run_tests(diff, "All 10 pairs", all_mask),
        _run_tests(diff, "PFC-LM1 pairs (3, primary)", pfc_lm1_mask),
        _run_tests(diff, "All LM1 pairs (4)", lm1_mask),
        _run_tests(diff, "Non-LM1 pairs", non_lm1_mask),
    ]

    for pi, pair_lbl in enumerate(UNIQUE_PAIR_LABELS):
        rows.append({
            "label": f"Pair: {pair_lbl}",
            "n_pairs": 1,
            "mean_diff": round(float(diff[pi]), 6) if np.isfinite(diff[pi]) else np.nan,
            "median_diff": round(float(diff[pi]), 6) if np.isfinite(diff[pi]) else np.nan,
            "n_positive_diff": int(diff[pi] > 0) if np.isfinite(diff[pi]) else 0,
            "n_negative_diff": int(diff[pi] < 0) if np.isfinite(diff[pi]) else 0,
            "wilcoxon_stat": np.nan, "wilcoxon_p_twosided": np.nan,
            "wilcoxon_significant_twosided": False,
            "wilcoxon_p_onesided_greater": np.nan, "wilcoxon_significant_onesided": False,
            "sign_p_onesided_greater": np.nan, "sign_significant_onesided": False,
            "t_stat": np.nan, "t_p_twosided": np.nan, "t_significant": False,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("test_cross_pair_ai [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# H_AI3: LM1-focused pre-specified analysis + permutation test
# ---------------------------------------------------------------------------

def test_lm1_focused_ai(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
    n_permutations: int = _N_PERMUTATIONS,
    rng_seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    H_AI3: Pre-specified targeted tests for the three PFC-LM1 pairs, one
    method. Individual tests (MWU + Welch's t, uncorrected) for each of the
    3 pairs, plus a pooled permutation test (21 Expert vs 18 Novice values).

    Returns
    -------
    Optional[pd.DataFrame]
        3 individual-pair rows + 1 pooled row.
    """
    from scipy.stats import mannwhitneyu, ttest_ind

    if config is None:
        config = load_config()

    out_path = _ai_stats_dir(method_key, config) / "ai_lm1_focused.csv"
    if not overwrite and out_path.exists():
        logger.info("test_lm1_focused_ai [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    alpha = float(_stats_cfg(config)["alpha"])

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None

    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)
    rng = np.random.default_rng(rng_seed)

    def _permutation_p(exp_vals: np.ndarray, nov_vals: np.ndarray, n_perm: int) -> tuple[float, float, float]:
        exp_clean = exp_vals[np.isfinite(exp_vals)]
        nov_clean = nov_vals[np.isfinite(nov_vals)]
        n_exp, n_nov = len(exp_clean), len(nov_clean)
        if n_exp < 2 or n_nov < 2:
            return np.nan, np.nan, np.nan
        pooled = np.concatenate([exp_clean, nov_clean])
        obs = float(np.mean(exp_clean) - np.mean(nov_clean))
        perm_diffs = np.empty(n_perm)
        for k in range(n_perm):
            perm = rng.permutation(pooled)
            perm_diffs[k] = np.mean(perm[:n_exp]) - np.mean(perm[n_exp:])
        p_2s = float(np.mean(np.abs(perm_diffs) >= abs(obs)))
        p_1s = float(np.mean(perm_diffs >= obs))
        return obs, p_2s, p_1s

    def _individual_test(exp_vals: np.ndarray, nov_vals: np.ndarray, pair_lbl: str) -> dict:
        exp_clean = exp_vals[np.isfinite(exp_vals)]
        nov_clean = nov_vals[np.isfinite(nov_vals)]
        row: dict = {
            "pair": pair_lbl, "test_type": "individual",
            "n_expert": len(exp_clean), "n_novice": len(nov_clean),
            "exp_mean_ai": round(float(np.nanmean(exp_clean)), 6) if len(exp_clean) > 0 else np.nan,
            "nov_mean_ai": round(float(np.nanmean(nov_clean)), 6) if len(nov_clean) > 0 else np.nan,
            "ai_diff_exp_minus_nov": np.nan,
        }
        if len(exp_clean) > 0 and len(nov_clean) > 0:
            row["ai_diff_exp_minus_nov"] = round(float(np.nanmean(exp_clean) - np.nanmean(nov_clean)), 6)

        if len(exp_clean) >= 2 and len(nov_clean) >= 2:
            try:
                U, p_2s = mannwhitneyu(exp_clean, nov_clean, alternative="two-sided")
                _, p_1s = mannwhitneyu(exp_clean, nov_clean, alternative="greater")
                row["mwu_U"] = round(float(U), 4)
                row["mwu_p_twosided"] = round(float(p_2s), 6)
                row["mwu_p_onesided_greater"] = round(float(p_1s), 6)
                row["mwu_significant_2s"] = bool(p_2s < alpha)
                row["mwu_significant_1s"] = bool(p_1s < alpha)
                row["mwu_r"] = round(_rank_biserial(float(U), len(exp_clean), len(nov_clean)), 4)
            except Exception:
                for k in ("mwu_U", "mwu_p_twosided", "mwu_p_onesided_greater", "mwu_r"):
                    row[k] = np.nan
                row["mwu_significant_2s"] = row["mwu_significant_1s"] = False

            try:
                t, tp = ttest_ind(exp_clean, nov_clean, equal_var=False)
                row["t_stat"] = round(float(t), 4)
                row["t_p_twosided"] = round(float(tp), 6)
                row["t_significant"] = bool(tp < alpha)
                row["cohens_d"] = round(_cohens_d(exp_clean, nov_clean), 4)
            except Exception:
                for k in ("t_stat", "t_p_twosided", "cohens_d"):
                    row[k] = np.nan
                row["t_significant"] = False

            obs, p_2s_perm, p_1s_perm = _permutation_p(exp_clean, nov_clean, n_permutations)
            row["perm_obs_diff"] = round(float(obs), 6) if np.isfinite(obs) else np.nan
            row["perm_p_twosided"] = round(float(p_2s_perm), 5) if np.isfinite(p_2s_perm) else np.nan
            row["perm_p_onesided_greater"] = round(float(p_1s_perm), 5) if np.isfinite(p_1s_perm) else np.nan
            row["perm_significant_2s"] = bool(np.isfinite(p_2s_perm) and p_2s_perm < alpha)
            row["perm_significant_1s"] = bool(np.isfinite(p_1s_perm) and p_1s_perm < alpha)
        else:
            for k in ("mwu_U", "mwu_p_twosided", "mwu_p_onesided_greater", "mwu_r",
                      "t_stat", "t_p_twosided", "cohens_d",
                      "perm_obs_diff", "perm_p_twosided", "perm_p_onesided_greater"):
                row[k] = np.nan
            for k in ("mwu_significant_2s", "mwu_significant_1s", "t_significant",
                      "perm_significant_2s", "perm_significant_1s"):
                row[k] = False

        return row

    rows = []
    for pi in PFC_LM1_PAIR_INDICES:
        pair_lbl = UNIQUE_PAIR_LABELS[pi]
        rows.append(_individual_test(ai_exp[:, pi], ai_nov[:, pi], pair_lbl))

    exp_lm1_pooled = np.concatenate([ai_exp[:, pi] for pi in PFC_LM1_PAIR_INDICES])
    nov_lm1_pooled = np.concatenate([ai_nov[:, pi] for pi in PFC_LM1_PAIR_INDICES])
    exp_lm1_clean = exp_lm1_pooled[np.isfinite(exp_lm1_pooled)]
    nov_lm1_clean = nov_lm1_pooled[np.isfinite(nov_lm1_pooled)]

    pooled_row: dict = {
        "pair": "PFC-LM1 - 3 pairs pooled (primary)",
        "test_type": "pooled_permutation",
        "n_expert": len(exp_lm1_clean), "n_novice": len(nov_lm1_clean),
        "exp_mean_ai": round(float(np.nanmean(exp_lm1_clean)), 6) if len(exp_lm1_clean) > 0 else np.nan,
        "nov_mean_ai": round(float(np.nanmean(nov_lm1_clean)), 6) if len(nov_lm1_clean) > 0 else np.nan,
        "ai_diff_exp_minus_nov": (
            round(float(np.nanmean(exp_lm1_clean) - np.nanmean(nov_lm1_clean)), 6)
            if (len(exp_lm1_clean) > 0 and len(nov_lm1_clean) > 0) else np.nan
        ),
    }

    if len(exp_lm1_clean) >= 2 and len(nov_lm1_clean) >= 2:
        try:
            U_pool, p_pool_2s = mannwhitneyu(exp_lm1_clean, nov_lm1_clean, alternative="two-sided")
            _, p_pool_1s = mannwhitneyu(exp_lm1_clean, nov_lm1_clean, alternative="greater")
            pooled_row["mwu_U"] = round(float(U_pool), 4)
            pooled_row["mwu_p_twosided"] = round(float(p_pool_2s), 6)
            pooled_row["mwu_p_onesided_greater"] = round(float(p_pool_1s), 6)
            pooled_row["mwu_significant_2s"] = bool(p_pool_2s < alpha)
            pooled_row["mwu_significant_1s"] = bool(p_pool_1s < alpha)
            pooled_row["mwu_r"] = round(_rank_biserial(float(U_pool), len(exp_lm1_clean), len(nov_lm1_clean)), 4)
        except Exception:
            for k in ("mwu_U", "mwu_p_twosided", "mwu_p_onesided_greater", "mwu_r"):
                pooled_row[k] = np.nan
            pooled_row["mwu_significant_2s"] = pooled_row["mwu_significant_1s"] = False

        try:
            t, tp = ttest_ind(exp_lm1_clean, nov_lm1_clean, equal_var=False)
            pooled_row["t_stat"] = round(float(t), 4)
            pooled_row["t_p_twosided"] = round(float(tp), 6)
            pooled_row["t_significant"] = bool(tp < alpha)
            pooled_row["cohens_d"] = round(_cohens_d(exp_lm1_clean, nov_lm1_clean), 4)
        except Exception:
            for k in ("t_stat", "t_p_twosided", "cohens_d"):
                pooled_row[k] = np.nan
            pooled_row["t_significant"] = False

        obs, p_2s_perm, p_1s_perm = _permutation_p(exp_lm1_clean, nov_lm1_clean, n_permutations)
        pooled_row["perm_obs_diff"] = round(float(obs), 6) if np.isfinite(obs) else np.nan
        pooled_row["perm_p_twosided"] = round(float(p_2s_perm), 5) if np.isfinite(p_2s_perm) else np.nan
        pooled_row["perm_p_onesided_greater"] = round(float(p_1s_perm), 5) if np.isfinite(p_1s_perm) else np.nan
        pooled_row["perm_significant_2s"] = bool(np.isfinite(p_2s_perm) and p_2s_perm < alpha)
        pooled_row["perm_significant_1s"] = bool(np.isfinite(p_1s_perm) and p_1s_perm < alpha)
    else:
        for k in ("mwu_U", "mwu_p_twosided", "mwu_p_onesided_greater", "mwu_r",
                  "t_stat", "t_p_twosided", "cohens_d",
                  "perm_obs_diff", "perm_p_twosided", "perm_p_onesided_greater"):
            pooled_row[k] = np.nan
        for k in ("mwu_significant_2s", "mwu_significant_1s", "t_significant",
                  "perm_significant_2s", "perm_significant_1s"):
            pooled_row[k] = False

    rows.append(pooled_row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("test_lm1_focused_ai [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def write_ai_statistics_summary(method_key: str, config: Optional[dict] = None) -> Path:
    """Aggregate key numbers from all three AI tests into a summary JSON."""
    if config is None:
        config = load_config()

    stats_dir = _ai_stats_dir(method_key, config)
    alpha = float(_stats_cfg(config)["alpha"])

    def _load(name: str) -> Optional[pd.DataFrame]:
        p = stats_dir / name
        return pd.read_csv(p) if p.exists() else None

    def _count_significant(df: Optional[pd.DataFrame], col: str) -> int:
        if df is None or col not in df.columns:
            return 0
        return int(df[col].fillna(False).sum())

    df_per = _load("ai_group_comparison.csv")
    df_cross = _load("ai_cross_pair_test.csv")
    df_lm1 = _load("ai_lm1_focused.csv")

    entry = _get_registry_entry(method_key, config)
    summary: dict = {
        "created_at": datetime.now().isoformat(),
        "method_key": method_key,
        "method_label": entry.get("label", method_key),
        "alpha": alpha,
        "n_permutations": _N_PERMUTATIONS,
        "h_ai1_per_connection": {
            "status": "ok" if df_per is not None else "missing",
            "n_pairs": int(len(df_per)) if df_per is not None else 0,
            "mwu_significant_fdr": _count_significant(df_per, "mwu_significant"),
            "t_significant_fdr": _count_significant(df_per, "t_significant"),
            "lm1_pairs": (
                df_per[df_per["is_lm1"] == True][
                    ["pair", "mwu_p_raw", "mwu_p_fdr", "mwu_r", "cohens_d",
                     "exp_mean_ai", "nov_mean_ai", "ai_diff"]
                ].to_dict(orient="records")
                if df_per is not None else []
            ),
        },
        "h_ai2_cross_pair": {
            "status": "ok" if df_cross is not None else "missing",
            "summary_rows": (
                df_cross[~df_cross["label"].str.startswith("Pair:")][
                    ["label", "n_pairs", "mean_diff", "median_diff",
                     "n_positive_diff", "n_negative_diff",
                     "wilcoxon_p_twosided", "wilcoxon_significant_twosided",
                     "wilcoxon_p_onesided_greater", "wilcoxon_significant_onesided",
                     "sign_p_onesided_greater", "sign_significant_onesided",
                     "t_p_twosided", "t_significant"]
                ].to_dict(orient="records")
                if df_cross is not None else []
            ),
        },
        "h_ai3_lm1_focused": {
            "status": "ok" if df_lm1 is not None else "missing",
            "pfc_lm1_pairs": PFC_LM1_PAIR_LABELS,
            "results": (
                df_lm1[["pair", "test_type", "n_expert", "n_novice",
                        "exp_mean_ai", "nov_mean_ai", "ai_diff_exp_minus_nov",
                        "mwu_p_twosided", "mwu_p_onesided_greater",
                        "mwu_significant_2s", "mwu_significant_1s", "mwu_r",
                        "perm_p_twosided", "perm_p_onesided_greater",
                        "perm_significant_2s", "perm_significant_1s"]].to_dict(orient="records")
                if df_lm1 is not None else []
            ),
        },
    }

    out_path = stats_dir / "ai_statistics_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)
    logger.info("write_ai_statistics_summary [%s]: saved -> %s", method_key, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_ai_per_subject(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Grouped box + scatter of per-subject AI for all 10 pairs, one method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if config is None:
        config = load_config()

    out_path = _ai_viz_dir(method_key, config) / "group_ai_box_per_subject.png"
    if not overwrite and out_path.exists():
        logger.info("plot_ai_per_subject [%s]: cache hit.", method_key)
        return out_path

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None
    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)

    comparison_csv = _ai_stats_dir(method_key, config) / "ai_group_comparison.csv"
    p_values: list[Optional[float]] = [None] * len(UNIQUE_PAIRS)
    if comparison_csv.exists():
        df_cmp = pd.read_csv(comparison_csv)
        for pi, lbl in enumerate(UNIQUE_PAIR_LABELS):
            row = df_cmp[df_cmp["pair"] == lbl]
            if not row.empty and "mwu_p_raw" in row.columns and pd.notna(row["mwu_p_raw"].iloc[0]):
                p_values[pi] = float(row["mwu_p_raw"].iloc[0])

    entry = _get_registry_entry(method_key, config)
    method_label = entry.get("label", method_key)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    n_pairs = len(UNIQUE_PAIRS)
    x = np.arange(n_pairs)
    width = 0.28
    jitter = 0.07
    color_exp, color_nov = "#1f77b4", "#d62728"

    for pi in LM1_PAIR_INDICES:
        ax.axvspan(pi - 0.5, pi + 0.5, color="#fffacd", alpha=0.55, zorder=0, label="_nolegend_")

    for pi in range(n_pairs):
        exp_vals = ai_exp[:, pi][np.isfinite(ai_exp[:, pi])]
        nov_vals = ai_nov[:, pi][np.isfinite(ai_nov[:, pi])]
        for vals, offset, color in [(exp_vals, -width / 2, color_exp), (nov_vals, width / 2, color_nov)]:
            if len(vals) >= 2:
                ax.boxplot(
                    vals, positions=[pi + offset], widths=width * 0.85, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.8),
                    boxprops=dict(facecolor=color, alpha=0.25),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.2),
                )
            rng_scatter = np.random.default_rng(pi * 100 + (0 if offset < 0 else 1))
            jx = pi + offset + rng_scatter.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(jx, vals, color=color, s=28, zorder=4, alpha=0.75, linewidths=0)

    for pi in range(n_pairs):
        p = p_values[pi]
        if p is None:
            continue
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "\u2020" if p < 0.1 else None
        if stars is None:
            continue
        exp_vals = ai_exp[:, pi][np.isfinite(ai_exp[:, pi])]
        nov_vals = ai_nov[:, pi][np.isfinite(ai_nov[:, pi])]
        all_vals = np.concatenate([exp_vals, nov_vals]) if (len(exp_vals) > 0 and len(nov_vals) > 0) else np.array([0.0])
        ax.text(pi, np.nanmax(all_vals) + 0.05, stars, ha="center", va="bottom",
                fontsize=10, color="black", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(UNIQUE_PAIR_LABELS, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Asymmetry Index (AI)\n(positive -> first-named ROI leads)", fontsize=10)
    ax.set_ylim(-1.15, 1.25)
    ax.set_xlim(-0.6, n_pairs - 0.4)

    for pi in LM1_PAIR_INDICES:
        ax.text(pi, -1.12, "LM1", ha="center", va="bottom", fontsize=7.5, color="#8B6914", style="italic")

    exp_patch = mpatches.Patch(facecolor=color_exp, alpha=0.7, label="Expert (n=7)")
    nov_patch = mpatches.Patch(facecolor=color_nov, alpha=0.7, label="Novice (n=6)")
    lm1_patch = mpatches.Patch(facecolor="#fffacd", alpha=0.9, edgecolor="#ccc", label="LM1 pair (pre-specified)")
    ax.legend(handles=[exp_patch, nov_patch, lm1_patch], fontsize=9, loc="upper right", framealpha=0.9)

    ax.text(0.01, 0.98, "\u2020 p<0.10  * p<0.05  ** p<0.01  *** p<0.001  (MWU, raw, no FDR)",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#555")

    ax.set_title(
        f"Per-Subject Asymmetry Index \u2014 {method_label}\n"
        "Expert vs Novice  |  AI = (fwd - rev)/(fwd + rev)  |  LM1 pairs shaded",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_ai_per_subject [%s]: saved -> %s", method_key, out_path)
    return out_path


def _stars_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return f"p={p:.3f}"


def plot_lm1_comparison(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Focused LM1 comparison plot: per-subject AI for the 3 PFC-LM1 pairs, one method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if config is None:
        config = load_config()

    out_path = _ai_viz_dir(method_key, config) / "group_ai_bar_lm1_pairs.png"
    if not overwrite and out_path.exists():
        logger.info("plot_lm1_comparison [%s]: cache hit.", method_key)
        return out_path

    B_Exp, B_Nov = load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return None
    ai_exp = compute_subject_ai(B_Exp)
    ai_nov = compute_subject_ai(B_Nov)

    lm1_csv = _ai_stats_dir(method_key, config) / "ai_lm1_focused.csv"
    perm_p: dict[str, float] = {}
    mwu_p: dict[str, float] = {}
    if lm1_csv.exists():
        df_lm1 = pd.read_csv(lm1_csv)
        for _, row in df_lm1.iterrows():
            if str(row.get("test_type")) == "individual":
                pair = str(row["pair"])
                if pd.notna(row.get("perm_p_twosided")):
                    perm_p[pair] = float(row["perm_p_twosided"])
                if pd.notna(row.get("mwu_p_twosided")):
                    mwu_p[pair] = float(row["mwu_p_twosided"])

    color_exp, color_nov = "#1f77b4", "#d62728"
    jitter = 0.08
    entry = _get_registry_entry(method_key, config)
    method_label = entry.get("label", method_key)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)

    for ax_idx, pi in enumerate(PFC_LM1_PAIR_INDICES):
        ax = axes[ax_idx]
        pair_lbl = UNIQUE_PAIR_LABELS[pi]
        exp_vals = ai_exp[:, pi][np.isfinite(ai_exp[:, pi])]
        nov_vals = ai_nov[:, pi][np.isfinite(ai_nov[:, pi])]
        rng = np.random.default_rng(pi * 7 + 13)

        for vals, xpos, color in [(exp_vals, 0.0, color_exp), (nov_vals, 1.0, color_nov)]:
            if len(vals) == 0:
                continue
            jx = xpos + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(jx, vals, color=color, s=45, zorder=4, alpha=0.8, linewidths=0.5, edgecolors="white")
            mean_v = np.nanmean(vals)
            sem_v = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax.errorbar(xpos, mean_v, yerr=sem_v, fmt="D", color=color, markersize=9, capsize=4, linewidth=2, zorder=5)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Expert", "Novice"], fontsize=10)
        ax.set_title(pair_lbl, fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Asymmetry Index (AI)", fontsize=10)

        def _stars(p: Optional[float]) -> str:
            if p is None:
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            if p < 0.1:
                return f"\u2020 p={p:.3f}"
            return f"p={p:.3f}"

        p_perm = perm_p.get(pair_lbl)
        p_mwu = mwu_p.get(pair_lbl)
        all_vals = np.concatenate([exp_vals, nov_vals]) if (len(exp_vals) > 0 and len(nov_vals) > 0) else np.array([0.0])
        y_max = np.nanmax(all_vals) + 0.08
        if p_perm is not None:
            ax.plot([0, 0, 1, 1], [y_max, y_max + 0.04, y_max + 0.04, y_max], lw=1.2, color="black")
            ax.text(0.5, y_max + 0.05,
                    f"MWU: {_stars(p_mwu) or _stars_p(p_mwu)}\nPerm: {_stars(p_perm) or _stars_p(p_perm)}",
                    ha="center", va="bottom", fontsize=8.5, color="black")

        ax.grid(axis="y", alpha=0.25, linestyle=":")

    exp_patch = mpatches.Patch(color=color_exp, alpha=0.8, label="Expert (n=7)")
    nov_patch = mpatches.Patch(color=color_nov, alpha=0.8, label="Novice (n=6)")
    fig.legend(handles=[exp_patch, nov_patch], loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.98), framealpha=0.9)

    fig.suptitle(
        f"LM1 Connection Asymmetry Index \u2014 Expert vs Novice\n{method_label}  |  "
        "AI>0: first-named ROI leads  |  Diamond = mean\u00b1SEM  |  Dots = individual subjects",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_lm1_comparison [%s]: saved -> %s", method_key, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Master runners (Section 10.9-10.10)
# ---------------------------------------------------------------------------

def run_ai_for_method(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> Optional[dict]:
    """
    Run the full H_AI1-H_AI3 battery + visualizations for one method.

    Returns
    -------
    Optional[dict]
        Keys: ``per_subject, per_connection, cross_pair, lm1, summary,
        fig_grouped, fig_lm1``. ``None`` if the method's arrays are
        unavailable (e.g. a WIRE ONLY track not yet batch-run) — logged as
        a warning, not raised, so ``run_ai_all_methods`` can skip cleanly.
    """
    if config is None:
        config = load_config()

    logger.info("=== run_ai_for_method[%s]: building per-subject AI ===", method_key)
    df_sub = build_ai_per_subject(method_key, config=config, overwrite=overwrite)
    if df_sub is None:
        logger.warning("run_ai_for_method [%s]: SKIPPED (no data available).", method_key)
        return None

    logger.info("=== run_ai_for_method[%s]: H_AI1 per-connection ===", method_key)
    df_per = test_per_connection_ai(method_key, config=config, overwrite=overwrite)

    logger.info("=== run_ai_for_method[%s]: H_AI2 cross-pair ===", method_key)
    df_cross = test_cross_pair_ai(method_key, config=config, overwrite=overwrite)

    logger.info("=== run_ai_for_method[%s]: H_AI3 LM1-focused ===", method_key)
    df_lm1 = test_lm1_focused_ai(method_key, config=config, overwrite=overwrite)

    summary_path = write_ai_statistics_summary(method_key, config=config)

    fig_grouped = plot_ai_per_subject(method_key, config=config, overwrite=overwrite)
    fig_lm1 = plot_lm1_comparison(method_key, config=config, overwrite=overwrite)

    return {
        "per_subject": df_sub, "per_connection": df_per, "cross_pair": df_cross,
        "lm1": df_lm1, "summary": summary_path, "fig_grouped": fig_grouped, "fig_lm1": fig_lm1,
    }


def run_ai_all_methods(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, Optional[dict]]:
    """
    Run AI for every registered track (``ai_track: true`` in the registry).

    This is the single entry point exercised by Section 10.12's
    extensibility proof: registering a new ``ai_track: true`` entry with a
    ``"symbolic_mean_npz"`` loader is picked up here with zero code changes.

    Returns
    -------
    dict[str, Optional[dict]]
        Keyed by method_key. Value is ``None`` for methods skipped due to
        missing data.
    """
    if config is None:
        config = load_config()

    tracks = registered_ai_tracks(config)
    logger.info("run_ai_all_methods: registered AI tracks = %s", tracks)

    results: dict[str, Optional[dict]] = {}
    ran, skipped = [], []
    for method_key in tracks:
        entry = _get_registry_entry(method_key, config)
        raw_kind = "ETE (floor-corrected)" if "ete" in method_key else "raw magnitude"
        logger.info("--- AI track: %s (%s) [%s] ---", method_key, entry.get("label", method_key), raw_kind)
        result = run_ai_for_method(method_key, config=config, overwrite=overwrite)
        results[method_key] = result
        (ran if result is not None else skipped).append(method_key)

    logger.info("run_ai_all_methods: DONE  ran=%s  skipped=%s", ran, skipped)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage E Section 10.9-10.10 -- Method-agnostic Asymmetry Index core"
    )
    parser.add_argument("--method", type=str, default=None,
                         help="Registry method_key to run (e.g. gc, phase_2, phase_4_raw, phase_4_ete).")
    parser.add_argument("--all", dest="run_all", action="store_true",
                         help="Run every registered ai_track (run_ai_all_methods()).")
    parser.set_defaults(overwrite=False)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    cfg = load_config()

    if args.run_all or args.method is None:
        results = run_ai_all_methods(config=cfg, overwrite=args.overwrite)
        for method_key, out in results.items():
            if out is None:
                logger.info("[%s] SKIPPED (no data)", method_key)
                continue
            df_per = out["per_connection"]
            n_sig_mwu = int(df_per["mwu_significant"].fillna(False).sum()) if df_per is not None else 0
            logger.info("[%s] H_AI1: %d/%d pairs significant (MWU FDR)", method_key, n_sig_mwu,
                        len(df_per) if df_per is not None else 0)
        return

    out = run_ai_for_method(args.method, config=cfg, overwrite=args.overwrite)
    if out is None:
        logger.warning("[%s] no data available -- nothing computed.", args.method)
        return
    df_per = out["per_connection"]
    n_sig_mwu = int(df_per["mwu_significant"].fillna(False).sum())
    n_sig_t = int(df_per["t_significant"].fillna(False).sum())
    logger.info("[%s] H_AI1: %d/%d significant (MWU FDR), %d (Welch FDR)", args.method, n_sig_mwu, len(df_per), n_sig_t)
    logger.info("[%s] Figures: %s  %s", args.method, out["fig_grouped"], out["fig_lm1"])


if __name__ == "__main__":
    _cli_main()
