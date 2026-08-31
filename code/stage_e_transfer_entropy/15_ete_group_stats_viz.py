"""
Stage E — Section 10.6: Method-Parameterized ETE Group Arrays, Statistics,
and Visualizations.

**Method-agnostic by design** — every public function here takes a
``method_key`` (e.g. ``"phase_4"``, later ``"phase_2"``) and resolves all
paths through the ``stage_e_transfer_entropy.significance_methods``
registry (``config.yaml``). No function body contains a hardcoded
``"phase_4"``/``"amplitude"`` string literal that would need to change to
support a second method — the **only** place a per-method detail lives is
the tiny ``_METHOD_ETE_FILENAME_TEMPLATES`` dict below (the per-subject
``.npz`` filename differs: ``..._amplitude_ete_connectivity.npz`` for
Phase 4 vs. ``..._multivariate_ete_connectivity.npz`` for Phase 2) — exactly the
"one small loader function" the registry's own docstring anticipates.

Mirrors the structure of ``phase_4/15_amplitude_dste_group.py`` (group
array aggregation) and ``phase_4/17_amplitude_dste_statistics.py``
(statistics), but generalised to operate on *any* registered symbolic-TE
method's ETE output schema (Section 10.5): ``connectivity`` (raw, kept for
comparison), ``ete_connectivity``, ``null_mean``, ``null_std``,
``p_value``, ``z_score``, ``delays``.

Functions
---------
``build_ete_group_arrays(method_key, config, overwrite)``
    Orchestrates all 3 aggregation strategies (all-windows / subject-mean /
    first-window) across all 7 array keys. Saves per-strategy, per-array-key
    ``.npz`` files to the method's ``ete_group_dir``.

``run_ete_statistics(method_key, config, overwrite)``
    Shapiro-Wilk normality, H1 Expert-vs-Novice (Mann-Whitney U + FDR) on
    ETE with a raw-TE comparison column, H0 within-trial / HIV across-trial
    stability, and the noise-floor-fraction diagnostic
    (``floor_fraction = mean(null_mean) / mean(raw_te)`` per connection per
    group).

``run_ete_visualizations(method_key, config, overwrite)``
    Null-distribution histograms (Gaussian approximation from the
    persisted ``null_mean``/``null_std`` — see note below) for the most
    significant connections, floor-vs-signal stacked bars, and an
    Expert/Novice ETE delta bar chart.

Note on null-distribution histograms
-------------------------------------
The Section 10.5 output schema intentionally persists only
``null_mean``/``null_std`` per connection per window (not the full
``n_surrogates``-length empirical null array — that would be 500× larger
per connection and was not in the specified schema). The visualization
therefore draws a **Gaussian approximation** ``N(null_mean, null_std)``
rather than the exact empirical histogram. With ``n_surrogates=500``, the
null distribution of a frequency-count TE statistic is well-approximated
by a normal distribution per the CLT, so this is a reasonable and clearly
labeled simplification — the overlaid ``raw_te`` marker and annotated
``p_value`` are always the exact values from the real surrogate test, only
the *background curve* is an approximation for illustration.

Called this pass with ``method_key="phase_4"`` only (10.6). Verified (grep)
that no function body below contains the literal strings ``"phase_4"`` or
``"amplitude"`` outside of the ``_METHOD_ETE_FILENAME_TEMPLATES`` lookup
table and CLI default argument — confirming the method-parameterization
completion check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).parent.parent          # code/
_STAGE_E_ROOT = Path(__file__).parent              # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_N_TRIALS_CANONICAL = 5

# The 7 array keys present in every method's ETE per-subject .npz (Section 10.5).
_ARRAY_KEYS = [
    "connectivity", "ete_connectivity", "null_mean", "null_std",
    "p_value", "z_score", "delays",
]

# The ONE place a per-method literal lives (see module docstring). Adding a
# 3rd method requires exactly one new entry here — no other function below
# needs to change.
_METHOD_ETE_FILENAME_TEMPLATES: dict[str, str] = {
    "phase_4": "sub-{sid}_amplitude_ete_connectivity.npz",
    "phase_2": "sub-{sid}_multivariate_ete_connectivity.npz",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _registry_entry(method_key: str, config: dict) -> dict:
    """Look up a method's entry in ``stage_e_transfer_entropy.significance_methods``."""
    te_cfg = config.get("stage_e_transfer_entropy", {})
    sm = te_cfg.get("significance_methods", {})
    if method_key not in sm:
        raise KeyError(
            f"_registry_entry: method_key={method_key!r} not found in "
            f"stage_e_transfer_entropy.significance_methods (available: "
            f"{list(sm.keys())})"
        )
    entry = sm[method_key]
    if not entry.get("has_ete", False):
        raise ValueError(
            f"_registry_entry: method_key={method_key!r} has has_ete=false — "
            "no ETE track registered for this method."
        )
    return entry


def _ete_conn_dir(method_key: str, config: dict) -> Path:
    """Resolve the method's per-subject ETE connectivity directory."""
    from utils_te import get_te_output_dir

    entry = _registry_entry(method_key, config)
    return get_te_output_dir(entry["ete_connectivity_dir"], config=config)


def _ete_group_dir(method_key: str, config: dict) -> Path:
    """Resolve (and create) the method's ETE group-array output directory."""
    from utils_te import get_te_output_dir

    entry = _registry_entry(method_key, config)
    d = get_te_output_dir(entry["ete_group_dir"], config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ete_stats_dir(method_key: str, config: dict) -> Path:
    """Resolve (and create) the method's ETE statistics output directory."""
    from utils_te import get_te_output_dir

    entry = _registry_entry(method_key, config)
    stats_key = entry.get("ete_group_dir", "").replace("_group", "_statistics")
    d = get_te_output_dir(stats_key, config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ete_viz_dir(method_key: str, config: dict) -> Path:
    """Resolve (and create) the method's ETE visualization output directory."""
    from utils_te import get_te_output_dir

    entry = _registry_entry(method_key, config)
    viz_key = entry.get("ete_group_dir", "").replace("_group", "_viz")
    d = get_te_output_dir(viz_key, config=config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _connection_labels() -> list[str]:
    from utils_te import get_connection_labels

    return get_connection_labels()


def _subject_filename(method_key: str, subject_id: str) -> str:
    """Resolve a subject's per-method ETE .npz filename."""
    if method_key not in _METHOD_ETE_FILENAME_TEMPLATES:
        raise KeyError(
            f"_subject_filename: no filename template registered for "
            f"method_key={method_key!r}. Add one to "
            f"_METHOD_ETE_FILENAME_TEMPLATES."
        )
    return _METHOD_ETE_FILENAME_TEMPLATES[method_key].format(sid=subject_id)


def _load_subject_ete_npz_lists(
    method_key: str,
    expert_ids: list[str],
    novice_ids: list[str],
    conn_dir: Path,
) -> tuple[list[Optional[dict]], list[Optional[dict]]]:
    """Load per-subject ETE ``.npz`` dicts for Expert and Novice groups."""

    def _load(subject_ids: list[str]) -> list[Optional[dict]]:
        results: list[Optional[dict]] = []
        for sid in subject_ids:
            fpath = conn_dir / _subject_filename(method_key, sid)
            if not fpath.exists():
                logger.warning(
                    "_load_subject_ete_npz_lists [%s]: missing file for subject %s (%s)",
                    method_key, sid, fpath.name,
                )
                results.append(None)
            else:
                results.append(dict(np.load(fpath, allow_pickle=True)))
        return results

    return _load(expert_ids), _load(novice_ids)


def _stats_cfg(config: dict) -> dict:
    return config["stage_e_transfer_entropy"]["statistics"]


# ===========================================================================
# Section 10.6a — Group array aggregation (3 strategies × 7 array keys)
# ===========================================================================

def build_ete_group_arrays_all_windows(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build all-windows group arrays for every array key, for one method.

    Parameters
    ----------
    method_key : str
        Registry key (e.g. ``"phase_4"``).
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, load cached ``.npz`` files when available.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{array_key: {'Exp': {trial_id: (n_subs, 20, max_wins)},
        'Nov': {trial_id: (n_subs, 20, max_wins)}}}``.

    Notes
    -----
    Saved files (in the method's ``ete_group_dir``):
    ``B_Exp_ete_all_{array_key}.npz``, ``B_Nov_ete_all_{array_key}.npz``
    for each of the 7 array keys, plus ``ete_group_summary.json``.
    """
    if config is None:
        config = load_config()

    group_dir = _ete_group_dir(method_key, config)
    exp_paths = {k: group_dir / f"B_Exp_ete_all_{k}.npz" for k in _ARRAY_KEYS}
    nov_paths = {k: group_dir / f"B_Nov_ete_all_{k}.npz" for k in _ARRAY_KEYS}

    if not overwrite and all(p.exists() for p in list(exp_paths.values()) + list(nov_paths.values())):
        logger.info("build_ete_group_arrays_all_windows [%s]: cache hit.", method_key)
        result: dict = {}
        for k in _ARRAY_KEYS:
            exp_d = dict(np.load(exp_paths[k], allow_pickle=True))
            nov_d = dict(np.load(nov_paths[k], allow_pickle=True))
            result[k] = {
                "Exp": {t: v for t, v in exp_d.items() if t.startswith("Tr_")},
                "Nov": {t: v for t, v in nov_d.items() if t.startswith("Tr_")},
            }
        return result

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _ete_conn_dir(method_key, config)
    expert_npz_list, novice_npz_list = _load_subject_ete_npz_lists(
        method_key, expert_ids, novice_ids, conn_dir
    )
    all_npz = [d for d in expert_npz_list + novice_npz_list if d is not None]
    if not all_npz:
        raise RuntimeError(
            f"build_ete_group_arrays_all_windows [{method_key}]: no subject "
            f"ETE files found in {conn_dir}. Run the batch pipeline first."
        )

    max_trials = max(int(d["n_trials"]) for d in all_npz)
    max_wins_per_trial: dict[str, int] = {}
    for trial_idx in range(max_trials):
        trial_id = f"Tr_{trial_idx + 1}"
        max_w = 0
        for d in all_npz:
            key = f"{trial_id}_connectivity"
            if key in d and d[key].ndim == 2 and d[key].shape[1] > 0:
                max_w = max(max_w, int(d[key].shape[1]))
        max_wins_per_trial[trial_id] = max_w

    n_conn = len(_connection_labels())
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {
        k: {"Exp": {}, "Nov": {}} for k in _ARRAY_KEYS
    }

    for array_key in _ARRAY_KEYS:
        for trial_id, max_w in max_wins_per_trial.items():
            if max_w == 0:
                continue
            B_Exp = np.full((len(expert_ids), n_conn, max_w), np.nan)
            B_Nov = np.full((len(novice_ids), n_conn, max_w), np.nan)

            for sub_idx, npz_data in enumerate(expert_npz_list):
                if npz_data is None:
                    continue
                key = f"{trial_id}_{array_key}"
                if key in npz_data:
                    arr = npz_data[key]
                    if arr.ndim == 2 and arr.shape[1] > 0:
                        B_Exp[sub_idx, :, : arr.shape[1]] = arr

            for sub_idx, npz_data in enumerate(novice_npz_list):
                if npz_data is None:
                    continue
                key = f"{trial_id}_{array_key}"
                if key in npz_data:
                    arr = npz_data[key]
                    if arr.ndim == 2 and arr.shape[1] > 0:
                        B_Nov[sub_idx, :, : arr.shape[1]] = arr

            result[array_key]["Exp"][trial_id] = B_Exp
            result[array_key]["Nov"][trial_id] = B_Nov

        exp_save = dict(result[array_key]["Exp"])
        exp_save["subject_ids"] = np.array(expert_ids)
        exp_save["strategy"] = np.array("all_windows")
        np.savez(exp_paths[array_key], **exp_save)

        nov_save = dict(result[array_key]["Nov"])
        nov_save["subject_ids"] = np.array(novice_ids)
        nov_save["strategy"] = np.array("all_windows")
        np.savez(nov_paths[array_key], **nov_save)

    logger.info(
        "build_ete_group_arrays_all_windows [%s]: saved %d array keys × 2 groups "
        "(trials=%s)", method_key, len(_ARRAY_KEYS), sorted(max_wins_per_trial.keys()),
    )
    return result


def build_ete_group_arrays_subject_mean(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build subject-mean group arrays (shape ``(n_subs, 20, 5)``) for every
    array key, for one method. Primary input for the H1 test (10.6).

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        ``{array_key: (B_Exp_mean, B_Nov_mean)}``.
    """
    if config is None:
        config = load_config()

    group_dir = _ete_group_dir(method_key, config)
    exp_paths = {k: group_dir / f"B_Exp_ete_mean_{k}.npz" for k in _ARRAY_KEYS}
    nov_paths = {k: group_dir / f"B_Nov_ete_mean_{k}.npz" for k in _ARRAY_KEYS}

    if not overwrite and all(p.exists() for p in list(exp_paths.values()) + list(nov_paths.values())):
        logger.info("build_ete_group_arrays_subject_mean [%s]: cache hit.", method_key)
        return {
            k: (
                np.load(exp_paths[k], allow_pickle=True)["data"],
                np.load(nov_paths[k], allow_pickle=True)["data"],
            )
            for k in _ARRAY_KEYS
        }

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _ete_conn_dir(method_key, config)
    expert_npz_list, novice_npz_list = _load_subject_ete_npz_lists(
        method_key, expert_ids, novice_ids, conn_dir
    )
    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]], array_key: str) -> np.ndarray:
        """Average windows within each trial for one group and one array key.

        Each subject's ETE ``.npz`` stores ``Tr_N_{array_key}`` as
        ``(20, n_windows)``. This helper collapses the window axis with
        ``nanmean`` so H1 can use one observation per subject × trial.

        Parameters
        ----------
        subject_ids : list of str
            Group membership in row order (Expert or Novice IDs). Length
            sets ``arr.shape[0]``; IDs themselves are unused beyond that.
        npz_list : list of dict or None
            Parallel to ``subject_ids``. Each dict is one loaded subject
            ``.npz``; ``None`` leaves that subject's row as NaN.
        array_key : str
            One of ``_ARRAY_KEYS`` (e.g. ``"ete_connectivity"``). Combined
            with the trial id to form keys like ``Tr_1_ete_connectivity``.

        Returns
        -------
        np.ndarray, shape (n_subjects, 20, 5)
            Window-averaged values. Axis 0 = subject, axis 1 = directed
            connection, axis 2 = trial (``Tr_1`` … ``Tr_5``). Missing
            files, missing trial keys, or empty window axes stay NaN.
        """
        arr = np.full((len(subject_ids), n_conn, _N_TRIALS_CANONICAL), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                continue
            for trial_idx in range(_N_TRIALS_CANONICAL):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_{array_key}"
                if key not in npz_data:
                    continue
                col = npz_data[key]
                if col.ndim == 2 and col.shape[1] > 0:
                    arr[sub_idx, :, trial_idx] = np.nanmean(col, axis=1)
        return arr

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for array_key in _ARRAY_KEYS:
        B_Exp_mean = _build(expert_ids, expert_npz_list, array_key)
        B_Nov_mean = _build(novice_ids, novice_npz_list, array_key)
        np.savez(exp_paths[array_key], data=B_Exp_mean, subject_ids=np.array(expert_ids),
                  strategy=np.array("subject_mean"))
        np.savez(nov_paths[array_key], data=B_Nov_mean, subject_ids=np.array(novice_ids),
                  strategy=np.array("subject_mean"))
        result[array_key] = (B_Exp_mean, B_Nov_mean)

    logger.info(
        "build_ete_group_arrays_subject_mean [%s]: saved %d array keys",
        method_key, len(_ARRAY_KEYS),
    )
    return result


def build_ete_group_arrays_first_window(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build first-window group arrays (shape ``(n_subs, 20, 5)``) for every
    array key, for one method. Sensitivity check for the H1 test.
    """
    if config is None:
        config = load_config()

    group_dir = _ete_group_dir(method_key, config)
    exp_paths = {k: group_dir / f"B_Exp_ete_first_{k}.npz" for k in _ARRAY_KEYS}
    nov_paths = {k: group_dir / f"B_Nov_ete_first_{k}.npz" for k in _ARRAY_KEYS}

    if not overwrite and all(p.exists() for p in list(exp_paths.values()) + list(nov_paths.values())):
        logger.info("build_ete_group_arrays_first_window [%s]: cache hit.", method_key)
        return {
            k: (
                np.load(exp_paths[k], allow_pickle=True)["data"],
                np.load(nov_paths[k], allow_pickle=True)["data"],
            )
            for k in _ARRAY_KEYS
        }

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    conn_dir = _ete_conn_dir(method_key, config)
    expert_npz_list, novice_npz_list = _load_subject_ete_npz_lists(
        method_key, expert_ids, novice_ids, conn_dir
    )
    n_conn = len(_connection_labels())

    def _build(subject_ids: list[str], npz_list: list[Optional[dict]], array_key: str) -> np.ndarray:
        """Keep only the first window of each trial for one group and one array key.

        Same layout as the nested ``_build`` in
        ``build_ete_group_arrays_subject_mean``, but instead of averaging
        windows this takes ``col[:, 0]`` — a sensitivity check for H1
        (does the Expert-vs-Novice result hold on the first window alone).

        Parameters
        ----------
        subject_ids : list of str
            Group membership in row order (Expert or Novice IDs). Length
            sets ``arr.shape[0]``; IDs themselves are unused beyond that.
        npz_list : list of dict or None
            Parallel to ``subject_ids``. Each dict is one loaded subject
            ``.npz``; ``None`` leaves that subject's row as NaN.
        array_key : str
            One of ``_ARRAY_KEYS`` (e.g. ``"ete_connectivity"``). Combined
            with the trial id to form keys like ``Tr_1_ete_connectivity``.

        Returns
        -------
        np.ndarray, shape (n_subjects, 20, 5)
            First-window values. Axis 0 = subject, axis 1 = directed
            connection, axis 2 = trial (``Tr_1`` … ``Tr_5``). Missing
            files, missing trial keys, or empty window axes stay NaN.
        """
        arr = np.full((len(subject_ids), n_conn, _N_TRIALS_CANONICAL), np.nan)
        for sub_idx, npz_data in enumerate(npz_list):
            if npz_data is None:
                continue
            for trial_idx in range(_N_TRIALS_CANONICAL):
                trial_id = f"Tr_{trial_idx + 1}"
                key = f"{trial_id}_{array_key}"
                if key not in npz_data:
                    continue
                col = npz_data[key]
                if col.ndim == 2 and col.shape[1] > 0:
                    arr[sub_idx, :, trial_idx] = col[:, 0]
        return arr

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for array_key in _ARRAY_KEYS:
        B_Exp_first = _build(expert_ids, expert_npz_list, array_key)
        B_Nov_first = _build(novice_ids, novice_npz_list, array_key)
        np.savez(exp_paths[array_key], data=B_Exp_first, subject_ids=np.array(expert_ids),
                  strategy=np.array("first_window"))
        np.savez(nov_paths[array_key], data=B_Nov_first, subject_ids=np.array(novice_ids),
                  strategy=np.array("first_window"))
        result[array_key] = (B_Exp_first, B_Nov_first)

    logger.info(
        "build_ete_group_arrays_first_window [%s]: saved %d array keys",
        method_key, len(_ARRAY_KEYS),
    )
    return result


def build_ete_group_arrays(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Orchestrate all 3 ETE group-array aggregation strategies (Section 10.6).

    Parameters
    ----------
    method_key : str
        Registry key (e.g. ``"phase_4"``).
    config : dict, optional
        Parsed pipeline configuration.
    overwrite : bool
        If ``False``, skip strategies whose output files already exist.

    Returns
    -------
    dict
        ``{'all_windows': ..., 'subject_mean': ..., 'first_window': ...}``.
    """
    if config is None:
        config = load_config()

    results: dict = {}
    logger.info("build_ete_group_arrays [%s]: strategy=all_windows ...", method_key)
    results["all_windows"] = build_ete_group_arrays_all_windows(
        method_key, config=config, overwrite=overwrite
    )
    logger.info("build_ete_group_arrays [%s]: strategy=subject_mean ...", method_key)
    results["subject_mean"] = build_ete_group_arrays_subject_mean(
        method_key, config=config, overwrite=overwrite
    )
    logger.info("build_ete_group_arrays [%s]: strategy=first_window ...", method_key)
    results["first_window"] = build_ete_group_arrays_first_window(
        method_key, config=config, overwrite=overwrite
    )

    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    expert_ids = list(subj_cfg.get("Expert", []))
    novice_ids = list(subj_cfg.get("Novice", []))

    group_dir = _ete_group_dir(method_key, config)
    summary = {
        "created_at": datetime.now().isoformat(),
        "method_key": method_key,
        "n_experts": len(expert_ids),
        "n_novices": len(novice_ids),
        "expert_ids": expert_ids,
        "novice_ids": novice_ids,
        "array_keys": _ARRAY_KEYS,
        "connection_labels": _connection_labels(),
    }
    with open(group_dir / "ete_group_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    logger.info("build_ete_group_arrays [%s]: DONE -> %s", method_key, group_dir)
    return results


# ===========================================================================
# Section 10.6b — Statistics (Shapiro-Wilk, H1, H0, HIV, floor-fraction)
# ===========================================================================

def _import_p1_stats():
    """Import Phase 1's ``04_bivariate_dspte_statistics.py`` for reusable generic helpers."""
    p = _STAGE_E_ROOT / "phase_1" / "04_bivariate_dspte_statistics.py"
    spec = importlib.util.spec_from_file_location("bivariate_dspte_statistics_p1_for_ete", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_ete_normality(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Shapiro-Wilk normality test on ETE subject-mean arrays (one method)."""
    if config is None:
        config = load_config()

    out_path = _ete_stats_dir(method_key, config) / "ete_normality_test.csv"
    if not overwrite and out_path.exists():
        logger.info("run_ete_normality [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    p1_stats = _import_p1_stats()
    mean_arrays = build_ete_group_arrays_subject_mean(method_key, config=config, overwrite=False)
    B_Exp_ete, B_Nov_ete = mean_arrays["ete_connectivity"]
    df = p1_stats.run_normality_tests_te(B_Exp_ete, B_Nov_ete, _connection_labels(), config)
    df.to_csv(out_path, index=False)
    logger.info("run_ete_normality [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


def _run_mwu_matrix(
    B_Exp: np.ndarray,
    B_Nov: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Mann-Whitney U (mwu) per (trial, connection); returns (p_raw, p_fdr, U_stat)."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection

    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])
    min_n = int(cfg.get("min_subjects_anova", 3))
    n_trials = B_Exp.shape[2]
    n_conn = B_Exp.shape[1]

    p_raw_mat = np.full((n_trials, n_conn), np.nan)
    p_fdr_mat = np.full((n_trials, n_conn), np.nan)
    u_stat_mat = np.full((n_trials, n_conn), np.nan)

    for t_idx in range(n_trials):
        p_raw_trial = np.full(n_conn, np.nan)
        u_trial = np.full(n_conn, np.nan)
        for ci in range(n_conn):
            exp_vals = B_Exp[:, ci, t_idx]
            nov_vals = B_Nov[:, ci, t_idx]
            exp_vals = exp_vals[~np.isnan(exp_vals)]
            nov_vals = nov_vals[~np.isnan(nov_vals)]
            if len(exp_vals) < min_n or len(nov_vals) < min_n:
                continue
            try:
                stat, p = mannwhitneyu(exp_vals, nov_vals, alternative="two-sided")
                u_trial[ci] = float(stat)
                p_raw_trial[ci] = float(p)
            except Exception:
                pass

        valid = ~np.isnan(p_raw_trial)
        p_fdr_trial = np.full(n_conn, np.nan)
        if valid.sum() > 0:
            _, p_corr = fdrcorrection(p_raw_trial[valid], alpha=alpha, method=fdr_method)
            p_fdr_trial[valid] = p_corr

        p_raw_mat[t_idx] = p_raw_trial
        p_fdr_mat[t_idx] = p_fdr_trial
        u_stat_mat[t_idx] = u_trial

    return p_raw_mat, p_fdr_mat, u_stat_mat


def run_ete_h1(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    H1 Expert-vs-Novice on ETE (Mann-Whitney U + FDR), with a raw-TE
    comparison column (Section 10.6).

    Returns
    -------
    pd.DataFrame
        Columns: ``trial, connection, stat_U, p_raw, p_fdr, significant,
        p_fw, p_fdr_fw, significant_fw, strategy_agreement,
        raw_te_p_raw, raw_te_significant, ete_sharper_than_raw``.
    """
    if config is None:
        config = load_config()

    out_path = _ete_stats_dir(method_key, config) / "ete_H1_expert_vs_novice.csv"
    if not overwrite and out_path.exists():
        logger.info("run_ete_h1 [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    alpha = float(cfg["alpha"])
    conn_labels = _connection_labels()
    n_conn = len(conn_labels)

    mean_arrays = build_ete_group_arrays_subject_mean(method_key, config=config, overwrite=False)
    first_arrays = build_ete_group_arrays_first_window(method_key, config=config, overwrite=False)

    B_Exp_ete_mean, B_Nov_ete_mean = mean_arrays["ete_connectivity"]
    B_Exp_ete_fw, B_Nov_ete_fw = first_arrays["ete_connectivity"]
    B_Exp_raw_mean, B_Nov_raw_mean = mean_arrays["connectivity"]

    p_raw_mean, p_fdr_mean, u_mean = _run_mwu_matrix(B_Exp_ete_mean, B_Nov_ete_mean, cfg)
    p_raw_fw, p_fdr_fw, _ = _run_mwu_matrix(B_Exp_ete_fw, B_Nov_ete_fw, cfg)
    p_raw_raw_te, p_fdr_raw_te, _ = _run_mwu_matrix(B_Exp_raw_mean, B_Nov_raw_mean, cfg)

    rows = []
    for t_idx in range(_N_TRIALS_CANONICAL):
        trial_id = f"Tr_{t_idx + 1}"
        for ci in range(n_conn):
            p_r = p_raw_mean[t_idx, ci]
            p_f = p_fdr_mean[t_idx, ci]
            p_r_fw = p_raw_fw[t_idx, ci]
            p_f_fw = p_fdr_fw[t_idx, ci]
            p_raw_te = p_raw_raw_te[t_idx, ci]
            p_fdr_te = p_fdr_raw_te[t_idx, ci]

            sig_ete = bool(p_f < alpha) if not np.isnan(p_f) else False
            # NOTE: significance is judged on the FDR-corrected p-value
            # (p_fdr_te), matching the convention used everywhere else in
            # this pipeline (e.g. 17_amplitude_dste_statistics.py's H1) —
            # NOT the uncorrected raw_te_p_raw, which is retained only for
            # transparency/diagnostic purposes.
            sig_raw = bool(p_fdr_te < alpha) if not np.isnan(p_fdr_te) else False

            rows.append({
                "trial": trial_id,
                "connection": conn_labels[ci],
                "stat_U": float(u_mean[t_idx, ci]) if not np.isnan(u_mean[t_idx, ci]) else np.nan,
                "p_raw": round(float(p_r), 6) if not np.isnan(p_r) else np.nan,
                "p_fdr": round(float(p_f), 6) if not np.isnan(p_f) else np.nan,
                "significant": sig_ete,
                "p_fw": round(float(p_r_fw), 6) if not np.isnan(p_r_fw) else np.nan,
                "p_fdr_fw": round(float(p_f_fw), 6) if not np.isnan(p_f_fw) else np.nan,
                "significant_fw": bool(p_f_fw < alpha) if not np.isnan(p_f_fw) else False,
                "strategy_agreement": (
                    sig_ete == (bool(p_f_fw < alpha) if not np.isnan(p_f_fw) else False)
                ),
                "raw_te_p_raw": round(float(p_raw_te), 6) if not np.isnan(p_raw_te) else np.nan,
                "raw_te_p_fdr": round(float(p_fdr_te), 6) if not np.isnan(p_fdr_te) else np.nan,
                "raw_te_significant": sig_raw,
                "ete_sharper_than_raw": sig_ete and not sig_raw,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("run_ete_h1 [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


def _run_anova_stability(
    B_group_dict: dict[str, dict[str, np.ndarray]],
    conn_labels: list[str],
    cfg: dict,
    axis: str,
) -> pd.DataFrame:
    """
    Shared ANOVA-based stability test core for H0 (within-trial, windows)
    and HIV (across-trial, per window).

    Parameters
    ----------
    B_group_dict : dict
        ``{'Exp': {trial_id: (n_subs, 20, n_wins)}, 'Nov': {...}}`` for the
        ``ete_connectivity`` array key.
    axis : str
        ``'within_trial'`` (ANOVA across windows, one row per trial) or
        ``'across_trial'`` (ANOVA across trials, one row per window).
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multitest import fdrcorrection

    min_n = int(cfg["min_subjects_anova"])
    min_groups = int(cfg["min_groups_anova"])
    alpha = float(cfg["alpha"])
    fdr_method = str(cfg["fdr_method"])
    n_conn = len(conn_labels)

    def _round(v, n=6):
        return round(float(v), n) if not np.isnan(v) else np.nan

    def _apply_fdr(p_values: np.ndarray) -> np.ndarray:
        p_fdr = np.full_like(p_values, np.nan)
        valid = ~np.isnan(p_values)
        if valid.sum() > 0:
            _, p_corr = fdrcorrection(p_values[valid], alpha=alpha, method=fdr_method)
            p_fdr[valid] = p_corr
        return p_fdr

    rows = []
    for group_name, B_group in [("Expert", B_group_dict["Exp"]), ("Novice", B_group_dict["Nov"])]:
        if axis == "within_trial":
            for trial_id, arr in sorted(B_group.items()):
                n_wins_total = arr.shape[2]
                p_raw_vec = np.full(n_conn, np.nan)
                f_stat_vec = np.full(n_conn, np.nan)
                for ci in range(n_conn):
                    groups = []
                    for win_idx in range(n_wins_total):
                        x = arr[:, ci, win_idx]
                        x = x[~np.isnan(x)]
                        if len(x) >= min_n:
                            groups.append(x)
                    if len(groups) >= min_groups:
                        f_stat, p_raw = f_oneway(*groups)
                        p_raw_vec[ci] = float(p_raw)
                        f_stat_vec[ci] = float(f_stat)
                p_fdr_vec = _apply_fdr(p_raw_vec)
                for ci in range(n_conn):
                    rows.append({
                        "group": group_name, "trial": trial_id,
                        "connection": conn_labels[ci], "n_windows": n_wins_total,
                        "F_stat": _round(f_stat_vec[ci]), "p_raw": _round(p_raw_vec[ci]),
                        "p_fdr": _round(p_fdr_vec[ci]),
                        "significant": bool(p_fdr_vec[ci] < alpha) if not np.isnan(p_fdr_vec[ci]) else False,
                    })
        else:  # across_trial
            hiv_max_windows = int(cfg.get("hiv_max_windows", 3))
            hiv_min_trials = int(cfg.get("hiv_min_trials", 2))
            all_trial_ids = sorted(B_group.keys())
            for win_idx in range(hiv_max_windows):
                p_raw_vec = np.full(n_conn, np.nan)
                f_stat_vec = np.full(n_conn, np.nan)
                n_trials_used_vec = np.zeros(n_conn, dtype=int)
                for ci in range(n_conn):
                    trial_groups = []
                    for trial_id in all_trial_ids:
                        arr = B_group[trial_id]
                        if win_idx >= arr.shape[2]:
                            continue
                        x = arr[:, ci, win_idx]
                        x_valid = x[~np.isnan(x)]
                        if len(x_valid) >= min_n:
                            trial_groups.append(x_valid)
                    n_trials_used_vec[ci] = len(trial_groups)
                    if len(trial_groups) >= max(min_groups, hiv_min_trials):
                        f_stat, p_raw = f_oneway(*trial_groups)
                        f_stat_vec[ci] = float(f_stat)
                        p_raw_vec[ci] = float(p_raw)
                p_fdr_vec = _apply_fdr(p_raw_vec)
                for ci in range(n_conn):
                    rows.append({
                        "group": group_name, "window": f"Win_{win_idx + 1}",
                        "connection": conn_labels[ci],
                        "n_trials_used": int(n_trials_used_vec[ci]),
                        "F_stat": _round(f_stat_vec[ci]), "p_raw": _round(p_raw_vec[ci]),
                        "p_fdr": _round(p_fdr_vec[ci]),
                        "significant": bool(p_fdr_vec[ci] < alpha) if not np.isnan(p_fdr_vec[ci]) else False,
                    })

    return pd.DataFrame(rows)


def run_ete_h0(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """H0 within-trial (across-window) stability ANOVA on ETE (Section 10.6)."""
    if config is None:
        config = load_config()

    out_path = _ete_stats_dir(method_key, config) / "ete_H0_within_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("run_ete_h0 [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    all_arrays = build_ete_group_arrays_all_windows(method_key, config=config, overwrite=False)
    B_dict = all_arrays["ete_connectivity"]
    df = _run_anova_stability(B_dict, _connection_labels(), cfg, axis="within_trial")
    df.to_csv(out_path, index=False)
    logger.info("run_ete_h0 [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


def run_ete_hiv(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """HIV across-trial stability ANOVA on ETE (Section 10.6)."""
    if config is None:
        config = load_config()

    out_path = _ete_stats_dir(method_key, config) / "ete_HIV_across_trial_stability.csv"
    if not overwrite and out_path.exists():
        logger.info("run_ete_hiv [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    cfg = _stats_cfg(config)
    all_arrays = build_ete_group_arrays_all_windows(method_key, config=config, overwrite=False)
    B_dict = all_arrays["ete_connectivity"]
    df = _run_anova_stability(B_dict, _connection_labels(), cfg, axis="across_trial")
    df.to_csv(out_path, index=False)
    logger.info("run_ete_hiv [%s]: saved %s (%d rows)", method_key, out_path.name, len(df))
    return df


def run_ete_floor_fraction(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute the noise-floor-fraction diagnostic (Section 10.6):

        floor_fraction = mean(null_mean) / mean(raw_te)

    per connection per group — the direct quantitative answer to "how much
    of what we measured is chance-level floor". A value near 1.0 means the
    raw TE is almost entirely floor (little/no true signal); a value near
    0.0 means the floor is negligible relative to the raw signal.

    Returns
    -------
    pd.DataFrame
        Columns: ``group, connection, mean_raw_te, mean_null_mean,
        mean_ete, floor_fraction``.
    """
    if config is None:
        config = load_config()

    out_path = _ete_stats_dir(method_key, config) / "ete_floor_fraction.csv"
    if not overwrite and out_path.exists():
        logger.info("run_ete_floor_fraction [%s]: cache hit.", method_key)
        return pd.read_csv(out_path)

    mean_arrays = build_ete_group_arrays_subject_mean(method_key, config=config, overwrite=False)
    B_Exp_raw, B_Nov_raw = mean_arrays["connectivity"]
    B_Exp_null, B_Nov_null = mean_arrays["null_mean"]
    B_Exp_ete, B_Nov_ete = mean_arrays["ete_connectivity"]
    conn_labels = _connection_labels()
    n_conn = len(conn_labels)

    rows = []
    for group_name, B_raw, B_null, B_ete in [
        ("Expert", B_Exp_raw, B_Exp_null, B_Exp_ete),
        ("Novice", B_Nov_raw, B_Nov_null, B_Nov_ete),
    ]:
        for ci in range(n_conn):
            mean_raw = float(np.nanmean(B_raw[:, ci, :]))
            mean_null = float(np.nanmean(B_null[:, ci, :]))
            mean_ete = float(np.nanmean(B_ete[:, ci, :]))
            floor_fraction = mean_null / mean_raw if abs(mean_raw) > 1e-12 else np.nan
            rows.append({
                "group": group_name,
                "connection": conn_labels[ci],
                "mean_raw_te": round(mean_raw, 6),
                "mean_null_mean": round(mean_null, 6),
                "mean_ete": round(mean_ete, 6),
                "floor_fraction": round(floor_fraction, 4) if not np.isnan(floor_fraction) else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(
        "run_ete_floor_fraction [%s]: saved %s (%d rows, mean floor_fraction=%.3f)",
        method_key, out_path.name, len(df), df["floor_fraction"].mean(),
    )
    return df


def run_ete_statistics(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Execute the full ETE statistics workflow for one method (Section 10.6).

    Builds group arrays (idempotent with cache), then runs normality, H1,
    H0, HIV, and the floor-fraction diagnostic; writes a JSON summary.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by ``'normality'``, ``'h1'``, ``'h0'``, ``'hiv'``, ``'floor_fraction'``.
    """
    if config is None:
        config = load_config()

    build_ete_group_arrays(method_key, config=config, overwrite=overwrite)

    df_normal = run_ete_normality(method_key, config=config, overwrite=overwrite)
    df_h1 = run_ete_h1(method_key, config=config, overwrite=overwrite)
    df_h0 = run_ete_h0(method_key, config=config, overwrite=overwrite)
    df_hiv = run_ete_hiv(method_key, config=config, overwrite=overwrite)
    df_floor = run_ete_floor_fraction(method_key, config=config, overwrite=overwrite)

    out = {"normality": df_normal, "h1": df_h1, "h0": df_h0, "hiv": df_hiv, "floor_fraction": df_floor}

    stats_dir = _ete_stats_dir(method_key, config)
    summary = {
        "created_at": datetime.now().isoformat(),
        "method_key": method_key,
        "n_h1_significant_ete": int(df_h1["significant"].fillna(False).sum()),
        "n_h1_significant_raw_te": int(df_h1["raw_te_significant"].fillna(False).sum()),
        "n_ete_sharper_than_raw": int(df_h1["ete_sharper_than_raw"].fillna(False).sum()),
        "mean_floor_fraction_expert": round(
            float(df_floor[df_floor["group"] == "Expert"]["floor_fraction"].mean()), 4
        ),
        "mean_floor_fraction_novice": round(
            float(df_floor[df_floor["group"] == "Novice"]["floor_fraction"].mean()), 4
        ),
        "n_normality_rows": int(len(df_normal)),
        "n_h0_significant": int(df_h0["significant"].fillna(False).sum()),
        "n_hiv_significant": int(df_hiv["significant"].fillna(False).sum()),
    }
    with open(stats_dir / "ete_statistics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    logger.info(
        "run_ete_statistics [%s]: DONE  H1_ete_sig=%d  H1_raw_sig=%d  "
        "mean_floor_fraction(Exp/Nov)=%.3f/%.3f",
        method_key, summary["n_h1_significant_ete"], summary["n_h1_significant_raw_te"],
        summary["mean_floor_fraction_expert"], summary["mean_floor_fraction_novice"],
    )
    return out


# ===========================================================================
# Section 10.6c — Visualizations
# ===========================================================================

def run_ete_visualizations(
    method_key: str,
    config: Optional[dict] = None,
    overwrite: bool = False,
    n_representative: int = 4,
) -> Path:
    """
    Generate ETE diagnostic visualizations for one method (Section 10.6).

    Produces one multi-panel figure:
      1. Null-distribution histograms (Gaussian approximation — see module
         docstring) for the ``n_representative`` most-significant
         connections (lowest grand-mean p-value), with observed raw TE and
         p-value annotated.
      2. Floor-vs-signal stacked bar chart (mean ``null_mean`` floor +
         mean ``ete`` signal = mean raw TE) per connection, per group.
      3. Expert/Novice ETE delta bar chart (mean ETE Expert − mean ETE
         Novice) across all 20 connections.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import norm as _norm

    if config is None:
        config = load_config()

    viz_dir = _ete_viz_dir(method_key, config)
    out_path = viz_dir / "ete_diagnostics.png"
    if not overwrite and out_path.exists():
        logger.info("run_ete_visualizations [%s]: cache hit -> %s", method_key, out_path)
        return out_path

    conn_labels = _connection_labels()
    n_conn = len(conn_labels)

    mean_arrays = build_ete_group_arrays_subject_mean(method_key, config=config, overwrite=False)
    B_Exp_raw, B_Nov_raw = mean_arrays["connectivity"]
    B_Exp_null, B_Nov_null = mean_arrays["null_mean"]
    B_Exp_std, B_Nov_std = mean_arrays["null_std"]
    B_Exp_ete, B_Nov_ete = mean_arrays["ete_connectivity"]
    B_Exp_p, B_Nov_p = mean_arrays["p_value"]

    grand_raw = np.nanmean(np.concatenate([B_Exp_raw, B_Nov_raw], axis=0), axis=(0, 2))
    grand_null_mean = np.nanmean(np.concatenate([B_Exp_null, B_Nov_null], axis=0), axis=(0, 2))
    grand_null_std = np.nanmean(np.concatenate([B_Exp_std, B_Nov_std], axis=0), axis=(0, 2))
    grand_ete = np.nanmean(np.concatenate([B_Exp_ete, B_Nov_ete], axis=0), axis=(0, 2))
    grand_p = np.nanmean(np.concatenate([B_Exp_p, B_Nov_p], axis=0), axis=(0, 2))

    order = np.argsort(grand_p)
    rep_idx = [i for i in order if not np.isnan(grand_p[i])][:n_representative]

    fig = plt.figure(figsize=(15, 13))
    n_hist_cols = min(len(rep_idx), 4) if rep_idx else 1
    gs = gridspec.GridSpec(
        3, max(n_hist_cols, 1),
        height_ratios=[1.0, 1.1, 1.1],
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.08, left=0.06, right=0.98,
    )

    # --- Panel 1: null-distribution histograms (Gaussian approximation) ---
    for k, ci in enumerate(rep_idx):
        ax = fig.add_subplot(gs[0, k])
        mu, sigma = grand_null_mean[ci], max(grand_null_std[ci], 1e-9)
        x = np.linspace(mu - 4 * sigma, max(mu + 4 * sigma, grand_raw[ci] + sigma), 300)
        ax.plot(x, _norm.pdf(x, mu, sigma), color="steelblue", lw=1.5,
                label="Null (Gaussian approx.)")
        ax.fill_between(x, _norm.pdf(x, mu, sigma), color="steelblue", alpha=0.25)
        ax.axvline(grand_raw[ci], color="darkred", lw=2, linestyle="--",
                   label=f"raw TE = {grand_raw[ci]:.4f}")
        ax.axvline(mu, color="gray", lw=1, linestyle=":", label=f"floor = {mu:.4f}")
        ax.set_title(
            f"{conn_labels[ci]}\np={grand_p[ci]:.3f}  ETE={grand_ete[ci]:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("TE value", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right")

    # --- Panel 2: floor-vs-signal stacked bars ---
    ax_stack = fig.add_subplot(gs[1, :])
    x_pos = np.arange(n_conn)
    exp_floor = np.nanmean(B_Exp_null, axis=(0, 2))
    exp_signal = np.nanmean(B_Exp_ete, axis=(0, 2))
    nov_floor = np.nanmean(B_Nov_null, axis=(0, 2))
    nov_signal = np.nanmean(B_Nov_ete, axis=(0, 2))
    width = 0.35

    ax_stack.bar(x_pos - width / 2, exp_floor, width, color="#cccccc", label="Expert floor (null_mean)")
    ax_stack.bar(x_pos - width / 2, exp_signal, width, bottom=exp_floor, color="#1f77b4", label="Expert ETE (signal)")
    ax_stack.bar(x_pos + width / 2, nov_floor, width, color="#e8b4b4", label="Novice floor (null_mean)")
    ax_stack.bar(x_pos + width / 2, nov_signal, width, bottom=nov_floor, color="#d62728", label="Novice ETE (signal)")

    ax_stack.set_xticks(x_pos)
    ax_stack.set_xticklabels(conn_labels, rotation=60, ha="right", fontsize=7)
    ax_stack.set_ylabel("Mean TE", fontsize=9)
    ax_stack.set_title(
        "Floor (surrogate null mean) vs. Signal (ETE) per connection — "
        "raw TE = floor + ETE (stacked bar height)",
        fontsize=10, fontweight="bold",
    )
    ax_stack.legend(fontsize=7.5, ncol=4, loc="upper right")
    ax_stack.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Panel 3: Expert/Novice ETE delta bar ---
    ax_delta = fig.add_subplot(gs[2, :])
    delta = exp_signal - nov_signal
    colors = ["#1f77b4" if d >= 0 else "#d62728" for d in delta]
    ax_delta.bar(x_pos, delta, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(x_pos)
    ax_delta.set_xticklabels(conn_labels, rotation=60, ha="right", fontsize=7)
    ax_delta.set_ylabel("ETE(Expert) − ETE(Novice)", fontsize=9)
    ax_delta.set_title(
        "Expert − Novice ETE Delta per Connection  (blue = Expert > Novice, red = Novice > Expert)",
        fontsize=10, fontweight="bold",
    )
    ax_delta.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Effective Transfer Entropy Diagnostics — method: {method_key}\n"
        f"ETE = raw TE − mean(circular-shift surrogate null)",
        fontsize=12, fontweight="bold", y=0.985,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("run_ete_visualizations [%s]: saved -> %s", method_key, out_path)
    return out_path


# ===========================================================================
# CLI
# ===========================================================================

def _cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stage E Section 10.6 -- Method-parameterized ETE group/stats/viz"
    )
    parser.add_argument("--method", type=str, default="phase_4",
                         help="Registry key in stage_e_transfer_entropy.significance_methods (default: phase_4)")
    parser.add_argument("--step", choices=["group", "statistics", "visualizations", "all"],
                         default="all")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    cfg = load_config()

    if args.step in ("group", "all"):
        build_ete_group_arrays(args.method, config=cfg, overwrite=args.overwrite)
    if args.step in ("statistics", "all"):
        out = run_ete_statistics(args.method, config=cfg, overwrite=args.overwrite)
        logger.info(
            "[%s] statistics: normality=%d h1=%d h0=%d hiv=%d floor_fraction=%d",
            args.method, len(out["normality"]), len(out["h1"]), len(out["h0"]),
            len(out["hiv"]), len(out["floor_fraction"]),
        )
    if args.step in ("visualizations", "all"):
        fig_path = run_ete_visualizations(args.method, config=cfg, overwrite=args.overwrite)
        logger.info("[%s] visualizations -> %s", args.method, fig_path)


if __name__ == "__main__":
    _cli_main()
