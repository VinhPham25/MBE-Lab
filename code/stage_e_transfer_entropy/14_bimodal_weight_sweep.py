"""
Stage E — Phase 3 Bimodal Weight Sweep Utility.

This script evaluates candidate bimodal fusion weights using already-saved Phase 3
branch connectivity outputs:
  - `sub-*_phase_branch_connectivity.npz`
  - `sub-*_amplitude_branch_connectivity.npz`

For each candidate `w_phase` (with `w_amplitude = 1 - w_phase`), it computes
window-level fused connectivity vectors and scores them using directionality
asymmetry and related summary metrics.

Primary objective (default)
---------------------------
Maximise mean matrix antisymmetry:
`mean(|EC[i,j] - EC[j,i]|)` over upper-triangle pairs.

Output
------
Writes ranked JSON to Phase 3 group directory, e.g.:
`outputs/stage_e_transfer_entropy/.../phase_3_bimodal_dspte/group/bimodal_weight_sweep.json`
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

_CODE_ROOT = Path(__file__).parent.parent
_STAGE_E_ROOT = Path(__file__).parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def _load_config_if_none(config: Optional[dict]) -> dict:
    """
    Load project configuration when not provided.

    Parameters
    ----------
    config : dict, optional
        Existing parsed config dictionary.

    Returns
    -------
    dict
        Parsed `config.yaml` dictionary.
    """
    if config is not None:
        return config
    return load_config()


def _load_bimodal_module():
    """
    Load `11_bimodal_dspte.py` via importlib.

    Returns
    -------
    module
        Loaded bimodal module object.
    """
    path = _STAGE_E_ROOT / "11_bimodal_dspte.py"
    spec = importlib.util.spec_from_file_location("bimodal_dspte_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalise_weights(w_phase: float, w_amplitude: float) -> tuple[float, float]:
    """
    Normalise non-negative fusion weights to sum to 1.

    Parameters
    ----------
    w_phase : float
        Raw phase branch weight.
    w_amplitude : float
        Raw amplitude branch weight.

    Returns
    -------
    tuple[float, float]
        Normalised `(w_phase, w_amplitude)` pair.
    """
    wp = max(0.0, float(w_phase))
    wa = max(0.0, float(w_amplitude))
    s = wp + wa
    if s <= 0:
        return 0.5, 0.5
    return wp / s, wa / s


def _get_subjects(config: dict, subjects: Optional[list[str]]) -> list[str]:
    """
    Resolve subject list for sweep.

    Parameters
    ----------
    config : dict
        Parsed pipeline configuration.
    subjects : list[str], optional
        Optional user-provided subject IDs.

    Returns
    -------
    list[str]
        Subject IDs to evaluate.
    """
    if subjects:
        return [s.strip() for s in subjects if s.strip()]
    te_cfg = config.get("stage_e_transfer_entropy", {})
    subj_cfg = te_cfg.get("subjects", {})
    return list(subj_cfg.get("Expert", [])) + list(subj_cfg.get("Novice", []))


def _parse_trial_filter(training_trials_raw: str) -> set[str]:
    """
    Parse trial filter string into a set of trial IDs.

    Parameters
    ----------
    training_trials_raw : str
        Comma-separated trial IDs (e.g., `Tr_1,Tr_2`).

    Returns
    -------
    set[str]
        Allowed trial IDs.
    """
    items = [x.strip() for x in training_trials_raw.split(",") if x.strip()]
    return set(items)


def _connection_schema_from_labels(conn_labels: list[str]) -> tuple[list[str], dict]:
    """
    Build ROI schema and directed edge index from connection labels.

    Parameters
    ----------
    conn_labels : list[str]
        Connection labels in `SRC→TGT` format.

    Returns
    -------
    tuple[list[str], dict]
        - ROI name list
        - mapping `{(src_name, tgt_name): vec_index}`
    """
    roi_names: list[str] = []
    edge_to_index: dict = {}
    for idx, lbl in enumerate(conn_labels):
        parts = lbl.split("→")
        if len(parts) != 2:
            continue
        src, tgt = parts[0], parts[1]
        if src not in roi_names:
            roi_names.append(src)
        if tgt not in roi_names:
            roi_names.append(tgt)
        edge_to_index[(src, tgt)] = idx
    return roi_names, edge_to_index


def _vec20_to_matrix(vec: np.ndarray, conn_labels: list[str]) -> np.ndarray:
    """
    Reconstruct directed connectivity matrix from a 20-connection vector.

    Parameters
    ----------
    vec : np.ndarray
        Connectivity vector, expected shape `(n_conn,)`.
    conn_labels : list[str]
        Label list aligned with vector indices.

    Returns
    -------
    np.ndarray
        Reconstructed square connectivity matrix `(n_rois, n_rois)`.
    """
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    roi_names, edge_to_index = _connection_schema_from_labels(conn_labels)
    n_rois = len(roi_names)
    mat = np.zeros((n_rois, n_rois), dtype=np.float64)
    roi_to_idx = {name: i for i, name in enumerate(roi_names)}

    for (src, tgt), vec_idx in edge_to_index.items():
        if vec_idx >= vec.size:
            continue
        i = roi_to_idx[src]
        j = roi_to_idx[tgt]
        mat[i, j] = vec[vec_idx]
    return mat


def _matrix_abs_asymmetry(ec_matrix: np.ndarray) -> float:
    """
    Compute mean absolute antisymmetry for one connectivity matrix.

    Parameters
    ----------
    ec_matrix : np.ndarray
        Directed connectivity matrix `(n_rois, n_rois)`.

    Returns
    -------
    float
        `mean(|EC[i,j] - EC[j,i]|)` over upper-triangle pairs.
    """
    diff = np.abs(ec_matrix - ec_matrix.T)
    tri = np.triu_indices(ec_matrix.shape[0], k=1)
    vals = diff[tri]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))


def _matrix_net_asymmetry(ec_matrix: np.ndarray) -> float:
    """
    Compute signed mean asymmetry for one connectivity matrix.

    Parameters
    ----------
    ec_matrix : np.ndarray
        Directed connectivity matrix `(n_rois, n_rois)`.

    Returns
    -------
    float
        `mean(EC[i,j] - EC[j,i])` over upper-triangle pairs.
    """
    diff = ec_matrix - ec_matrix.T
    tri = np.triu_indices(ec_matrix.shape[0], k=1)
    vals = diff[tri]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))


def _load_subject_branch_data(
    subject_id: str,
    config: dict,
) -> Optional[dict]:
    """
    Load phase/amplitude branch data for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    config : dict
        Parsed pipeline configuration.

    Returns
    -------
    Optional[dict]
        Dictionary with `phase`, `amp`, `conn_labels`, and `trial_ids`,
        or `None` when required files are missing.
    """
    from stage_e_transfer_entropy.utils_te import get_te_output_dir

    conn_dir = get_te_output_dir("phase_3_connectivity", config=config)
    phase_path = conn_dir / f"sub-{subject_id}_phase_branch_connectivity.npz"
    amp_path = conn_dir / f"sub-{subject_id}_amplitude_branch_connectivity.npz"

    if not phase_path.exists() or not amp_path.exists():
        return None

    phase = dict(np.load(phase_path, allow_pickle=True))
    amp = dict(np.load(amp_path, allow_pickle=True))

    conn_labels = [str(x) for x in phase.get("connection_labels", [])]
    trial_ids = [str(x) for x in phase.get("trial_ids", [])]
    return {
        "phase": phase,
        "amp": amp,
        "conn_labels": conn_labels,
        "trial_ids": trial_ids,
    }


def _collect_window_samples(
    subjects: list[str],
    config: dict,
    trial_filter: Optional[set[str]] = None,
) -> tuple[list[dict], dict]:
    """
    Collect aligned phase/amplitude connectivity vectors per subject/trial/window.

    Parameters
    ----------
    subjects : list[str]
        Subject IDs to scan.
    config : dict
        Parsed pipeline configuration.
    trial_filter : set[str], optional
        Optional allowed trial IDs (e.g., `{Tr_1, Tr_2}`).

    Returns
    -------
    tuple[list[dict], dict]
        - List of window sample dicts with keys:
          `subject_id`, `trial_id`, `window_idx`, `phase_vec`, `amp_vec`, `conn_labels`
        - Metadata dictionary summarising inclusion/exclusion counts.
    """
    samples: list[dict] = []
    subject_meta: dict = {}

    for sid in subjects:
        loaded = _load_subject_branch_data(sid, config)
        if loaded is None:
            subject_meta[sid] = {"status": "missing_branch_files", "n_windows": 0}
            continue

        phase = loaded["phase"]
        amp = loaded["amp"]
        conn_labels = loaded["conn_labels"]
        trial_ids = loaded["trial_ids"]

        n_sub_windows = 0
        for tr_id in trial_ids:
            if trial_filter is not None and tr_id not in trial_filter:
                continue
            key = f"{tr_id}_connectivity"
            if key not in phase or key not in amp:
                continue

            phase_arr = np.asarray(phase[key], dtype=np.float64)
            amp_arr = np.asarray(amp[key], dtype=np.float64)
            if phase_arr.ndim != 2 or amp_arr.ndim != 2:
                continue

            n_wins = min(phase_arr.shape[1], amp_arr.shape[1])
            for i in range(n_wins):
                samples.append(
                    {
                        "subject_id": sid,
                        "trial_id": tr_id,
                        "window_idx": i,
                        "phase_vec": phase_arr[:, i],
                        "amp_vec": amp_arr[:, i],
                        "conn_labels": conn_labels,
                    }
                )
                n_sub_windows += 1

        subject_meta[sid] = {
            "status": "ok" if n_sub_windows > 0 else "no_valid_windows",
            "n_windows": int(n_sub_windows),
        }

    meta = {
        "n_subjects_requested": len(subjects),
        "n_subjects_with_windows": int(sum(1 for v in subject_meta.values() if v["status"] == "ok")),
        "subjects": subject_meta,
        "n_windows_total": int(len(samples)),
    }
    return samples, meta


def _evaluate_weight(
    w_phase: float,
    samples: list[dict],
) -> dict:
    """
    Evaluate one fusion weight on collected window samples.

    Parameters
    ----------
    w_phase : float
        Phase weight in `[0, 1]`.
    samples : list[dict]
        Collected branch samples from `_collect_window_samples()`.

    Returns
    -------
    dict
        Metric dictionary for this weight.
    """
    wp, wa = _normalise_weights(w_phase, 1.0 - w_phase)

    asym_vals: list[float] = []
    net_asym_vals: list[float] = []
    fused_mean_vals: list[float] = []
    phase_contrib_vals: list[float] = []
    amp_contrib_vals: list[float] = []

    for s in samples:
        phase_vec = np.asarray(s["phase_vec"], dtype=np.float64)
        amp_vec = np.asarray(s["amp_vec"], dtype=np.float64)
        fused_vec = wp * phase_vec + wa * amp_vec

        mat = _vec20_to_matrix(fused_vec, s["conn_labels"])
        asym_vals.append(_matrix_abs_asymmetry(mat))
        net_asym_vals.append(_matrix_net_asymmetry(mat))
        fused_mean_vals.append(float(np.nanmean(fused_vec)))

        phase_contrib_vals.append(float(np.nanmean(wp * phase_vec)))
        amp_contrib_vals.append(float(np.nanmean(wa * amp_vec)))

    return {
        "w_phase": float(wp),
        "w_amplitude": float(wa),
        "n_windows": int(len(samples)),
        "mean_abs_asymmetry": float(np.nanmean(asym_vals)) if asym_vals else 0.0,
        "std_abs_asymmetry": float(np.nanstd(asym_vals)) if asym_vals else 0.0,
        "mean_signed_asymmetry": float(np.nanmean(net_asym_vals)) if net_asym_vals else 0.0,
        "mean_fused_te": float(np.nanmean(fused_mean_vals)) if fused_mean_vals else 0.0,
        "std_fused_te": float(np.nanstd(fused_mean_vals)) if fused_mean_vals else 0.0,
        "mean_phase_contribution": float(np.nanmean(phase_contrib_vals)) if phase_contrib_vals else 0.0,
        "mean_amplitude_contribution": float(np.nanmean(amp_contrib_vals)) if amp_contrib_vals else 0.0,
    }


def sweep_bimodal_weights(
    subjects: Optional[list[str]] = None,
    w_phase_values: Optional[list[float]] = None,
    use_training_trials_only: bool = True,
    training_trials: Optional[list[str]] = None,
    config: Optional[dict] = None,
    output_filename: str = "bimodal_weight_sweep.json",
) -> dict:
    """
    Run a grid sweep over candidate bimodal weights and rank results.

    Parameters
    ----------
    subjects : list[str], optional
        Subject IDs to include. Defaults to configured Expert+Novice lists.
    w_phase_values : list[float], optional
        Candidate phase weights in `[0,1]`. Defaults to `np.arange(0, 1.01, 0.05)`.
    use_training_trials_only : bool
        If `True`, evaluate only training trials (default `Tr_1`, `Tr_2`).
    training_trials : list[str], optional
        Trial IDs to include when `use_training_trials_only=True`.
    config : dict, optional
        Parsed pipeline configuration.
    output_filename : str
        Output JSON filename written under Phase 3 group directory.

    Returns
    -------
    dict
        Sweep summary payload including ranked results and selected best weight.
    """
    config = _load_config_if_none(config)

    if w_phase_values is None:
        w_phase_values = [float(x) for x in np.arange(0.0, 1.0001, 0.05)]

    if training_trials is None:
        training_trials = ["Tr_1", "Tr_2"]

    subject_list = _get_subjects(config, subjects)
    trial_filter = set(training_trials) if use_training_trials_only else None

    samples, sample_meta = _collect_window_samples(
        subjects=subject_list,
        config=config,
        trial_filter=trial_filter,
    )
    if not samples:
        raise RuntimeError(
            "sweep_bimodal_weights: no valid phase/amplitude branch windows found. "
            "Run Phase 3 branch generation first."
        )

    results = []
    for w in w_phase_values:
        results.append(_evaluate_weight(w_phase=float(w), samples=samples))

    results_sorted = sorted(results, key=lambda x: x["mean_abs_asymmetry"], reverse=True)
    best = results_sorted[0]

    from stage_e_transfer_entropy.utils_te import get_te_output_dir

    group_dir = get_te_output_dir("phase_3_group", config=config)
    group_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(),
        "phase": "phase_3_bimodal_dspte",
        "objective": "max_mean_abs_asymmetry",
        "use_training_trials_only": bool(use_training_trials_only),
        "training_trials": training_trials if use_training_trials_only else None,
        "subjects_requested": subject_list,
        "sample_metadata": sample_meta,
        "n_candidate_weights": int(len(w_phase_values)),
        "best_weight": best,
        "ranked_results": results_sorted,
    }

    out_path = group_dir / output_filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "sweep_bimodal_weights: saved %s (best w_phase=%.3f, asym=%.6f)",
        out_path,
        best["w_phase"],
        best["mean_abs_asymmetry"],
    )
    return payload


def _parse_weights_arg(weights_raw: Optional[str], step: float) -> list[float]:
    """
    Parse CLI weight candidates.

    Parameters
    ----------
    weights_raw : str, optional
        Comma-separated explicit weights, e.g. `0.1,0.3,0.5`.
    step : float
        Grid step used when explicit weights are not provided.

    Returns
    -------
    list[float]
        Candidate phase weights in `[0,1]`.
    """
    if weights_raw:
        vals = []
        for x in weights_raw.split(","):
            x = x.strip()
            if not x:
                continue
            vals.append(float(x))
        return vals

    if step <= 0:
        raise ValueError("step must be > 0")
    grid = np.arange(0.0, 1.0 + 1e-12, step)
    return [float(v) for v in grid]

'''
Example CLI usage:
python3 code/stage_e_transfer_entropy/14_bimodal_weight_sweep.py \
  --weights 0.0,0.25,0.5,0.75,1.0 \
  --output bimodal_weight_sweep_A1.json
'''

def _cli_main() -> None:
    """CLI entrypoint for bimodal weight sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Sweep Phase 3 bimodal fusion weights")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional explicit subject IDs (default: all configured).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated candidate w_phase values (default uses --step grid)." \
        "The inputted value is the w_phase weight; w_amplitude is computed as (1 - w_phase) for each candidate.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step for w_phase when --weights is omitted.",
    )
    parser.add_argument(
        "--training-trials",
        type=str,
        default="Tr_1,Tr_2",
        help="Comma-separated trial IDs for sweep when not using --all-trials.",
    )
    parser.add_argument(
        "--all-trials",
        action="store_true",
        help="Use all available trials instead of training-only trials.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bimodal_weight_sweep.json",
        help="Output JSON filename under phase_3_group/.",
    )
    args = parser.parse_args()

    cfg = _load_config_if_none(None)
    w_values = _parse_weights_arg(args.weights, args.step)
    training_trials = [x.strip() for x in args.training_trials.split(",") if x.strip()]

    payload = sweep_bimodal_weights(
        subjects=args.subjects,
        w_phase_values=w_values,
        use_training_trials_only=not args.all_trials,
        training_trials=training_trials,
        config=cfg,
        output_filename=args.output,
    )

    best = payload["best_weight"]
    logger.info(
        "Best weight: w_phase=%.3f w_amplitude=%.3f mean_abs_asymmetry=%.6f",
        best["w_phase"],
        best["w_amplitude"],
        best["mean_abs_asymmetry"],
    )


if __name__ == "__main__":
    _cli_main()
