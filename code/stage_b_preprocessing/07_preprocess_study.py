"""
Stage B — Step B7: Full Study Batch Processing
Purpose: Process all subjects in a study through the complete Stage B pipeline
         (B1 OD → B2 bandpass → B3 motion detect → B4 rPCA → B5 MBLL → B6 GLM/OLS)
         and produce BIDS-compliant .fif outputs plus a study-level summary.

Task coverage:
  3.8.1 — Batch processing script (parameterised for any study)
  3.8.2 — Process all 14 Expert/Novice subjects
  3.8.3 — Track and document processing success/failures
  3.8.4 — Generate processing summary report

Output layout  (example: expert-novice)
---------------------------------------
outputs/stage_b_processed/data/expert-novice/
  sub-A1_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif
  sub-A2_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif
  ...  (14 BIDS-compliant .fif files)
  expert-novice_master_summary.csv
  expert-novice_group_visualization.png
  expert-novice_provenance.json

BIDS naming convention
----------------------
  sub-{subject_id}_task-{task}_desc-preprocessed_nirs_raw.fif

Study parameter table
---------------------
  expert_novice  → folder: expert-novice   | TechEn CW6 | PPF {690:6.4, 830:5.8}
  learning_curve → folder: learning-curve  | TechEn CW6 | PPF {690:6.4, 830:5.8}
  short_term_tes → folder: short-term-tES  | NIRx        | PPF {760:6.0, 850:6.0}*
  long_term_tes  → folder: long-term-tES   | NIRx        | PPF {760:6.0, 850:6.0}*

  * tES studies require Section 3.9 adaptation before processing.
    Parameters listed here are placeholders only.

Reference: PROGRESS.md Section 3.8
Upstream:  06_ols_regression.py (B1–B6 full chain)
"""

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_dataset_config, get_output_path, get_snirf_files, load_config

# ---------------------------------------------------------------------------
# Dynamic import: run_ols_regression from 06_ols_regression.py
# ---------------------------------------------------------------------------
import importlib.util

_spec_b6 = importlib.util.spec_from_file_location(
    "ols_regression",
    Path(__file__).parent / "06_ols_regression.py",
)
_b6 = importlib.util.module_from_spec(_spec_b6)
_spec_b6.loader.exec_module(_b6)
run_ols_regression = _b6.run_ols_regression

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ===========================================================================
# Build study metadata from config.yaml (Task 3.8.1)
# ===========================================================================

def _build_study_meta(config: dict) -> dict:
    """
    Construct the per-study metadata table from config.yaml.

    Reads bids_task, output_folder, system, and ppf from each dataset entry
    so that changes to config.yaml automatically propagate here.
    """
    datasets = config["datasets"]
    meta = {}
    for key, ds in datasets.items():
        ppf_raw = ds.get("ppf", {})
        if isinstance(ppf_raw, dict):
            ppf_dict = {int(k): float(v) for k, v in ppf_raw.items()}
        else:
            ppf_dict = {}
        meta[key] = {
            "output_folder": ds.get("output_folder", key.replace("_", "-")),
            "task_name":     ds.get("bids_task", key),
            "system":        ds.get("system", ""),
            "ppf_dict":      ppf_dict,
        }
    return meta


_CONFIG = load_config()
STUDY_META: dict = _build_study_meta(_CONFIG)

# rPCA parameters — sourced from config.yaml stage_b.rpca
# Homer3-faithful values (audit fix): nSV=0.97, OR-union mask (min_ch=1), maxIter=5
_RPCA_CFG = _CONFIG.get("stage_b", {}).get("rpca", {})
DEFAULT_N_COMPONENTS:   float = float(_RPCA_CFG.get("n_components",         0.97))
DEFAULT_MIN_CHANNELS:   int   = int(  _RPCA_CFG.get("min_channels_flagged", 1))
DEFAULT_MAX_ITER:       int   = int(  _RPCA_CFG.get("max_iterations",       5))

# Short-channel separation threshold — sourced from config.yaml stage_b.short_channel_regression
_SCR_CFG = _CONFIG.get("stage_b", {}).get("short_channel_regression", {})
SHORT_SEP_THRESHOLD_MM: float = float(_SCR_CFG.get("short_separation_threshold", 15.0))

# SNR thresholds — sourced from config.yaml stage_a.snr
_SNR_CFG = _CONFIG.get("stage_a", {}).get("snr", {})
_SNR_LOW:  float = float(_SNR_CFG.get("threshold_low",  80.0))
_SNR_HIGH: float = float(_SNR_CFG.get("threshold_high", 140.0))


# ===========================================================================
# Stage A exclusion list helpers
# ===========================================================================

def _ensure_exclusion_list(dataset_name: str, subject_id: str) -> dict:
    """
    Return Stage A signal-amplitude exclusion dict for subject_id.

    If the JSON already exists (generated during Stage A per-subject run),
    it is read directly.  Otherwise the amplitude analysis is performed
    inline using the raw SNIRF file and the result is saved to the Stage A
    output directory for reuse.

    The exclusion list identifies channels outside the 80–140 dB signal
    amplitude range (Step B1 prerequisite).
    """
    exclusion_dir = get_output_path("stage_a", "channel_signal_quality_exclusion")
    json_path = exclusion_dir / f"{subject_id}_exclusion_list_preview.json"

    if json_path.exists():
        with open(json_path) as fh:
            return json.load(fh)

    print(f"  [Stage A] Exclusion list missing for {subject_id} — generating now...")

    # Load raw SNIRF to compute amplitude
    snirf_files = get_snirf_files(dataset_name, subject_id)
    raw = mne.io.read_raw_snirf(str(snirf_files[0]), preload=True, verbose=False)

    data        = raw.get_data()           # (n_ch, n_times), raw intensity
    ch_names    = raw.ch_names
    amp_db      = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(data ** 2, axis=1)), 1e-30)
    )

    # MNE NIRS requirement: both wavelengths of a source-detector pair must be
    # consistently marked bad.  If only one wavelength fails the SNR criterion,
    # we expand the exclusion to cover the whole pair.
    def _pair_base(ch_name: str) -> str:
        for suffix in (" 690", " 830", " 760", " 850"):
            if ch_name.endswith(suffix):
                return ch_name[: -len(suffix)]
        return ch_name

    amp_db_map = dict(zip(ch_names, amp_db))

    # Channels directly failing the threshold (thresholds from config)
    direct_bad = {ch for ch, a in amp_db_map.items()
                  if a < _SNR_LOW or a > _SNR_HIGH}

    # Expand to full pairs: if either wavelength is bad, mark both bad
    bad_pair_bases = {_pair_base(ch) for ch in direct_bad}
    all_bad = {ch for ch in ch_names if _pair_base(ch) in bad_pair_bases}

    bad_channels  = sorted(all_bad)
    good_channels = sorted(ch for ch in ch_names if ch not in all_bad)
    excl_reasons  = {}
    for ch in bad_channels:
        a = amp_db_map[ch]
        if a < _SNR_LOW:
            excl_reasons[ch] = f"below_{_SNR_LOW:.0f}db"
        elif a > _SNR_HIGH:
            excl_reasons[ch] = f"above_{_SNR_HIGH:.0f}db"
        else:
            # Paired with a channel that failed; exclude for NIRS consistency
            partner_failed = [
                p for p in direct_bad if _pair_base(p) == _pair_base(ch)
            ]
            reason = excl_reasons.get(partner_failed[0], "pair_consistency") \
                     if partner_failed else "pair_consistency"
            excl_reasons[ch] = f"pair_excluded_with_{reason}"

    exclusion_json = {
        "subject_id":        subject_id,
        "bad_channels":      sorted(bad_channels),
        "good_channels":     sorted(good_channels),
        "exclusion_reasons": excl_reasons,
        "thresholds_db":     {"low": _SNR_LOW, "high": _SNR_HIGH},
        "stage_a_note": (
            "Generated inline by 07_preprocess_study.py batch script. "
            "Stage A documentation only: channels listed as exclusion candidates "
            "for Stage B. No channels excluded from Stage A QA analysis."
        ),
    }

    with open(json_path, "w") as fh:
        json.dump(exclusion_json, fh, indent=2)
    print(f"    Saved exclusion list: {json_path.name}  "
          f"({len(bad_channels)} bad, {len(good_channels)} good)")

    return exclusion_json


# ===========================================================================
# Single-subject processing  (Task 3.8.2 / 3.8.3)
# ===========================================================================

def process_subject(
    dataset_name:        str,
    subject_id:          str,
    ppf_dict:            dict,
    n_components:        float,
    min_channels_flagged: int,
    study_dir:           Path,
    task_name:           str,
    verbose:             bool = False,
) -> dict:
    """
    Run the complete B1–B6 pipeline for one subject and return a metrics dict.

    Steps
    -----
    1. Ensure Stage A exclusion list exists (generate if absent).
    2. Run run_ols_regression() — full B1→B6 chain.
    3. Save BIDS-compliant .fif to study_dir.
    4. Read post-rPCA motion ratio from saved effectiveness JSON.
    5. Compile per-subject metrics.

    Returns
    -------
    metrics : dict
        subject_id, n_channels_excluded, n_bad_channels, mean_MR_post_rPCA,
        var_removed_HbO_pct, var_removed_HbR_pct,
        mean_r_HbO_HbR_before, mean_r_HbO_HbR_after,
        status, notes, bids_fif_path
    """
    metrics = {
        "subject_id":           subject_id,
        "n_channels_excluded":  None,
        "n_bad_channels":       None,
        "mean_MR_post_rPCA":    None,
        "var_removed_HbO_pct":  None,
        "var_removed_HbR_pct":  None,
        "mean_r_HbO_HbR_before": None,
        "mean_r_HbO_HbR_after":  None,
        "status":               "Fail",
        "notes":                "",
        "bids_fif_path":        None,
        "bids_glm_hrf_path":    None,
    }

    try:
        # ---- 1. Ensure Stage A exclusion list --------------------------------
        excl_json = _ensure_exclusion_list(dataset_name, subject_id)
        n_bad_pairs = len(set(
            ch.replace(" 690", "").replace(" 830", "")
            .replace(" 760", "").replace(" 850", "")
            for ch in excl_json.get("bad_channels", [])
        ))
        metrics["n_bad_channels"]      = n_bad_pairs
        metrics["n_channels_excluded"] = n_bad_pairs

        # ---- 2. Run B1–B6 pipeline -------------------------------------------
        results = run_ols_regression(
            dataset_name         = dataset_name,
            subject_id           = subject_id,
            short_threshold_mm   = SHORT_SEP_THRESHOLD_MM,
            n_components         = n_components,
            min_channels_flagged = min_channels_flagged,
            ppf_dict             = ppf_dict,
            verbose              = verbose,
        )
        raw_corrected = results["raw_corrected"]
        reg_results   = results["reg_results"]
        corr_df       = results["corr_df"]

        # ---- 3. Save BIDS-compliant continuous .fif to study directory -------
        bids_fname = f"sub-{subject_id}_task-{task_name}_desc-preprocessed_nirs_raw.fif"
        bids_path  = study_dir / bids_fname
        raw_corrected.save(str(bids_path), overwrite=True, verbose=False)
        metrics["bids_fif_path"] = str(bids_path)

        # ---- 3b. Save BIDS-sidecar GLM HRF evoked -----------------------------
        glm_evoked = results.get("glm_primary_evoked", None)
        if glm_evoked is not None:
            glm_fname = f"sub-{subject_id}_task-{task_name}_desc-glmhrf_nirs-ave.fif"
            glm_path = study_dir / glm_fname
            glm_evoked.save(str(glm_path), overwrite=True, verbose=False)
            metrics["bids_glm_hrf_path"] = str(glm_path)

        # ---- 4. Post-rPCA motion ratio from saved effectiveness JSON ---------
        rpca_json_path = (
            get_output_path("stage_b", f"rPCA_correction/{subject_id}")
            / f"{subject_id}_rpca_effectiveness.json"
        )
        if rpca_json_path.exists():
            with open(rpca_json_path) as fh:
                rpca_eff = json.load(fh)
            metrics["mean_MR_post_rPCA"] = round(
                float(rpca_eff.get("mean_motion_ratio_after_dual", np.nan)), 4
            )

        # ---- 5. OLS regression metrics ---------------------------------------
        hbo_r = reg_results[reg_results["ch_type"] == "hbo"].dropna(subset=["beta_norm"])
        hbr_r = reg_results[reg_results["ch_type"] == "hbr"].dropna(subset=["beta_norm"])

        if len(hbo_r):
            metrics["var_removed_HbO_pct"] = round(
                float(hbo_r["var_removed_frac"].mean() * 100), 2
            )
        if len(hbr_r):
            metrics["var_removed_HbR_pct"] = round(
                float(hbr_r["var_removed_frac"].mean() * 100), 2
            )

        pair_df = corr_df[corr_df["ch_type"] == "hbo"].dropna(
            subset=["hbo_hbr_r_before"]
        )
        if len(pair_df):
            metrics["mean_r_HbO_HbR_before"] = round(
                float(pair_df["hbo_hbr_r_before"].mean()), 4
            )
            metrics["mean_r_HbO_HbR_after"] = round(
                float(pair_df["hbo_hbr_r_after"].mean()), 4
            )

        metrics["status"] = "Pass"
        metrics["notes"]  = ""

        # Safe format helpers — metrics can be None when no valid channels
        # survive the dropna filters (e.g. R10 trailing-zero artifact, or
        # subjects where all channels lack a matched short-sep pair).
        def _fv(v, spec):
            return format(v, spec) if v is not None else "N/A"

        print(f"  ✅ {subject_id}: Pass | "
              f"bad={n_bad_pairs} | "
              f"MR={_fv(metrics['mean_MR_post_rPCA'], '.3f')} | "
              f"varHbO={_fv(metrics['var_removed_HbO_pct'], '.1f')}% | "
              f"varHbR={_fv(metrics['var_removed_HbR_pct'], '.1f')}% | "
              f"r={_fv(metrics['mean_r_HbO_HbR_after'], '.3f')}")

    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"
        metrics["status"] = "Fail"
        metrics["notes"]  = err_msg
        print(f"  ❌ {subject_id}: FAIL — {err_msg}")
        if verbose:
            traceback.print_exc()

    return metrics


# ===========================================================================
# Summary outputs  (Task 3.8.4)
# ===========================================================================

def generate_master_summary(
    records:      list,
    study_dir:    Path,
    study_folder: str,
) -> Path:
    """
    Task 3.8.4: Save per-subject master summary CSV.

    Columns
    -------
    subject_id | n_channels_excluded | mean_MR_post_rPCA | var_removed_HbO_pct |
    var_removed_HbR_pct | mean_r_HbO_HbR_before | mean_r_HbO_HbR_after | status | notes
    """
    df = pd.DataFrame(records)

    # Reorder / rename columns for readability
    col_order = [
        "subject_id",
        "n_channels_excluded",
        "mean_MR_post_rPCA",
        "var_removed_HbO_pct",
        "var_removed_HbR_pct",
        "mean_r_HbO_HbR_before",
        "mean_r_HbO_HbR_after",
        "status",
        "notes",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = study_dir / f"{study_folder}_master_summary.csv"
    df.to_csv(str(csv_path), index=False)

    # Console summary
    n_pass = (df["status"] == "Pass").sum()
    n_fail = (df["status"] == "Fail").sum()
    print(f"\n  Master summary saved: {csv_path.name}")
    print(f"  {n_pass}/{len(df)} subjects passed  ({n_fail} failed)")

    pass_df = df[df["status"] == "Pass"]
    if len(pass_df):
        print(f"\n  Group means (passing subjects):")
        for col in ["n_channels_excluded", "mean_MR_post_rPCA",
                    "var_removed_HbO_pct", "var_removed_HbR_pct",
                    "mean_r_HbO_HbR_before", "mean_r_HbO_HbR_after"]:
            vals = pd.to_numeric(pass_df[col], errors="coerce").dropna()
            if len(vals):
                print(f"    {col:30s}: {vals.mean():.4f}  (SD={vals.std():.4f})")

    return csv_path


def generate_group_visualization(
    records:      list,
    study_dir:    Path,
    study_folder: str,
) -> Path:
    """
    Task 3.8.4: Two-panel group-level visualization.

    Panel A — Variance removed distribution (box plot)
        HbO and HbR variance-removed-% across passing subjects.
        Shows range and consistency of OLS regression effectiveness.

    Panel B — HbO/HbR anti-correlation shift (scatter)
        Each point = one subject; x = mean r(HbO,HbR) before OLS,
        y = mean r(HbO,HbR) after OLS.
        A shift toward more negative values indicates scalp-noise removal.
    """
    pass_records = [r for r in records if r["status"] == "Pass"]
    if not pass_records:
        print("  ⚠  No passing subjects — group visualization skipped.")
        return None

    sub_ids = [r["subject_id"] for r in pass_records]
    n = len(pass_records)

    # Filter None values from variance metrics (e.g. R10 trailing-zero artefact
    # leaves var_removed_frac NaN → metric stays None).
    var_hbo_pairs = [(r["subject_id"], r["var_removed_HbO_pct"])
                     for r in pass_records if r["var_removed_HbO_pct"] is not None]
    var_hbr_pairs = [(r["subject_id"], r["var_removed_HbR_pct"])
                     for r in pass_records if r["var_removed_HbR_pct"] is not None]
    var_hbo = [v for _, v in var_hbo_pairs]
    var_hbr = [v for _, v in var_hbr_pairs]

    # Only include subjects where both r_before and r_after are available
    # (subjects with no valid short-sep pair have None for correlation metrics).
    r_records = [r for r in pass_records
                 if r["mean_r_HbO_HbR_before"] is not None
                 and r["mean_r_HbO_HbR_after"] is not None]
    r_sub_ids = [r["subject_id"]           for r in r_records]
    r_before  = [r["mean_r_HbO_HbR_before"] for r in r_records]
    r_after   = [r["mean_r_HbO_HbR_after"]  for r in r_records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel A: Variance removed box plot ---------------------------------
    ax = axes[0]
    if var_hbo or var_hbr:
        bp = ax.boxplot(
            [var_hbo or [0], var_hbr or [0]],
            tick_labels=["HbO", "HbR"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="+", markersize=7, color="black"),
        )
        bp["boxes"][0].set_facecolor("#ffb3b3")   # light red for HbO
        bp["boxes"][1].set_facecolor("#b3c6ff")   # light blue for HbR

        # Overlay individual subject points
        for i, (vals, color) in enumerate(
            [(var_hbo, "#d62728"), (var_hbr, "#1f77b4")], start=1
        ):
            if vals:
                x_jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(
                    [i + x_jitter[j] for j in range(len(vals))],
                    vals,
                    c=color, alpha=0.7, zorder=3, s=40,
                )

        all_vals = var_hbo + var_hbr
        ax.set_ylim(0, max(max(all_vals, default=30) * 1.2, 30))
        hbo_mean_str = f"{np.mean(var_hbo):.1f}%" if var_hbo else "N/A"
        hbr_mean_str = f"{np.mean(var_hbr):.1f}%" if var_hbr else "N/A"
        ax.annotate(
            f"HbO mean: {hbo_mean_str}\nHbR mean: {hbr_mean_str}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85),
        )
    else:
        ax.text(0.5, 0.5, "No variance data available",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("Variance Removed (%)", fontsize=11)
    ax.set_title(
        f"(A) OLS Variance Removed\n"
        f"(n={len(var_hbo)} HbO / {len(var_hbr)} HbR subjects with data)",
        fontsize=11, fontweight="bold",
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.grid(axis="y", alpha=0.3)

    # ---- Panel B: Anti-correlation shift scatter ----------------------------
    ax = axes[1]

    if r_records:
        ax.scatter(r_before, r_after, c="#2ca02c", s=60, alpha=0.8, zorder=3)

        for sid, xv, yv in zip(r_sub_ids, r_before, r_after):
            ax.annotate(sid, (xv, yv), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, alpha=0.8)

        lim = [-1.0, 0.5]
        ax.plot(lim, lim, "k--", lw=0.9, label="No change (y = x)")
        ax.axhline(-0.5, color="darkgreen", lw=1.0, ls=":", label="Target ≤ −0.5")
        ax.axhline(-1.0, color="darkgreen", lw=0.8, ls="--")

        mean_bef = float(np.mean(r_before))
        mean_aft = float(np.mean(r_after))
        ax.annotate(
            "",
            xy=(mean_bef, mean_aft),
            xytext=(mean_bef, mean_bef),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        )
        ax.scatter([mean_bef], [mean_aft], c="purple", s=120, zorder=5,
                   label=f"Group mean: {mean_bef:.3f} → {mean_aft:.3f}")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5,
                f"No paired r(HbO,HbR) data\n({n} subjects processed,\n"
                "0 with valid short-sep pairs)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("Mean r(HbO, HbR)  before OLS", fontsize=11)
    ax.set_ylabel("Mean r(HbO, HbR)  after OLS",  fontsize=11)
    ax.set_title(
        f"(B) HbO/HbR Anti-Correlation Shift\n"
        f"(n={len(r_records)} subjects with paired correlation data)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Stage B Group-Level Preprocessing Summary — {study_folder}  (n={n})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    fig_path = study_dir / f"{study_folder}_group_visualization.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Group visualization saved: {fig_path.name}")
    return fig_path


def save_provenance_log(
    dataset_name:        str,
    records:             list,
    study_dir:           Path,
    study_folder:        str,
    n_components:        float,
    min_channels_flagged: int,
    max_iterations:      int,
    ppf_dict:            dict,
    system:              str,
) -> Path:
    """
    Task 3.8.4: Save provenance JSON documenting exact parameters used.

    The provenance log enables full reproducibility: given the input data
    and this file, every parameter can be re-applied identically.
    """
    config   = load_config()
    b_params = config.get("stage_b", {})

    n_pass = sum(1 for r in records if r["status"] == "Pass")
    n_fail = len(records) - n_pass
    failed = [r["subject_id"] for r in records if r["status"] == "Fail"]

    # Sort PPF dict for serialisation
    ppf_serialisable = {str(k): v for k, v in ppf_dict.items()}

    b3_params = b_params.get("motion_detection", {})
    bl_params = b_params.get("beer_lambert", {})
    scr_params = b_params.get("short_channel_regression", {})
    snr_params = config.get("stage_a", {}).get("snr", {})

    provenance = {
        "study_name":         study_folder,
        "dataset_key":        dataset_name,
        "system":             system,
        "date_processed":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_version":   "Stage B v1.0",
        "pipeline_steps":     ["B1_OD", "B2_bandpass", "B3_motion_detect",
                               "B4_rPCA", "B5_MBLL", "B6_GLM_OLS"],
        "n_subjects_attempted": len(records),
        "n_subjects_passed":  n_pass,
        "n_subjects_failed":  n_fail,
        "failed_subjects":    failed,
        "parameters": {
            "B1_signal_amplitude_exclusion": {
                "low_db":  float(snr_params.get("threshold_low",  80.0)),
                "high_db": float(snr_params.get("threshold_high", 140.0)),
            },
            "B2_bandpass_filter": {
                "l_freq":  b_params.get("filter", {}).get("hpf", 0.0),
                "h_freq":  b_params.get("filter", {}).get("lpf", 0.5),
                "method":  b_params.get("filter", {}).get("method", "iir"),
            },
            "B3_motion_detection": {
                "AMPthresh":              float(b3_params.get("AMPthresh", 5.0)),
                "STDEVthresh_multiplier": float(b3_params.get("STDEVthresh", 20.0)),
                "tMotion":                float(b3_params.get("tMotion", 0.5)),
                "t_mask":                 float(b3_params.get("t_mask", 0.5)),
                "use_amp_only":           bool(b3_params.get("use_amp_only", False)),
            },
            "B4_rPCA": {
                "n_components":         n_components,
                "min_channels_flagged": min_channels_flagged,
                "max_iterations":       max_iterations,
            },
            "B5_beer_lambert": {
                "ppf_values":  ppf_serialisable,
                "system":      system,
                "hb_range_um": {
                    "min": float(bl_params.get("hb_range_min", -100)),
                    "max": float(bl_params.get("hb_range_max",  100)),
                },
            },
            "B6_glm_ols": {
                "method": "homer3_style_glm_ols",
                "short_sep_threshold_mm": float(scr_params.get("short_separation_threshold", 15.0)),
                "trange_s": [
                    float(scr_params.get("trange", [config.get("stage_c", {}).get("epochs", {}).get("tmin", -25), config.get("stage_c", {}).get("epochs", {}).get("tmax", 100)])[0]),
                    float(scr_params.get("trange", [config.get("stage_c", {}).get("epochs", {}).get("tmin", -25), config.get("stage_c", {}).get("epochs", {}).get("tmax", 100)])[1]),
                ],
                "gaussian_std_s": float(scr_params.get("gaussian_std_s", 1.0)),
                "gaussian_step_s": float(scr_params.get("gaussian_step_s", 1.0)),
                "drift_order": int(scr_params.get("drift_order", 3)),
                "flag_nuisance_r_method": int(scr_params.get("flag_nuisance_r_method", scr_params.get("flagNuisanceRMethod", 1))),
                "glm_solve_method": int(scr_params.get("glm_solve_method", 1)),
                "event_ids": scr_params.get("event_ids", [1]),
            },
        },
        "output_format":       "MNE .fif",
        "bids_naming":         "sub-{subject_id}_task-{task}_desc-preprocessed_nirs_raw.fif",
    }

    json_path = study_dir / f"{study_folder}_provenance.json"
    with open(json_path, "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"  Provenance log saved:  {json_path.name}")
    return json_path


# ===========================================================================
# Main orchestrator  (Task 3.8.1 / 3.8.2 / 3.8.3 / 3.8.4)
# ===========================================================================

def run_preprocess_study(
    dataset_name:         str  = "expert_novice",
    n_components:         float = DEFAULT_N_COMPONENTS,
    min_channels_flagged: int   = DEFAULT_MIN_CHANNELS,
    subjects_override:    list  = None,
    verbose:              bool  = False,
) -> dict:
    """
    Batch-process all subjects in a study through the full Stage B pipeline.

    Parameters
    ----------
    dataset_name : str
        One of: 'expert_novice', 'learning_curve', 'short_term_tes', 'long_term_tes'.
        Determines subjects list, PPF values, output folder, and BIDS task name.
    n_components : float
        rPCA variance threshold (default 0.99 — from tuning in Section 3.4.5).
    min_channels_flagged : int
        rPCA vote threshold (default 10 — from tuning in Section 3.4.5).
    subjects_override : list[str], optional
        Process only these subjects instead of the full list.
        Useful for testing or re-running failed subjects.
    verbose : bool
        If True, pass verbose=True to run_ols_regression (very detailed output).

    Returns
    -------
    summary : dict
        {
          'records':       list[dict],   per-subject metrics
          'csv_path':      Path,
          'fig_path':      Path,
          'provenance':    Path,
          'n_pass':        int,
          'n_fail':        int,
          'study_dir':     Path,
        }

    Notes
    -----
    tES studies (short_term_tes, long_term_tes) require additional adaptation
    (Section 3.9) before processing.  Their entries in STUDY_META are parameter
    stubs; results may not be physiologically valid.
    """
    # ---- Validate study name ------------------------------------------------
    if dataset_name not in STUDY_META:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(STUDY_META.keys())}"
        )

    meta        = STUDY_META[dataset_name]
    task_name   = meta["task_name"]
    ppf_dict    = meta["ppf_dict"]
    system      = meta["system"]
    study_folder = meta["output_folder"]

    # ---- Subject list -------------------------------------------------------
    config   = load_config()
    ds_cfg   = config["datasets"][dataset_name]
    subjects = subjects_override or ds_cfg["subjects"]

    # ---- Study output directory ---------------------------------------------
    study_dir = (
        PROJECT_ROOT
        / config["paths"].get("processed_root", "outputs/stage_b_processed/preprocessed_data")
        / study_folder
    )
    study_dir.mkdir(parents=True, exist_ok=True)

    # ---- Banner -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STAGE B — BATCH PREPROCESSING  |  Study: {study_folder}")
    print("=" * 70)
    print(f"  Dataset:        {dataset_name}")
    print(f"  System:         {system}")
    print(f"  Subjects:       {len(subjects)}")
    print(f"  PPF values:     {ppf_dict}")
    print(f"  rPCA params:    n_components={n_components}, "
          f"min_channels_flagged={min_channels_flagged}, "
          f"max_iterations={DEFAULT_MAX_ITER}  (Homer3-faithful)")
    print(f"  Output dir:     {study_dir}")
    print(f"  BIDS naming:    sub-{{id}}_task-{task_name}_desc-preprocessed_nirs_raw.fif")
    print("=" * 70 + "\n")

    # ---- Per-subject processing  (Tasks 3.8.2 + 3.8.3) --------------------
    records = []
    for i, subj in enumerate(subjects, start=1):
        print(f"\n[{i}/{len(subjects)}] Processing subject: {subj}")
        metrics = process_subject(
            dataset_name         = dataset_name,
            subject_id           = subj,
            ppf_dict             = ppf_dict,
            n_components         = n_components,
            min_channels_flagged = min_channels_flagged,
            study_dir            = study_dir,
            task_name            = task_name,
            verbose              = verbose,
        )
        records.append(metrics)

    # ---- Summary outputs  (Task 3.8.4) -------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING STUDY SUMMARY OUTPUTS")
    print("=" * 70)

    csv_path  = generate_master_summary(records, study_dir, study_folder)
    fig_path  = generate_group_visualization(records, study_dir, study_folder)
    prov_path = save_provenance_log(
        dataset_name         = dataset_name,
        records              = records,
        study_dir            = study_dir,
        study_folder         = study_folder,
        n_components         = n_components,
        min_channels_flagged = min_channels_flagged,
        max_iterations       = DEFAULT_MAX_ITER,
        ppf_dict             = ppf_dict,
        system               = system,
    )

    # ---- Final report  (Task 3.8.3 + 3.8.4) -------------------------------
    n_pass = sum(1 for r in records if r["status"] == "Pass")
    n_fail = len(records) - n_pass

    print("\n" + "=" * 70)
    print(f"✅ BATCH COMPLETE — {study_folder}")
    print("=" * 70)
    print(f"  Processed:     {len(records)} subjects")
    print(f"  Pass:          {n_pass}")
    print(f"  Fail:          {n_fail}")
    if n_fail:
        failed = [r["subject_id"] for r in records if r["status"] == "Fail"]
        print(f"  Failed IDs:    {', '.join(failed)}")
    print(f"\n  Outputs in:    {study_dir}")
    print(f"    .fif files:  {n_pass} BIDS-compliant hemoglobin time series")
    print(f"    Summary CSV: {csv_path.name}")
    if fig_path:
        print(f"    Group viz:   {fig_path.name}")
    print(f"    Provenance:  {prov_path.name}")
    print("=" * 70 + "\n")

    return {
        "records":    records,
        "csv_path":   csv_path,
        "fig_path":   fig_path,
        "provenance": prov_path,
        "n_pass":     n_pass,
        "n_fail":     n_fail,
        "study_dir":  study_dir,
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage B batch preprocessing for one study."
    )

    #Ex: python code/stage_b_preprocessing/07_preprocess_study.py --study short_term_tes
    #expert_novice, learning_curve, short_term_tes, long_term_tes
    parser.add_argument(
        "--study",
        default="expert_novice",
        choices=list(STUDY_META.keys()),
        help="Dataset to process (default: expert_novice)",
    )
    #Ex: python code/stage_b_preprocessing/07_preprocess_study.py --n_components 0.99
    parser.add_argument(
        "--n_components",
        type=float,
        default=DEFAULT_N_COMPONENTS,
        help=f"rPCA variance threshold (default {DEFAULT_N_COMPONENTS} from config.yaml stage_b.rpca.n_components)",
    )
    #Ex: python code/stage_b_preprocessing/07_preprocess_study.py --min_channels 10
    parser.add_argument(
        "--min_channels",
        type=int,
        default=DEFAULT_MIN_CHANNELS,
        help=f"rPCA min channels flagged in union mask (default {DEFAULT_MIN_CHANNELS} from config.yaml stage_b.rpca.min_channels_flagged)",
    )
    #Ex: python code/stage_b_preprocessing/07_preprocess_study.py --subjects A1 R10
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Process only these subjects (default: all in study)",
    )
    #Ex: python code/stage_b_preprocessing/07_preprocess_study.py --verbose
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose pipeline output",
    )
    args = parser.parse_args()

    run_preprocess_study(
        dataset_name         = args.study,
        n_components         = args.n_components,
        min_channels_flagged = args.min_channels,
        subjects_override    = args.subjects,
        verbose              = args.verbose,
    )
