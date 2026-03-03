"""
pipeline_combo_audit.py
Stage B/C Audit — Preprocessing Combination CNR Sweep

Purpose
-------
Systematically test all meaningful combinations of the Stage B preprocessing
pipeline (excluding channel pruning) to identify which steps move the
reproduced CNR closest to the ASTaUND paper reference (Figure 4i).

Paper reference
---------------
  "To give an accurate assessment of the quality of the data, no channels
  have been excluded from the analysis provided here and no denoising methods
  have been used prior to analysis."

  "First, we calculate the group mean HRFs for each population using the
  processing pipelines specified for each study in Homer3 (excluding channel
  pruning)"

  Target (Figure 4i): median CNR ≈ 0.45, max > 2.25

Pipeline components under test
-------------------------------
  Always included:
    B1   Intensity → Optical Density (MNE natural-log)
    B2   Bandpass filter 0.01–0.5 Hz (IIR, zero-phase) — current baseline
    B5   Modified Beer-Lambert Law (PPF: 690→6.4, 830→5.8)

  Toggled (2 independent flags → 4 conditions):
    rPCA  B3 motion detection + B4 rPCA correction (on OD, before MBLL)
    OLS   B6 Gaussian-basis OLS short-channel regression (on haemo, after MBLL)

  Always excluded:
    Stage A 80–140 dB channel pruning

Conditions (4 total)
--------------------
  01_baseline    no rPCA  | no OLS   ← current techval branch reference
  02_ols_only    no rPCA  | + OLS
  03_rpca_only   + rPCA   | no OLS
  04_rpca_ols    + rPCA   | + OLS    ← full pipeline without channel pruning

For each condition the script computes group-HRF CNR (Expert vs Novice,
all long-separation channels, HbO+HbR pooled) and saves:
  - an individual CNR boxplot   (cnr_boxplot_{condition}.png)

After all conditions:
  - a combined multi-box comparison plot (cnr_all_conditions_boxplot.png)
  - a summary CSV with 5-number summary + mean + range per condition
    (pipeline_combo_audit_table.csv)

Outputs (outputs/audit/pipeline_combo_audit/)
---------------------------------------------
  cnr_boxplot_01_baseline.png
  cnr_boxplot_02_ols_only.png
  cnr_boxplot_03_rpca_only.png
  cnr_boxplot_04_rpca_ols.png
  cnr_all_conditions_boxplot.png
  pipeline_combo_audit_table.csv

Usage
-----
  # Full run (≈ 15–30 min)
  python code/audit/pipeline_combo_audit.py

  # Force recompute cached evokeds
  python code/audit/pipeline_combo_audit.py --force

Reference: PROGRESS.md (Audit Issue 3/4), AUDIT_PROGRESS.md
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root + shared utilities
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from utils import get_snirf_files, load_config


# ---------------------------------------------------------------------------
# Dynamic imports — Stage B preprocessing modules
# ---------------------------------------------------------------------------
def _load_b(name: str, filename: str):
    stage_b = PROJECT_ROOT / "code" / "stage_b_preprocessing"
    spec = importlib.util.spec_from_file_location(name, stage_b / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b3 = _load_b("motion_correct",       "03_motion_correct.py")
_b4 = _load_b("rpca_correct",         "04_rPCA_motion_correct.py")
_b6 = _load_b("ols_regression",       "06_ols_regression.py")

detect_motion_artifacts = _b3.detect_motion_artifacts
apply_rpca_correction   = _b4.apply_rpca_correction
apply_ols_regression    = _b6.apply_ols_regression


# ---------------------------------------------------------------------------
# Dynamic imports — Stage C analysis helpers
# ---------------------------------------------------------------------------
def _load_c(name: str, filename: str):
    stage_c = PROJECT_ROOT / "code" / "stage_c_validation"
    spec = importlib.util.spec_from_file_location(name, stage_c / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hrf = _load_c("hrf_extraction",  "01_hrf_extraction.py")
_grp = _load_c("group_hrf",       "02_group_hrf_analysis.py")
_cnr = _load_c("cnr_calc",        "04_calculate_cnr.py")

extract_epochs         = _hrf.extract_epochs
get_long_channel_names = _hrf.get_long_channel_names
compute_group_hrf_cnr  = _cnr.compute_group_hrf_cnr
_harmonise_evoked      = _grp._harmonise_evoked


# ---------------------------------------------------------------------------
# Configuration — sourced from config.yaml
# ---------------------------------------------------------------------------
_CFG  = load_config()
_DS   = _CFG["datasets"]["expert_novice"]

STUDY  = "expert_novice"
GROUPS: dict[str, list[str]] = (
    {grp: list(subs) for grp, subs in _DS.get("groups", {}).items()}
    if _DS.get("groups") else {
        "Expert": ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"],
        "Novice": ["R1", "R2", "R3", "R4", "R5", "R6"],
    }
)

_ppf_raw = _DS.get("ppf", {})
PPF_DICT: dict = (
    {int(k): float(v) for k, v in _ppf_raw.items()}
    if isinstance(_ppf_raw, dict) else {690: 6.4, 830: 5.8}
)

_EPOCHS_CFG  = _CFG["stage_c"]["epochs"]
TMIN         = float(_EPOCHS_CFG["tmin"])
TMAX         = float(_EPOCHS_CFG["tmax"])
BASELINE     = tuple(float(v) for v in _EPOCHS_CFG["baseline"])
PEAK_WINDOW  = (5.0, 15.0)
TARGET_SFREQ = 25.0

# Bandpass parameters (baseline = current techval branch)
_FILT_CFG = _CFG.get("stage_b", {}).get("filter", {})
BP_L_FREQ   = float(_FILT_CFG.get("l_freq",  0.01))
BP_H_FREQ   = float(_FILT_CFG.get("h_freq",  0.5))
BP_METHOD   = str(_FILT_CFG.get("method",   "iir"))

# rPCA parameters from config
_RPCA_CFG       = _CFG.get("stage_b", {}).get("rpca", {})
RPCA_N_COMP     = float(_RPCA_CFG.get("n_components",        0.99))
RPCA_MIN_CH     = int(_RPCA_CFG.get("min_channels_flagged",  10))

# Output directory
OUT_DIR = PROJECT_ROOT / "outputs" / "audit" / "pipeline_combo_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper targets
PAPER_MEDIAN_TARGET = 0.45
PAPER_MAX_TARGET    = 2.25
CNR_THRESHOLD       = float(
    _CFG.get("stage_c", {}).get("cnr", {}).get("threshold_detectable", 1.0)
)


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
CONDITIONS: list[dict] = [
    {
        "name":        "01_baseline",
        "label":       "Baseline\n(OD → BP → MBLL)",
        "include_rpca": False,
        "include_ols":  False,
        "note":        "Current technical-validation branch: no motion correction, no OLS",
    },
    {
        "name":        "02_ols_only",
        "label":       "Baseline\n+ OLS",
        "include_rpca": False,
        "include_ols":  True,
        "note":        "Add Gaussian-basis OLS short-channel regression only",
    },
    {
        "name":        "03_rpca_only",
        "label":       "Baseline\n+ rPCA",
        "include_rpca": True,
        "include_ols":  False,
        "note":        "Add motion detection + rPCA correction only",
    },
    {
        "name":        "04_rpca_ols",
        "label":       "Full Pipeline\n(rPCA + OLS)",
        "include_rpca": True,
        "include_ols":  True,
        "note":        "Full preprocessing without channel pruning: rPCA + OLS",
    },
]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _mbll(raw_od_filt: mne.io.BaseRaw, ppf_dict: dict) -> mne.io.BaseRaw:
    """OD → HbO / HbR via Modified Beer-Lambert Law."""
    wavelengths = sorted({
        int(ch.split()[-1])
        for ch in raw_od_filt.ch_names
        if ch.split()[-1].isdigit()
    })
    if len(wavelengths) != 2:
        raise ValueError(f"Expected 2 wavelengths, found {wavelengths}")
    wl1, wl2  = wavelengths
    ppf_tuple = (ppf_dict.get(wl1, 6.0), ppf_dict.get(wl2, 6.0))
    return mne.preprocessing.nirs.beer_lambert_law(raw_od_filt, ppf=ppf_tuple)


def preprocess_subject(
    raw_intensity: mne.io.BaseRaw,
    include_rpca: bool,
    include_ols: bool,
    ppf_dict: dict,
    subject_id: str = "?",
) -> Optional[mne.io.BaseRaw]:
    """
    Apply one combination of the preprocessing pipeline.

    Chain:
      B1  Intensity → OD
      B2  Bandpass filter (0.01–0.5 Hz, IIR, zero-phase)
      [B3/B4  Motion detection + rPCA on OD]   ← if include_rpca=True
      B5  MBLL → HbO/HbR
      [B6  Gaussian-basis OLS regression]       ← if include_ols=True

    No channel pruning applied at any step.
    Returns the preprocessed Raw (haemo) or None on failure.
    """
    try:
        # B1 — OD
        raw_od = mne.preprocessing.nirs.optical_density(raw_intensity, verbose=False)

        # B2 — Bandpass filter
        raw_od_filt = raw_od.copy().filter(
            l_freq=BP_L_FREQ, h_freq=BP_H_FREQ,
            method=BP_METHOD, verbose=False,
        )

        # B3 + B4 — Motion detection + rPCA (optional)
        if include_rpca:
            motion_results = detect_motion_artifacts(
                raw_od_filt,
                use_amp_only=False,   # dual criterion (amp + stdev) for mask quality
                verbose=False,
            )
            raw_od_filt, _rpca_summary = apply_rpca_correction(
                raw_od_filt,
                motion_results,
                n_components=RPCA_N_COMP,
                min_channels_flagged=RPCA_MIN_CH,
                verbose=False,
            )

        # B5 — MBLL
        raw_haemo = _mbll(raw_od_filt, ppf_dict)

        # B6 — OLS (optional)
        if include_ols:
            raw_haemo, _ols_results = apply_ols_regression(
                raw_haemo,
                verbose=False,
            )

        return raw_haemo

    except Exception as exc:
        print(f"    [{subject_id}] ⚠ preprocessing failed: {exc}")
        return None


def evoked_from_haemo(
    subject_id: str,
    raw_haemo: mne.io.BaseRaw,
) -> Optional[mne.Evoked]:
    """Extract epochs → average → Evoked.  Returns None on failure."""
    try:
        epochs = extract_epochs(raw_haemo, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
    except Exception as exc:
        print(f"    [{subject_id}] ⚠ epoch extraction failed: {exc}")
        return None
    if len(epochs) == 0:
        print(f"    [{subject_id}] ⚠ no epochs found")
        return None
    return epochs.average()


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _cnr_stats(values: list[float]) -> dict:
    """Compute 5-number summary, mean, and range from a list of CNR values."""
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return {k: np.nan for k in ["n", "min", "Q1", "median", "Q3", "max", "mean", "range"]}
    return {
        "n":      int(len(v)),
        "min":    round(float(np.min(v)),             4),
        "Q1":     round(float(np.percentile(v, 25)),  4),
        "median": round(float(np.median(v)),           4),
        "Q3":     round(float(np.percentile(v, 75)),  4),
        "max":    round(float(np.max(v)),              4),
        "mean":   round(float(np.mean(v)),             4),
        "range":  round(float(np.max(v) - np.min(v)), 4),
    }


# ---------------------------------------------------------------------------
# Per-condition CNR computation
# ---------------------------------------------------------------------------

def run_condition(
    cond: dict,
    raw_cache: dict[str, mne.io.BaseRaw],
    long_hbo: list[str],
    long_hbr: list[str],
) -> list[float]:
    """
    Run one preprocessing condition over all subjects and return pooled CNR values.

    Returns
    -------
    all_cnr : list[float]
        Pooled group-HRF CNR for HbO + HbR long channels.
    """
    include_rpca = cond["include_rpca"]
    include_ols  = cond["include_ols"]

    group_evokeds: dict[str, list[mne.Evoked]] = {g: [] for g in GROUPS}

    for g_name, subjects in GROUPS.items():
        print(f"\n  [{g_name}]")
        for sid in subjects:
            if sid not in raw_cache:
                print(f"    [{sid}] ⚠ skipped (no cached SNIRF)")
                continue

            raw_haemo = preprocess_subject(
                raw_cache[sid].copy(),     # always work on a copy
                include_rpca=include_rpca,
                include_ols=include_ols,
                ppf_dict=PPF_DICT,
                subject_id=sid,
            )
            if raw_haemo is None:
                continue

            ev = evoked_from_haemo(sid, raw_haemo)
            if ev is None:
                continue

            ev = _harmonise_evoked(ev, TARGET_SFREQ)
            group_evokeds[g_name].append(ev)
            print(f"    [{sid}] ✓  ({len(group_evokeds[g_name])} evoked in {g_name})")

    g1_name, g2_name = list(GROUPS.keys())
    g1_ev = group_evokeds[g1_name]
    g2_ev = group_evokeds[g2_name]
    print(f"\n  Subjects: {g1_name}={len(g1_ev)} | {g2_name}={len(g2_ev)}")

    all_cnr: list[float] = []
    for chromo, long_chs in [("hbo", long_hbo), ("hbr", long_hbr)]:
        label = "HbO" if chromo == "hbo" else "HbR"
        cnr_ch, _ = compute_group_hrf_cnr(g1_ev, g2_ev, long_chs, chromo, PEAK_WINDOW)
        valid = [v for v in cnr_ch.values() if np.isfinite(v)]
        all_cnr.extend(valid)
        if valid:
            print(f"  {label}: n={len(valid)} | "
                  f"median={np.median(valid):.3f} | max={np.max(valid):.3f}")

    return all_cnr


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_PAPER_PALETTE = {
    "01_baseline": "#AEC6E8",   # light blue  — baseline
    "02_ols_only": "#98DF8A",   # light green — OLS only
    "03_rpca_only": "#FFBB78",  # light orange — rPCA only
    "04_rpca_ols": "#FF9896",   # light red   — full pipeline
}


def _make_boxplot(
    cnr_values: list[float],
    cond: dict,
    stats: dict,
    save_path: Path,
) -> None:
    """Save an individual CNR boxplot for a single condition."""
    arr = np.array([v for v in cnr_values if np.isfinite(v)])
    color = _PAPER_PALETTE.get(cond["name"], "#C5B0D5")

    fig, ax = plt.subplots(figsize=(4.5, 6))

    bp = ax.boxplot(
        [arr],
        positions=[1],
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor=color, color="#333333", linewidth=1.5),
        medianprops=dict(color="#D62728", linewidth=2.5),
        whiskerprops=dict(color="#333333", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#333333", linewidth=1.5),
        flierprops=dict(marker="+", color="#D62728", markersize=7),
    )

    ax.axhline(PAPER_MEDIAN_TARGET, color="#2CA02C", lw=1.8, ls="--",
               label=f"Paper target median (≈ {PAPER_MEDIAN_TARGET})")
    ax.axhline(CNR_THRESHOLD, color="#FF7F0E", lw=1.2, ls=":",
               label=f"Detectable threshold (CNR = {CNR_THRESHOLD:.0f})")

    # Annotate stats inside the plot
    stat_lines = [
        f"n = {stats['n']}",
        f"min  = {stats['min']:.3f}",
        f"Q1   = {stats['Q1']:.3f}",
        f"med  = {stats['median']:.3f}",
        f"Q3   = {stats['Q3']:.3f}",
        f"max  = {stats['max']:.3f}",
        f"mean = {stats['mean']:.3f}",
        f"range= {stats['range']:.3f}",
    ]
    ax.text(
        0.97, 0.98, "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xticks([1])
    ax.set_xticklabels([cond["label"]], fontsize=10)
    ax.set_ylabel("CNR", fontsize=12)
    ax.set_title(
        f"CNR — Expert vs Novice\n{cond['note']}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.30)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _make_combined_boxplot(
    cnr_arrays: dict[str, list[float]],
    conditions: list[dict],
    save_path: Path,
) -> None:
    """Save a combined boxplot comparing all conditions side-by-side."""
    ordered = [c for c in conditions if c["name"] in cnr_arrays]
    labels  = [c["label"] for c in ordered]
    data    = [np.array([v for v in cnr_arrays[c["name"]] if np.isfinite(v)])
               for c in ordered]
    colors  = [_PAPER_PALETTE.get(c["name"], "#C5B0D5") for c in ordered]

    fig_w = max(10, 2.8 * len(data))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bp = ax.boxplot(
        data,
        positions=range(1, len(data) + 1),
        patch_artist=True,
        widths=0.60,
        boxprops=dict(color="#333333", linewidth=1.5),
        medianprops=dict(color="#D62728", linewidth=2.5),
        whiskerprops=dict(color="#333333", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#333333", linewidth=1.5),
        flierprops=dict(marker="+", color="#D62728", markersize=6),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Highlight baseline with a thicker border
    if bp["boxes"]:
        bp["boxes"][0].set_linewidth(2.5)

    # Reference lines
    ax.axhline(PAPER_MEDIAN_TARGET, color="#2CA02C", lw=1.8, ls="--",
               label=f"Paper target median (≈ {PAPER_MEDIAN_TARGET})")
    ax.axhline(CNR_THRESHOLD, color="#FF7F0E", lw=1.2, ls=":",
               label=f"Detectable threshold (CNR = {CNR_THRESHOLD:.0f})")

    # Annotate median above each box
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            med = float(np.median(arr))
            ax.text(i, med + 0.012, f"{med:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color="#D62728",
                    fontweight="bold")

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CNR", fontsize=12)
    ax.set_title(
        "CNR Pipeline Combination Audit — Expert vs Novice\n"
        "Group-HRF method · All long-separation channels · HbO + HbR pooled",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.30)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.3, len(data) + 0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], save_path: Path) -> None:
    """Save summary statistics CSV with 5-number summary, mean, and range."""
    col_order = [
        "condition", "label", "include_rpca", "include_ols",
        "n", "min", "Q1", "median", "Q3", "max", "mean", "range",
        "note",
    ]
    df = pd.DataFrame(rows)[col_order]
    df.to_csv(save_path, index=False)

    # Console table
    display_cols = ["condition", "include_rpca", "include_ols",
                    "n", "min", "Q1", "median", "Q3", "max", "mean", "range"]
    print("\n" + "=" * 110)
    print("  PIPELINE COMBINATION AUDIT — SUMMARY TABLE")
    print(f"  Paper target: median ≈ {PAPER_MEDIAN_TARGET}  |  max > {PAPER_MAX_TARGET}")
    print("=" * 110)
    print(df[display_cols].to_string(index=False))
    print("=" * 110)
    print(f"\n  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def run_audit(force_recompute: bool = False) -> None:
    """
    Run all 4 pipeline-combination conditions and save outputs.

    Parameters
    ----------
    force_recompute : bool
        Unused in this script (all processing is done in-memory from SNIRF
        cache, no evoked cache is used). Kept for CLI parity with other audit
        scripts.
    """
    all_subjects = [s for g in GROUPS.values() for s in g]

    # ── Pre-load all SNIRF files once ──────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Pipeline Combination CNR Audit")
    print(f"  Pre-loading {len(all_subjects)} SNIRF files …")
    print(f"{'='*68}")

    raw_cache: dict[str, mne.io.BaseRaw] = {}
    for sid in all_subjects:
        try:
            snirf_files = get_snirf_files(STUDY, sid)
            raw = mne.io.read_raw_snirf(str(snirf_files[0]), preload=True, verbose=False)
            raw_cache[sid] = raw
            print(f"  [{sid}]  {len(raw.ch_names)} ch  {raw.times[-1]:.0f} s  "
                  f"@ {raw.info['sfreq']} Hz")
        except Exception as exc:
            print(f"  [{sid}] ⚠ SNIRF load failed: {exc}")

    # ── Identify long-separation channels from a reference run ─────────────
    ref_sid = next(s for s in all_subjects if s in raw_cache)
    _ref_od    = mne.preprocessing.nirs.optical_density(raw_cache[ref_sid], verbose=False)
    _ref_filt  = _ref_od.copy().filter(BP_L_FREQ, BP_H_FREQ, method=BP_METHOD, verbose=False)
    _ref_haemo = _mbll(_ref_filt, PPF_DICT)
    long_hbo, long_hbr = get_long_channel_names(_ref_haemo)
    print(f"\n  Long-separation channels: {len(long_hbo)} HbO + {len(long_hbr)} HbR "
          f"= {len(long_hbo) + len(long_hbr)} total")
    del _ref_od, _ref_filt, _ref_haemo

    # ── Iterate over conditions ─────────────────────────────────────────────
    cnr_arrays: dict[str, list[float]] = {}
    summary_rows: list[dict] = []

    for idx, cond in enumerate(CONDITIONS):
        t0 = time.time()
        print(f"\n{'─'*68}")
        print(f"  [{idx+1}/{len(CONDITIONS)}]  {cond['name']}")
        print(f"  {cond['note']}")
        print(f"  rPCA: {'ON' if cond['include_rpca'] else 'off'}  |  "
              f"OLS: {'ON' if cond['include_ols'] else 'off'}")
        print(f"{'─'*68}")

        all_cnr = run_condition(cond, raw_cache, long_hbo, long_hbr)
        elapsed = round(time.time() - t0, 1)

        stats = _cnr_stats(all_cnr)
        cnr_arrays[cond["name"]] = all_cnr

        print(
            f"\n  CNR  n={stats['n']}  "
            f"min={stats['min']:.3f}  Q1={stats['Q1']:.3f}  "
            f"median={stats['median']:.3f}  mean={stats['mean']:.3f}  "
            f"Q3={stats['Q3']:.3f}  max={stats['max']:.3f}  "
            f"range={stats['range']:.3f}  [{elapsed}s]"
        )

        # Per-condition boxplot
        plot_path = OUT_DIR / f"cnr_boxplot_{cond['name']}.png"
        _make_boxplot(all_cnr, cond, stats, plot_path)

        summary_rows.append({
            "condition":    cond["name"],
            "label":        cond["label"].replace("\n", " "),
            "include_rpca": cond["include_rpca"],
            "include_ols":  cond["include_ols"],
            **stats,
            "note":         cond["note"],
        })

    # ── Combined boxplot ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Generating combined comparison boxplot …")
    combined_path = OUT_DIR / "cnr_all_conditions_boxplot.png"
    _make_combined_boxplot(cnr_arrays, CONDITIONS, combined_path)

    # ── Summary CSV ─────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "pipeline_combo_audit_table.csv"
    _save_csv(summary_rows, csv_path)

    # ── Final ranking ───────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  RANKING BY MEDIAN CNR")
    print(f"  Paper target: median ≈ {PAPER_MEDIAN_TARGET}  |  max > {PAPER_MAX_TARGET}")
    print(f"  {'#':<3}  {'Condition':<18}  {'rPCA':<6}  {'OLS':<6}  "
          f"{'Median':>8}  {'Δ vs base':>10}  {'Max':>8}")
    print("  " + "-" * 64)

    df_sorted = pd.DataFrame(summary_rows).sort_values("median", ascending=False)
    base_med  = float(
        pd.DataFrame(summary_rows).loc[
            pd.DataFrame(summary_rows)["condition"] == "01_baseline", "median"
        ].values[0]
    )

    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        delta  = float(row["median"]) - base_med
        marker = "  ← BASELINE" if row["condition"] == "01_baseline" else ""
        delta_s = f"{delta:+.4f}" if row["condition"] != "01_baseline" else "    —"
        print(
            f"  {rank:<3}  {row['condition']:<18}  "
            f"{'✓' if row['include_rpca'] else '—':<6}  "
            f"{'✓' if row['include_ols'] else '—':<6}  "
            f"{float(row['median']):>8.4f}  {delta_s:>10}  "
            f"{float(row['max']):>8.4f}{marker}"
        )

    print(f"{'='*68}\n")
    print(f"  Outputs written to: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline combination audit: test all preprocessing combos for CNR"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute (reserved for future cache support; currently all "
             "processing is done in-memory from raw SNIRF files).",
    )
    args = parser.parse_args()
    run_audit(force_recompute=args.force)
