"""
audit_cnr.py
Stage B Preprocessing — CNR Parameter Audit

Purpose: Systematically vary preprocessing parameters (OD conversion, bandpass
         filter method / phase / order / frequency bounds) to identify which
         choices cause the reproduced CNR to deviate from the ASTaUND paper
         target (Figure 4i: median ≈ 0.45, max > 2.25).

Pipeline under audit
--------------------
  Raw Intensity → OD → [optional bandpass] → MBLL → Epochs → Group-HRF CNR

Conditions tested (11 total)
-----------------------------
  OD conversion  : natural log (MNE default)  vs  log₁₀ (scale-invariance check)
  Filter method  : IIR Butterworth  vs  FIR windowed-sinc
  Filter phase   : zero-phase (filtfilt)  vs  causal (forward / minimum-phase)
  Filter order   : MNE auto, 2, 6, 8  (IIR Butterworth)
  Frequency band : current config (0.001–0.5 Hz)  vs  paper spec (0.01–0.5 Hz)
  No bandpass    : OD → MBLL with no filtering

Each condition processes all 14 Expert-Novice subjects from raw SNIRF and
computes group-HRF CNR (Expert vs Novice, all long-separation channels, HbO+HbR).

Outputs
-------
  outputs/audit/cnr_audit/cnr_audit_table.csv    — per-condition CNR statistics
  outputs/audit/cnr_audit/cnr_audit_boxplot.png  — multi-box comparison figure
  outputs/audit/cnr_audit/cnr_audit_report.txt   — ranked summary + analysis

Usage
-----
  # All 11 conditions (≈ 10–20 min)
  python code/stage_b_preprocessing/audit_cnr.py

  # Fast mode: 5 key conditions only (≈ 4–8 min)
  python code/stage_b_preprocessing/audit_cnr.py --fast

Paper reference
---------------
  Kamat et al. (2023) ASTaUND, Figure 4i:  median CNR ≈ 0.45,  max > 2.25
  Current baseline (config.yaml):           median CNR ≈ 0.33
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
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from utils import get_snirf_files, load_config


def _load_stage_c(name: str, filename: str):
    stage_c = PROJECT_ROOT / "code" / "stage_c_validation"
    spec = importlib.util.spec_from_file_location(name, stage_c / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hrf = _load_stage_c("hrf_extraction", "01_hrf_extraction.py")
_grp = _load_stage_c("group_hrf",      "02_group_hrf_analysis.py")
_cnr = _load_stage_c("cnr_calc",       "04_calculate_cnr.py")

extract_epochs        = _hrf.extract_epochs
get_long_channel_names = _hrf.get_long_channel_names
compute_group_hrf_cnr  = _cnr.compute_group_hrf_cnr
_harmonise_evoked      = _grp._harmonise_evoked

# ---------------------------------------------------------------------------
# Study constants (sourced from config.yaml)
# ---------------------------------------------------------------------------
_CFG  = load_config()
_DS   = _CFG["datasets"]["expert_novice"]

STUDY  = "expert_novice"
GROUPS = {
    "Expert": _DS.get("groups", {}).get(
        "Expert", ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"]
    ),
    "Novice": _DS.get("groups", {}).get(
        "Novice", ["R1", "R2", "R3", "R4", "R5", "R6"]
    ),
}

_PPF_RAW = _DS.get("ppf", {})
PPF_DICT = (
    {int(k): float(v) for k, v in _PPF_RAW.items()}
    if _PPF_RAW else {690: 6.4, 830: 5.8}
)

_EPOCHS_CFG  = _CFG["stage_c"]["epochs"]
TMIN         = float(_EPOCHS_CFG["tmin"])
TMAX         = float(_EPOCHS_CFG["tmax"])
BASELINE     = tuple(float(v) for v in _EPOCHS_CFG["baseline"])
PEAK_WINDOW  = (5.0, 15.0)
TARGET_SFREQ = 25.0

OUT_DIR = PROJECT_ROOT / "outputs" / "audit" / "cnr_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audit conditions
# ---------------------------------------------------------------------------
# Each condition dict:
#   name        short identifier (used as dict key, safe for filenames)
#   label       multi-line display label for the boxplot x-axis
#   od          'ln' (MNE default) or 'log10'
#   bp_method   'iir', 'fir', or None  (None = skip filter)
#   bp_phase    'zero' (filtfilt), 'forward' (IIR causal), 'minimum' (FIR causal)
#   bp_order    int or None  (None = MNE auto)
#   l_freq      float or None
#   h_freq      float or None
#   note        one-line description for the report

ALL_CONDITIONS: list[dict] = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    {
        "name":      "01_baseline",
        "label":     "Baseline\n(ln, IIR, zero-ph\n0.001–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  None,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "Current config.yaml settings (reference point)",
    },
    # ── Paper bandpass frequencies ────────────────────────────────────────────
    {
        "name":      "02_paper_bp",
        "label":     "Paper BP\n(ln, IIR, zero-ph\n0.01–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  None,
        "l_freq":    0.01,
        "h_freq":    0.5,
        "note":      "Paper-specified bandpass range (0.01–0.5 Hz) — isolates freq effect",
    },
    # ── Causal IIR (Homer3-style) ─────────────────────────────────────────────
    {
        "name":      "03_causal_iir",
        "label":     "Causal IIR\n(ln, IIR, forward\n0.001–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "forward",
        "bp_order":  None,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "Homer3-style: causal (forward-only) IIR — isolates phase effect",
    },
    # ── Paper BP + causal (closest to Homer3 pipeline) ────────────────────────
    {
        "name":      "04_paper_bp_causal",
        "label":     "Paper BP\n+ Causal\n(ln, IIR, fwd\n0.01–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "forward",
        "bp_order":  None,
        "l_freq":    0.01,
        "h_freq":    0.5,
        "note":      "Homer3 approximation: paper bandpass frequencies + causal IIR",
    },
    # ── FIR zero-phase ────────────────────────────────────────────────────────
    # Note: l_freq=0.01 Hz used because FIR at 0.001 Hz requires a filter kernel
    # longer than the recording duration (~1000 s at 25 Hz), which is infeasible.
    {
        "name":      "05_fir_zero",
        "label":     "FIR zero-ph\n(ln, FIR, zero\n0.01–0.5 Hz)",
        "od":        "ln",
        "bp_method": "fir",
        "bp_phase":  "zero",
        "bp_order":  None,
        "l_freq":    0.01,
        "h_freq":    0.5,
        "note":      "FIR windowed-sinc, zero-phase (l_freq raised to 0.01 Hz for FIR feasibility)",
    },
    # ── FIR minimum-phase (causal) ────────────────────────────────────────────
    {
        "name":      "06_fir_causal",
        "label":     "FIR min-ph\n(ln, FIR, min\n0.01–0.5 Hz)",
        "od":        "ln",
        "bp_method": "fir",
        "bp_phase":  "minimum",
        "bp_order":  None,
        "l_freq":    0.01,
        "h_freq":    0.5,
        "note":      "FIR minimum-phase (causal) filter (l_freq=0.01 Hz for feasibility)",
    },
    # ── IIR order sweep ───────────────────────────────────────────────────────
    {
        "name":      "07_iir_order2",
        "label":     "IIR order 2\n(ln, zero-ph\n0.001–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  2,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "Lower-order Butterworth: gentler rolloff, less ringing",
    },
    {
        "name":      "08_iir_order6",
        "label":     "IIR order 6\n(ln, zero-ph\n0.001–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  6,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "Higher-order Butterworth: sharper rolloff",
    },
    {
        "name":      "09_iir_order8",
        "label":     "IIR order 8\n(ln, zero-ph\n0.001–0.5 Hz)",
        "od":        "ln",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  8,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "Highest IIR order tested — approaches ideal brick-wall response",
    },
    # ── log10 OD (scale-invariance check) ────────────────────────────────────
    # Theory: CNR = |μ1-μ2| / √(σ1²+σ2²) is scale-invariant, so dividing all
    # values by ln(10) ≈ 2.303 should leave CNR unchanged.  If CNR DOES change,
    # there is an unexpected interaction with the MBLL or filter nonlinearity.
    {
        "name":      "10_log10_od",
        "label":     "log₁₀ OD\n(IIR, zero-ph\n0.001–0.5 Hz)",
        "od":        "log10",
        "bp_method": "iir",
        "bp_phase":  "zero",
        "bp_order":  None,
        "l_freq":    0.001,
        "h_freq":    0.5,
        "note":      "log10 OD (÷2.303 vs ln) — CNR should be identical if scale-invariant",
    },
    # ── No bandpass filter ────────────────────────────────────────────────────
    {
        "name":      "11_no_filter",
        "label":     "No filter\n(ln, OD→MBLL\nno BP)",
        "od":        "ln",
        "bp_method": None,
        "bp_phase":  None,
        "bp_order":  None,
        "l_freq":    None,
        "h_freq":    None,
        "note":      "Skip bandpass entirely: Intensity → OD → MBLL with no filtering",
    },
]

# Fast mode: 5 conditions that cover the main hypotheses
FAST_NAMES = {"01_baseline", "02_paper_bp", "03_causal_iir", "04_paper_bp_causal", "10_log10_od"}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _optical_density(raw_intensity: mne.io.BaseRaw, use_log10: bool) -> mne.io.BaseRaw:
    """
    Intensity → OD.

    use_log10=False : MNE default  OD = -ln(I / mean(I))
    use_log10=True  : log10 scale  OD = -log10(I / mean(I)) = OD_ln / ln(10)

    The log10 version is implemented as a post-scale of MNE's output, which
    ensures channel type metadata is set correctly by MNE.
    """
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity, verbose=False)
    if use_log10:
        raw_od._data /= np.log(10)      # ln(10) ≈ 2.3026
    return raw_od


def _bandpass(
    raw_od: mne.io.BaseRaw,
    l_freq: Optional[float],
    h_freq: Optional[float],
    method: Optional[str],
    phase: Optional[str],
    order: Optional[int],
) -> mne.io.BaseRaw:
    """
    Apply bandpass filter to OD data.  Returns a copy.
    If method is None, returns the OD unchanged (no filter applied).
    Falls back to unfiltered OD if MNE raises an error (e.g. FIR kernel longer
    than data — logged with a warning).
    """
    if method is None:
        return raw_od.copy()

    kwargs: dict = dict(
        l_freq=l_freq, h_freq=h_freq,
        method=method, verbose=False,
    )
    if method == "iir":
        kwargs["phase"] = phase
        if order is not None:
            kwargs["iir_params"] = {"order": order, "ftype": "butter", "output": "sos"}
    elif method == "fir":
        kwargs["phase"] = phase

    try:
        return raw_od.copy().filter(**kwargs)
    except Exception as exc:
        print(f"      ⚠ filter failed ({exc}) — using unfiltered OD for this subject")
        return raw_od.copy()


def _mbll(raw_od_filt: mne.io.BaseRaw, ppf_dict: dict) -> mne.io.BaseRaw:
    """OD → HbO / HbR via Modified Beer-Lambert Law."""
    wavelengths = sorted({
        int(ch.split()[-1])
        for ch in raw_od_filt.ch_names
        if ch.split()[-1].isdigit()
    })
    if len(wavelengths) != 2:
        raise ValueError(f"Expected 2 wavelengths, found {wavelengths}")
    wl1, wl2   = wavelengths
    ppf_tuple  = (ppf_dict.get(wl1, 6.0), ppf_dict.get(wl2, 6.0))
    return mne.preprocessing.nirs.beer_lambert_law(raw_od_filt, ppf=ppf_tuple)


def preprocess_one(
    raw_intensity: mne.io.BaseRaw,
    cond: dict,
    ppf_dict: dict,
) -> mne.io.BaseRaw:
    """Apply one condition's preprocessing chain to a raw intensity recording."""
    raw_od      = _optical_density(raw_intensity, use_log10=(cond["od"] == "log10"))
    raw_od_filt = _bandpass(
        raw_od,
        l_freq=cond["l_freq"], h_freq=cond["h_freq"],
        method=cond["bp_method"], phase=cond["bp_phase"], order=cond["bp_order"],
    )
    return _mbll(raw_od_filt, ppf_dict)


def evoked_from_haemo(
    subject_id: str,
    raw_haemo: mne.io.BaseRaw,
) -> Optional[mne.Evoked]:
    """Extract epochs → average → Evoked.  Returns None on failure."""
    try:
        epochs = extract_epochs(raw_haemo, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
    except Exception as exc:
        print(f"      [{subject_id}] ⚠ epoch failed: {exc}")
        return None
    if len(epochs) == 0:
        print(f"      [{subject_id}] ⚠ no epochs found")
        return None
    return epochs.average()


# ---------------------------------------------------------------------------
# CNR statistics helper
# ---------------------------------------------------------------------------

def _cnr_stats(values: list[float]) -> dict:
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return {k: np.nan for k in ["n", "min", "Q1", "median", "mean", "Q3", "max"]}
    return {
        "n":      int(len(v)),
        "min":    round(float(np.min(v)),              4),
        "Q1":     round(float(np.percentile(v, 25)),   4),
        "median": round(float(np.median(v)),            4),
        "mean":   round(float(np.mean(v)),              4),
        "Q3":     round(float(np.percentile(v, 75)),   4),
        "max":    round(float(np.max(v)),               4),
    }


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def run_audit(conditions: list[dict]) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """
    Process every condition and return a stats DataFrame + per-condition CNR arrays.
    """
    all_subjects = [s for g in GROUPS.values() for s in g]

    # ── Pre-load all SNIRF files once ─────────────────────────────────────────
    print(f"\n{'='*68}")
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

    # ── Identify long-separation channels from a reference MBLL output ────────
    ref_sid = next(s for s in all_subjects if s in raw_cache)
    _ref_od   = mne.preprocessing.nirs.optical_density(raw_cache[ref_sid], verbose=False)
    _ref_filt = _ref_od.copy().filter(0.001, 0.5, method="iir", verbose=False)
    _ref_haemo = _mbll(_ref_filt, PPF_DICT)
    hbo_long, hbr_long = get_long_channel_names(_ref_haemo)
    print(f"\n  Long channels: {len(hbo_long)} HbO + {len(hbr_long)} HbR "
          f"= {len(hbo_long)+len(hbr_long)} total")
    del _ref_od, _ref_filt, _ref_haemo

    # ── Iterate over conditions ───────────────────────────────────────────────
    rows: list[dict]               = []
    cnr_arrays: dict[str, list]    = {}

    for idx, cond in enumerate(conditions):
        t0 = time.time()
        print(f"\n{'─'*68}")
        print(f"  [{idx+1}/{len(conditions)}]  {cond['name']}")
        print(f"  {cond['note']}")
        print(f"{'─'*68}")

        group_evokeds: dict[str, list[mne.Evoked]] = {g: [] for g in GROUPS}

        for g_name, subjects in GROUPS.items():
            print(f"\n  [{g_name}]")
            for sid in subjects:
                if sid not in raw_cache:
                    print(f"    [{sid}] ⚠ skipped (no cached SNIRF)")
                    continue
                try:
                    raw_haemo = preprocess_one(raw_cache[sid], cond, PPF_DICT)
                    ev = evoked_from_haemo(sid, raw_haemo)
                    if ev is not None:
                        ev = _harmonise_evoked(ev, TARGET_SFREQ)
                        group_evokeds[g_name].append(ev)
                        n = len(group_evokeds[g_name])
                        print(f"    [{sid}] ✓  ({n} evoked)")
                except Exception as exc:
                    print(f"    [{sid}] ⚠ failed: {exc}")

        g1, g2 = list(GROUPS.keys())
        g1_ev  = group_evokeds[g1]
        g2_ev  = group_evokeds[g2]
        print(f"\n  Subjects used: {g1}={len(g1_ev)}  {g2}={len(g2_ev)}")

        # ── Compute group-HRF CNR (HbO + HbR combined) ───────────────────────
        all_cnr: list[float] = []
        for chromo, long_chs in [("hbo", hbo_long), ("hbr", hbr_long)]:
            cnr_ch, _ = compute_group_hrf_cnr(g1_ev, g2_ev, long_chs, chromo, PEAK_WINDOW)
            all_cnr.extend(v for v in cnr_ch.values() if np.isfinite(v))

        stats   = _cnr_stats(all_cnr)
        elapsed = round(time.time() - t0, 1)

        rows.append({
            "condition":  cond["name"],
            "label":      cond["label"].replace("\n", " "),
            "od":         cond["od"],
            "bp_method":  cond["bp_method"] or "none",
            "bp_phase":   cond["bp_phase"]  or "—",
            "bp_order":   str(cond["bp_order"]) if cond["bp_order"] else "auto",
            "l_freq":     str(cond["l_freq"]) if cond["l_freq"] is not None else "—",
            "h_freq":     str(cond["h_freq"]) if cond["h_freq"] is not None else "—",
            **stats,
            "elapsed_s":  elapsed,
            "note":       cond["note"],
        })
        cnr_arrays[cond["name"]] = all_cnr

        print(
            f"\n  CNR  n={stats['n']}  "
            f"min={stats['min']:.3f}  Q1={stats['Q1']:.3f}  "
            f"median={stats['median']:.3f}  mean={stats['mean']:.3f}  "
            f"Q3={stats['Q3']:.3f}  max={stats['max']:.3f}  "
            f"[{elapsed}s]"
        )

    return pd.DataFrame(rows), cnr_arrays


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_table(df: pd.DataFrame) -> Path:
    csv_path = OUT_DIR / "cnr_audit_table.csv"
    df.to_csv(csv_path, index=False)

    cols = [
        "condition", "od", "bp_method", "bp_phase", "bp_order",
        "l_freq", "h_freq", "n", "min", "Q1", "median", "mean", "Q3", "max",
    ]
    print("\n" + "="*108)
    print("  CNR AUDIT — STATISTICS TABLE")
    print("="*108)
    print(df[cols].to_string(index=False))
    print("="*108)
    print(f"\n  Saved: {csv_path.name}")
    return csv_path


def _save_boxplot(cnr_arrays: dict, conditions: list[dict]) -> Path:
    plot_path = OUT_DIR / "cnr_audit_boxplot.png"

    ordered = [c for c in conditions if c["name"] in cnr_arrays]
    labels  = [c["label"]  for c in ordered]
    data    = [np.array([v for v in cnr_arrays[c["name"]] if np.isfinite(v)])
               for c in ordered]

    fig_w = max(14, 2.4 * len(data))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bp = ax.boxplot(
        data,
        positions=range(1, len(data) + 1),
        patch_artist=True,
        widths=0.60,
        boxprops=dict(facecolor="white", color="#1F77B4", linewidth=1.5),
        medianprops=dict(color="#D62728", linewidth=2.2),
        whiskerprops=dict(color="#1F77B4", linewidth=1.5, linestyle="--"),
        capprops=dict(color="#1F77B4", linewidth=1.5),
        flierprops=dict(marker="+", color="#D62728", markersize=6),
    )

    # Highlight baseline box
    if data:
        bp["boxes"][0].set_facecolor("#D9EAF7")
        bp["boxes"][0].set_linewidth(2.2)

    # Reference lines
    ax.axhline(0.45, color="#2CA02C", lw=1.8, ls="--",
               label="Paper target median (≈ 0.45)")
    ax.axhline(1.0,  color="#FF7F0E", lw=1.2, ls=":",
               label="Detectable threshold (CNR = 1.0)")

    # Annotate median value above each box
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            med = float(np.median(arr))
            ax.text(i, med + 0.015, f"{med:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#D62728",
                    fontweight="bold")

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CNR", fontsize=12)
    ax.set_title(
        "CNR Parameter Audit — Expert vs Novice\n"
        "Group-HRF method · All long-separation channels · HbO + HbR pooled",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.30)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.3, len(data) + 0.7)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path.name}")
    return plot_path


def _save_report(df: pd.DataFrame, conditions: list[dict]) -> Path:
    """
    Write a ranked analysis report:  ranking table + per-parameter commentary.
    """
    report_path = OUT_DIR / "cnr_audit_report.txt"

    df_sorted = df.sort_values("median", ascending=False).reset_index(drop=True)

    baseline_row = df.loc[df["condition"] == "01_baseline"]
    baseline_med = float(baseline_row["median"].values[0]) if len(baseline_row) else np.nan

    def _get(name: str, col: str) -> Optional[float]:
        row = df.loc[df["condition"] == name]
        if len(row) == 0:
            return None
        v = row[col].values[0]
        return float(v) if np.isfinite(v) else None

    lines: list[str] = []

    lines += [
        "=" * 72,
        "  CNR PARAMETER AUDIT REPORT",
        "  ASTaUND Phase 1 — Expert vs Novice (group-HRF CNR)",
        "=" * 72,
        "",
        f"  Paper target   : median ≈ 0.45   max > 2.25  (Figure 4i)",
        f"  Baseline (01)  : median = {baseline_med:.4f}",
        f"  Gap to target  : {0.45 - baseline_med:+.4f}",
        "",
        "  RANKING BY MEDIAN CNR (highest → lowest):",
        f"  {'#':<3}  {'Condition':<48}  {'Median':>7}  {'Δ vs base':>10}  {'Max':>7}",
        "  " + "-" * 76,
    ]

    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        delta = row["median"] - baseline_med
        marker = "  ← BASELINE" if row["condition"] == "01_baseline" else ""
        delta_s = f"{delta:+.4f}" if row["condition"] != "01_baseline" else "    —"
        lines.append(
            f"  {rank:<3}  {row['label'][:48]:<48}  {row['median']:>7.4f}  "
            f"{delta_s:>10}  {row['max']:>7.4f}{marker}"
        )

    # ── Parameter-by-parameter analysis ──────────────────────────────────────
    lines += ["", "  PARAMETER EFFECTS ANALYSIS:", "  " + "-" * 72]

    # 1. OD log type
    log10_med = _get("10_log10_od", "median")
    if log10_med is not None:
        delta = log10_med - baseline_med
        lines += [
            "",
            f"  [1] OD CONVERSION  ln vs log₁₀",
            f"      ln (baseline): {baseline_med:.4f}   log₁₀: {log10_med:.4f}   "
            f"Δ = {delta:+.4f}",
        ]
        if abs(delta) < 0.002:
            lines.append(
                "      → CNR is SCALE-INVARIANT as expected. Dividing all OD values by\n"
                "        ln(10) ≈ 2.303 scales μ and σ identically, so the ratio cancels.\n"
                "        The OD log convention does NOT explain the CNR gap."
            )
        else:
            lines.append(
                f"      → Unexpected Δ = {delta:.4f}. Investigate MBLL interaction or\n"
                f"        numerical precision issue with log10 OD scale."
            )

    # 2. Filter frequency band
    paper_bp_med = _get("02_paper_bp", "median")
    if paper_bp_med is not None:
        delta = paper_bp_med - baseline_med
        lines += [
            "",
            f"  [2] BANDPASS FREQUENCY  0.001–0.5 Hz (baseline) vs 0.01–0.5 Hz (paper)",
            f"      Baseline: {baseline_med:.4f}   Paper BP: {paper_bp_med:.4f}   "
            f"Δ = {delta:+.4f}",
        ]
        if delta > 0.01:
            lines.append(
                f"      → Paper bandpass RAISES CNR by {delta:.4f}.\n"
                "        The 0.5–0.5 Hz band adds HF noise to σ without adding HRF signal,\n"
                "        deflating CNR.  Switching to the paper range is beneficial."
            )
        elif delta < -0.01:
            lines.append(
                f"      → Narrower paper bandpass LOWERS CNR by {abs(delta):.4f}.\n"
                "        The 0.001–0.01 Hz ultra-low-frequency content contributes positively\n"
                "        to the HRF shape captured in [5, 15] s peak window."
            )
        else:
            lines.append(
                f"      → Filter frequency band has MINIMAL impact (Δ = {delta:.4f}).\n"
                "        This parameter alone does not explain the CNR gap."
            )

    # 3. Filter phase
    causal_med = _get("03_causal_iir", "median")
    if causal_med is not None:
        delta = causal_med - baseline_med
        lines += [
            "",
            f"  [3] FILTER PHASE  zero-phase (filtfilt) vs causal (forward-only)",
            f"      Zero-phase: {baseline_med:.4f}   Causal: {causal_med:.4f}   "
            f"Δ = {delta:+.4f}",
        ]
        if delta > 0.02:
            lines.append(
                f"      → Causal filter INCREASES CNR by {delta:.4f}.\n"
                "        Zero-phase (filtfilt) applies the filter twice, doubling the effective\n"
                "        order and introducing different transient behaviour at epoch boundaries.\n"
                "        The Homer3/Homer2 pipeline uses a causal (forward-only) filter."
            )
        elif delta < -0.02:
            lines.append(
                f"      → Causal filter DECREASES CNR by {abs(delta):.4f}.\n"
                "        Zero-phase filtfilt suppresses noise more effectively for CNR."
            )
        else:
            lines.append(
                f"      → Filter phase has MINIMAL impact (Δ = {delta:.4f})."
            )

    # 4. Combined Homer3 approximation
    homer3_med = _get("04_paper_bp_causal", "median")
    if homer3_med is not None:
        delta = homer3_med - baseline_med
        lines += [
            "",
            f"  [4] HOMER3 APPROXIMATION  paper BP (0.01–0.5 Hz) + causal IIR",
            f"      Baseline: {baseline_med:.4f}   Homer3 approx: {homer3_med:.4f}   "
            f"Δ = {delta:+.4f}",
        ]
        if delta > 0.02:
            lines.append(
                f"      → Combined paper-style settings INCREASE CNR by {delta:.4f}.\n"
                "        This condition is closest to how Homer2/3 processes the data\n"
                "        and represents the most actionable correction to the pipeline."
            )
        elif abs(delta) <= 0.02:
            lines.append(
                f"      → Combined Homer3 approximation has minimal additional effect\n"
                f"        beyond individual parameter changes (Δ = {delta:.4f})."
            )

    # 5. Filter order
    order2 = _get("07_iir_order2", "median")
    order6 = _get("08_iir_order6", "median")
    order8 = _get("09_iir_order8", "median")
    if any(v is not None for v in [order2, order6, order8]):
        lines += ["", "  [5] IIR BUTTERWORTH ORDER  2 / auto / 6 / 8"]
        row_auto = df.loc[df["condition"] == "01_baseline"]
        auto_med = float(row_auto["bp_order"].values[0]) if len(row_auto) else "auto"
        parts = [f"order auto (baseline): {baseline_med:.4f}"]
        for o, m in [(2, order2), (6, order6), (8, order8)]:
            if m is not None:
                parts.append(f"order {o}: {m:.4f} (Δ{m - baseline_med:+.4f})")
        lines.append("      " + "   ".join(parts))
        diffs = [abs(m - baseline_med) for m in [order2, order6, order8] if m is not None]
        if max(diffs, default=0) < 0.01:
            lines.append(
                "      → Filter order has MINIMAL impact on CNR for fNIRS HRF data.\n"
                "        The slow HRF signal (0.001–0.5 Hz) is well within the passband\n"
                "        regardless of Butterworth order."
            )
        else:
            lines.append(
                f"      → Filter order DOES affect CNR (max Δ = {max(diffs):.4f}).\n"
                "        Check whether higher orders cause ringing artifacts at epoch edges."
            )

    # 6. FIR vs IIR (comparing at similar frequency ranges)
    fir_zero = _get("05_fir_zero", "median")
    if fir_zero is not None and paper_bp_med is not None:
        delta = fir_zero - paper_bp_med
        lines += [
            "",
            f"  [6] FIR vs IIR  (both at similar l_freq=0.01 Hz, zero-phase)",
            f"      IIR 0.01–0.5 Hz: {paper_bp_med:.4f}   FIR 0.01–0.5 Hz: {fir_zero:.4f}   "
            f"Δ = {delta:+.4f}",
            "      (Note: h_freq differs 0.5 vs 0.5 Hz, so this compares filter design)"
        ]
        if abs(delta) < 0.02:
            lines.append(
                "      → FIR vs IIR design has minimal impact. Both methods produce\n"
                "        comparable CNR for slow fNIRS signals."
            )
        else:
            lines.append(
                f"      → Non-trivial difference (Δ = {delta:.4f}). FIR/IIR design choice\n"
                "        may be relevant depending on filter edge characteristics."
            )

    # 7. No filter
    no_filt = _get("11_no_filter", "median")
    if no_filt is not None:
        delta = no_filt - baseline_med
        lines += [
            "",
            f"  [7] NO BANDPASS FILTER  (OD → MBLL only)",
            f"      Baseline: {baseline_med:.4f}   No filter: {no_filt:.4f}   "
            f"Δ = {delta:+.4f}",
        ]
        if delta > 0.05:
            lines.append(
                "      → Removing the filter INCREASES CNR. The bandpass may be\n"
                "        distorting the HRF peak or introducing baseline artifacts.\n"
                "        Review filter edge effects at the start/end of epochs."
            )
        elif delta < -0.05:
            lines.append(
                "      → Removing the filter significantly DECREASES CNR. The bandpass\n"
                "        is doing important noise rejection (drift + HF noise).\n"
                "        Low-frequency drift and high-frequency noise are major confounds."
            )
        else:
            lines.append(
                f"      → No filter gives similar CNR (Δ = {delta:.4f}).\n"
                "        The bandpass filter has limited effect on CNR for this dataset."
            )

    # ── Overall conclusion ────────────────────────────────────────────────────
    best_row   = df_sorted.iloc[0]
    best_delta = best_row["median"] - baseline_med

    lines += [
        "",
        "=" * 72,
        "  OVERALL CONCLUSION",
        "=" * 72,
        "",
        f"  Best condition    : {best_row['label'].replace(chr(10), ' ')}",
        f"  Best median CNR   : {best_row['median']:.4f}",
        f"  Baseline median   : {baseline_med:.4f}",
        f"  Improvement       : {best_delta:+.4f} ({best_delta/baseline_med*100:+.1f}%)",
        f"  Paper target      : ~0.45",
        f"  Remaining gap     : {0.45 - best_row['median']:.4f}",
        "",
        "  UNRESOLVED FACTORS (not tested by this script):",
        "  ─────────────────────────────────────────────────",
        "  • Molar extinction coefficients: MNE uses Matcher et al. (1994) / internal",
        "    tables. Homer2/3 uses GetExtinctions() from Prahl (1998) / Zijlstra (1990).",
        "    Different ε values change HbO/HbR cross-talk which could affect CNR.",
        "  • PPF values: config.yaml uses PPF={690:6.4, 830:5.8} — verify against the",
        "    paper's reported PPF for the TechEn CW6 system.",
        "  • Epoch baseline window: current (-25, 0) s vs paper's possible (-5, 0) s.",
        "    A longer baseline could introduce more drift into the baseline correction.",
        "  • OD denominator: MNE uses whole-recording mean; Homer3 uses a shorter",
        "    pre-stimulus reference window per trial.",
        "  • Subject-level HRF averaging: current pipeline uses equal-weighted",
        "    average across epochs; Homer3 weights may differ.",
        "=" * 72,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n  Saved: {report_path.name}")
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNR parameter audit — vary OD, bandpass, and MBLL settings"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help=f"Run only {len(FAST_NAMES)} key conditions instead of all {len(ALL_CONDITIONS)}",
    )
    args = parser.parse_args()

    conditions = (
        [c for c in ALL_CONDITIONS if c["name"] in FAST_NAMES]
        if args.fast else ALL_CONDITIONS
    )

    if args.fast:
        print(f"\n  Fast mode: {len(conditions)}/{len(ALL_CONDITIONS)} conditions")

    t_total = time.time()
    df_results, cnr_arrays = run_audit(conditions)

    print(f"\n{'='*68}")
    print("  Saving outputs …")
    print(f"{'='*68}")

    _save_table(df_results)
    _save_boxplot(cnr_arrays, conditions)
    _save_report(df_results, conditions)

    total = round(time.time() - t_total, 0)
    print(f"\n{'='*68}")
    print(f"  Done in {int(total)}s   Output: {OUT_DIR}")
    print(f"{'='*68}\n")
