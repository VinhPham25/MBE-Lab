"""
Stage B — Step B8: rPCA Over-Correction Quality Control Audit (Issue 8)
Purpose: Detect and flag subjects where rPCA motion correction may have
         removed genuine neural hemodynamic signal rather than motion artifacts.

Task coverage:
  8.1 — Define guardrails for acceptable variance removal and residual signal structure
  8.2 — Per-subject QC flags for excessive removal (HbO/HbR)
  8.3 — Fail/hold logic for subjects crossing guardrails
  8.4 — Recompute group HRF outputs with and without flagged subjects; delta report

Guardrails
----------
The rPCA algorithm (Homer3: hmrR_PCAfilter) is designed to correct discrete
motion artifacts.  Since the exact rPCA parameters used in Kamat et al. 2023
are unknown, we define physiologically-grounded guardrails to detect
over-correction — the silent removal of true neural signal.

G1 — Mask Coverage (rPCA-specific)
  PASS    : pct_flagged_union ≤ 35%
  WARNING : 35% < pct_flagged_union ≤ 55%
  HOLD    : pct_flagged_union > 55%
  Rationale: If >55% of time points are "corrected", rPCA transitions from
  artifact correction to global signal transformation, which risks removing
  genuine hemodynamics.

G2 — Neural Band OD Power Preservation (0.01–0.1 Hz, pre vs post rPCA)
  PASS    : neural_band_ratio ≥ 0.65  (≤35% reduction)
  WARNING : 0.45 ≤ neural_band_ratio < 0.65  (35–55% reduction)
  HOLD    : neural_band_ratio < 0.45  (>55% reduction)
  Rationale: Neural hemodynamics are concentrated in 0.01–0.1 Hz. Significant
  power loss in this band after rPCA indicates true signal removal.

G3 — HbO/HbR Anti-Correlation After Full Pipeline (post-OLS)
  PASS    : mean_r_HbO_HbR_after ≤ 0.0   (physiological)
  WARNING : 0.0 < mean_r_HbO_HbR_after ≤ 0.25
  HOLD    : mean_r_HbO_HbR_after > 0.25   (pathological positive coupling)
  Rationale: After rPCA+OLS, HbO and HbR should be anti-correlated (neural
  signal dominates). Positive coupling indicates corrupted or flattened data.

G4 — Total Pipeline Variance Removed by OLS (OLS var_removed_frac)
  PASS    : max(varHbO, varHbR) ≤ 70%
  WARNING : 70% < max(varHbO, varHbR) ≤ 85%
  HOLD    : max(varHbO, varHbR) > 85%
  Rationale: The OLS regression removes scalp hemodynamics from the rPCA-
  corrected signal. Extreme OLS removal (>85%) suggests rPCA already flattened
  the signal, leaving the OLS regressor and the long-channel artificially
  correlated with noise.

Overall Flag Logic
------------------
  HOLD    : any single guardrail triggers HOLD
  WARNING : any guardrail triggers WARNING (no HOLD)
  PASS    : all guardrails pass

Outputs
-------
outputs/stage_b_processed/rpca_qc/
  rpca_qc_report.csv           — per-subject guardrail values and flags
  rpca_qc_report.json          — structured JSON with full metadata
  rpca_qc_summary_plot.png     — group visualisation of guardrail metrics
  rpca_neural_band_heatmap.png — per-subject neural-band preservation heatmap
  rpca_hrf_delta_report.csv    — group HRF amplitude with/without HOLD subjects
  rpca_hrf_delta_plot.png      — before/after group HRF comparison plot

Reference: AUDIT_PROGRESS.md Issue 8
Upstream:  04_rPCA_motion_correct.py (B4), 06_ols_regression.py (B6)
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_snirf_files, load_config

# ---------------------------------------------------------------------------
# Dynamic imports — B2 bandpass filter (needed to reconstruct pre-rPCA OD)
# ---------------------------------------------------------------------------
_spec_b3 = importlib.util.spec_from_file_location(
    "motion_correct",
    Path(__file__).parent / "03_motion_correct.py",
)
_b3 = importlib.util.module_from_spec(_spec_b3)
_spec_b3.loader.exec_module(_b3)
run_bandpass_filter = _b3.run_bandpass_filter   # re-exported from B2 inside B3

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ===========================================================================
# Guardrail Thresholds (8.1)
# ===========================================================================

GUARDRAIL_THRESHOLDS = {
    "G1_mask_coverage": {
        "pass_max":    35.0,   # pct_flagged_union ≤ 35%  → PASS
        "warning_max": 55.0,   # ≤ 55%                     → WARNING
        "hold":        55.0,   # > 55%                     → HOLD
    },
    "G2_neural_band_ratio": {
        "pass_min":    0.65,   # ratio ≥ 0.65 → PASS
        "warning_min": 0.45,   # ratio ≥ 0.45 → WARNING
        "hold":        0.45,   # ratio < 0.45 → HOLD
    },
    "G3_hbo_hbr_coupling": {
        "pass_max":    0.00,   # r ≤ 0.00  → PASS
        "warning_max": 0.25,   # r ≤ 0.25  → WARNING
        "hold":        0.25,   # r > 0.25  → HOLD
    },
    "G4_ols_var_removed": {
        "pass_max":    70.0,   # max(varHbO, varHbR) ≤ 70% → PASS
        "warning_max": 85.0,   # ≤ 85%                      → WARNING
        "hold":        85.0,   # > 85%                      → HOLD
    },
}

NEURAL_BAND_HZ = (0.01, 0.10)   # neural hemodynamic band


# ===========================================================================
# 8.2 — Neural band power computation
# ===========================================================================

def compute_neural_band_power(data: np.ndarray, sfreq: float,
                              band: tuple = NEURAL_BAND_HZ) -> float:
    """
    Compute mean neural-band power across channels using Welch PSD.

    Parameters
    ----------
    data  : (n_channels, n_times) array
    sfreq : sampling frequency
    band  : (low_hz, high_hz) tuple

    Returns
    -------
    mean_band_power : float  (mean PSD power in band, averaged across channels)
    """
    n_times = data.shape[1]
    # Welch segment length: 60 s window or half the signal, whichever is smaller
    nperseg = min(int(60 * sfreq), n_times // 2)
    nperseg = max(nperseg, 256)   # floor at 256 samples

    band_powers = []
    for ch_data in data:
        if not np.all(np.isfinite(ch_data)):
            continue
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if mask.sum() == 0:
            continue
        # np.trapezoid is the canonical name in NumPy ≥2.0; trapz is legacy alias
        try:
            bp = np.trapezoid(psd[mask], freqs[mask])
        except AttributeError:
            bp = np.trapz(psd[mask], freqs[mask])  # fallback for NumPy <2.0
        band_powers.append(bp)

    if not band_powers:
        return np.nan
    return float(np.mean(band_powers))


def load_pre_rpca_od(dataset_name: str, subject_id: str) -> mne.io.Raw:
    """
    Load or recompute the bandpass-filtered OD (B2 output) — the signal
    that enters rPCA.  This function recomputes from raw SNIRF rather than
    loading a cached file (B2 output is not persisted to disk).

    Returns
    -------
    raw_od_filt : mne.io.Raw  (bandpass-filtered OD, pre-rPCA)
    """
    b2_results = run_bandpass_filter(dataset_name, subject_id, verbose=False)
    return b2_results['raw_od_filt']


def load_post_rpca_od(subject_id: str) -> mne.io.Raw:
    """
    Load the saved rPCA-corrected OD .fif file.

    Returns
    -------
    raw_od_corrected : mne.io.Raw
    """
    rpca_dir = get_output_path("stage_b", f"rPCA_correction/{subject_id}")
    fif_path = rpca_dir / f"{subject_id}_od_corrected_rpca_raw.fif"
    if not fif_path.exists():
        raise FileNotFoundError(f"Post-rPCA OD not found: {fif_path}")
    return mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)


def compute_neural_band_ratio(dataset_name: str, subject_id: str,
                              verbose: bool = True) -> dict:
    """
    Compute neural-band OD power before and after rPCA for one subject.

    Returns
    -------
    dict with keys:
        neural_band_power_before  : float
        neural_band_power_after   : float
        neural_band_ratio         : float   (after / before)
        total_var_before          : float
        total_var_after           : float
        total_var_ratio           : float   (after / before)
        n_good_channels           : int
    """
    if verbose:
        print(f"  [{subject_id}] Loading pre-rPCA OD (B1+B2 chain)...", flush=True)

    raw_before = load_pre_rpca_od(dataset_name, subject_id)

    if verbose:
        print(f"  [{subject_id}] Loading post-rPCA OD (.fif)...", flush=True)

    raw_after = load_post_rpca_od(subject_id)

    # Align channel sets (post-rPCA may contain bad-channel channels unchanged)
    ch_names  = raw_before.ch_names
    sfreq     = raw_before.info['sfreq']
    data_b    = raw_before.get_data()    # (n_ch, n_times)
    data_a    = raw_after.get_data()

    # Use only channels present in both
    common_chs = [ch for ch in ch_names if ch in raw_after.ch_names]
    idx_b = [raw_before.ch_names.index(ch) for ch in common_chs]
    idx_a = [raw_after.ch_names.index(ch)  for ch in common_chs]
    data_b = data_b[idx_b]
    data_a = data_a[idx_a]

    # Drop channels with non-finite values
    finite_mask = np.all(np.isfinite(data_b), axis=1) & np.all(np.isfinite(data_a), axis=1)
    data_b = data_b[finite_mask]
    data_a = data_a[finite_mask]
    n_good = int(finite_mask.sum())

    # Neural band power
    power_before = compute_neural_band_power(data_b, sfreq)
    power_after  = compute_neural_band_power(data_a, sfreq)
    ratio_nb     = (power_after / power_before) if (power_before > 0 and np.isfinite(power_before)) else np.nan

    # Total signal variance
    var_before = float(np.var(data_b, axis=1).mean())
    var_after  = float(np.var(data_a, axis=1).mean())
    ratio_var  = (var_after / var_before) if var_before > 0 else np.nan

    result = {
        'neural_band_power_before': round(float(power_before), 8) if np.isfinite(power_before) else None,
        'neural_band_power_after':  round(float(power_after), 8)  if np.isfinite(power_after)  else None,
        'neural_band_ratio':        round(float(ratio_nb), 4)     if np.isfinite(ratio_nb)      else None,
        'total_var_before':         round(float(var_before), 8),
        'total_var_after':          round(float(var_after), 8),
        'total_var_ratio':          round(float(ratio_var), 4)    if np.isfinite(ratio_var)     else None,
        'n_good_channels':          n_good,
    }

    if verbose:
        print(f"  [{subject_id}] Neural-band ratio = {ratio_nb:.3f}  "
              f"| Total-var ratio = {ratio_var:.3f}  "
              f"| n_good = {n_good}")

    return result


# ===========================================================================
# 8.2 / 8.3 — Guardrail evaluation and QC flagging
# ===========================================================================

def _rate_g1(pct_flagged: float) -> str:
    t = GUARDRAIL_THRESHOLDS["G1_mask_coverage"]
    if pct_flagged > t["hold"]:
        return "HOLD"
    if pct_flagged > t["pass_max"]:
        return "WARNING"
    return "PASS"


def _rate_g2(ratio: float) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "WARNING"   # data missing → conservative
    t = GUARDRAIL_THRESHOLDS["G2_neural_band_ratio"]
    if ratio < t["hold"]:
        return "HOLD"
    if ratio < t["pass_min"]:
        return "WARNING"
    return "PASS"


def _rate_g3(r_after: float) -> str:
    if r_after is None or not np.isfinite(r_after):
        return "WARNING"
    t = GUARDRAIL_THRESHOLDS["G3_hbo_hbr_coupling"]
    if r_after > t["hold"]:
        return "HOLD"
    if r_after > t["pass_max"]:
        return "WARNING"
    return "PASS"


def _rate_g4(var_hbo: float, var_hbr: float) -> str:
    if var_hbo is None or var_hbr is None:
        return "WARNING"
    max_var = max(var_hbo, var_hbr)
    t = GUARDRAIL_THRESHOLDS["G4_ols_var_removed"]
    if max_var > t["hold"]:
        return "HOLD"
    if max_var > t["pass_max"]:
        return "WARNING"
    return "PASS"


_RANK = {"HOLD": 2, "WARNING": 1, "PASS": 0}


def _overall_flag(g1: str, g2: str, g3: str, g4: str) -> str:
    return max([g1, g2, g3, g4], key=lambda x: _RANK[x])


def evaluate_guardrails(
    subject_id:     str,
    rpca_summary:   dict,
    master_row:     dict,
    neural_metrics: dict,
) -> dict:
    """
    Apply all four guardrails to one subject and return a QC record.

    Parameters
    ----------
    rpca_summary  : loaded from {subj}_rpca_summary.json
    master_row    : one row from expert-novice_master_summary.csv as dict
    neural_metrics: output of compute_neural_band_ratio()

    Returns
    -------
    dict — all guardrail values, ratings, and overall flag
    """
    pct_flagged = rpca_summary.get('pct_flagged_union', np.nan)
    nb_ratio    = neural_metrics.get('neural_band_ratio')
    var_ratio   = neural_metrics.get('total_var_ratio')
    r_after     = master_row.get('mean_r_HbO_HbR_after')
    var_hbo     = master_row.get('var_removed_HbO_pct')
    var_hbr     = master_row.get('var_removed_HbR_pct')

    g1 = _rate_g1(pct_flagged)
    g2 = _rate_g2(nb_ratio)
    g3 = _rate_g3(r_after)
    g4 = _rate_g4(var_hbo, var_hbr)
    overall = _overall_flag(g1, g2, g3, g4)

    return {
        'subject_id':           subject_id,
        # G1
        'G1_pct_flagged':       round(pct_flagged, 2) if np.isfinite(pct_flagged) else None,
        'G1_rating':            g1,
        # G2
        'G2_neural_band_ratio': nb_ratio,
        'G2_total_var_ratio':   var_ratio,
        'G2_rating':            g2,
        # G3
        'G3_r_HbO_HbR_after':  r_after,
        'G3_rating':            g3,
        # G4
        'G4_var_removed_HbO':   var_hbo,
        'G4_var_removed_HbR':   var_hbr,
        'G4_rating':            g4,
        # Summary
        'overall_flag':         overall,
        # Additional context
        'n_components_used':    rpca_summary.get('n_components_used'),
        'n_good_channels':      neural_metrics.get('n_good_channels'),
        'r_HbO_HbR_before':     master_row.get('mean_r_HbO_HbR_before'),
        'n_channels_excluded':  master_row.get('n_channels_excluded'),
    }


# ===========================================================================
# Reporting helpers
# ===========================================================================

def save_qc_report(records: list, output_dir: Path) -> tuple:
    """Save CSV and JSON QC reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    csv_path = output_dir / "rpca_qc_report.csv"
    df.to_csv(csv_path, index=False)

    report = {
        "guardrail_thresholds": GUARDRAIL_THRESHOLDS,
        "neural_band_hz":       list(NEURAL_BAND_HZ),
        "flag_legend":          {
            "PASS":    "All guardrails met — subject cleared for group analysis.",
            "WARNING": "One or more guardrails breached at WARNING level — "
                       "include in analysis but annotate in report.",
            "HOLD":    "At least one guardrail breached at HOLD level — "
                       "exclude from primary group analysis; include in "
                       "sensitivity analysis.",
        },
        "subjects": records,
    }
    json_path = output_dir / "rpca_qc_report.json"
    with open(json_path, 'w') as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  QC report CSV  : {csv_path}")
    print(f"  QC report JSON : {json_path}")
    return csv_path, json_path


def plot_qc_summary(records: list, output_dir: Path) -> Path:
    """
    Four-panel guardrail summary visualisation.

    Each panel shows all subjects' metric for one guardrail,
    coloured by their rating (green=PASS, orange=WARNING, red=HOLD).
    """
    df = pd.DataFrame(records).set_index('subject_id')
    subjects = df.index.tolist()
    n_subj   = len(subjects)
    x        = np.arange(n_subj)

    COLOR_MAP = {'PASS': '#2ca02c', 'WARNING': '#ff7f0e', 'HOLD': '#d62728'}

    def _bar_colors(ratings):
        return [COLOR_MAP.get(r, 'gray') for r in ratings]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Issue 8 — rPCA QC Guardrail Audit\n"
        "Green = PASS | Orange = WARNING | Red = HOLD",
        fontsize=13, fontweight='bold'
    )

    # G1: Mask coverage
    ax = axes[0, 0]
    vals = df['G1_pct_flagged'].fillna(0).tolist()
    colors = _bar_colors(df['G1_rating'].tolist())
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(GUARDRAIL_THRESHOLDS["G1_mask_coverage"]["pass_max"],
               color='orange', lw=1.2, ls='--', label='WARNING threshold (35%)')
    ax.axhline(GUARDRAIL_THRESHOLDS["G1_mask_coverage"]["hold"],
               color='red',    lw=1.2, ls='--', label='HOLD threshold (55%)')
    ax.set_title('G1: rPCA Mask Coverage (%)', fontweight='bold')
    ax.set_ylabel('% samples flagged')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # G2: Neural band ratio
    ax = axes[0, 1]
    vals = df['G2_neural_band_ratio'].fillna(0).tolist()
    colors = _bar_colors(df['G2_rating'].tolist())
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(GUARDRAIL_THRESHOLDS["G2_neural_band_ratio"]["pass_min"],
               color='orange', lw=1.2, ls='--', label='WARNING threshold (0.65)')
    ax.axhline(GUARDRAIL_THRESHOLDS["G2_neural_band_ratio"]["warning_min"],
               color='red',    lw=1.2, ls='--', label='HOLD threshold (0.45)')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':', alpha=0.5, label='No change (1.0)')
    ax.set_title('G2: Neural Band Power Preservation\n(0.01–0.1 Hz, after/before rPCA)',
                 fontweight='bold')
    ax.set_ylabel('Power ratio (after / before)')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # G3: HbO/HbR coupling
    ax = axes[1, 0]
    vals = df['G3_r_HbO_HbR_after'].fillna(0).tolist()
    colors = _bar_colors(df['G3_rating'].tolist())
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.axhline(GUARDRAIL_THRESHOLDS["G3_hbo_hbr_coupling"]["pass_max"],
               color='orange', lw=1.2, ls='--', label='WARNING threshold (0.0)')
    ax.axhline(GUARDRAIL_THRESHOLDS["G3_hbo_hbr_coupling"]["hold"],
               color='red',    lw=1.2, ls='--', label='HOLD threshold (0.25)')
    ax.axhline(0, color='k', lw=0.6, ls='-', alpha=0.3)
    ax.set_title('G3: HbO/HbR Coupling After Pipeline\n(post-OLS; negative = physiological)',
                 fontweight='bold')
    ax.set_ylabel('Mean r(HbO, HbR) after OLS')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # G4: OLS variance removed
    ax = axes[1, 1]
    hbo_vals = df['G4_var_removed_HbO'].fillna(0).tolist()
    hbr_vals = df['G4_var_removed_HbR'].fillna(0).tolist()
    colors = _bar_colors(df['G4_rating'].tolist())
    w = 0.35
    ax.bar(x - w/2, hbo_vals, width=w, color=colors, alpha=0.85, label='HbO')
    ax.bar(x + w/2, hbr_vals, width=w, color=colors, alpha=0.45, hatch='//', label='HbR')
    ax.axhline(GUARDRAIL_THRESHOLDS["G4_ols_var_removed"]["pass_max"],
               color='orange', lw=1.2, ls='--', label='WARNING threshold (70%)')
    ax.axhline(GUARDRAIL_THRESHOLDS["G4_ols_var_removed"]["hold"],
               color='red',    lw=1.2, ls='--', label='HOLD threshold (85%)')
    ax.set_title('G4: OLS Variance Removed (%)\n(reflects combined rPCA+OLS impact)',
                 fontweight='bold')
    ax.set_ylabel('Variance removed (%)')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8, ncol=2); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "rpca_qc_summary_plot.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  QC summary plot : {fig_path}")
    return fig_path


def plot_neural_band_heatmap(records: list, output_dir: Path) -> Path:
    """
    Heatmap: per-subject neural-band power (before/after) as a 2-column matrix,
    plus the ratio column with red-green colouring.
    """
    df = pd.DataFrame(records).set_index('subject_id')
    subjects = df.index.tolist()

    nb_before = df['G2_neural_band_ratio']   # ratio only — we don't have raw power stored in record
    # Build a text table instead
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    col_labels = ['Subject', 'G1 Mask%', 'G2 NB Ratio', 'G2 VarRatio',
                  'G3 r_after', 'G4 varHbO%', 'G4 varHbR%', 'Overall']
    rows = []
    cell_colors = []
    _cmap = {'PASS': '#c8f5c8', 'WARNING': '#ffe4a0', 'HOLD': '#ffb3b3', None: '#ffffff'}

    for subj, rec in zip(subjects, records):
        r = rec
        row = [
            subj,
            f"{r['G1_pct_flagged']:.1f}%" if r['G1_pct_flagged'] is not None else '—',
            f"{r['G2_neural_band_ratio']:.3f}" if r['G2_neural_band_ratio'] is not None else '—',
            f"{r['G2_total_var_ratio']:.3f}"   if r['G2_total_var_ratio']   is not None else '—',
            f"{r['G3_r_HbO_HbR_after']:.3f}"  if r['G3_r_HbO_HbR_after']  is not None else '—',
            f"{r['G4_var_removed_HbO']:.1f}%"  if r['G4_var_removed_HbO']  is not None else '—',
            f"{r['G4_var_removed_HbR']:.1f}%"  if r['G4_var_removed_HbR']  is not None else '—',
            r['overall_flag'],
        ]
        cell_row_colors = [
            '#ffffff',
            _cmap.get(r['G1_rating'], '#ffffff'),
            _cmap.get(r['G2_rating'], '#ffffff'),
            _cmap.get(r['G2_rating'], '#ffffff'),
            _cmap.get(r['G3_rating'], '#ffffff'),
            _cmap.get(r['G4_rating'], '#ffffff'),
            _cmap.get(r['G4_rating'], '#ffffff'),
            _cmap.get(r['overall_flag'], '#ffffff'),
        ]
        rows.append(row)
        cell_colors.append(cell_row_colors)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    ax.set_title(
        "Issue 8 — rPCA QC Report: Per-Subject Guardrail Values\n"
        "Green=PASS | Orange=WARNING | Red=HOLD",
        fontsize=12, fontweight='bold', pad=20,
    )
    plt.tight_layout()
    fig_path = output_dir / "rpca_neural_band_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Neural-band heatmap : {fig_path}")
    return fig_path


# ===========================================================================
# 8.4 — Group HRF delta report (with / without HOLD subjects)
# ===========================================================================

def load_subject_evoked(subject_id: str, dataset_name: str = "expert-novice") -> mne.Evoked | None:
    """Load the per-subject MNE Evoked from Stage C output."""
    hrf_dir = (
        PROJECT_ROOT
        / "outputs" / "stage_c_validation" / "hrf" / dataset_name / subject_id
    )
    ave_path = hrf_dir / f"{subject_id}-ave.fif"
    if not ave_path.exists():
        print(f"  ⚠  Evoked not found for {subject_id}: {ave_path}")
        return None
    try:
        evokeds = mne.read_evokeds(str(ave_path), verbose=False)
        return evokeds[0] if evokeds else None
    except Exception as exc:
        print(f"  ⚠  Failed to load evoked for {subject_id}: {exc}")
        return None


def compute_group_hrf_delta(
    records:       list,
    dataset_name:  str  = "expert-novice",
    hrf_window:    tuple = (5, 40),   # (tmin, tmax) in seconds for peak search
    output_dir:    Path  = None,
) -> tuple:
    """
    Load per-subject Stage C evokeds; compute group HRF with and without
    HOLD subjects; report peak amplitude differences.

    Parameters
    ----------
    records      : list of QC dicts (from evaluate_guardrails)
    dataset_name : str
    hrf_window   : (tmin, tmax) seconds to search for peak HbO amplitude
    output_dir   : Path for saving outputs

    Returns
    -------
    delta_df : pd.DataFrame — per-channel HRF amplitude with/without HOLDs
    fig_path : Path
    """
    hold_subjects    = [r['subject_id'] for r in records if r['overall_flag'] == 'HOLD']
    warning_subjects = [r['subject_id'] for r in records if r['overall_flag'] == 'WARNING']
    all_subjects     = [r['subject_id'] for r in records]

    print(f"\n  HOLD subjects   : {hold_subjects}")
    print(f"  WARNING subjects: {warning_subjects}")

    # Load evokeds
    evokeds_all  = {}
    evokeds_clean = {}
    for subj in all_subjects:
        ev = load_subject_evoked(subj, dataset_name)
        if ev is not None:
            evokeds_all[subj] = ev
            if subj not in hold_subjects:
                evokeds_clean[subj] = ev

    if not evokeds_all:
        print("  ⚠  No evokeds found — skipping HRF delta analysis.")
        return pd.DataFrame(), None

    def _grand_avg(evoked_dict: dict) -> mne.Evoked | None:
        """Grand average that handles subjects with different time arrays.
        Uses the first evoked as the reference time grid, interpolating others."""
        evs = list(evoked_dict.values())
        if not evs:
            return None
        # Use the reference evoked (shortest recording to ensure common window)
        ref = min(evs, key=lambda e: len(e.times)).copy()
        ref_times = ref.times
        # Accumulate resampled data across subjects
        n_ch    = len(ref.ch_names)
        accum   = np.zeros((n_ch, len(ref_times)))
        n_valid = 0
        for ev in evs:
            # For each channel, interpolate to ref time grid
            data_interp = np.zeros_like(accum)
            for ci in range(len(ev.ch_names)):
                if ev.ch_names[ci] in ref.ch_names:
                    ref_ci = ref.ch_names.index(ev.ch_names[ci])
                    # Interpolate subject trace onto reference time grid
                    data_interp[ref_ci] = np.interp(
                        ref_times, ev.times, ev.data[ci],
                        left=np.nan, right=np.nan
                    )
            # Add only if all channels have valid values
            if np.all(np.isfinite(data_interp)):
                accum   += data_interp
                n_valid += 1
        if n_valid == 0:
            return None
        ref._data = accum / n_valid
        ref.nave  = n_valid
        return ref

    grand_all   = _grand_avg(evokeds_all)
    grand_clean = _grand_avg(evokeds_clean)

    if grand_all is None or grand_clean is None:
        print("  ⚠  Could not compute grand average.")
        return pd.DataFrame(), None

    # Extract HbO channels and peak amplitude in hrf_window
    # Both grand averages now use the same reference time grid (from _grand_avg)
    # Align grand_clean to grand_all's time axis if needed
    if len(grand_all.times) != len(grand_clean.times):
        # Interpolate grand_clean to grand_all's time grid
        times_all = grand_all.times
        data_realigned = np.zeros((len(grand_clean.ch_names), len(times_all)))
        for ci in range(len(grand_clean.ch_names)):
            data_realigned[ci] = np.interp(
                times_all, grand_clean.times, grand_clean.data[ci]
            )
        grand_clean._data = data_realigned

    times = grand_all.times
    t_mask = (times >= hrf_window[0]) & (times <= hrf_window[1])

    def _peak_amp(evoked: mne.Evoked, ch_type: str) -> dict:
        """Return {channel: peak_amplitude_uM} for a given ch_type."""
        picks = mne.pick_types(evoked.info, fnirs=ch_type)
        result = {}
        for pick in picks:
            ch_name = evoked.ch_names[pick]
            trace   = evoked.data[pick, t_mask] * 1e6   # → μM
            result[ch_name] = float(np.max(np.abs(trace)))
        return result

    peaks_all_hbo   = _peak_amp(grand_all,   'hbo')
    peaks_clean_hbo = _peak_amp(grand_clean, 'hbo')
    peaks_all_hbr   = _peak_amp(grand_all,   'hbr')
    peaks_clean_hbr = _peak_amp(grand_clean, 'hbr')

    records_delta = []
    all_channels  = sorted(set(peaks_all_hbo) | set(peaks_clean_hbo))
    for ch in all_channels:
        p_all   = peaks_all_hbo.get(ch)
        p_clean = peaks_clean_hbo.get(ch)
        delta   = (p_clean - p_all) if (p_all is not None and p_clean is not None) else None
        records_delta.append({
            'channel':                ch,
            'ch_type':                'hbo',
            'peak_uM_all_subjects':   round(p_all,   4) if p_all   is not None else None,
            'peak_uM_without_HOLD':   round(p_clean, 4) if p_clean is not None else None,
            'delta_uM':               round(delta,   4) if delta   is not None else None,
        })

    all_channels_hbr = sorted(set(peaks_all_hbr) | set(peaks_clean_hbr))
    for ch in all_channels_hbr:
        p_all   = peaks_all_hbr.get(ch)
        p_clean = peaks_clean_hbr.get(ch)
        delta   = (p_clean - p_all) if (p_all is not None and p_clean is not None) else None
        records_delta.append({
            'channel':                ch,
            'ch_type':                'hbr',
            'peak_uM_all_subjects':   round(p_all,   4) if p_all   is not None else None,
            'peak_uM_without_HOLD':   round(p_clean, 4) if p_clean is not None else None,
            'delta_uM':               round(delta,   4) if delta   is not None else None,
        })

    delta_df = pd.DataFrame(records_delta)

    # Save CSV
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "rpca_hrf_delta_report.csv"
        delta_df.to_csv(csv_path, index=False)
        print(f"  HRF delta CSV : {csv_path}")

    # Group-level summary
    hbo_df = delta_df[delta_df['ch_type'] == 'hbo'].dropna(subset=['delta_uM'])
    hbr_df = delta_df[delta_df['ch_type'] == 'hbr'].dropna(subset=['delta_uM'])

    if len(hbo_df):
        print(f"\n  Group HbO peak amplitude (μM):")
        print(f"    All subjects  : median={hbo_df['peak_uM_all_subjects'].median():.4f}  "
              f"mean={hbo_df['peak_uM_all_subjects'].mean():.4f}")
        print(f"    Without HOLDs : median={hbo_df['peak_uM_without_HOLD'].median():.4f}  "
              f"mean={hbo_df['peak_uM_without_HOLD'].mean():.4f}")
        print(f"    Δ (clean−all) : median={hbo_df['delta_uM'].median():.4f}  "
              f"mean={hbo_df['delta_uM'].mean():.4f}")

    # Plot: Before/after group HRF traces
    fig_path = None
    if output_dir and grand_all is not None and grand_clean is not None:
        fig_path = _plot_hrf_delta(
            grand_all, grand_clean,
            hold_subjects, warning_subjects,
            n_all=len(evokeds_all), n_clean=len(evokeds_clean),
            output_dir=output_dir,
        )

    return delta_df, fig_path


def _plot_hrf_delta(
    grand_all,
    grand_clean,
    hold_subjects:    list,
    warning_subjects: list,
    n_all:    int,
    n_clean:  int,
    output_dir: Path,
) -> Path:
    """
    Plot group-average HRF traces (HbO + HbR) for all-subjects vs clean-only,
    averaged across long-separation channels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    def _mean_trace(evoked: mne.Evoked, ch_type: str) -> np.ndarray:
        picks = mne.pick_types(evoked.info, fnirs=ch_type)
        return evoked.data[picks].mean(axis=0) * 1e6   # → μM

    times     = grand_all.times

    for ax, ch_type, color_all, color_clean, label in [
        (axes[0], 'hbo', '#1f77b4', '#2ca02c', 'HbO'),
        (axes[1], 'hbr', '#d62728', '#ff7f0e', 'HbR'),
    ]:
        trace_all   = _mean_trace(grand_all,   ch_type)
        trace_clean = _mean_trace(grand_clean, ch_type)

        ax.plot(times, trace_all,   color=color_all,   lw=1.8, alpha=0.9,
                label=f'All subjects (n={n_all})')
        ax.plot(times, trace_clean, color=color_clean, lw=1.8, alpha=0.9,
                ls='--', label=f'Without HOLD (n={n_clean})')
        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.4, label='Task onset')
        ax.axhline(0, color='k', lw=0.5, ls='-', alpha=0.2)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Δ[Hb] (μM)', fontsize=10)
        ax.set_title(f'{label} — Group HRF\nAll vs Without HOLD subjects', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Issue 8.4 — rPCA Over-Correction: Group HRF Delta\n"
        f"HOLD: {hold_subjects}  |  WARNING: {warning_subjects}",
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    fig_path = output_dir / "rpca_hrf_delta_plot.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  HRF delta plot : {fig_path}")
    return fig_path


# ===========================================================================
# Main orchestrator — runs all Issue 8 sub-tasks
# ===========================================================================

def run_rpca_qc_audit(
    dataset_name:  str  = "expert_novice",
    hrf_folder:    str  = "expert-novice",
    subjects_override: list = None,
    verbose:       bool = True,
) -> dict:
    """
    Issue 8 full pipeline:
      8.1 — Guardrails defined (see module-level constants)
      8.2 — Per-subject QC flags computed
      8.3 — HOLD/WARNING/PASS logic applied
      8.4 — Group HRF delta report produced

    Parameters
    ----------
    dataset_name       : str   internal key (e.g. 'expert_novice')
    hrf_folder         : str   Stage C HRF folder name (e.g. 'expert-novice')
    subjects_override  : list  process only these subjects
    verbose            : bool

    Returns
    -------
    dict with keys:
        qc_records : list of per-subject QC dicts
        df         : pd.DataFrame (summary table)
        delta_df   : pd.DataFrame (HRF delta report)
        output_dir : Path
    """
    config   = load_config()
    ds_cfg   = config["datasets"][dataset_name]
    subjects = subjects_override or ds_cfg["subjects"]

    output_dir = PROJECT_ROOT / "outputs" / "stage_b_processed" / "rpca_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load master summary ------------------------------------------------
    study_folder = hrf_folder
    master_csv = (
        PROJECT_ROOT
        / config["paths"].get("processed_root", "outputs/stage_b_processed/preprocessed_data")
        / study_folder
        / f"{study_folder}_master_summary.csv"
    )
    master_df = pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()

    print("\n" + "=" * 70)
    print("STAGE B — ISSUE 8: rPCA OVER-CORRECTION QC AUDIT")
    print("=" * 70)
    print(f"  Subjects    : {len(subjects)}")
    print(f"  Dataset     : {dataset_name}")
    print(f"  Output dir  : {output_dir}")
    print("\n  Guardrail thresholds:")
    for gk, gv in GUARDRAIL_THRESHOLDS.items():
        print(f"    {gk}: {gv}")
    print("=" * 70 + "\n")

    # ---- Per-subject processing ---------------------------------------------
    qc_records = []
    for subj in subjects:
        print(f"\n[Subject: {subj}]")

        # Load rPCA summary
        rpca_json = (
            get_output_path("stage_b", f"rPCA_correction/{subj}")
            / f"{subj}_rpca_summary.json"
        )
        if not rpca_json.exists():
            print(f"  ⚠  rPCA summary missing for {subj} — skipping.")
            continue
        with open(rpca_json) as fh:
            rpca_summary = json.load(fh)

        # Master summary row
        if not master_df.empty and 'subject_id' in master_df.columns:
            row_df = master_df[master_df['subject_id'] == subj]
            master_row = row_df.iloc[0].to_dict() if len(row_df) else {}
        else:
            master_row = {}

        # Neural band power (8.2 computation)
        try:
            neural_metrics = compute_neural_band_ratio(
                dataset_name, subj, verbose=verbose
            )
        except Exception as exc:
            print(f"  ⚠  Neural band computation failed for {subj}: {exc}")
            neural_metrics = {
                'neural_band_ratio': None,
                'total_var_ratio':   None,
                'n_good_channels':   None,
            }

        # Apply guardrails (8.2 / 8.3)
        record = evaluate_guardrails(subj, rpca_summary, master_row, neural_metrics)
        qc_records.append(record)

        # Print subject summary
        flag_emoji = {'PASS': '✅', 'WARNING': '⚠️', 'HOLD': '🔴'}.get(record['overall_flag'], '?')
        print(f"  {flag_emoji} {subj}: {record['overall_flag']}")
        print(f"     G1 mask%={record['G1_pct_flagged']}% [{record['G1_rating']}]  "
              f"G2 NB_ratio={record['G2_neural_band_ratio']} [{record['G2_rating']}]  "
              f"G3 r={record['G3_r_HbO_HbR_after']} [{record['G3_rating']}]  "
              f"G4 varHbO={record['G4_var_removed_HbO']}% [{record['G4_rating']}]")

    # ---- Summary statistics -------------------------------------------------
    n_pass    = sum(1 for r in qc_records if r['overall_flag'] == 'PASS')
    n_warning = sum(1 for r in qc_records if r['overall_flag'] == 'WARNING')
    n_hold    = sum(1 for r in qc_records if r['overall_flag'] == 'HOLD')

    print("\n" + "=" * 70)
    print("QC AUDIT SUMMARY")
    print("=" * 70)
    print(f"  PASS    : {n_pass} / {len(qc_records)}")
    print(f"  WARNING : {n_warning} / {len(qc_records)}")
    print(f"  HOLD    : {n_hold} / {len(qc_records)}")
    hold_ids = [r['subject_id'] for r in qc_records if r['overall_flag'] == 'HOLD']
    warn_ids = [r['subject_id'] for r in qc_records if r['overall_flag'] == 'WARNING']
    if hold_ids:
        print(f"  HOLD subjects    : {hold_ids}")
    if warn_ids:
        print(f"  WARNING subjects : {warn_ids}")

    # ---- 8.1 Guardrail explanation printed ---------------------------------
    print("\n  Guardrail definitions:")
    print("    G1 — Mask coverage > 55%  → rPCA touching majority of signal")
    print("    G2 — Neural band (0.01–0.1 Hz) power ratio < 0.45  → >55% neural power lost")
    print("    G3 — HbO/HbR positive coupling (r > 0.25)  → signal destroyed / residual noise")
    print("    G4 — OLS var removed > 85%  → combined pipeline stripped too much signal")

    # ---- Save reports -------------------------------------------------------
    csv_path, json_path = save_qc_report(qc_records, output_dir)
    fig_summary = plot_qc_summary(qc_records, output_dir)
    fig_table   = plot_neural_band_heatmap(qc_records, output_dir)

    # ---- 8.4 Group HRF delta ------------------------------------------------
    print("\n[8.4] Computing group HRF delta: all subjects vs without HOLD subjects...")
    delta_df, fig_delta = compute_group_hrf_delta(
        qc_records,
        dataset_name = hrf_folder,
        output_dir   = output_dir,
    )

    # ---- Save QC guardrail summary to JSON for audit log -------------------
    audit_meta = {
        "issue":              "Issue 8 — rPCA Over-Correction QC",
        "date_run":           pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_subjects":         len(qc_records),
        "n_pass":             n_pass,
        "n_warning":          n_warning,
        "n_hold":             n_hold,
        "hold_subjects":      hold_ids,
        "warning_subjects":   warn_ids,
        "guardrail_thresholds": GUARDRAIL_THRESHOLDS,
        "hrf_delta_median_hbo": (
            float(delta_df[delta_df['ch_type'] == 'hbo']['delta_uM'].median())
            if not delta_df.empty and 'ch_type' in delta_df.columns
            else None
        ),
    }
    meta_path = output_dir / "rpca_qc_audit_meta.json"
    with open(meta_path, 'w') as fh:
        json.dump(audit_meta, fh, indent=2)
    print(f"\n  Audit metadata : {meta_path}")

    # ---- Final report -------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ ISSUE 8 — rPCA QC AUDIT COMPLETE")
    print("=" * 70)
    print(f"  Outputs saved to: {output_dir}")
    print(f"    QC report CSV       : rpca_qc_report.csv")
    print(f"    QC report JSON      : rpca_qc_report.json")
    print(f"    Guardrail plot      : rpca_qc_summary_plot.png")
    print(f"    Per-subject table   : rpca_neural_band_heatmap.png")
    print(f"    HRF delta CSV       : rpca_hrf_delta_report.csv")
    print(f"    HRF delta plot      : rpca_hrf_delta_plot.png")
    print(f"    Audit metadata      : rpca_qc_audit_meta.json")
    print("=" * 70 + "\n")

    return {
        "qc_records":  qc_records,
        "df":          pd.DataFrame(qc_records),
        "delta_df":    delta_df,
        "output_dir":  output_dir,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    run_rpca_qc_audit(
        dataset_name = "expert_novice",
        hrf_folder   = "expert-novice",
        verbose      = True,
    )
