"""
Stage A - Step A4: Generate Stage A QA Report
Purpose: Create comprehensive QA report combining SNR and motion ratio metrics

Tasks:
  2.4.1 - Create comprehensive metrics DataFrame (SNR + motion ratio + QA flags)
  2.4.2 - SNR box plot visualization
  2.4.3 - Motion ratio box plot visualization
  2.4.4 - Summary statistics calculator and console report generator
  2.4.5 - Generate complete Stage A report for subject A1

Reference: PROJECT.md Section 4.5, Paper Figure 4

IMPORTANT: No channels are excluded from this analysis. All channels are
included to give an accurate assessment of data quality, as stated in the
paper: "To give an accurate assessment of the quality of the data, no
channels have been excluded from the analysis provided here and no denoising
methods have been used prior to analysis."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import json
import argparse
import importlib.util
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, load_config

# Dynamically import Stage A dependencies used for dataset batch processing.
_spec_load = importlib.util.spec_from_file_location(
    "load_data", Path(__file__).parent / "01_load_data.py"
)
load_data = importlib.util.module_from_spec(_spec_load)
_spec_load.loader.exec_module(load_data)
load_raw_snirf = load_data.load_raw_snirf

_spec_snr = importlib.util.spec_from_file_location(
    "calculate_snr", Path(__file__).parent / "05_calculate_snr.py"
)
calculate_snr = importlib.util.module_from_spec(_spec_snr)
_spec_snr.loader.exec_module(calculate_snr)

_spec_motion = importlib.util.spec_from_file_location(
    "calculate_motion", Path(__file__).parent / "06_calculate_motion_ratio.py"
)
calculate_motion = importlib.util.module_from_spec(_spec_motion)
_spec_motion.loader.exec_module(calculate_motion)


# ==============================================================================
# TASK 2.4.1 - Comprehensive Metrics DataFrame
# ==============================================================================

def create_metrics_dataframe(
    snr_csv_path: Path,
    motion_csv_path: Path,
    subject_id: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create comprehensive metrics DataFrame combining SNR, motion ratio,
    and QA quality flags.

    Parameters
    ----------
    snr_csv_path : Path
        Path to SNR CSV (output of 05_calculate_snr.py)
    motion_csv_path : Path
        Path to motion ratio CSV (output of 06_calculate_motion_ratio.py)
    subject_id : str
        Subject identifier (e.g., 'A1')
    verbose : bool
        Print loading information

    Returns
    -------
    metrics_df : pd.DataFrame
        Columns:
          subject_id, channel, wavelength, channel_type,
          mean_intensity, std_intensity, snr_db,
          motion_ratio, n_flagged_samples, n_total_samples,
          snr_quality_flag,    # 'low' | 'acceptable' | 'high'
          motion_quality_flag, # 'low' | 'elevated'
          qa_note              # text note for documentation

    Notes
    -----
    Quality thresholds (for documentation only — NO exclusion occurs here):
      SNR < 10 dB   → 'low'        (reference line in Figure 4)
      SNR > 30 dB   → 'high'       (reference line in Figure 4)
      10–30 dB      → 'acceptable'
      motion > 0.10 → 'elevated'   (>10% of samples flagged)
    Exclusion for Stage B (80–140 dB amplitude range) occurs in task 2.6.
    """
    if verbose:
        print("=" * 65)
        print("TASK 2.4.1 — Comprehensive Metrics DataFrame")
        print("=" * 65)

    # ------------------------------------------------------------------
    # Load SNR results
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nLoading SNR data: {snr_csv_path}")
    snr_df = pd.read_csv(snr_csv_path)
    if verbose:
        print(f"  {len(snr_df)} channels loaded")

    # ------------------------------------------------------------------
    # Load motion ratio results
    # ------------------------------------------------------------------
    if verbose:
        print(f"Loading motion data: {motion_csv_path}")
    motion_df = pd.read_csv(motion_csv_path)
    if verbose:
        print(f"  {len(motion_df)} channels loaded")

    # ------------------------------------------------------------------
    # Rename columns before merge to avoid ambiguity
    # ------------------------------------------------------------------
    snr_renamed = snr_df.rename(columns={
        'mean':         'mean_intensity',
        'std':          'std_intensity',
        'channel_type': 'channel_type',
    })

    motion_renamed = motion_df.rename(columns={
        'n_excluded':   'n_flagged_samples',
        'n_total':      'n_total_samples',
        'channel_type': 'channel_type_motion',
    })

    # ------------------------------------------------------------------
    # Merge on channel + wavelength (inner join — both must be present)
    # ------------------------------------------------------------------
    metrics_df = pd.merge(
        snr_renamed,
        motion_renamed,
        on=['channel', 'wavelength'],
        how='inner',
    )

    # Drop duplicate channel_type column from motion side
    if 'channel_type_motion' in metrics_df.columns:
        metrics_df.drop(columns=['channel_type_motion'], inplace=True)

    # ------------------------------------------------------------------
    # Add subject_id as first column
    # ------------------------------------------------------------------
    metrics_df.insert(0, 'subject_id', subject_id)

    # ------------------------------------------------------------------
    # SNR quality flag  (documentation only, not an exclusion)
    # Thresholds match the reference lines shown in paper Figure 4
    # ------------------------------------------------------------------
    def snr_flag(snr: float) -> str:
        if pd.isna(snr):
            return 'invalid'
        if snr < 10.0:
            return 'low'
        if snr > 30.0:
            return 'high'
        return 'acceptable'

    metrics_df['snr_quality_flag'] = metrics_df['snr_db'].apply(snr_flag)

    # ------------------------------------------------------------------
    # Motion quality flag  (>10 % of samples flagged as motion)
    # ------------------------------------------------------------------
    metrics_df['motion_quality_flag'] = metrics_df['motion_ratio'].apply(
        lambda mr: 'elevated' if mr > 0.10 else 'low'
    )

    # ------------------------------------------------------------------
    # Human-readable QA note for each channel
    # ------------------------------------------------------------------
    def make_qa_note(row: pd.Series) -> str:
        notes = []
        if row['snr_quality_flag'] == 'low':
            notes.append(f"SNR below 10 dB ({row['snr_db']:.1f} dB)")
        elif row['snr_quality_flag'] == 'high':
            notes.append(f"SNR above 30 dB ({row['snr_db']:.1f} dB)")
        if row['motion_quality_flag'] == 'elevated':
            notes.append(f"motion ratio {row['motion_ratio']*100:.1f}%")
        return '; '.join(notes) if notes else 'OK'

    metrics_df['qa_note'] = metrics_df.apply(make_qa_note, axis=1)

    # ------------------------------------------------------------------
    # Reorder columns for readability
    # ------------------------------------------------------------------
    col_order = [
        'subject_id', 'channel', 'wavelength', 'channel_type',
        'mean_intensity', 'std_intensity', 'snr_db',
        'motion_ratio', 'n_flagged_samples', 'n_total_samples',
        'snr_quality_flag', 'motion_quality_flag', 'qa_note',
    ]
    # Keep only columns that actually exist (guard against schema changes)
    col_order = [c for c in col_order if c in metrics_df.columns]
    metrics_df = metrics_df[col_order]

    if verbose:
        print(f"\n  Combined metrics DataFrame: {len(metrics_df)} rows × "
              f"{len(metrics_df.columns)} columns")
        snr_counts = metrics_df['snr_quality_flag'].value_counts()
        print(f"\n  SNR quality flags:")
        for flag, cnt in snr_counts.items():
            print(f"    {flag:12s}: {cnt} channels")
        motion_counts = metrics_df['motion_quality_flag'].value_counts()
        print(f"\n  Motion quality flags:")
        for flag, cnt in motion_counts.items():
            print(f"    {flag:12s}: {cnt} channels")
        print(f"\n  NOTE: Zero channels excluded — all included for QA transparency.")

    return metrics_df


def save_metrics_dataframe(
    metrics_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
) -> Path:
    """
    Save combined metrics DataFrame to CSV.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of create_metrics_dataframe()
    subject_id : str
        Subject identifier
    output_dir : Path
        Directory where CSV is written

    Returns
    -------
    output_path : Path
        Saved file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_qa_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"  Metrics DataFrame saved: {output_path}")
    return output_path


# ==============================================================================
# TASK 2.4.2 - SNR Box Plot
# ==============================================================================

def plot_snr_boxplot(
    metrics_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    save_fig: bool = True,
    verbose: bool = True,
) -> plt.Figure:
    """
    Create SNR box plot visualization matching paper Figure 4 style.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of create_metrics_dataframe()
    subject_id : str
        Subject identifier (used in title and filename)
    output_dir : Path
        Directory where the figure is saved
    save_fig : bool
        Whether to save the figure to disk
    verbose : bool
        Print status messages

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if verbose:
        print("\n" + "=" * 65)
        print("TASK 2.4.2 — SNR Box Plot")
        print("=" * 65)

    # Filter to intensity channels only (wavelength > 0)
    plot_df = metrics_df[metrics_df['wavelength'] > 0].copy()

    # Determine bad channels (SNR < 10 dB, below the low-quality reference)
    bad_mask = plot_df['snr_quality_flag'] == 'low'

    # ---------------------------------------------------------------
    # Figure setup — two panels: one per wavelength
    # ---------------------------------------------------------------
    wavelengths = sorted(plot_df['wavelength'].unique())
    n_wl = len(wavelengths)

    fig, axes = plt.subplots(1, n_wl, figsize=(6 * n_wl, 6), sharey=False)
    if n_wl == 1:
        axes = [axes]

    for ax, wl in zip(axes, wavelengths):
        wl_df = plot_df[plot_df['wavelength'] == wl].copy()

        # Separate good / bad channels
        good_snr = wl_df.loc[~bad_mask[wl_df.index], 'snr_db'].values
        bad_snr  = wl_df.loc[ bad_mask[wl_df.index], 'snr_db'].values

        # Build box data: one box per subject label (here just subject_id)
        box_data  = [wl_df['snr_db'].values]
        positions = [1]
        labels    = [subject_id]

        # Draw box plot — white fill like paper
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color='blue', linewidth=2),
            boxprops=dict(facecolor='white', color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(
                marker='+',
                color='red',
                markersize=8,
                linestyle='none',
            ),
        )

        # Overlay bad channel points in red
        if len(bad_snr) > 0:
            jitter_x = np.random.default_rng(42).uniform(-0.15, 0.15, len(bad_snr))
            ax.scatter(
                np.ones(len(bad_snr)) + jitter_x,
                bad_snr,
                color='red',
                s=40,
                zorder=5,
                label='Low SNR (< 10 dB)',
            )

        # Reference lines matching Figure 4 dashed threshold lines
        ax.axhline(y=10,  color='gray', linestyle='--', linewidth=1.2,
                   label='10 dB (low quality)')
        ax.axhline(y=30,  color='gray', linestyle=':',  linewidth=1.2,
                   label='30 dB (high quality)')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Subject Label')
        ax.set_ylabel('SNR (dB)')
        ax.set_title(f'SNR — {subject_id} ({wl} nm)')
        ax.legend(fontsize=8, loc='upper right')

        if verbose:
            n_total = len(wl_df)
            n_bad   = bad_mask[wl_df.index].sum()
            print(f"  {wl} nm: {n_total} channels, "
                  f"{n_bad} below 10 dB, "
                  f"median={wl_df['snr_db'].median():.1f} dB, "
                  f"range=[{wl_df['snr_db'].min():.1f}, "
                  f"{wl_df['snr_db'].max():.1f}] dB")

    fig.suptitle(f'Expert/Novice SNR — Subject {subject_id}', fontsize=13)
    fig.tight_layout()

    if save_fig:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{subject_id}_snr_boxplot.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"  SNR box plot saved: {fig_path}")

    return fig


# ==============================================================================
# TASK 2.4.3 - Motion Ratio Box Plot
# ==============================================================================

def plot_motion_boxplot(
    metrics_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    save_fig: bool = True,
    verbose: bool = True,
) -> plt.Figure:
    """
    Create motion ratio box plot visualization matching paper Figure 4 style.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of create_metrics_dataframe()
    subject_id : str
        Subject identifier (used in title and filename)
    output_dir : Path
        Directory where figure is saved
    save_fig : bool
        Whether to save the figure to disk
    verbose : bool
        Print status messages

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if verbose:
        print("\n" + "=" * 65)
        print("TASK 2.4.3 — Motion Ratio Box Plot")
        print("=" * 65)

    # Filter to intensity channels only (wavelength > 0)
    plot_df = metrics_df[metrics_df['wavelength'] > 0].copy()

    # Elevated motion channels
    elev_mask = plot_df['motion_quality_flag'] == 'elevated'

    wavelengths = sorted(plot_df['wavelength'].unique())
    n_wl = len(wavelengths)

    fig, axes = plt.subplots(1, n_wl, figsize=(6 * n_wl, 6), sharey=False)
    if n_wl == 1:
        axes = [axes]

    for ax, wl in zip(axes, wavelengths):
        wl_df  = plot_df[plot_df['wavelength'] == wl].copy()
        elev   = wl_df.loc[ elev_mask[wl_df.index], 'motion_ratio'].values
        normal = wl_df.loc[~elev_mask[wl_df.index], 'motion_ratio'].values

        box_data  = [wl_df['motion_ratio'].values]
        positions = [1]
        labels    = [subject_id]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color='blue', linewidth=2),
            boxprops=dict(facecolor='white', color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(
                marker='+',
                color='red',
                markersize=8,
                linestyle='none',
            ),
        )

        # Overlay elevated motion channels in red
        if len(elev) > 0:
            jitter_x = np.random.default_rng(42).uniform(-0.15, 0.15, len(elev))
            ax.scatter(
                np.ones(len(elev)) + jitter_x,
                elev,
                color='red',
                s=40,
                zorder=5,
                label='Elevated motion (> 10%)',
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Subject Label')
        ax.set_ylabel('Motion Ratio')
        ax.set_title(f'Motion Ratio — {subject_id} ({wl} nm)')
        if len(elev) > 0:
            ax.legend(fontsize=8, loc='upper right')

        if verbose:
            n_total = len(wl_df)
            n_elev  = elev_mask[wl_df.index].sum()
            print(f"  {wl} nm: {n_total} channels, "
                  f"{n_elev} elevated, "
                  f"median={wl_df['motion_ratio'].median():.4f}, "
                  f"range=[{wl_df['motion_ratio'].min():.4f}, "
                  f"{wl_df['motion_ratio'].max():.4f}]")

    fig.suptitle(f'Expert/Novice Motion Ratio — Subject {subject_id}',
                 fontsize=13)
    fig.tight_layout()

    if save_fig:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{subject_id}_motion_boxplot.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"  Motion ratio box plot saved: {fig_path}")

    return fig


# ==============================================================================
# TASK 2.4.4 - Summary Statistics and Console Report
# ==============================================================================

def calculate_qa_summary_statistics(
    metrics_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Calculate summary statistics across all channels for the QA report.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of create_metrics_dataframe()
    verbose : bool
        Print the report to console

    Returns
    -------
    summary : dict
        Nested dict with overall and per-wavelength stats for SNR and MR
    """
    plot_df = metrics_df[metrics_df['wavelength'] > 0].copy()

    summary = {
        'n_channels_total': len(plot_df),
        'subject_id': metrics_df['subject_id'].iloc[0],
        'snr': {},
        'motion': {},
    }

    # Overall SNR
    snr_vals = plot_df['snr_db'].dropna()
    summary['snr']['overall'] = {
        'mean':   float(snr_vals.mean()),
        'median': float(snr_vals.median()),
        'std':    float(snr_vals.std()),
        'min':    float(snr_vals.min()),
        'max':    float(snr_vals.max()),
        'q25':    float(snr_vals.quantile(0.25)),
        'q75':    float(snr_vals.quantile(0.75)),
        'n_low':        int((plot_df['snr_quality_flag'] == 'low').sum()),
        'n_high':       int((plot_df['snr_quality_flag'] == 'high').sum()),
        'n_acceptable': int((plot_df['snr_quality_flag'] == 'acceptable').sum()),
    }

    # Overall Motion
    mr_vals = plot_df['motion_ratio']
    summary['motion']['overall'] = {
        'mean':     float(mr_vals.mean()),
        'median':   float(mr_vals.median()),
        'std':      float(mr_vals.std()),
        'min':      float(mr_vals.min()),
        'max':      float(mr_vals.max()),
        'q25':      float(mr_vals.quantile(0.25)),
        'q75':      float(mr_vals.quantile(0.75)),
        'n_elevated': int((plot_df['motion_quality_flag'] == 'elevated').sum()),
    }

    # Per-wavelength breakdown
    for wl in sorted(plot_df['wavelength'].unique()):
        wl_df = plot_df[plot_df['wavelength'] == wl]
        wl_snr = wl_df['snr_db'].dropna()
        wl_mr  = wl_df['motion_ratio']

        summary['snr'][wl] = {
            'mean':   float(wl_snr.mean()),
            'median': float(wl_snr.median()),
            'std':    float(wl_snr.std()),
            'min':    float(wl_snr.min()),
            'max':    float(wl_snr.max()),
            'n_channels': len(wl_snr),
        }
        summary['motion'][wl] = {
            'mean':     float(wl_mr.mean()),
            'median':   float(wl_mr.median()),
            'std':      float(wl_mr.std()),
            'min':      float(wl_mr.min()),
            'max':      float(wl_mr.max()),
            'n_channels': len(wl_mr),
        }

    if verbose:
        _print_qa_console_report(summary)

    return summary


def _print_qa_console_report(summary: dict):
    """Print formatted QA summary to console."""
    sid  = summary['subject_id']
    n_ch = summary['n_channels_total']
    snr  = summary['snr']['overall']
    mr   = summary['motion']['overall']

    print("\n" + "=" * 65)
    print(f"  STAGE A QA SUMMARY — Subject {sid}")
    print("=" * 65)
    print(f"\n  Total channels analyzed : {n_ch}  (NO channels excluded)")

    print(f"\n  SNR (Raw Intensity):")
    print(f"    Mean   : {snr['mean']:.2f} dB")
    print(f"    Median : {snr['median']:.2f} dB")
    print(f"    Std    : {snr['std']:.2f} dB")
    print(f"    Range  : [{snr['min']:.2f}, {snr['max']:.2f}] dB")
    print(f"    IQR    : [{snr['q25']:.2f}, {snr['q75']:.2f}] dB")
    print(f"    Quality flags:")
    print(f"      < 10 dB (low)       : {snr['n_low']} channels")
    print(f"      10–30 dB (acceptable): {snr['n_acceptable']} channels")
    print(f"      > 30 dB (high)      : {snr['n_high']} channels")

    print(f"\n  Motion Ratio (Optical Density):")
    print(f"    Mean   : {mr['mean']:.4f}  ({mr['mean']*100:.2f}%)")
    print(f"    Median : {mr['median']:.4f}  ({mr['median']*100:.2f}%)")
    print(f"    Std    : {mr['std']:.4f}")
    print(f"    Range  : [{mr['min']:.4f}, {mr['max']:.4f}]")
    print(f"    IQR    : [{mr['q25']:.4f}, {mr['q75']:.4f}]")
    print(f"    Elevated (> 10%) : {mr['n_elevated']} channels")

    print(f"\n  Paper Figure 4 reference ranges:")
    print(f"    SNR    : 15–30 dB for typical channels")
    print(f"    Motion : ~0 median expected for clean subjects")

    print("=" * 65 + "\n")


def save_summary_statistics(
    summary: dict,
    subject_id: str,
    output_dir: Path,
) -> Path:
    """Save summary statistics dict as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}_qa_summary.json"

    def _make_serializable(obj):
        """Recursively convert numpy scalar keys/values to native Python types."""
        if isinstance(obj, dict):
            return {
                (int(k) if hasattr(k, 'item') else k): _make_serializable(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if hasattr(obj, 'item'):   # numpy scalar
            return obj.item()
        return obj

    with open(out_path, 'w') as f:
        json.dump(_make_serializable(summary), f, indent=2)
    print(f"  Summary statistics saved: {out_path}")
    return out_path


# ==============================================================================
# TASK 2.4.5 - Generate Complete Stage A Report for Subject A1
# ==============================================================================

def generate_stage_a_report(
    subject_id: str = 'A1',
    dataset: str = 'expert_novice',
    verbose: bool = True,
) -> dict:
    """
    Generate the complete Stage A QA report for a single subject.

    Runs all sub-tasks:
      2.4.1 — build metrics DataFrame
      2.4.2 — SNR box plot
      2.4.3 — motion ratio box plot
      2.4.4 — summary statistics + console report

    Parameters
    ----------
    subject_id : str
        Subject identifier (default 'A1')
    dataset : str
        Dataset name (default 'expert_novice')
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        {
          'metrics_df'      : pd.DataFrame,
          'summary'         : dict,
          'snr_fig'         : plt.Figure,
          'motion_fig'      : plt.Figure,
          'metrics_csv_path': Path,
          'summary_json_path': Path,
        }
    """
    output_dir = get_output_path('stage_a', 'qa_reports')

    if verbose:
        print("=" * 65)
        print(f"  GENERATING STAGE A QA REPORT — Subject {subject_id}")
        print("=" * 65)

    # ------------------------------------------------------------------
    # Locate input CSVs (produced by tasks 2.2 and 2.3)
    # ------------------------------------------------------------------
    snr_csv    = output_dir / f"{subject_id}_snr_raw.csv"
    motion_csv = output_dir / f"{subject_id}_motion_od.csv"

    for p in (snr_csv, motion_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input file not found: {p}\n"
                f"Run 05_calculate_snr.py and 06_calculate_motion_ratio.py first."
            )

    # ------------------------------------------------------------------
    # 2.4.1 — Metrics DataFrame
    # ------------------------------------------------------------------
    metrics_df = create_metrics_dataframe(
        snr_csv, motion_csv, subject_id, verbose=verbose
    )
    metrics_csv_path = save_metrics_dataframe(metrics_df, subject_id, output_dir)

    # ------------------------------------------------------------------
    # 2.4.2 — SNR Box Plot
    # ------------------------------------------------------------------
    snr_fig = plot_snr_boxplot(metrics_df, subject_id, output_dir,
                               save_fig=True, verbose=verbose)

    # ------------------------------------------------------------------
    # 2.4.3 — Motion Ratio Box Plot
    # ------------------------------------------------------------------
    motion_fig = plot_motion_boxplot(metrics_df, subject_id, output_dir,
                                     save_fig=True, verbose=verbose)

    # ------------------------------------------------------------------
    # 2.4.4 — Summary Statistics + Console Report
    # ------------------------------------------------------------------
    summary = calculate_qa_summary_statistics(metrics_df, verbose=verbose)
    summary_json_path = save_summary_statistics(summary, subject_id, output_dir)

    if verbose:
        print("\n" + "=" * 65)
        print(f"  Stage A QA report complete for subject {subject_id}")
        print(f"  Output directory: {output_dir}")
        print(f"  Files generated:")
        print(f"    {subject_id}_qa_metrics.csv")
        print(f"    {subject_id}_snr_boxplot.png")
        print(f"    {subject_id}_motion_boxplot.png")
        print(f"    {subject_id}_qa_summary.json")
        print("=" * 65)

    return {
        'metrics_df':        metrics_df,
        'summary':           summary,
        'snr_fig':           snr_fig,
        'motion_fig':        motion_fig,
        'metrics_csv_path':  metrics_csv_path,
        'summary_json_path': summary_json_path,
    }


# ==============================================================================
# TASK 2.6.1 - Dataset Aggregation (all available subjects)
# ==============================================================================

def aggregate_dataset_metrics(
    dataset: str = 'expert_novice',
    subject_ids: list | None = None,
    output_subdir: str = 'qa_reports',
    verbose: bool = True,
) -> dict:
    """
    Aggregate Stage A QA metrics across subjects and compute channel-wise stats.

    Parameters
    ----------
    dataset : str
        Dataset key from config.yaml (default: 'expert_novice')
    subject_ids : list | None
        Explicit subject IDs to aggregate. If None, use config order.
    output_subdir : str
        Stage A output subdirectory containing per-subject QA metrics CSVs.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    results : dict
        {
          'dataset': str,
          'subject_ids_expected': list[str],
          'subject_ids_found': list[str],
          'subject_ids_missing': list[str],
          'all_metrics_df': pd.DataFrame,
          'channel_stats_df': pd.DataFrame,
          'subject_stats_df': pd.DataFrame,
          'all_metrics_csv_path': Path,
          'channel_stats_csv_path': Path,
          'subject_stats_csv_path': Path,
        }
    """
    config = load_config()
    ds_cfg = config['datasets'][dataset]

    if subject_ids is None:
        subject_ids = list(ds_cfg['subjects'])

    output_dir = get_output_path('stage_a', output_subdir)
    full_report_dir = output_dir / f"{dataset}_full_report"
    full_report_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print(f"TASK 2.6.1 — DATASET AGGREGATION ({dataset})")
        print("=" * 70)
        print(f"Expected subjects: {len(subject_ids)}")
        print(f"Metrics directory: {output_dir}")
        print(f"Full report directory: {full_report_dir}")

    metrics_frames = []
    found_subjects = []
    missing_subjects = []

    for sid in subject_ids:
        metrics_path = output_dir / f"{sid}_qa_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            # Ensure subject_id column exists and is consistent.
            df['subject_id'] = sid
            metrics_frames.append(df)
            found_subjects.append(sid)
            if verbose:
                print(f"  ✓ loaded {sid}: {len(df)} rows")
        else:
            missing_subjects.append(sid)
            if verbose:
                print(f"  - missing {sid}: {metrics_path.name}")

    if not metrics_frames:
        raise FileNotFoundError(
            "No subject QA metrics found for aggregation. "
            "Run per-subject Stage A QA generation first."
        )

    all_metrics_df = pd.concat(metrics_frames, ignore_index=True)

    # Keep only intensity channels (wavelength > 0) for dataset stats.
    agg_df = all_metrics_df[all_metrics_df['wavelength'] > 0].copy()

    # Subject-level summary for quick dataset checks.
    subject_stats_df = (
        agg_df.groupby('subject_id', as_index=False)
        .agg(
            n_channels=('channel', 'count'),
            snr_mean=('snr_db', 'mean'),
            snr_median=('snr_db', 'median'),
            snr_std=('snr_db', 'std'),
            motion_mean=('motion_ratio', 'mean'),
            motion_median=('motion_ratio', 'median'),
            motion_std=('motion_ratio', 'std'),
        )
    )

    # Channel-wise statistics across subjects.
    channel_stats_df = (
        agg_df.groupby(['channel', 'wavelength'], as_index=False)
        .agg(
            n_subjects=('subject_id', 'nunique'),
            snr_mean=('snr_db', 'mean'),
            snr_median=('snr_db', 'median'),
            snr_std=('snr_db', 'std'),
            snr_q25=('snr_db', lambda x: float(x.quantile(0.25))),
            snr_q75=('snr_db', lambda x: float(x.quantile(0.75))),
            snr_min=('snr_db', 'min'),
            snr_max=('snr_db', 'max'),
            motion_mean=('motion_ratio', 'mean'),
            motion_median=('motion_ratio', 'median'),
            motion_std=('motion_ratio', 'std'),
            motion_q25=('motion_ratio', lambda x: float(x.quantile(0.25))),
            motion_q75=('motion_ratio', lambda x: float(x.quantile(0.75))),
            motion_min=('motion_ratio', 'min'),
            motion_max=('motion_ratio', 'max'),
        )
    )

    # Save aggregation outputs.
    all_metrics_csv_path = full_report_dir / f"{dataset}_all_subjects_qa_metrics.csv"
    channel_stats_csv_path = full_report_dir / f"{dataset}_channel_stats.csv"
    subject_stats_csv_path = full_report_dir / f"{dataset}_subject_stats.csv"

    all_metrics_df.to_csv(all_metrics_csv_path, index=False)
    channel_stats_df.to_csv(channel_stats_csv_path, index=False)
    subject_stats_df.to_csv(subject_stats_csv_path, index=False)

    if verbose:
        print("\nAggregation summary:")
        print(f"  Subjects found: {len(found_subjects)}/{len(subject_ids)}")
        print(f"  Combined rows: {len(all_metrics_df)}")
        print(f"  Channel stats rows: {len(channel_stats_df)}")
        print(f"  Subject stats rows: {len(subject_stats_df)}")
        print(f"  Saved: {all_metrics_csv_path.name}")
        print(f"  Saved: {channel_stats_csv_path.name}")
        print(f"  Saved: {subject_stats_csv_path.name}")
        if missing_subjects:
            print(f"  Missing subjects ({len(missing_subjects)}): {missing_subjects}")

    return {
        'dataset': dataset,
        'full_report_dir': full_report_dir,
        'subject_ids_expected': subject_ids,
        'subject_ids_found': found_subjects,
        'subject_ids_missing': missing_subjects,
        'all_metrics_df': all_metrics_df,
        'channel_stats_df': channel_stats_df,
        'subject_stats_df': subject_stats_df,
        'all_metrics_csv_path': all_metrics_csv_path,
        'channel_stats_csv_path': channel_stats_csv_path,
        'subject_stats_csv_path': subject_stats_csv_path,
    }


# ==============================================================================
# TASK 2.6.2 - Dataset-level SNR/MR visualizations (Figure 4 style)
# ==============================================================================

def _subject_order_for_dataset(dataset: str, subject_ids: list[str]) -> list[str]:
    """Return plotting order. Expert/Novice uses R1..R12, A1, A2."""
    unique_ids = list(dict.fromkeys(subject_ids))
    if dataset != 'expert_novice':
        return unique_ids

    r_ids = sorted(
        [sid for sid in unique_ids if sid.startswith('R')],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )
    a_ids = sorted(
        [sid for sid in unique_ids if sid.startswith('A')],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )
    others = [sid for sid in unique_ids if sid not in r_ids + a_ids]
    return r_ids + a_ids + others


def plot_dataset_level_boxplots(
    all_metrics_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    show_errorbars: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Create dataset-level SNR and motion-ratio box plots across subjects.

    Parameters
    ----------
    all_metrics_df : pd.DataFrame
        Aggregated QA metrics from aggregate_dataset_metrics()
    dataset : str
        Dataset key (e.g., 'expert_novice')
    output_dir : Path
        Output directory where figures are saved
    show_errorbars : bool
        Overlay mean ± std marker on each subject box
    verbose : bool
        Print plotting summary

    Returns
    -------
    dict
        Paths and figure objects for SNR and motion plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = all_metrics_df[all_metrics_df['wavelength'] > 0].copy()
    subject_order = _subject_order_for_dataset(dataset, plot_df['subject_id'].tolist())

    if verbose:
        print("\n" + "=" * 70)
        print(f"TASK 2.6.2 — DATASET VISUALIZATION ({dataset})")
        print("=" * 70)
        print(f"Subjects in plot: {subject_order}")

    # Shared style
    box_kwargs = dict(
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color='blue', linewidth=1.8),
        boxprops=dict(facecolor='white', color='black'),
        whiskerprops=dict(color='black', linestyle='--', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        flierprops=dict(marker='+', markeredgecolor='red', markersize=6, linestyle='none'),
    )

    # -----------------------------
    # SNR plot
    # -----------------------------
    snr_data = [
        plot_df.loc[plot_df['subject_id'] == sid, 'snr_db'].dropna().values
        for sid in subject_order
    ]
    snr_positions = np.arange(1, len(subject_order) + 1)

    snr_fig, snr_ax = plt.subplots(figsize=(12, 7))
    snr_ax.boxplot(snr_data, positions=snr_positions, **box_kwargs)
    snr_ax.set_title('Expert/Novice SNR' if dataset == 'expert_novice' else f'{dataset} SNR')
    snr_ax.set_xlabel('Subject Label')
    snr_ax.set_ylabel('SNR')
    snr_ax.set_xticks(snr_positions)
    snr_ax.set_xticklabels(subject_order, rotation=35, ha='right')
    snr_ax.axhline(y=10, color='gray', linestyle='--', linewidth=1.0)
    snr_ax.axhline(y=30, color='gray', linestyle='--', linewidth=1.0)

    if show_errorbars:
        snr_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in snr_data]
        snr_stds = [np.std(vals) if len(vals) > 0 else np.nan for vals in snr_data]
        snr_ax.errorbar(
            snr_positions, snr_means, yerr=snr_stds, fmt='o',
            color='black', markersize=3, linewidth=1.0, capsize=2, alpha=0.65
        )

    snr_fig.tight_layout()
    snr_path = output_dir / f"{dataset}_snr_boxplot.png"
    snr_fig.savefig(snr_path, dpi=150, bbox_inches='tight')

    # -----------------------------
    # Motion plot
    # -----------------------------
    mr_data = [
        plot_df.loc[plot_df['subject_id'] == sid, 'motion_ratio'].dropna().values
        for sid in subject_order
    ]
    mr_positions = np.arange(1, len(subject_order) + 1)

    mr_fig, mr_ax = plt.subplots(figsize=(12, 7))
    mr_ax.boxplot(mr_data, positions=mr_positions, **box_kwargs)
    mr_ax.set_title('Expert/Novice MR' if dataset == 'expert_novice' else f'{dataset} Motion Ratio')
    mr_ax.set_xlabel('Subject Label')
    mr_ax.set_ylabel('Motion Ratio')
    mr_ax.set_xticks(mr_positions)
    mr_ax.set_xticklabels(subject_order, rotation=35, ha='right')

    if show_errorbars:
        mr_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in mr_data]
        mr_stds = [np.std(vals) if len(vals) > 0 else np.nan for vals in mr_data]
        mr_ax.errorbar(
            mr_positions, mr_means, yerr=mr_stds, fmt='o',
            color='black', markersize=3, linewidth=1.0, capsize=2, alpha=0.65
        )

    mr_fig.tight_layout()
    mr_path = output_dir / f"{dataset}_motion_boxplot.png"
    mr_fig.savefig(mr_path, dpi=150, bbox_inches='tight')

    if verbose:
        print(f"  Saved: {snr_path}")
        print(f"  Saved: {mr_path}")

    return {
        'subject_order': subject_order,
        'snr_fig': snr_fig,
        'motion_fig': mr_fig,
        'snr_plot_path': snr_path,
        'motion_plot_path': mr_path,
    }


def build_cumulative_dataset_outputs(
    dataset: str = 'expert_novice',
    output_subdir: str = 'qa_reports',
    verbose: bool = True,
) -> dict:
    """
    Build cumulative CSV/JSON/PNG outputs from subject-level Stage A artifacts.

    This aggregates per-subject files into study-level files under:
      outputs/stage_a_qa/qa_reports/{dataset}_full_report/

    Aggregated artifacts include:
      - {dataset}_all_subjects_snr_raw.csv
      - {dataset}_all_subjects_motion_od.csv
      - {dataset}_all_subjects_qa_metrics.csv
      - {dataset}_all_subjects_qa_summary.json
      - {dataset}_snr_boxplot.png
      - {dataset}_motion_boxplot.png

    Parameters
    ----------
    dataset : str
        Dataset key from config.yaml
    output_subdir : str
        Stage A output subdirectory containing per-subject artifacts
    verbose : bool
        Print progress logs

    Returns
    -------
    dict
        Aggregation summary with output paths and any missing subjects.
    """
    config = load_config()
    subject_ids = list(config['datasets'][dataset]['subjects'])

    qa_dir = get_output_path('stage_a', output_subdir)
    full_report_dir = qa_dir / f"{dataset}_full_report"
    full_report_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print(f"CUMULATIVE DATASET OUTPUTS ({dataset})")
        print("=" * 70)
        print(f"Source directory: {qa_dir}")
        print(f"Target directory: {full_report_dir}")

    snr_frames = []
    motion_frames = []
    metrics_frames = []
    summary_by_subject = {}
    missing_subjects = []

    for sid in subject_ids:
        snr_path = qa_dir / f"{sid}_snr_raw.csv"
        motion_path = qa_dir / f"{sid}_motion_od.csv"
        metrics_path = qa_dir / f"{sid}_qa_metrics.csv"
        summary_path = qa_dir / f"{sid}_qa_summary.json"

        required = [snr_path, motion_path, metrics_path, summary_path]
        if not all(p.exists() for p in required):
            missing_subjects.append(sid)
            if verbose:
                missing_files = [p.name for p in required if not p.exists()]
                print(f"  - missing {sid}: {missing_files}")
            continue

        snr_df = pd.read_csv(snr_path)
        snr_df.insert(0, 'subject_id', sid)
        snr_frames.append(snr_df)

        motion_df = pd.read_csv(motion_path)
        motion_df.insert(0, 'subject_id', sid)
        motion_frames.append(motion_df)

        metrics_df = pd.read_csv(metrics_path)
        metrics_df['subject_id'] = sid
        metrics_frames.append(metrics_df)

        with open(summary_path, 'r') as f:
            summary_by_subject[sid] = json.load(f)

        if verbose:
            print(f"  ✓ aggregated {sid}")

    if not metrics_frames:
        raise FileNotFoundError(
            "No complete subject artifact sets found to aggregate. "
            "Run Stage A subject processing first."
        )

    all_snr_df = pd.concat(snr_frames, ignore_index=True)
    all_motion_df = pd.concat(motion_frames, ignore_index=True)
    all_metrics_df = pd.concat(metrics_frames, ignore_index=True)

    # Save cumulative CSVs.
    all_snr_csv_path = full_report_dir / f"{dataset}_all_subjects_snr_raw.csv"
    all_motion_csv_path = full_report_dir / f"{dataset}_all_subjects_motion_od.csv"
    all_metrics_csv_path = full_report_dir / f"{dataset}_all_subjects_qa_metrics.csv"
    all_snr_df.to_csv(all_snr_csv_path, index=False)
    all_motion_df.to_csv(all_motion_csv_path, index=False)
    all_metrics_df.to_csv(all_metrics_csv_path, index=False)

    # Save cumulative summary JSON.
    cumulative_summary = {
        'dataset': dataset,
        'n_subjects_expected': len(subject_ids),
        'n_subjects_aggregated': len(summary_by_subject),
        'subjects_aggregated': sorted(summary_by_subject.keys()),
        'subjects_missing': missing_subjects,
        'subject_summaries': summary_by_subject,
    }
    cumulative_summary_json_path = full_report_dir / f"{dataset}_all_subjects_qa_summary.json"
    with open(cumulative_summary_json_path, 'w') as f:
        json.dump(cumulative_summary, f, indent=2)

    # Refresh channel/subject stats from cumulative metrics and save in full report dir.
    agg_results = aggregate_dataset_metrics(
        dataset=dataset,
        subject_ids=sorted(summary_by_subject.keys()),
        output_subdir=output_subdir,
        verbose=False,
    )

    # Regenerate cumulative paper-style box plots.
    plot_results = plot_dataset_level_boxplots(
        all_metrics_df=all_metrics_df,
        dataset=dataset,
        output_dir=full_report_dir,
        show_errorbars=True,
        verbose=False,
    )
    plt.close(plot_results['snr_fig'])
    plt.close(plot_results['motion_fig'])

    if verbose:
        print("\nCumulative outputs written:")
        print(f"  - {all_snr_csv_path}")
        print(f"  - {all_motion_csv_path}")
        print(f"  - {all_metrics_csv_path}")
        print(f"  - {cumulative_summary_json_path}")
        print(f"  - {plot_results['snr_plot_path']}")
        print(f"  - {plot_results['motion_plot_path']}")
        print(f"Aggregated subjects: {len(summary_by_subject)}/{len(subject_ids)}")
        if missing_subjects:
            print(f"Missing subjects: {missing_subjects}")

    return {
        'dataset': dataset,
        'full_report_dir': full_report_dir,
        'missing_subjects': missing_subjects,
        'all_snr_csv_path': all_snr_csv_path,
        'all_motion_csv_path': all_motion_csv_path,
        'all_metrics_csv_path': all_metrics_csv_path,
        'all_summary_json_path': cumulative_summary_json_path,
        'snr_plot_path': plot_results['snr_plot_path'],
        'motion_plot_path': plot_results['motion_plot_path'],
        'channel_stats_csv_path': agg_results['channel_stats_csv_path'],
        'subject_stats_csv_path': agg_results['subject_stats_csv_path'],
    }


# ==============================================================================
# TASK 2.6.3 - Process all subjects through Stage A pipeline
# ==============================================================================

def process_dataset_stage_a_reports(
    dataset: str = 'expert_novice',
    output_subdir: str = 'qa_reports',
    verbose: bool = True,
) -> dict:
    """
    Run Stage A QA report generation for all subjects in a dataset.

    This ensures each subject has:
      - {subject}_snr_raw.csv
      - {subject}_motion_od.csv
      - {subject}_qa_metrics.csv
      - {subject}_qa_summary.json
      - {subject}_snr_boxplot.png
      - {subject}_motion_boxplot.png

    Parameters
    ----------
    dataset : str
        Dataset key in config.yaml (default: expert_novice)
    output_subdir : str
        Stage A output subdirectory where QA artifacts are stored.
    verbose : bool
        Print progress for each subject.

    Returns
    -------
    dict
        Batch processing summary including success/failure lists.
    """
    config = load_config()
    subject_ids = list(config['datasets'][dataset]['subjects'])
    output_dir = get_output_path('stage_a', output_subdir)

    if verbose:
        print("\n" + "=" * 70)
        print(f"TASK 2.6.3 — BATCH STAGE A PROCESSING ({dataset})")
        print("=" * 70)
        print(f"Subjects to process: {len(subject_ids)}")
        print(f"Output directory: {output_dir}")

    succeeded = []
    failed = []
    errors = {}

    for idx, subject_id in enumerate(subject_ids, start=1):
        try:
            if verbose:
                print(f"\n[{idx}/{len(subject_ids)}] Processing subject {subject_id}")

            # Load raw once and compute Stage A prerequisite inputs.
            raw, _ = load_raw_snirf(dataset, subject_id, preload=True, verbose=False)

            snr_df = calculate_snr.calculate_snr_raw(raw, verbose=False)
            calculate_snr.save_snr_results(snr_df, subject_id, output_dir)

            raw_od = calculate_motion.convert_to_optical_density(raw, verbose=False)
            motion_df = calculate_motion.calculate_motion_ratio_od(raw_od, verbose=False)
            calculate_motion.save_motion_results(motion_df, subject_id, output_dir)

            # Generate full per-subject Stage A QA report.
            report_results = generate_stage_a_report(
                subject_id=subject_id, dataset=dataset, verbose=False
            )
            # Prevent figure accumulation during long batch runs.
            plt.close(report_results['snr_fig'])
            plt.close(report_results['motion_fig'])

            succeeded.append(subject_id)
            if verbose:
                print(f"  ✓ Completed {subject_id}")

        except Exception as exc:
            failed.append(subject_id)
            errors[subject_id] = str(exc)
            if verbose:
                print(f"  ✗ Failed {subject_id}: {exc}")

    summary = {
        'dataset': dataset,
        'n_subjects_expected': len(subject_ids),
        'n_subjects_succeeded': len(succeeded),
        'n_subjects_failed': len(failed),
        'subjects_succeeded': succeeded,
        'subjects_failed': failed,
        'errors': errors,
        'output_dir': str(output_dir),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Succeeded: {len(succeeded)}/{len(subject_ids)}")
        if failed:
            print(f"Failed: {len(failed)} -> {failed}")
        else:
            print("Failed: 0")
        print("=" * 70 + "\n")

    return summary


def cleanup_subject_level_outputs(
    dataset: str,
    output_subdir: str = 'qa_reports',
    verbose: bool = True,
) -> dict:
    """
    Delete per-subject Stage A QA artifacts after cumulative aggregation.

    This keeps only study-level outputs in {dataset}_full_report.
    """
    config = load_config()
    subject_ids = list(config['datasets'][dataset]['subjects'])
    qa_dir = get_output_path('stage_a', output_subdir)

    suffixes = [
        '_snr_raw.csv',
        '_motion_od.csv',
        '_qa_metrics.csv',
        '_qa_summary.json',
        '_snr_boxplot.png',
        '_motion_boxplot.png',
    ]

    deleted = []
    missing = []
    errors = {}

    for sid in subject_ids:
        for suffix in suffixes:
            path = qa_dir / f"{sid}{suffix}"
            if path.exists():
                try:
                    path.unlink()
                    deleted.append(str(path))
                except Exception as exc:
                    errors[str(path)] = str(exc)
            else:
                missing.append(str(path))

    if verbose:
        print("\n" + "=" * 70)
        print("SUBJECT-LEVEL CLEANUP")
        print("=" * 70)
        print(f"Deleted files: {len(deleted)}")
        print(f"Already missing: {len(missing)}")
        if errors:
            print(f"Delete errors: {len(errors)}")
        print("=" * 70 + "\n")

    return {
        'dataset': dataset,
        'deleted_count': len(deleted),
        'missing_count': len(missing),
        'error_count': len(errors),
        'errors': errors,
    }

# ==============================================================================
# Entry point — dataset-level Stage A processing
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage A QA for an entire dataset and build cumulative "
            "study-level outputs."
        )
    )
    #specifies which dataset to process
    parser.add_argument(
        '--dataset',
        type=str,
        default='expert_novice',
        help=(
            "Dataset key from config.yaml "
            "(e.g., expert_novice, learning_curve, short_term_tes, long_term_tes)."
        ),
    )

    #specifies the output subdirectory for the stage a reports
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='qa_reports',
        help="Stage A output subdirectory under outputs/stage_a_qa/.",
    )

    #if present, skips the per-subject processing and only builds the cumulative outputs from existing subject artifacts
    parser.add_argument(
        '--skip-batch',
        action='store_true',
        help=(
            "Skip per-subject processing and only build cumulative outputs "
            "from existing subject artifacts."
        ),
    )

    #if present, reduces the console logging
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Reduce console logging.",
    )

    #if present, deletes the per-subject CSV/JSON/PNG artifacts from the outputs/stage_a_qa/qa_reports/ directory
    parser.add_argument(
        '--cleanup-subject-files',
        action='store_true',
        help=(
            "After cumulative outputs are built, delete per-subject CSV/JSON/PNG "
            "artifacts from outputs/stage_a_qa/qa_reports/."
        ),
    )
    args = parser.parse_args()

    # To run expert-novice dataset and delete per-subject artifacts, run: 
    # python code/stage_a_raw_qa/07_generate_qa_report.py --dataset expert_novice --cleanup-subject-files

    verbose = not args.quiet

    if not args.skip_batch:
        batch_summary = process_dataset_stage_a_reports(
            dataset=args.dataset,
            output_subdir=args.output_subdir,
            verbose=verbose,
        )
        if batch_summary['n_subjects_failed'] > 0:
            print(
                f"\nWARNING: {batch_summary['n_subjects_failed']} subject(s) failed. "
                "Cumulative outputs will include successful subjects only."
            )

    cumulative = build_cumulative_dataset_outputs(
        dataset=args.dataset,
        output_subdir=args.output_subdir,
        verbose=verbose,
    )

    if args.cleanup_subject_files:
        cleanup_subject_level_outputs(
            dataset=args.dataset,
            output_subdir=args.output_subdir,
            verbose=verbose,
        )

    print("\nStage A dataset run complete.")
    print(f"Dataset: {args.dataset}")
    print(f"Study folder: {cumulative['full_report_dir']}")
    print(f"SNR plot: {cumulative['snr_plot_path']}")
    print(f"MR plot: {cumulative['motion_plot_path']}")
