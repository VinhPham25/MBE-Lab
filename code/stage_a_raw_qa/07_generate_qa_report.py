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
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path


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
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
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
# Entry point — runs 2.4.5 which orchestrates all sub-tasks
# ==============================================================================

if __name__ == '__main__':
    results = generate_stage_a_report(subject_id='A1', verbose=True)

    snr_s = results['summary']['snr']['overall']
    mr_s  = results['summary']['motion']['overall']

    print("\n  VERIFICATION vs PAPER FIGURE 4:")
    print(f"  SNR  — median {snr_s['median']:.1f} dB  "
          f"(paper: ~15–30 dB; "
          f"{'PASS' if 10 < snr_s['median'] < 40 else 'REVIEW'})")
    print(f"  MR   — median {mr_s['median']:.4f}  "
          f"(paper: ~0;  "
          f"{'PASS' if mr_s['median'] < 0.05 else 'REVIEW'})")
