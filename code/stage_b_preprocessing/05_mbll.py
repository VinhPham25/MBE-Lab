"""
Stage B — Step B5: Modified Beer-Lambert Law (MBLL) Conversion
Purpose: Convert rPCA-corrected optical density (OD) data to hemoglobin
         concentration time series (HbO and HbR) using wavelength-specific
         partial path-length factors (PPF) for the TechEn CW6 system.

Task coverage:
  3.5.1 — Implement Beer-Lambert conversion with TechEn PPF values
  3.5.2 — Verify hemoglobin concentration ranges (±50 μM typical, >±100 μM flagged)
  3.5.3 — Visualize HbO/HbR time series and verify inverse relationship
  3.5.4 — Test on subject A1 end-to-end

Algorithm
---------
Modified Beer-Lambert Law (Delpy et al. 1988):

    ΔOD_λ = Σ_i (ε_i_λ × ΔC_i) × d × PPF_λ

Rearranged for a two-wavelength (690 nm, 830 nm) system:

    [ΔHbO]       [ε_HbO_690  ε_HbR_690]⁻¹   [ΔOD_690 / PPF_690]
    [ΔHbR]   =   [ε_HbO_830  ε_HbR_830]    × [ΔOD_830 / PPF_830]

PPF parameters (TechEn CW6, from paper — Kamat et al. 2023):
  690 nm → PPF = 6.4
  830 nm → PPF = 5.8

MNE 1.7+ supports per-wavelength PPF via a tuple:
  mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=(6.4, 5.8))
  → tuple order matches MNE's ascending wavelength sort: (λ1=690, λ2=830)

Output units: mol/L (MNE convention); multiply × 1e6 for μM in plots/analysis.

Upstream:  04_rPCA_motion_correct.py  (optimal: n_components=0.99, min_ch=10)
Reference: PROGRESS.md Section 3.5; PROJECT.md Section 5.6
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

# ---------------------------------------------------------------------------
# Import Step B4 pipeline via importlib
# ---------------------------------------------------------------------------
import importlib.util

_spec_b4 = importlib.util.spec_from_file_location(
    "rpca_motion_correct",
    Path(__file__).parent / "04_rPCA_motion_correct.py",
)
_b4 = importlib.util.module_from_spec(_spec_b4)
_spec_b4.loader.exec_module(_b4)

run_rpca_correction = _b4.run_rpca_correction

# ---------------------------------------------------------------------------
# TechEn PPF constants
# ---------------------------------------------------------------------------
TECHEN_PPF = {690: 6.4, 830: 5.8}    # partial path-length factors (paper values)
NIRX_PPF   = {760: 6.0, 850: 6.0}    # NIRx defaults (used for tES datasets)


# ===========================================================================
# Task 3.5.1 — Beer-Lambert conversion
# ===========================================================================

def apply_beer_lambert(
    raw_od_corrected,
    ppf_dict: dict = None,
    verbose: bool = True,
):
    """
    Task 3.5.1: Convert rPCA-corrected OD to hemoglobin concentration.

    Uses MNE's beer_lambert_law with per-wavelength PPF (tuple, MNE ≥1.7).
    Output channels are renamed to the standard MNE convention:
      '<source>_<detector> hbo' and '<source>_<detector> hbr'.

    Parameters
    ----------
    raw_od_corrected : mne.io.Raw
        Filtered, rPCA-corrected OD data (Step B4 output).
        Must contain paired 690 nm and 830 nm channels.
    ppf_dict : dict, optional
        {wavelength_nm: ppf_value}.  Defaults to TECHEN_PPF = {690: 6.4, 830: 5.8}.
    verbose : bool

    Returns
    -------
    raw_haemo : mne.io.Raw
        Hemoglobin data (channel types 'hbo' and 'hbr', units mol/L).
        Multiply data by 1e6 to obtain μM values.
    ppf_tuple : tuple
        (ppf_wl1, ppf_wl2) passed to MNE, ordered ascending by wavelength.
    """
    if ppf_dict is None:
        ppf_dict = TECHEN_PPF

    # Determine the two wavelengths present in the data
    all_chs = raw_od_corrected.ch_names
    wavelengths = sorted({
        int(ch.split()[-1])
        for ch in all_chs
        if ch.split()[-1].isdigit()
    })

    if len(wavelengths) != 2:
        raise ValueError(
            f"Expected exactly 2 wavelengths, found {wavelengths}. "
            "Beer-Lambert requires paired dual-wavelength channels."
        )

    wl1, wl2 = wavelengths   # ascending order (e.g., 690, 830)
    ppf_wl1 = ppf_dict.get(wl1, 6.0)
    ppf_wl2 = ppf_dict.get(wl2, 6.0)
    ppf_tuple = (ppf_wl1, ppf_wl2)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.5.1 — MODIFIED BEER-LAMBERT LAW CONVERSION")
        print("=" * 70)
        print(f"  System wavelengths : {wl1} nm, {wl2} nm")
        print(f"  PPF ({wl1} nm)     : {ppf_wl1}  (paper value)")
        print(f"  PPF ({wl2} nm)     : {ppf_wl2}  (paper value)")
        print(f"  Input channels     : {len(all_chs)} "
              f"({len(raw_od_corrected.info['bads'])} bad)")
        print(f"  MNE ppf tuple      : {ppf_tuple}  (ascending wavelength order)")

    # Apply MNE beer_lambert_law — tuple PPF supported since MNE 1.7
    # Output: channel type changes from fnirs_od → hbo / hbr
    #         data units: mol/L  (× 1e6 → μM)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(
        raw_od_corrected, ppf=ppf_tuple
    )

    n_hbo = sum(1 for ch in raw_haemo.get_channel_types() if ch == 'hbo')
    n_hbr = sum(1 for ch in raw_haemo.get_channel_types() if ch == 'hbr')

    if verbose:
        print(f"\n  Output channels:")
        print(f"    HbO channels : {n_hbo}")
        print(f"    HbR channels : {n_hbr}")
        print(f"    Total        : {n_hbo + n_hbr}")
        print(f"  Data units: mol/L  (×1e6 = μM for analysis)")
        print("=" * 70 + "\n")

    return raw_haemo, ppf_tuple


# ===========================================================================
# Task 3.5.2 — Hemoglobin concentration range verification
# ===========================================================================

def verify_hb_ranges(
    raw_haemo,
    typical_range_um: float = 50.0,
    flag_threshold_um: float = 100.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Task 3.5.2: Verify per-channel hemoglobin concentration ranges.

    Checks that signal is within physiologically plausible bounds.
    Expected: ±50 μM range (task-evoked responses), flag if >±100 μM.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Output of apply_beer_lambert().  Units: mol/L.
    typical_range_um : float
        Typical task-evoked range (μM). Used for informational summary.
    flag_threshold_um : float
        Channels with |max| or |min| > this are flagged as out-of-range.
    verbose : bool

    Returns
    -------
    stats_df : pd.DataFrame
        Per-channel: channel, ch_type, mean_um, std_um, min_um, max_um,
                     range_um, within_typical, flagged.
    """
    data_mol_L = raw_haemo.get_data()        # (n_ch, n_times)
    data_um    = data_mol_L * 1e6            # convert mol/L → μM
    ch_names   = raw_haemo.ch_names
    ch_types   = raw_haemo.get_channel_types()

    records = []
    for i, (ch, ct) in enumerate(zip(ch_names, ch_types)):
        if ct not in ('hbo', 'hbr'):
            continue
        sig      = data_um[i]
        mean_um  = float(np.mean(sig))
        std_um   = float(np.std(sig))
        min_um   = float(np.min(sig))
        max_um   = float(np.max(sig))
        range_um = max_um - min_um
        within   = range_um <= (2 * typical_range_um)
        flagged  = (abs(min_um) > flag_threshold_um) or (abs(max_um) > flag_threshold_um)

        records.append({
            'channel':        ch,
            'ch_type':        ct,
            'mean_um':        round(mean_um, 4),
            'std_um':         round(std_um, 4),
            'min_um':         round(min_um, 4),
            'max_um':         round(max_um, 4),
            'range_um':       round(range_um, 4),
            'within_typical': within,
            'flagged':        flagged,
        })

    stats_df = pd.DataFrame(records)

    if verbose:
        hbo_df = stats_df[stats_df['ch_type'] == 'hbo']
        hbr_df = stats_df[stats_df['ch_type'] == 'hbr']
        n_flagged = int(stats_df['flagged'].sum())

        print("\n" + "=" * 70)
        print("TASK 3.5.2 — HEMOGLOBIN CONCENTRATION RANGE VERIFICATION")
        print("=" * 70)
        print(f"  Units: μM (micromolar).  Typical task range: ±{typical_range_um} μM")
        print(f"  Flag threshold: |value| > {flag_threshold_um} μM")
        print()
        print(f"  {'':8} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'Range':>9}")
        print(f"  {'-'*53}")
        print(f"  {'HbO':8} "
              f"{hbo_df['mean_um'].mean():>8.2f}  "
              f"{hbo_df['std_um'].mean():>8.2f}  "
              f"{hbo_df['min_um'].min():>8.2f}  "
              f"{hbo_df['max_um'].max():>8.2f}  "
              f"{hbo_df['range_um'].mean():>8.2f}")
        print(f"  {'HbR':8} "
              f"{hbr_df['mean_um'].mean():>8.2f}  "
              f"{hbr_df['std_um'].mean():>8.2f}  "
              f"{hbr_df['min_um'].min():>8.2f}  "
              f"{hbr_df['max_um'].max():>8.2f}  "
              f"{hbr_df['range_um'].mean():>8.2f}")
        print()

        if n_flagged > 0:
            print(f"  ⚠ {n_flagged} channels flagged (|value| > {flag_threshold_um} μM):")
            for _, row in stats_df[stats_df['flagged']].iterrows():
                print(f"    {row['channel']:30s}  [{row['ch_type']}]  "
                      f"min={row['min_um']:.1f}  max={row['max_um']:.1f} μM")
        else:
            print(f"  ✅ All channels within ±{flag_threshold_um} μM — physiologically plausible")

        print("=" * 70 + "\n")

    return stats_df


# ===========================================================================
# Task 3.5.3 — HbO/HbR time series visualization
# ===========================================================================

def plot_hbo_hbr_timeseries(
    raw_haemo,
    hb_stats_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    n_channels: int = 4,
    t_window: float = 300.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.5.3: Visualize HbO (red) and HbR (blue) time series.

    Selects n_channels representative long-separation channel pairs
    (chosen by median |HbO| std — spread across range of activity levels).
    Plots HbO and HbR overlaid on the same axes per channel.
    Inverse relationship (HbO ↑ while HbR ↓) is the key physiological check.

    Parameters
    ----------
    raw_haemo    : mne.io.Raw  (units mol/L)
    hb_stats_df  : pd.DataFrame  (output of verify_hb_ranges)
    subject_id   : str
    output_dir   : Path
    n_channels   : int  number of SD pairs to display
    t_window     : float  display window in seconds
    verbose      : bool

    Returns
    -------
    fig_path : Path
    """
    sfreq       = raw_haemo.info['sfreq']
    n_plot_samp = int(t_window * sfreq)
    times       = raw_haemo.times[:n_plot_samp]
    data_um     = raw_haemo.get_data() * 1e6   # mol/L → μM
    ch_names    = raw_haemo.ch_names
    ch_types    = raw_haemo.get_channel_types()

    # Build index maps for hbo and hbr
    hbo_map = {ch.replace(' hbo', ''): i
               for i, (ch, ct) in enumerate(zip(ch_names, ch_types))
               if ct == 'hbo'}
    hbr_map = {ch.replace(' hbr', ''): i
               for i, (ch, ct) in enumerate(zip(ch_names, ch_types))
               if ct == 'hbr'}

    # Pairs that have both hbo and hbr
    paired_sds = sorted(set(hbo_map.keys()) & set(hbr_map.keys()))

    # Select channels by spreading across HbO std quintiles
    hbo_stds = {sd: float(np.std(data_um[hbo_map[sd], :n_plot_samp]))
                for sd in paired_sds}
    sorted_sds = sorted(paired_sds, key=lambda sd: hbo_stds[sd])
    step   = max(1, len(sorted_sds) // n_channels)
    chosen = sorted_sds[::step][:n_channels]

    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 3.5 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, sd in zip(axes, chosen):
        hbo_sig = data_um[hbo_map[sd], :n_plot_samp]
        hbr_sig = data_um[hbr_map[sd], :n_plot_samp]

        # Compute correlation to quantify inverse relationship
        corr = float(np.corrcoef(hbo_sig, hbr_sig)[0, 1])

        ax.plot(times, hbo_sig, color='#d62728', lw=0.8, alpha=0.9,
                label='HbO (oxygenated)')
        ax.plot(times, hbr_sig, color='#1f77b4', lw=0.8, alpha=0.9,
                label='HbR (deoxygenated)')
        ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.35)
        ax.set_ylabel('Δ[Hb] (μM)', fontsize=8)

        corr_color = '#2ca02c' if corr < -0.3 else '#ff7f0e' if corr < 0 else '#d62728'
        ax.set_title(
            f'{sd}   |   HbO std={hbo_stds[sd]:.2f} μM   |   '
            f'HbO/HbR corr={corr:+.3f}  '
            f'({"inverse ✓" if corr < -0.3 else "weak inverse" if corr < 0 else "not inverse ⚠"})',
            fontsize=8.5, loc='left', color=corr_color,
        )

        if ax == axes[0]:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(
        f'Subject {subject_id} — HbO / HbR Time Series After MBLL Conversion\n'
        f'PPF: 690 nm=6.4, 830 nm=5.8  |  '
        f'First {t_window:.0f}s  |  '
        f'{n_channels} representative long-separation channels',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_hbo_hbr_timeseries.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved HbO/HbR time series : {fig_path}")

    return fig_path


# ===========================================================================
# Saving helpers
# ===========================================================================

def save_haemo_data(
    raw_haemo,
    hb_stats_df: pd.DataFrame,
    ppf_tuple: tuple,
    subject_id: str,
    output_dir: Path,
) -> tuple:
    """Save raw_haemo as .fif and per-channel stats as CSV + JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save haemo .fif
    fif_path = output_dir / f'{subject_id}_haemo.fif'
    raw_haemo.save(str(fif_path), overwrite=True, verbose=False)
    print(f"Saved haemo .fif       : {fif_path}")

    # Per-channel stats CSV
    csv_path = output_dir / f'{subject_id}_hb_range_stats.csv'
    hb_stats_df.to_csv(csv_path, index=False)
    print(f"Saved Hb range stats   : {csv_path}")

    # Conversion summary JSON
    hbo_df = hb_stats_df[hb_stats_df['ch_type'] == 'hbo']
    hbr_df = hb_stats_df[hb_stats_df['ch_type'] == 'hbr']
    n_total = len(hb_stats_df)
    summary = {
        'ppf_690nm': ppf_tuple[0],
        'ppf_830nm': ppf_tuple[1],
        'n_hbo_channels': int((hb_stats_df['ch_type'] == 'hbo').sum()),
        'n_hbr_channels': int((hb_stats_df['ch_type'] == 'hbr').sum()),
        'n_flagged': int(hb_stats_df['flagged'].sum()),
        'HbO': {
            'mean_mean_um': round(float(hbo_df['mean_um'].mean()), 4),
            'mean_std_um':  round(float(hbo_df['std_um'].mean()), 4),
            'min_um':       round(float(hbo_df['min_um'].min()), 4),
            'max_um':       round(float(hbo_df['max_um'].max()), 4),
            'mean_range_um': round(float(hbo_df['range_um'].mean()), 4),
        },
        'HbR': {
            'mean_mean_um': round(float(hbr_df['mean_um'].mean()), 4),
            'mean_std_um':  round(float(hbr_df['std_um'].mean()), 4),
            'min_um':       round(float(hbr_df['min_um'].min()), 4),
            'max_um':       round(float(hbr_df['max_um'].max()), 4),
            'mean_range_um': round(float(hbr_df['range_um'].mean()), 4),
        },
    }
    json_path = output_dir / f'{subject_id}_mbll_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved MBLL summary     : {json_path}")

    return fif_path, csv_path, json_path


# ===========================================================================
# End-to-end pipeline — Tasks 3.5.1–3.5.4
# ===========================================================================

def run_mbll(
    dataset_name: str,
    subject_id: str,
    ppf_dict: dict = None,
    n_components: float = 0.99,
    min_channels_flagged: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Task 3.5.4: Full MBLL pipeline for one subject.

    Chain:  B1 (OD) → B2 (bandpass) → B3 (motion detect) → B4 (rPCA)
            → B5 (Beer-Lambert) → verification → visualization → save.

    rPCA uses the tuning-optimal parameters (n_components=0.99, min_ch=10)
    unless overridden.  PPF defaults to TechEn CW6 values {690: 6.4, 830: 5.8}.

    Parameters
    ----------
    dataset_name : str
    subject_id   : str
    ppf_dict     : dict, optional  {wavelength_nm: ppf_value}
    n_components : float  rPCA variance threshold (default 0.99)
    min_channels_flagged : int  rPCA vote threshold (default 10)
    verbose      : bool

    Returns
    -------
    results : dict
        {
          'raw_od_corrected': mne.io.Raw,
          'raw_haemo':        mne.io.Raw,
          'hb_stats_df':      pd.DataFrame,
          'ppf_tuple':        tuple,
          'output_dir':       Path,
          'fig_path':         Path,
        }
    """
    if ppf_dict is None:
        ppf_dict = TECHEN_PPF

    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B5: MODIFIED BEER-LAMBERT LAW  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Steps B1–B4: rPCA-corrected OD ------------------------------------
    print("\n[1/4] Running upstream pipeline (B1→B2→B3→B4)...")
    b4_results = run_rpca_correction(
        dataset_name,
        subject_id,
        n_components         = n_components,
        min_channels_flagged = min_channels_flagged,
        verbose              = False,
    )
    raw_od_corrected = b4_results['raw_od_corrected']

    # ---- 3.5.1 Beer-Lambert conversion ------------------------------------
    print("[2/4] Applying Beer-Lambert conversion (Task 3.5.1)...")
    raw_haemo, ppf_tuple = apply_beer_lambert(
        raw_od_corrected,
        ppf_dict = ppf_dict,
        verbose  = verbose,
    )

    # ---- 3.5.2 Range verification -----------------------------------------
    print("[3/4] Verifying hemoglobin concentration ranges (Task 3.5.2)...")
    hb_stats_df = verify_hb_ranges(raw_haemo, verbose=verbose)

    # ---- Save ---------------------------------------------------------------
    output_dir = get_output_path('stage_b', f'mbll/{subject_id}')
    fif_path, csv_path, json_path = save_haemo_data(
        raw_haemo, hb_stats_df, ppf_tuple, subject_id, output_dir
    )

    # ---- 3.5.3 Visualization -----------------------------------------------
    print("[4/4] Visualizing HbO/HbR time series (Task 3.5.3)...")
    fig_path = plot_hbo_hbr_timeseries(
        raw_haemo,
        hb_stats_df,
        subject_id = subject_id,
        output_dir = output_dir,
        n_channels = 4,
        t_window   = 300.0,
        verbose    = verbose,
    )

    # ---- Console summary ---------------------------------------------------
    hbo_df = hb_stats_df[hb_stats_df['ch_type'] == 'hbo']
    hbr_df = hb_stats_df[hb_stats_df['ch_type'] == 'hbr']
    n_flag = int(hb_stats_df['flagged'].sum())

    print("\n" + "=" * 70)
    print(f"✅ Step B5 complete for subject {subject_id}")
    print(f"\n  PPF applied     : 690 nm={ppf_tuple[0]}, 830 nm={ppf_tuple[1]}")
    print(f"  HbO channels    : {(hb_stats_df['ch_type']=='hbo').sum()}")
    print(f"  HbR channels    : {(hb_stats_df['ch_type']=='hbr').sum()}")
    print(f"\n  HbO mean std    : {hbo_df['std_um'].mean():.2f} μM")
    print(f"  HbO range       : [{hbo_df['min_um'].min():.1f}, "
          f"{hbo_df['max_um'].max():.1f}] μM")
    print(f"  HbR mean std    : {hbr_df['std_um'].mean():.2f} μM")
    print(f"  HbR range       : [{hbr_df['min_um'].min():.1f}, "
          f"{hbr_df['max_um'].max():.1f}] μM")
    print(f"\n  Channels flagged (>±100 μM): {n_flag}")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od_corrected': raw_od_corrected,
        'raw_haemo':        raw_haemo,
        'hb_stats_df':      hb_stats_df,
        'ppf_tuple':        ppf_tuple,
        'output_dir':       output_dir,
        'fig_path':         fig_path,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    run_mbll(
        dataset_name = 'expert_novice',
        subject_id   = 'A1',
        verbose      = True,
    )
