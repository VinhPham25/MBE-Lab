"""
Stage B — Step B6: Short-Channel Regression (OLS)
Purpose: Remove scalp hemodynamics from long-separation fNIRS channels by
         regressing out the most-correlated short-separation channel signal
         (Ordinary Least Squares, per-channel).

Task coverage:
  3.6.1 — OLS regression: most-correlated short-ch selection + beta estimation
  3.6.2 — Channel count verification (25 long + 8 short for Expert/Novice)
  3.6.3 — Before/after comparison plots (Before = After + Removed)
  3.6.4 — Correlation reduction metrics (long–short and HbO/HbR pair correlations)
  3.6.5 — End-to-end test on subject A1

Algorithm
---------
Short-channel regression (Saager & Berger 2008; Scholkmann et al. 2014):

  For each long-separation channel L of type T (HbO or HbR):
    1. Collect all short-separation channels {S_i} of the same type T.
        - Ex: for a long HbO channel, the lagorithm only looks at the 8 short HbO channels available
    2. Compute Pearson r(L, S_i) for each S_i; select S* = argmax |r|.
        - Not all short channels are the same because of where they sit on the head.
        - By calculating the pearson coefficient between the long channel and every short channel, it finds the short channel that is most correlated with the noise in the long channel.
    3. Estimate OLS slope:
          β = Cov(L, S*) / Var(S*)
          - Cov(L, S*) = covariance between the long channel and the most correlated short channel
          - Var(S*) = variance of the most correlated short channel
          - After identifying the most correlated short channel, we need to scale that noise before subtracting it
          - β = scaling factor; for every 1 unit of change in the short channel, how many units of change in the long channel are there?
            - Ex: β = 1.2 means the noise in the long channel is 1.2 times the noise in the short channel
    4. Subtract scalp noise:
          L_corrected = L − β · S*
          - Subtract the scaled noise in the short channel from the long channel to get the corrected long channel

  Short channels are NOT corrected — they serve as scalp-noise references only.

Rationale: Short-separation channels (< 15 mm source-detector distance) sample
primarily scalp hemodynamics (cardiac, respiratory, vasomotor oscillations).
Long channels pick up brain activation plus the same scalp fluctuations.  OLS
regression removes the shared scalp component while preserving brain signal.

Short-channel threshold : 15 mm (config: stage_b.short_channel_regression.short_separation_threshold)
Upstream               : 05_mbll.py  (output: raw_haemo, haemo .fif)
Reference              : PROGRESS.md Section 3.6; PROJECT.md Section 5.7
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

# ---------------------------------------------------------------------------
# Import Step B5 pipeline via importlib
# ---------------------------------------------------------------------------
import importlib.util

_spec_b5 = importlib.util.spec_from_file_location(
    "mbll",
    Path(__file__).parent / "05_mbll.py",
)
_b5 = importlib.util.module_from_spec(_spec_b5)
_spec_b5.loader.exec_module(_b5)

run_mbll = _b5.run_mbll

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHORT_SEP_THRESHOLD_MM = 15.0   # channels with separation < 15 mm = short-separation


# ===========================================================================
# Task 3.6.2 — Channel separation classification & count verification
# ===========================================================================

def identify_channel_separation(
    raw_haemo,
    short_threshold_mm: float = SHORT_SEP_THRESHOLD_MM,
    expected_short_per_type: int = 8,
    expected_long_per_type: int = 25,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.6.2: Classify haemo channels as short or long by source-detector
    distance and verify expected counts for the dataset.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Hemoglobin data (hbo/hbr channels).
    short_threshold_mm : float
        Channels with separation < this value (mm) are classified as short.
    expected_short_per_type : int
        Expected number of short channels per channel type (default 8 for
        Expert/Novice TechEn CW6 — dedicated short-separation detectors D17-D24).
    expected_long_per_type : int
        Expected number of long channels per channel type (default 25).
    verbose : bool

    Returns
    -------
    short_idx : list[int]
        Indices of short-separation channels.
    long_idx : list[int]
        Indices of long-separation channels.
    dists_mm : np.ndarray, shape (n_channels,)
        Source-detector distances in mm.
    count_summary : dict
        Per-type counts and verification flags.
    """
    dists_m  = mne.preprocessing.nirs.source_detector_distances(raw_haemo.info)
    dists_mm = dists_m * 1000.0

    short_idx = [i for i, d in enumerate(dists_mm) if d < short_threshold_mm]
    long_idx  = [i for i, d in enumerate(dists_mm) if d >= short_threshold_mm]

    ch_types  = raw_haemo.get_channel_types()
    ch_names  = raw_haemo.ch_names

    n_short_hbo = sum(1 for i in short_idx if ch_types[i] == 'hbo')
    n_short_hbr = sum(1 for i in short_idx if ch_types[i] == 'hbr')
    n_long_hbo  = sum(1 for i in long_idx  if ch_types[i] == 'hbo')
    n_long_hbr  = sum(1 for i in long_idx  if ch_types[i] == 'hbr')

    ok_short = (n_short_hbo == expected_short_per_type and
                n_short_hbr == expected_short_per_type)
    ok_long  = (n_long_hbo  == expected_long_per_type  and
                n_long_hbr  == expected_long_per_type)

    count_summary = {
        'threshold_mm':        short_threshold_mm,
        'n_short_hbo':         n_short_hbo,
        'n_short_hbr':         n_short_hbr,
        'n_long_hbo':          n_long_hbo,
        'n_long_hbr':          n_long_hbr,
        'n_total':             len(dists_mm),
        'expected_short_per_type': expected_short_per_type,
        'expected_long_per_type':  expected_long_per_type,
        'count_ok':            ok_short and ok_long,
        'short_channels_hbo':  [ch_names[i] for i in short_idx if ch_types[i] == 'hbo'],
        'short_channels_hbr':  [ch_names[i] for i in short_idx if ch_types[i] == 'hbr'],
    }

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.6.2 — CHANNEL SEPARATION VERIFICATION")
        print("=" * 70)
        print(f"  Short-separation threshold : < {short_threshold_mm:.0f} mm")
        print()
        print(f"  {'Type':5}  {'Short':>7}  {'Long':>7}  {'Total':>7}")
        print(f"  {'-'*32}")
        print(f"  {'HbO':5}  {n_short_hbo:>7d}  {n_long_hbo:>7d}  "
              f"{n_short_hbo + n_long_hbo:>7d}")
        print(f"  {'HbR':5}  {n_short_hbr:>7d}  {n_long_hbr:>7d}  "
              f"{n_short_hbr + n_long_hbr:>7d}")
        print()

        print(f"  Expected (Expert/Novice config): "
              f"{expected_short_per_type} short / {expected_long_per_type} long per type")

        if ok_short and ok_long:
            print(f"  ✅ Channel counts match expected configuration.")
        else:
            if not ok_short:
                note = (
                    f"  ⚠  Short channels: found {n_short_hbo} HbO / {n_short_hbr} HbR, "
                    f"expected {expected_short_per_type}/type.\n"
                    f"     Note: With threshold < {short_threshold_mm:.0f} mm, one additional "
                    f"regular channel\n"
                    f"     (S8_D9 at ~10 mm) is included alongside the 8 dedicated\n"
                    f"     short-separation detectors (D17–D24, 5–8 mm). This channel\n"
                    f"     measures primarily scalp signal and is correctly treated as\n"
                    f"     short-separation for regression."
                )
                print(note)
            if not ok_long:
                print(f"  ⚠  Long channels: found {n_long_hbo}/{n_long_hbr}, "
                      f"expected {expected_long_per_type}/type.")

        short_names = sorted(set(ch_names[i].replace(' hbo', '').replace(' hbr', '')
                                 for i in short_idx))
        print(f"\n  Short-separation source-detector pairs:")
        for name in short_names:
            idx = next(i for i in short_idx if ch_names[i].startswith(name))
            print(f"    {name:<12}  {dists_mm[idx]:5.1f} mm")

        print("=" * 70 + "\n")

    return short_idx, long_idx, dists_mm, count_summary


# ===========================================================================
# Task 3.6.1 — OLS short-channel regression
# ===========================================================================

def apply_ols_regression(
    raw_haemo,
    short_threshold_mm: float = SHORT_SEP_THRESHOLD_MM,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.6.1: Apply OLS short-channel regression to all long-separation channels.

    For each long channel L:
      1. Select all short channels of the same type (HbO or HbR).
      2. Find S* = argmax |Pearson r(L, S_i)|.
      3. Compute β = Cov(L, S*) / Var(S*)  (OLS slope).
      4. Corrected signal: L_corrected = L − β · S*.

    Short channels are copied unchanged into the output Raw object.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Hemoglobin data (Step B5 output). Units: mol/L.
    short_threshold_mm : float
        Source-detector separation threshold (mm).
    verbose : bool

    Returns
    -------
    raw_corrected : mne.io.Raw
        Deep copy of raw_haemo with long channels OLS-corrected.
    reg_results : pd.DataFrame
        Per-long-channel results:
          channel, ch_type, dist_mm, best_short_ch, pearson_r_before,
          pearson_r_after, beta, var_removed_frac.
    """
    short_idx, long_idx, dists_mm, count_summary = identify_channel_separation(
        raw_haemo,
        short_threshold_mm=short_threshold_mm,
        verbose=verbose,
    )

    data      = raw_haemo.get_data().copy()   # (n_ch, n_times)
    ch_names  = raw_haemo.ch_names
    ch_types  = raw_haemo.get_channel_types()

    # Short-channel lookup by type
    short_hbo = [(i, ch_names[i]) for i in short_idx if ch_types[i] == 'hbo']
    short_hbr = [(i, ch_names[i]) for i in short_idx if ch_types[i] == 'hbr']
    short_by_type = {'hbo': short_hbo, 'hbr': short_hbr}

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.6.1 — OLS SHORT-CHANNEL REGRESSION")
        print("=" * 70)
        n_long_hbo = sum(1 for i in long_idx if ch_types[i] == 'hbo')
        n_long_hbr = sum(1 for i in long_idx if ch_types[i] == 'hbr')
        print(f"  Correcting {n_long_hbo} HbO + {n_long_hbr} HbR long channels")
        print(f"  Using {len(short_hbo)} HbO + {len(short_hbr)} HbR short channels")
        print(f"  Method: OLS  |  β = Cov(L, S*) / Var(S*)")
        print(f"  Correction:  L_corrected = L − β · S*")
        print()

    records = []

    for li in long_idx:
        L_sig   = data[li]
        ct      = ch_types[li]
        sc_list = short_by_type.get(ct, [])

        if not sc_list:
            records.append({
                'channel':          ch_names[li],
                'ch_type':          ct,
                'dist_mm':          round(float(dists_mm[li]), 2),
                'best_short_ch':    None,
                'pearson_r_before': np.nan,
                'pearson_r_after':  np.nan,
                'beta':             np.nan,
                'var_removed_frac': np.nan,
            })
            continue

        L_sig = L_sig.ravel()   # ensure 1D

        # Step 2: find most correlated short channel (by |r|)
        # Skip short channels with zero variance (bad/flat channels)
        best_r, best_si, best_sname = 0.0, None, None
        for si, sname in sc_list:
            S_cand = data[si].ravel()
            if np.std(S_cand, ddof=1) < 1e-15:   # flat / bad channel — skip
                continue
            r, _ = stats.pearsonr(L_sig.ravel(), S_cand)
            if np.isnan(r):
                continue
            if abs(r) > abs(best_r):
                best_r, best_si, best_sname = r, si, sname

        # No valid (non-flat) short channel found for this type → skip regression
        if best_si is None:
            records.append({
                'channel':          ch_names[li],
                'ch_type':          ct,
                'dist_mm':          round(float(dists_mm[li]), 2),
                'best_short_ch':    None,
                'pearson_r_before': np.nan,
                'pearson_r_after':  np.nan,
                'beta':             np.nan,
                'var_removed_frac': np.nan,
            })
            continue

        S_sig = data[best_si].ravel()
        L_sig = L_sig.ravel()

        # Step 3: OLS slope β = Cov(L, S*) / Var(S*)
        # Equivalent: β = r · std(L) / std(S)  (avoids np.cov 2D broadcasting issues)
        std_L = np.std(L_sig, ddof=1)
        std_S = np.std(S_sig, ddof=1)
        beta  = best_r * (std_L / std_S) if std_S > 0 else 0.0

        # Step 4: corrected signal
        L_corr = L_sig - beta * S_sig

        # Post-regression correlation (should approach 0 with the short channel)
        r_after, _ = stats.pearsonr(L_corr, S_sig)

        # Fraction of variance removed
        var_before   = np.var(L_sig,  ddof=1)
        var_after_ch = np.var(L_corr, ddof=1)
        var_removed  = max(0.0, (var_before - var_after_ch) / var_before)

        data[li] = L_corr

        records.append({
            'channel':          ch_names[li],
            'ch_type':          ct,
            'dist_mm':          round(float(dists_mm[li]), 2),
            'best_short_ch':    best_sname,
            'pearson_r_before': round(float(best_r), 4),
            'pearson_r_after':  round(float(r_after), 4),
            'beta':             round(float(beta), 6),
            'var_removed_frac': round(float(var_removed), 4),
        })

    reg_results = pd.DataFrame(records)

    # Build corrected Raw object (deep copy, replace data matrix)
    raw_corrected = raw_haemo.copy()
    raw_corrected._data = data

    if verbose:
        hbo_df = reg_results[reg_results['ch_type'] == 'hbo'].dropna(subset=['beta'])
        hbr_df = reg_results[reg_results['ch_type'] == 'hbr'].dropna(subset=['beta'])
        print(f"  {'Type':5}  {'|r| before':>12}  {'|r| after':>10}  "
              f"{'mean β':>10}  {'var removed':>12}")
        print(f"  {'-'*55}")
        for label, df in [('HbO', hbo_df), ('HbR', hbr_df)]:
            print(
                f"  {label:5}  "
                f"{df['pearson_r_before'].abs().mean():>12.4f}  "
                f"{df['pearson_r_after'].abs().mean():>10.4f}  "
                f"{df['beta'].abs().mean():>10.4f}  "
                f"{df['var_removed_frac'].mean() * 100:>11.1f}%"
            )
        print("\n  ✅ OLS regression complete.")
        print("=" * 70 + "\n")

    return raw_corrected, reg_results


# ===========================================================================
# Task 3.6.3 — Before/after comparison plots
# ===========================================================================

def plot_before_after_regression(
    raw_haemo,
    raw_corrected,
    reg_results: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    n_channels: int = 4,
    t_window: float = 300.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.6.3: Before/after OLS regression comparison visualization.

    Layout: one column per representative long channel (HbO).
    Each column has 3 rows:
      Row 1 — Before regression  (HbO only, red)
      Row 2 — After regression   (corrected HbO, dark red)
      Row 3 — Removed noise      (Before − After = β · S*)

    The matched short channel is overlaid on Row 1 and Row 3 (blue, secondary axis)
    to show the noise reference.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Uncorrected hemoglobin data.
    raw_corrected : mne.io.Raw
        OLS-corrected hemoglobin data.
    reg_results : pd.DataFrame
        Output of apply_ols_regression().
    subject_id : str
    output_dir : Path
    n_channels : int
        Number of long HbO channels to plot (evenly sampled by |r|).
    t_window : float
        Time (s) of data to display.
    verbose : bool

    Returns
    -------
    fig_path : Path
    """
    sfreq     = raw_haemo.info['sfreq']
    n_samples = int(min(t_window * sfreq, raw_haemo.n_times))
    times     = raw_haemo.times[:n_samples]

    data_before = raw_haemo.get_data()
    data_after  = raw_corrected.get_data()
    ch_names    = raw_haemo.ch_names

    # Select representative long HbO channels (range of |r_before| magnitudes)
    hbo_long = (
        reg_results[reg_results['ch_type'] == 'hbo']
        .dropna(subset=['best_short_ch'])
        .sort_values('pearson_r_before', key=abs, ascending=False)
    )

    if len(hbo_long) == 0:
        print("  ⚠  No HbO long channels with regression results — skipping plot.")
        return None

    # Sample channels: high-r, mid-r, and low-r cases
    n_plot = min(n_channels, len(hbo_long))
    indices = np.round(np.linspace(0, len(hbo_long) - 1, n_plot)).astype(int)
    selected = hbo_long.iloc[indices]

    fig, axes = plt.subplots(
        nrows=3, ncols=n_plot,
        figsize=(5 * n_plot, 9),
        sharey='row',
    )
    if n_plot == 1:
        axes = axes.reshape(3, 1)

    row_labels = ['Before Regression', 'After Regression', 'Removed Noise (β · S*)']
    row_colors = ['#d62728', '#8c1a1a', '#1f77b4']   # red, dark red, blue

    for col, (_, row) in enumerate(selected.iterrows()):
        ch_name   = row['channel']
        short_ch  = row['best_short_ch']
        r_before  = row['pearson_r_before']
        r_after   = row['pearson_r_after']
        beta      = row['beta']
        var_rem   = row['var_removed_frac'] * 100

        ch_idx      = ch_names.index(ch_name)
        short_idx_v = ch_names.index(short_ch) if short_ch else None

        sig_before   = data_before[ch_idx, :n_samples] * 1e6   # mol/L → μM
        sig_after    = data_after[ch_idx,  :n_samples] * 1e6
        sig_removed  = sig_before - sig_after                   # = β · S*

        axes[0, col].set_title(
            f'{ch_name}\nr={r_before:.3f}  |  β={beta:.4f}',
            fontsize=9,
        )

        for ax_row, sig, color, label in zip(
            [0, 1, 2],
            [sig_before, sig_after, sig_removed],
            row_colors,
            row_labels,
        ):
            ax = axes[ax_row, col]
            ax.plot(times, sig, color=color, lw=0.7, label=ch_name)

            # Overlay matched short channel on Before and Removed rows
            if short_idx_v is not None and ax_row != 1:
                s_sig = data_before[short_idx_v, :n_samples] * 1e6
                ax2 = ax.twinx()
                ax2.plot(times, s_sig, color='steelblue', lw=0.5,
                         alpha=0.5, label=short_ch)
                ax2.set_ylabel(short_ch, fontsize=6, color='steelblue')
                ax2.tick_params(axis='y', labelsize=6, colors='steelblue')

            if col == 0:
                ax.set_ylabel(f'{row_labels[ax_row]}\n[μM]', fontsize=8)

            ax.tick_params(axis='both', labelsize=7)
            ax.set_xlim([times[0], times[-1]])

        axes[2, col].set_xlabel('Time [s]', fontsize=8)

        # Annotate variance removed on After row
        axes[1, col].annotate(
            f'var removed: {var_rem:.1f}%\nr_after: {r_after:.3f}',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.7),
        )

    fig.suptitle(
        f'Subject {subject_id} — OLS Short-Channel Regression: Before vs After\n'
        f'(Before = After + Removed Noise)',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    fig_path = output_dir / f'{subject_id}_ols_before_after.png'
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved before/after plot : {fig_path}")

    return fig_path


# ===========================================================================
# Task 3.6.4 — Correlation reduction verification
# ===========================================================================

def calculate_correlation_reduction(
    raw_haemo,
    raw_corrected,
    reg_results: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    verbose: bool = True,
) -> tuple:
    """
    Task 3.6.4: Quantify how well OLS regression reduced scalp noise.

    Two complementary metrics:

    (A) Long–short channel correlation
        r(L, S*) before → r(L_corrected, S*) after.
        Successful regression drives |r_after| toward 0.

    (B) HbO / HbR intra-channel correlation
        r(HbO_i, HbR_i) for each long channel pair, before vs after.
        Scalp noise inflates this correlation (both go in same direction).
        After regression, brain signal dominates: expect more negative r
        (inverse HbO/HbR relationship), target range ≈ −0.5 to −1.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
    raw_corrected : mne.io.Raw
    reg_results : pd.DataFrame
    subject_id : str
    output_dir : Path
    verbose : bool

    Returns
    -------
    corr_df : pd.DataFrame
        Per long-channel: channel, ch_type, r_long_short_before,
        r_long_short_after, hbo_hbr_r_before, hbo_hbr_r_after.
    fig_path : Path
    """
    data_before = raw_haemo.get_data()
    data_after  = raw_corrected.get_data()
    ch_names    = raw_haemo.ch_names
    ch_types    = raw_haemo.get_channel_types()

    # Build HbO/HbR pairing by base channel name
    def _base(ch):
        return ch.replace(' hbo', '').replace(' hbr', '')

    hbo_base = {_base(ch_names[i]): i
                for i, t in enumerate(ch_types) if t == 'hbo'}
    hbr_base = {_base(ch_names[i]): i
                for i, t in enumerate(ch_types) if t == 'hbr'}

    records = []
    for _, row in reg_results[reg_results['best_short_ch'].notna()].iterrows():
        ch      = row['channel']
        ct      = row['ch_type']
        base    = _base(ch)
        short_n = row['best_short_ch']

        if short_n not in ch_names:
            continue

        si      = ch_names.index(ch)
        short_i = ch_names.index(short_n)

        r_ls_bef = row['pearson_r_before']
        r_ls_aft = row['pearson_r_after']

        # HbO/HbR pair correlation
        hbo_hbr_before = np.nan
        hbo_hbr_after  = np.nan
        if ct == 'hbo' and base in hbr_base:
            hbr_i = hbr_base[base]
            r1, _ = stats.pearsonr(data_before[si], data_before[hbr_i])
            r2, _ = stats.pearsonr(data_after[si],  data_after[hbr_i])
            hbo_hbr_before = round(float(r1), 4)
            hbo_hbr_after  = round(float(r2), 4)

        records.append({
            'channel':            ch,
            'ch_type':            ct,
            'best_short_ch':      short_n,
            'r_long_short_before': round(float(r_ls_bef), 4),
            'r_long_short_after':  round(float(r_ls_aft), 4),
            'hbo_hbr_r_before':   hbo_hbr_before,
            'hbo_hbr_r_after':    hbo_hbr_after,
        })

    corr_df = pd.DataFrame(records)

    # --- Visualization -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Long–short correlation before vs after
    ax = axes[0]
    hbo_corr = corr_df[corr_df['ch_type'] == 'hbo'].dropna(
        subset=['r_long_short_before'])
    hbr_corr = corr_df[corr_df['ch_type'] == 'hbr'].dropna(
        subset=['r_long_short_before'])

    for df_type, color, label in [
        (hbo_corr, '#d62728', 'HbO'),
        (hbr_corr, '#1f77b4', 'HbR'),
    ]:
        x = df_type['r_long_short_before'].values
        y = df_type['r_long_short_after'].values
        ax.scatter(x, y, c=color, alpha=0.7, s=30, label=label)

    lims = [-1, 1]
    ax.plot(lims, lims, 'k--', lw=0.8, label='No change')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('r(Long, Short*)  before regression', fontsize=10)
    ax.set_ylabel('r(Long, Short*)  after regression', fontsize=10)
    ax.set_title('(A) Long–Short Correlation\n(should move toward 0)', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Annotate mean shift
    all_bef = corr_df['r_long_short_before'].abs().mean()
    all_aft = corr_df['r_long_short_after'].abs().mean()
    ax.annotate(
        f'Mean |r| before: {all_bef:.3f}\nMean |r| after:  {all_aft:.3f}',
        xy=(0.04, 0.95), xycoords='axes fraction',
        va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8),
    )

    # Panel B: HbO/HbR pair correlation before vs after
    ax = axes[1]
    pair_df = corr_df[corr_df['ch_type'] == 'hbo'].dropna(
        subset=['hbo_hbr_r_before'])

    ax.scatter(
        pair_df['hbo_hbr_r_before'], pair_df['hbo_hbr_r_after'],
        c='#2ca02c', alpha=0.7, s=40, label='HbO/HbR pairs'
    )
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='No change')
    ax.axhline(0,    color='gray',      lw=0.5)
    ax.axhline(-0.5, color='darkgreen', lw=0.8, ls=':', label='Target ≥ −0.5')
    ax.axhline(-1.0, color='darkgreen', lw=0.8, ls='--', label='Max −1.0')
    ax.set_xlabel('r(HbO, HbR)  before regression', fontsize=10)
    ax.set_ylabel('r(HbO, HbR)  after regression',  fontsize=10)
    ax.set_title(
        '(B) HbO / HbR Intra-channel Correlation\n'
        '(more negative = brain signal dominates)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])

    mean_bef = pair_df['hbo_hbr_r_before'].mean()
    mean_aft = pair_df['hbo_hbr_r_after'].mean()
    target_ok = mean_aft <= -0.5
    status_str = '✅ in target range' if target_ok else '⚠ above target'
    ax.annotate(
        f'Mean r before: {mean_bef:.3f}\nMean r after:  {mean_aft:.3f}  {status_str}',
        xy=(0.04, 0.04), xycoords='axes fraction',
        va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.8),
    )

    fig.suptitle(
        f'Subject {subject_id} — OLS Regression Effectiveness',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    fig_path = output_dir / f'{subject_id}_ols_correlation_reduction.png'
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print("\n" + "=" * 70)
        print("TASK 3.6.4 — REGRESSION EFFECTIVENESS (CORRELATION REDUCTION)")
        print("=" * 70)

        print("\n  (A) Long–Short Channel Correlation (target: |r| → 0)")
        print(f"  Mean |r| before : {all_bef:.4f}")
        print(f"  Mean |r| after  : {all_aft:.4f}")
        print(f"  Reduction       : {(all_bef - all_aft) / all_bef * 100:.1f}%")

        print("\n  (B) HbO / HbR Intra-channel Correlation (target: ≤ −0.5)")
        print(f"  Mean r before   : {mean_bef:.4f}")
        print(f"  Mean r after    : {mean_aft:.4f}")

        target_ch = (pair_df['hbo_hbr_r_after'] <= -0.5).sum()
        print(f"  Channels in target range (r ≤ −0.5): "
              f"{target_ch}/{len(pair_df)}")
        if target_ok:
            print(f"  ✅ Mean HbO/HbR correlation ({mean_aft:.3f}) is in target range "
                  f"[−0.5, −1.0]")
        else:
            print(f"  ⚠  Mean HbO/HbR correlation ({mean_aft:.3f}) is above −0.5 target.")
            print(f"     Regression still reduces scalp noise; physiological inverse")
            print(f"     relationship may be partially masked by remaining noise.")

        print(f"\n  Saved correlation plot : {fig_path}")
        print("=" * 70 + "\n")

    return corr_df, fig_path


# ===========================================================================
# Save outputs
# ===========================================================================

def save_regression_data(
    raw_corrected,
    reg_results: pd.DataFrame,
    corr_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    verbose: bool = True,
) -> tuple:
    """
    Save OLS regression outputs:
      - <subject>_ols_haemo_raw.fif   : corrected hemoglobin data
      - <subject>_ols_reg_results.csv : per-channel regression metadata
      - <subject>_ols_corr_reduction.csv : correlation before/after
      - <subject>_ols_summary.json    : summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Corrected haemo .fif
    fif_path = output_dir / f'{subject_id}_ols_haemo_raw.fif'
    raw_corrected.save(str(fif_path), overwrite=True, verbose=False)

    # Regression metadata CSV
    reg_csv = output_dir / f'{subject_id}_ols_reg_results.csv'
    reg_results.to_csv(str(reg_csv), index=False)

    # Correlation reduction CSV
    corr_csv = output_dir / f'{subject_id}_ols_corr_reduction.csv'
    corr_df.to_csv(str(corr_csv), index=False)

    # Summary JSON
    hbo_r = reg_results[reg_results['ch_type'] == 'hbo'].dropna(subset=['beta'])
    hbr_r = reg_results[reg_results['ch_type'] == 'hbr'].dropna(subset=['beta'])
    pair_df = corr_df[corr_df['ch_type'] == 'hbo'].dropna(
        subset=['hbo_hbr_r_before'])

    summary = {
        'n_long_channels_corrected': len(reg_results.dropna(subset=['beta'])),
        'n_short_channels_used': reg_results['best_short_ch'].nunique(),
        'HbO_regression': {
            'mean_abs_r_before':    round(float(hbo_r['pearson_r_before'].abs().mean()), 4),
            'mean_abs_r_after':     round(float(hbo_r['pearson_r_after'].abs().mean()),  4),
            'mean_beta':            round(float(hbo_r['beta'].mean()), 6),
            'mean_var_removed_pct': round(float(hbo_r['var_removed_frac'].mean() * 100), 2),
        },
        'HbR_regression': {
            'mean_abs_r_before':    round(float(hbr_r['pearson_r_before'].abs().mean()), 4),
            'mean_abs_r_after':     round(float(hbr_r['pearson_r_after'].abs().mean()),  4),
            'mean_beta':            round(float(hbr_r['beta'].mean()), 6),
            'mean_var_removed_pct': round(float(hbr_r['var_removed_frac'].mean() * 100), 2),
        },
        'HbO_HbR_correlation': {
            'mean_r_before': round(float(pair_df['hbo_hbr_r_before'].mean()), 4),
            'mean_r_after':  round(float(pair_df['hbo_hbr_r_after'].mean()),  4),
            'n_in_target_range': int((pair_df['hbo_hbr_r_after'] <= -0.5).sum()),
            'n_total_pairs':     int(len(pair_df)),
        },
    }

    json_path = output_dir / f'{subject_id}_ols_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n  Saved .fif             : {fif_path}")
        print(f"  Saved reg results CSV  : {reg_csv}")
        print(f"  Saved corr reduction   : {corr_csv}")
        print(f"  Saved OLS summary JSON : {json_path}")

    return fif_path, reg_csv, corr_csv, json_path


# ===========================================================================
# Task 3.6.5 — End-to-end pipeline
# ===========================================================================

def run_ols_regression(
    dataset_name: str,
    subject_id: str,
    short_threshold_mm: float = SHORT_SEP_THRESHOLD_MM,
    n_components: float = 0.99,
    min_channels_flagged: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Task 3.6.5: Full OLS short-channel regression pipeline for one subject.

    Chain: B1 (OD) → B2 (bandpass) → B3 (motion detect) → B4 (rPCA)
           → B5 (Beer-Lambert) → B6 (OLS regression) → save.

    Parameters
    ----------
    dataset_name : str
    subject_id   : str
    short_threshold_mm : float
        Source-detector separation threshold (mm) for short-channel classification.
    n_components : float
        rPCA variance threshold (passed to B4).
    min_channels_flagged : int
        rPCA vote threshold (passed to B4).
    verbose : bool

    Returns
    -------
    results : dict
        {
          'raw_haemo'      : mne.io.Raw  (uncorrected, from B5),
          'raw_corrected'  : mne.io.Raw  (OLS-corrected),
          'reg_results'    : pd.DataFrame,
          'corr_df'        : pd.DataFrame,
          'output_dir'     : Path,
          'fif_path'       : Path,
          'fig_before_after': Path,
          'fig_corr'       : Path,
        }
    """
    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B6: OLS SHORT-CHANNEL REGRESSION  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Steps B1–B5: MBLL-converted haemo data --------------------------------
    print("\n[1/5] Running upstream pipeline (B1→B2→B3→B4→B5)...")
    b5_results = run_mbll(
        dataset_name,
        subject_id,
        n_components         = n_components,
        min_channels_flagged = min_channels_flagged,
        verbose              = False,
    )
    raw_haemo = b5_results['raw_haemo']

    # ---- 3.6.1 + 3.6.2 OLS regression ----------------------------------------
    print("[2/5] Applying OLS short-channel regression (Tasks 3.6.1, 3.6.2)...")
    raw_corrected, reg_results = apply_ols_regression(
        raw_haemo,
        short_threshold_mm = short_threshold_mm,
        verbose            = verbose,
    )

    # ---- 3.6.3 Before/after visualization -------------------------------------
    output_dir = get_output_path('stage_b', f'ols_regression/{subject_id}')
    print("[3/5] Creating before/after comparison plots (Task 3.6.3)...")
    fig_before_after = plot_before_after_regression(
        raw_haemo, raw_corrected, reg_results,
        subject_id = subject_id,
        output_dir = output_dir,
        n_channels = 4,
        t_window   = 300.0,
        verbose    = verbose,
    )

    # ---- 3.6.4 Correlation reduction ------------------------------------------
    print("[4/5] Calculating correlation reduction (Task 3.6.4)...")
    corr_df, fig_corr = calculate_correlation_reduction(
        raw_haemo, raw_corrected, reg_results,
        subject_id = subject_id,
        output_dir = output_dir,
        verbose    = verbose,
    )

    # ---- Save -----------------------------------------------------------------
    print("[5/5] Saving outputs...")
    fif_path, reg_csv, corr_csv, json_path = save_regression_data(
        raw_corrected, reg_results, corr_df,
        subject_id = subject_id,
        output_dir = output_dir,
        verbose    = verbose,
    )

    # ---- Console summary ------------------------------------------------------
    hbo_r = reg_results[reg_results['ch_type'] == 'hbo'].dropna(subset=['beta'])
    hbr_r = reg_results[reg_results['ch_type'] == 'hbr'].dropna(subset=['beta'])
    pair_df = corr_df[corr_df['ch_type'] == 'hbo'].dropna(
        subset=['hbo_hbr_r_before'])

    print("\n" + "=" * 70)
    print(f"✅ Step B6 complete for subject {subject_id}")
    print(f"\n  Short-separation threshold   : < {short_threshold_mm:.0f} mm")
    print(f"  Long channels corrected      : {len(reg_results.dropna(subset=['beta']))}")
    print(f"\n  HbO  mean |r| before/after   : "
          f"{hbo_r['pearson_r_before'].abs().mean():.4f}  →  "
          f"{hbo_r['pearson_r_after'].abs().mean():.4f}")
    print(f"  HbO  mean var removed        : "
          f"{hbo_r['var_removed_frac'].mean() * 100:.1f}%")
    print(f"\n  HbR  mean |r| before/after   : "
          f"{hbr_r['pearson_r_before'].abs().mean():.4f}  →  "
          f"{hbr_r['pearson_r_after'].abs().mean():.4f}")
    print(f"  HbR  mean var removed        : "
          f"{hbr_r['var_removed_frac'].mean() * 100:.1f}%")
    print(f"\n  HbO/HbR mean r before/after  : "
          f"{pair_df['hbo_hbr_r_before'].mean():.4f}  →  "
          f"{pair_df['hbo_hbr_r_after'].mean():.4f}")
    print(f"  Outputs saved to             : {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_haemo':       raw_haemo,
        'raw_corrected':   raw_corrected,
        'reg_results':     reg_results,
        'corr_df':         corr_df,
        'output_dir':      output_dir,
        'fif_path':        fif_path,
        'fig_before_after': fig_before_after,
        'fig_corr':        fig_corr,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    run_ols_regression(
        dataset_name = 'expert_novice',
        subject_id   = 'A1',
        verbose      = True,
    )
