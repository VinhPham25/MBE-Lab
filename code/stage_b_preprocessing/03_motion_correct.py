"""
Stage B - Step B3 & B4: Motion Artifact Detection and rPCA Correction
Purpose: Detect motion artifacts on filtered OD data (Homer3 algorithm) and
         correct them using recursive PCA (rPCA).

Task coverage in this file:
  3.3.1 - Adapt Homer3 motion detection algorithm for filtered OD data
  3.3.2 - Generate per-channel, per-sample motion mask; compute % corrupted
  3.3.3 - Visualize detected motion artifacts on OD traces for subject A1
  [3.4.x - rPCA correction will be added in the next section]

Key adaptation from Stage A
---------------------------
Stage A motion detection:
  - Input:  raw intensity (converted internally to OD)
  - Output: motion_ratio scalar per channel (QA metric only)

Stage B motion detection (this file):
  - Input:  filtered OD (output of Step B2, 0.01–0.5 Hz)
  - Output: per-sample boolean mask per channel (used for rPCA in Step B4)
  - Bad channels (raw_od_filt.info['bads']) are detected but flagged separately

Reference: PROJECT.md Section 5 Step B3; PROGRESS.md Section 3.3
Homer3 algorithm: hmrR_MotionArtifactByChannel
"""

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
# Load config once at import time — all tunable constants derive from here
# ---------------------------------------------------------------------------
_CONFIG = load_config()
_B3_CFG = _CONFIG.get("stage_b", {}).get("motion_detection", {})

_DEFAULT_AMP_THRESH   = float(_B3_CFG.get("AMPthresh",   5.0))
_DEFAULT_STDEV_THRESH = float(_B3_CFG.get("tStdev",     20.0))
_DEFAULT_T_MOTION     = float(_B3_CFG.get("tMotion",     0.5))
_DEFAULT_T_MASK       = float(_B3_CFG.get("t_mask",      0.5))
_DEFAULT_USE_AMP_ONLY = bool( _B3_CFG.get("use_amp_only", False))

# Import Step B2 pipeline to obtain filtered OD
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bandpass_filtering",
    Path(__file__).parent / "02_bandpass_filtering.py",
)
_b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b2)
run_bandpass_filter = _b2.run_bandpass_filter


# ===========================================================================
# Task 3.3.1 — Homer3 motion detection adapted for filtered OD
# ===========================================================================

def _detect_motion_homer3_single(
    od_signal: np.ndarray,
    sfreq: float,
    amp_thresh: float,
    stdev_thresh: float,
    t_motion: float,
    use_amp_only: bool = True,
) -> np.ndarray:
    """
    Homer3 hmrR_MotionArtifactByChannel algorithm applied to one OD channel.

    Stage B Adaptation Note
    -----------------------
    Stage A applied this on *unfiltered* OD (converted internally from raw
    intensity) and the dual-criterion (AMPthresh=5 + STDEVthresh=20×std)
    worked correctly.

    Stage B applies this on *filtered* OD (0.01–0.5 Hz, Step B2 output).
    After bandpass filtering the derivative std (std_diff) is much smaller
    because the signal changes very slowly between adjacent 25 Hz samples.
    This causes the stdev leg (mc_thresh = std_diff × 20) to shrink to a level
    that flags normal hemodynamic oscillations (~17.4% of A1 samples flagged,
    vs ~0% expected).  Using amplitude-only (use_amp_only=True) avoids this
    false-positive inflation while still catching genuine large-amplitude
    artifacts that survive the bandpass filter.

    Why not Option 1 (raise stdev_thresh)?
    The correct multiplier for filtered OD is not well-defined and would
    require dataset-specific calibration.  The amplitude criterion is
    stable across datasets because it operates in absolute OD units (5.0)
    rather than relative to signal noise.

    Algorithm
    ---------
    1. Sliding-window max absolute OD change over tMotion seconds:
         max_diff[t] = max( |OD[t] - OD[t-lag]| ) for lag in 1..tMotion*fs
    2a. [use_amp_only=True]  flag if  max_diff > amp_thresh
    2b. [use_amp_only=False] flag if  max_diff > stdev_thresh × std(diff(OD))
                                   OR max_diff > amp_thresh  (Homer3 dual-criterion)

    Parameters
    ----------
    od_signal : np.ndarray  shape (n_times,)
    sfreq : float
    amp_thresh : float
        Absolute OD-change threshold.  Homer3 default = 5.0 OD units.
        Applied to the maximum change within the tMotion window.
    stdev_thresh : float
        Derivative-std multiplier.  Only used when use_amp_only=False.
        Homer3 default = 20.0.
    t_motion : float
        Detection window width in seconds (Homer3 default 0.5 s).
    use_amp_only : bool
        True (default for Stage B filtered OD): amplitude criterion only.
        False: Homer3 dual-criterion (suitable for unfiltered OD).

    Returns
    -------
    motion_mask_deriv : np.ndarray  bool, shape (n_times - 1,)
        True at derivative indices where motion is detected.
    """
    od_signal = np.asarray(od_signal, dtype=np.float64)

    # Replace non-finite values with channel median
    if not np.isfinite(od_signal).all():
        finite_vals = od_signal[np.isfinite(od_signal)]
        fill = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
        od_signal = np.nan_to_num(od_signal, nan=fill, posinf=fill, neginf=fill)

    n = len(od_signal)
    if n < 2:
        return np.zeros(max(n - 1, 0), dtype=bool)

    # Sliding-window max absolute OD change
    window = int(np.round(t_motion * sfreq))
    max_diff = np.zeros(n - 1, dtype=np.float64)

    for lag in range(1, window + 1):
        if lag < n:
            abs_change = np.abs(od_signal[lag:] - od_signal[:-lag])
            padded = np.concatenate([np.zeros(lag - 1), abs_change])
            max_diff = np.maximum(max_diff, padded)

    if use_amp_only:
        motion_mask_deriv = max_diff > amp_thresh
    else:
        derivative = np.diff(od_signal)
        std_diff = float(np.std(derivative))
        mc_thresh = std_diff * stdev_thresh
        motion_mask_deriv = (max_diff > mc_thresh) | (max_diff > amp_thresh)

    return motion_mask_deriv

# takes a motion artifact mask that is defined in the derivative domain (i.e., it marks time points where the derivative of a signal indicates motion) (outputted from _detect_motion_homer3_single)
# and expand it into a signal-domain mask (i.e., it marks time points in the original signal that are affected by motion). It also adds a temporal buffer around each detected motion event.
def _expand_mask_to_signal(
    motion_mask_deriv: np.ndarray,
    n_times: int,
    sfreq: float,
    t_mask: float,
) -> np.ndarray:
    """
    Convert derivative-domain motion mask to signal-domain mask and apply
    temporal buffering of ±t_mask seconds around each detected event.

    Homer3 equivalent: tInc(1+bad_inds) = 0 + art_buffer extension.

    Parameters
    ----------
    motion_mask_deriv : np.ndarray  bool, shape (n_times - 1,)
    n_times : int
        Original signal length.
    sfreq : float
    t_mask : float
        Buffer duration in seconds.

    Returns
    -------
    signal_mask : np.ndarray  bool, shape (n_times,)
        True where motion is present (including buffer).
    """
    art_buffer = int(np.round(t_mask * sfreq))

    # Derivative index i → signal index i+1 (Homer3: tInc(1+bad_inds) = 0)
    bad_deriv = np.where(motion_mask_deriv)[0]
    if bad_deriv.size == 0:
        return np.zeros(n_times, dtype=bool)

    bad_signal = bad_deriv + 1  # shift to signal domain

    # Expand by ±art_buffer
    offsets = np.arange(-art_buffer, art_buffer + 1)
    expanded = (bad_signal[:, np.newaxis] + offsets[np.newaxis, :]).ravel()
    expanded = expanded[(expanded >= 0) & (expanded < n_times)]

    signal_mask = np.zeros(n_times, dtype=bool)
    signal_mask[expanded] = True
    return signal_mask


def detect_motion_artifacts(
    raw_od_filt,
    amp_thresh: float = _DEFAULT_AMP_THRESH,
    stdev_thresh: float = _DEFAULT_STDEV_THRESH,
    t_motion: float = _DEFAULT_T_MOTION,
    t_mask: float = _DEFAULT_T_MASK,
    use_amp_only: bool = _DEFAULT_USE_AMP_ONLY,
    verbose: bool = True,
) -> dict:
    """
    Task 3.3.1: Detect motion artifacts on filtered OD data (Step B3).

    Applies the Homer3 algorithm channel-by-channel.  Channels already
    marked bad (from Stage A) are processed but their masks are flagged
    separately so rPCA can skip them.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        Filtered OD data (output of Step B2).
    amp_thresh : float
        Absolute OD-change threshold (Homer3 default 5.0 OD units).
        Catches genuine spikes that survive the 0.01–0.5 Hz bandpass filter.
    stdev_thresh : float
        Derivative-std multiplier (Homer3 default 20.0).
        Only used when use_amp_only=False.
    t_motion : float
        Detection window (seconds, Homer3 default 0.5).
    t_mask : float
        Buffer around detected events (seconds, Homer3 default 0.5).
    use_amp_only : bool
        If True (default), use amplitude criterion only.  See
        _detect_motion_homer3_single docstring for full rationale.
    verbose : bool

    Returns
    -------
    motion_results : dict
        {
          'masks':        dict[str → np.ndarray bool (n_times,)]
                          per-channel signal-domain motion mask
          'params':       dict  (thresholds used)
          'sfreq':        float
          'n_times':      int
          'bad_channels': list[str]  (pre-existing bads from Stage A)
        }
    """
    sfreq = float(raw_od_filt.info['sfreq'])
    data = raw_od_filt.get_data()          # (n_channels, n_times)
    n_times = data.shape[1]
    bad_set = set(raw_od_filt.info['bads'])

    if verbose:
        n_good = len(raw_od_filt.ch_names) - len(bad_set)
        print("\n" + "=" * 70)
        print("MOTION ARTIFACT DETECTION — HOMER3 ALGORITHM (TASK 3.3.1)")
        print("=" * 70)
        print("Adaptation from Stage A:")
        print("  - Input: filtered OD (Stage A used unfiltered OD)")
        print("  - Output: per-sample boolean mask (Stage A output motion ratio only)")
        criterion = ("amplitude only  [use_amp_only=True]"
                     if use_amp_only else
                     "dual: amp + stdev  [use_amp_only=False]")
        print(f"\nParameters:")
        print(f"  AMPthresh  : {amp_thresh} OD units")
        print(f"  Criterion  : {criterion}")
        if not use_amp_only:
            print(f"  STDEVthresh: {stdev_thresh}× derivative std")
        print(f"  tMotion    : {t_motion} s  (detection window)")
        print(f"  tMask      : {t_mask} s  (buffer around events)")
        print(f"  sfreq      : {sfreq:.2f} Hz")
        print(f"\nChannels: {n_good} good + {len(bad_set)} Stage-A bad")

    masks = {}
    for i, ch_name in enumerate(raw_od_filt.ch_names):
        od_sig = data[i, :]

        motion_deriv = _detect_motion_homer3_single(
            od_sig, sfreq, amp_thresh, stdev_thresh, t_motion,
            use_amp_only=use_amp_only,
        )
        signal_mask = _expand_mask_to_signal(motion_deriv, n_times, sfreq, t_mask)
        masks[ch_name] = signal_mask

    if verbose:
        n_any = sum(1 for ch, m in masks.items()
                    if ch not in bad_set and m.any())
        ratios = [
            masks[ch].mean()
            for ch in raw_od_filt.ch_names
            if ch not in bad_set
        ]
        print(f"\nDetection complete:")
        print(f"  Channels with ≥1 flagged sample: {n_any} / "
              f"{len(raw_od_filt.ch_names) - len(bad_set)}")
        print(f"  Motion ratio (good chs) — "
              f"median={np.median(ratios):.4f}  "
              f"mean={np.mean(ratios):.4f}  "
              f"max={np.max(ratios):.4f}")
        print("=" * 70 + "\n")

    return {
        'masks': masks,
        'params': {
            'amp_thresh': amp_thresh,
            'stdev_thresh': stdev_thresh,
            't_motion': t_motion,
            't_mask': t_mask,
        },
        'sfreq': sfreq,
        'n_times': n_times,
        'bad_channels': list(bad_set),
    }


# ===========================================================================
# Task 3.3.2 — Motion mask summary and per-channel statistics
# ===========================================================================

def compute_motion_stats(
    motion_results: dict,
    raw_od_filt,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Task 3.3.2: Compute per-channel motion statistics from the mask dict.

    Returns a DataFrame with motion ratio, flagged-sample count, and a
    summary note (ok / elevated / bad_channel).  Also prints the overall
    percentage of motion-corrupted data across the good-channel set.

    Parameters
    ----------
    motion_results : dict
        Output of detect_motion_artifacts().
    raw_od_filt : mne.io.Raw
        Used for channel names, types, and wavelengths.
    verbose : bool

    Returns
    -------
    stats_df : pd.DataFrame
        Per-channel columns:
        channel | wavelength | channel_type | n_total | n_flagged |
        motion_ratio | pct_corrupted | motion_quality
    """
    masks = motion_results['masks']
    n_times = motion_results['n_times']
    bad_set = set(motion_results['bad_channels'])
    ch_types = raw_od_filt.get_channel_types()

    def _wavelength(ch_name):
        parts = ch_name.split()
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        return 0

    records = []
    for i, ch_name in enumerate(raw_od_filt.ch_names):
        mask = masks[ch_name]
        n_flagged = int(mask.sum())
        ratio = float(n_flagged / n_times) if n_times > 0 else 0.0

        if ch_name in bad_set:
            quality = 'bad_channel'
        elif ratio > 0.5:
            quality = 'high_motion'
        elif ratio > 0.1:
            quality = 'elevated_motion'
        else:
            quality = 'ok'

        records.append({
            'channel': ch_name,
            'wavelength': _wavelength(ch_name),
            'channel_type': ch_types[i],
            'n_total': n_times,
            'n_flagged': n_flagged,
            'motion_ratio': ratio,
            'pct_corrupted': round(ratio * 100, 2),
            'motion_quality': quality,
        })

    stats_df = pd.DataFrame(records)

    if verbose:
        good = stats_df[stats_df['motion_quality'] != 'bad_channel']
        n_ok = int((good['motion_quality'] == 'ok').sum())
        n_elev = int((good['motion_quality'] == 'elevated_motion').sum())
        n_high = int((good['motion_quality'] == 'high_motion').sum())
        n_bad = int((stats_df['motion_quality'] == 'bad_channel').sum())
        overall_pct = float(good['pct_corrupted'].mean())

        print("\n" + "=" * 70)
        print("MOTION MASK SUMMARY (TASK 3.3.2)")
        print("=" * 70)
        print(f"Total channels:          {len(stats_df)}")
        print(f"  ok (≤10% flagged):     {n_ok}")
        print(f"  elevated (10–50%):     {n_elev}")
        print(f"  high motion (>50%):    {n_high}")
        print(f"  bad_channel (Stage A): {n_bad}")
        print(f"\nMean % corrupted (good channels): {overall_pct:.2f}%")
        print(f"Median motion ratio:              "
              f"{good['motion_ratio'].median():.4f}")
        print(f"Max motion ratio:                 "
              f"{good['motion_ratio'].max():.4f}")

        flagged_chs = good[good['motion_quality'] != 'ok']
        if len(flagged_chs) > 0:
            print(f"\nChannels with elevated/high motion ({len(flagged_chs)}):")
            for _, row in flagged_chs.iterrows():
                print(f"  {row['channel']:30s}  "
                      f"{row['pct_corrupted']:5.1f}%  "
                      f"[{row['motion_quality']}]")
        print("=" * 70 + "\n")

    return stats_df


def save_motion_stats(
    stats_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
) -> Path:
    """Save per-channel motion statistics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}_motion_stats.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Saved motion stats: {out_path}")
    return out_path


# ===========================================================================
# Task 3.3.3 — Visualize motion artifacts on OD traces
# ===========================================================================

def plot_motion_artifacts(
    raw_od_filt,
    motion_results: dict,
    stats_df: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    n_good: int = 2,
    n_flagged: int = 2,
    t_window: float = 300.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.3.3: Visualize OD traces with motion-flagged regions highlighted.

    Shows ``n_good`` channels with low/no motion (blue) and ``n_flagged``
    channels with the most detected motion (orange/red).  Flagged segments
    are shaded in red.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        Filtered OD data.
    motion_results : dict
        Output of detect_motion_artifacts().
    stats_df : pd.DataFrame
        Output of compute_motion_stats().
    subject_id : str
    output_dir : Path
    n_good : int
        Good (low-motion) channels to show.
    n_flagged : int
        Most-corrupted channels to show.
    t_window : float
        Duration of signal to display (seconds). Default 300 s (5 min).
    verbose : bool

    Returns
    -------
    fig_path : Path
    """
    sfreq = raw_od_filt.info['sfreq']
    n_plot_samples = int(t_window * sfreq)
    times = raw_od_filt.times[:n_plot_samples]
    data = raw_od_filt.get_data()
    ch_names = raw_od_filt.ch_names
    bad_set = set(motion_results['bad_channels'])
    masks = motion_results['masks']

    # Select channels: low-motion good channels and highest-motion channels
    good_df = stats_df[stats_df['motion_quality'] != 'bad_channel'].copy()
    sorted_by_ratio = good_df.sort_values('motion_ratio')

    # Spread good sample channels evenly
    good_pool = sorted_by_ratio[sorted_by_ratio['motion_quality'] == 'ok']['channel'].tolist()
    step = max(1, len(good_pool) // n_good)
    sample_good = good_pool[::step][:n_good]

    # Pick highest-motion channels (skip channels with 0 flagged samples if possible)
    high_motion = sorted_by_ratio.sort_values('motion_ratio', ascending=False)
    sample_high = high_motion[high_motion['n_flagged'] > 0]['channel'].tolist()[:n_flagged]
    # fallback: just take top n_flagged even if ratio=0
    if len(sample_high) < n_flagged:
        sample_high = high_motion['channel'].tolist()[:n_flagged]

    plot_channels = sample_good + sample_high
    n_plots = len(plot_channels)

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, ch_name in zip(axes, plot_channels):
        ch_idx = ch_names.index(ch_name)
        seg = data[ch_idx, :n_plot_samples]
        mask = masks[ch_name][:n_plot_samples]
        row = stats_df[stats_df['channel'] == ch_name].iloc[0]

        is_bad = ch_name in bad_set
        is_high = row['motion_quality'] in ('elevated_motion', 'high_motion')

        if is_bad:
            line_color, bg = '#d62728', '#ffcccc'
            label_tag = f'[BAD (Stage A) | {row["pct_corrupted"]:.1f}% flagged]'
        elif is_high:
            line_color, bg = '#ff7f0e', '#fff0cc'
            label_tag = f'[{row["motion_quality"]} | {row["pct_corrupted"]:.1f}% flagged]'
        else:
            line_color, bg = '#1f77b4', '#ffffff'
            label_tag = f'[ok | {row["pct_corrupted"]:.1f}% flagged]'

        ax.set_facecolor(bg)
        ax.plot(times, seg, color=line_color, lw=0.7, alpha=0.85)

        # Shade flagged regions
        if mask.any():
            _shade_mask(ax, times, mask, color='red', alpha=0.25, label='Motion flagged')

        ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.3)
        ax.set_ylabel('OD', fontsize=8)
        ax.set_title(f'{ch_name}  {label_tag}', fontsize=8, loc='left')

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(
        f'Subject {subject_id} — Motion Artifact Detection on Filtered OD\n'
        f'(first {t_window:.0f} s; red shading = flagged regions)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_motion_artifacts.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved motion artifact plot: {fig_path}")

    return fig_path


def _shade_mask(ax, times, mask, color='red', alpha=0.25, label='flagged'):
    """Shade contiguous True blocks in mask as filled regions on ax."""
    in_block = False
    t_start = None
    label_used = False

    for i, flagged in enumerate(mask):
        if flagged and not in_block:
            in_block = True
            t_start = times[i]
        elif not flagged and in_block:
            in_block = False
            kw = {'label': label} if not label_used else {}
            ax.axvspan(t_start, times[i - 1], color=color, alpha=alpha, **kw)
            label_used = True

    if in_block:
        kw = {'label': label} if not label_used else {}
        ax.axvspan(t_start, times[-1], color=color, alpha=alpha, **kw)


# ===========================================================================
# End-to-end pipeline — Tasks 3.3.1–3.3.3
# ===========================================================================

def run_motion_detection(
    dataset_name: str,
    subject_id: str,
    verbose: bool = True,
) -> dict:
    """
    Run Step B3 motion detection pipeline (Tasks 3.3.1–3.3.3) for one subject.

    Steps:
    1. Run Steps B1+B2 to obtain filtered OD object
    2. Detect motion artifacts (3.3.1)
    3. Compute motion statistics + mask summary (3.3.2)
    4. Visualize motion artifacts on OD traces (3.3.3)

    Parameters
    ----------
    dataset_name : str
    subject_id : str
    verbose : bool

    Returns
    -------
    results : dict
        {
          'raw_od_filt':    mne.io.Raw (filtered OD),
          'motion_results': dict (masks + params),
          'stats_df':       pd.DataFrame (per-channel stats),
          'fig_path':       Path,
          'csv_path':       Path,
        }
    """
    # Motion detection criterion is configured via config.yaml
    # stage_b.motion_detection.use_amp_only:
    #   false (default) → Homer3 dual criterion (AMPthresh + STDEVthresh × std_diff)
    #   true            → Amplitude-only (AMPthresh OD units)
    # To change the method, edit config.yaml and re-run.

    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B3: MOTION DETECTION  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Steps B1+B2: filtered OD ------------------------------------------
    b2_results = run_bandpass_filter(dataset_name, subject_id, verbose=False)
    raw_od_filt = b2_results['raw_od_filt']

    # ---- 3.3.1 Detect motion artifacts -------------------------------------
    motion_results = detect_motion_artifacts(
        raw_od_filt,
        verbose=verbose,
    )

    # ---- 3.3.2 Compute stats -----------------------------------------------
    stats_df = compute_motion_stats(motion_results, raw_od_filt, verbose=verbose)

    # ---- Outputs directory -------------------------------------------------
    output_dir = get_output_path('stage_b', f'motion_correction/{subject_id}')

    csv_path = save_motion_stats(stats_df, subject_id, output_dir)

    # ---- 3.3.3 Visualize ---------------------------------------------------
    fig_path = plot_motion_artifacts(
        raw_od_filt,
        motion_results,
        stats_df,
        subject_id=subject_id,
        output_dir=output_dir,
        verbose=verbose,
    )

    # ---- Summary -----------------------------------------------------------
    good = stats_df[stats_df['motion_quality'] != 'bad_channel']
    print("\n" + "=" * 70)
    print(f"✅ Step B3 complete for subject {subject_id}")
    print(f"  Channels ok:           {(good['motion_quality'] == 'ok').sum()}")
    print(f"  Channels elevated:     {(good['motion_quality'] == 'elevated_motion').sum()}")
    print(f"  Channels high motion:  {(good['motion_quality'] == 'high_motion').sum()}")
    print(f"  Mean % corrupted:      {good['pct_corrupted'].mean():.2f}%")
    print(f"  Outputs saved: {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od_filt': raw_od_filt,
        'motion_results': motion_results,
        'stats_df': stats_df,
        'fig_path': fig_path,
        'csv_path': csv_path,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    run_motion_detection(
        dataset_name='expert_novice',
        subject_id='A1',
        verbose=True,
    )
