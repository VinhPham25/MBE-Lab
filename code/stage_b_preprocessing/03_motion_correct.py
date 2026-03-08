"""
Stage B — Step B3: Motion Artifact Detection on OD
Purpose: Detect motion artifacts on OD data using the Homer3
         hmrR_MotionArtifact algorithm and produce per-sample boolean masks
         for use by the B4 rPCA correction step.

Audit Fix (2026-03-05) — Discrepancy 3a resolved
-------------------------------------------------
The `max_diff` sliding-window computation previously placed zero-padding at
the START of each lag's difference array (Python behaviour).  In MATLAB
(hmrR_MotionArtifact / hmrR_MotionArtifactByChannel), zero-padding is placed
at the END, so that each difference |d[j+lag] − d[j]| is assigned to
derivative index j (the start of the comparison window).  This is the
mathematically correct alignment: max_diff[j] measures the maximum signal
change beginning at sample j over the next tMotion seconds.

With the old Python padding, every motion event was time-shifted forward by
(lag − 1) samples.  At the maximum lag of tMotion × sfreq this shift can be
up to tMotion × sfreq − 1 = 11 samples at 25 Hz (0.44 s) — nearly as large
as tMask itself.  The fix is a one-line change:

    # OLD (wrong):  zeros at START
    padded = np.concatenate([np.zeros(lag - 1), abs_change])

    # NEW (correct, MATLAB parity):  zeros at END
    padded = np.concatenate([abs_change, np.zeros(lag - 1)])

All other aspects of the Homer3 algorithm (dual-criterion detection, ±tMask
buffering, derivative→signal index shift) are unchanged.

Key adaptation from Stage A
---------------------------
Stage A motion detection:
  - Input:  raw intensity (converted internally to OD)
  - Output: motion_ratio scalar per channel (QA metric only)

Stage B motion detection (this file):
  - Input:  OD (output of Step B1)
  - Output: per-sample boolean mask per channel (used for rPCA in Step B4)
  - Bad channels (raw_od_filt.info['bads']) are detected but flagged separately

Reference: docs/reproduced_matlab_comparison.md §B3; PROGRESS.md §3.3
Homer3 algorithm: hmrR_MotionArtifact.m
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
from utils import get_output_path, load_config

# ---------------------------------------------------------------------------
# Load config once at import time — all tunable constants derive from here
# ---------------------------------------------------------------------------
_CONFIG = load_config()
_B3_CFG = _CONFIG.get("stage_b", {}).get("motion_detection", {})

_DEFAULT_AMP_THRESH   = float(_B3_CFG.get("AMPthresh",    5.0))
_DEFAULT_STDEV_THRESH = float(_B3_CFG.get("STDEVthresh", 20.0))
_DEFAULT_T_MOTION     = float(_B3_CFG.get("tMotion",      0.5))
_DEFAULT_T_MASK       = float(_B3_CFG.get("t_mask",       0.5))
_DEFAULT_USE_AMP_ONLY = bool( _B3_CFG.get("use_amp_only", False))

# Import Step B1 pipeline to obtain OD
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "intensity_to_od",
    Path(__file__).parent / "01_intensity_to_od.py",
)
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
run_intensity_to_od = _b1.run_intensity_to_od


# ===========================================================================
# Core Homer3 algorithm: single-channel motion detection
# ===========================================================================

def _detect_motion_homer3_single(
    od_signal: np.ndarray,
    sfreq: float,
    amp_thresh: float,
    stdev_thresh: float,
    t_motion: float,
    use_amp_only: bool = False,
) -> np.ndarray:
    """
    Homer3 hmrR_MotionArtifact algorithm applied to a single OD channel.

    Sliding-window max-absolute-change detection with optional dual criterion
    (amplitude AND/OR standard-deviation multiplier).  Matches the MATLAB
    hmrR_MotionArtifact.m loop exactly, including the padding orientation fix
    described in the module docstring (Discrepancy 3a).

    Algorithm
    ---------
    1. Compute sliding-window max absolute OD change for all lags in
       [1 .. round(tMotion × sfreq)]:
           max_diff[j] = max over lag of |OD[j+lag] − OD[j]|
       (MATLAB: zeros padding at END so differences sit at their START index j)
    2a. [use_amp_only=True]  flag if  max_diff > amp_thresh
    2b. [use_amp_only=False] flag if  max_diff > STDEVthresh × std(diff(OD))
                                   OR max_diff > amp_thresh  (Homer3 default)

    Parameters
    ----------
    od_signal : np.ndarray, shape (n_times,)
        Single-channel filtered optical density time series.
    sfreq : float
        Sampling frequency in Hz.
    amp_thresh : float
        Absolute OD-change threshold.  Homer3/ASTaUND default: 5.0 OD units.
        Read from config.yaml → stage_b.motion_detection.AMPthresh.
        A time point is flagged if the signal changes by more than this value
        within the tMotion window regardless of noise level.
    stdev_thresh : float
        Derivative-std multiplier (Homer3 STDEVthresh).  Only used when
        use_amp_only=False.  Homer3/ASTaUND default: 20.0.
        Read from config.yaml → stage_b.motion_detection.STDEVthresh.
        A time point is also flagged if max_diff > stdev_thresh × std(diff(OD)).
    t_motion : float
        Detection window width in seconds.  Homer3 default: 0.5 s.
        Read from config.yaml → stage_b.motion_detection.tMotion.
    use_amp_only : bool
        If True, skip the STDEVthresh criterion and use amplitude only.
        If False (Homer3 default), apply dual criterion (amplitude OR stdev).
        Read from config.yaml → stage_b.motion_detection.use_amp_only.

    Returns
    -------
    motion_mask_deriv : np.ndarray, bool, shape (n_times − 1,)
        True at derivative index j when the signal change starting at sample j
        exceeds the detection threshold.  Convert to signal domain via
        _expand_mask_to_signal().
    """
    od_signal = np.asarray(od_signal, dtype=np.float64)

    # Replace non-finite values with channel median to avoid propagating NaN
    if not np.isfinite(od_signal).all():
        finite_vals = od_signal[np.isfinite(od_signal)]
        fill = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
        od_signal = np.nan_to_num(od_signal, nan=fill, posinf=fill, neginf=fill)

    n = len(od_signal)
    if n < 2:
        return np.zeros(max(n - 1, 0), dtype=bool)

    # ------------------------------------------------------------------
    # Sliding-window max absolute OD change (Homer3 parity)
    # ------------------------------------------------------------------
    # For each lag ii in 1..window, compute |OD[j+ii] − OD[j]| for all valid j
    # and update max_diff[j] = running maximum across lags.
    #
    # MATLAB places the (n-ii) differences at positions 0..n-ii-1 with ii-1
    # zeros at the END.  This ensures max_diff[j] represents the largest change
    # *starting from* sample j (forward-looking).
    #
    # Previous Python code placed zeros at the START (wrong), shifting every
    # detected event forward by up to (window-1) samples.  Fixed below.
    # ------------------------------------------------------------------
    window = int(np.round(t_motion * sfreq))
    max_diff = np.zeros(n - 1, dtype=np.float64)

    for lag in range(1, window + 1):
        if lag < n:
            abs_change = np.abs(od_signal[lag:] - od_signal[:-lag])  # n-lag elements
            # MATLAB parity: differences at START (indices 0..n-lag-1), zeros at END
            padded = np.concatenate([abs_change, np.zeros(lag - 1)])
            max_diff = np.maximum(max_diff, padded)

    # ------------------------------------------------------------------
    # Apply detection threshold (dual criterion or amplitude-only)
    # ------------------------------------------------------------------
    if use_amp_only:
        motion_mask_deriv = max_diff > amp_thresh
    else:
        # Homer3 dual criterion: flag if EITHER condition is met
        derivative = np.diff(od_signal)
        std_diff = float(np.std(derivative))
        mc_thresh = std_diff * stdev_thresh
        motion_mask_deriv = (max_diff > mc_thresh) | (max_diff > amp_thresh)

    return motion_mask_deriv


def _expand_mask_to_signal(
    motion_mask_deriv: np.ndarray,
    n_times: int,
    sfreq: float,
    t_mask: float,
) -> np.ndarray:
    """
    Convert a derivative-domain motion mask to a signal-domain mask and apply
    a temporal buffer of ±t_mask seconds around each detected event.

    Homer3 equivalent: tInc(1 + bad_inds) = 0  +  ±art_buffer expansion.

    The derivative-domain index i corresponds to signal-domain index i+1
    (the end of the change window), matching MATLAB's ``tInc(1+bad_inds)=0``.

    Parameters
    ----------
    motion_mask_deriv : np.ndarray, bool, shape (n_times − 1,)
        Output of _detect_motion_homer3_single().  True at derivative index j
        where motion is detected.
    n_times : int
        Length of the original OD signal.
    sfreq : float
        Sampling frequency in Hz.
    t_mask : float
        Half-width of the temporal buffer around each detected event, in
        seconds.  Homer3/ASTaUND default: 0.5 s.
        Read from config.yaml → stage_b.motion_detection.t_mask.

    Returns
    -------
    signal_mask : np.ndarray, bool, shape (n_times,)
        True at every signal-domain sample that is either directly flagged or
        within ±t_mask seconds of a flagged sample.
    """
    art_buffer = int(np.round(t_mask * sfreq))

    # Derivative index i → signal index i+1 (Homer3: tInc(1+bad_inds) = 0)
    bad_deriv = np.where(motion_mask_deriv)[0]
    if bad_deriv.size == 0:
        return np.zeros(n_times, dtype=bool)

    bad_signal = bad_deriv + 1  # shift to signal domain

    # Expand by ±art_buffer samples
    offsets = np.arange(-art_buffer, art_buffer + 1)
    expanded = (bad_signal[:, np.newaxis] + offsets[np.newaxis, :]).ravel()
    expanded = expanded[(expanded >= 0) & (expanded < n_times)]

    signal_mask = np.zeros(n_times, dtype=bool)
    signal_mask[expanded] = True
    return signal_mask


# ===========================================================================
# Task 3.3.1 — Full-dataset motion detection (all channels)
# ===========================================================================

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
    Task 3.3.1: Detect motion artifacts on OD data (Step B3).

    Applies the Homer3 hmrR_MotionArtifact algorithm channel-by-channel.
    Channels already marked bad (from Stage A) are processed but their masks
    are stored with a ``bad_channel`` flag so that Step B4 rPCA can skip them.

    All parameters default to values read from config.yaml at import time.
    Changing config.yaml and re-running will automatically pick up new values.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        OD data (output of Step B1).
        Should be of type ``fnirs_od``.
    amp_thresh : float, optional
        Absolute OD-change threshold.  Default from config.yaml
        (stage_b.motion_detection.AMPthresh), typically 5.0 OD units.
    stdev_thresh : float, optional
        Derivative-std multiplier (Homer3 STDEVthresh).  Only active when
        use_amp_only=False.  Default from config.yaml
        (stage_b.motion_detection.STDEVthresh), typically 20.0.
    t_motion : float, optional
        Detection window in seconds.  Default from config.yaml
        (stage_b.motion_detection.tMotion), typically 0.5 s.
    t_mask : float, optional
        Temporal buffer around detected events in seconds.  Default from
        config.yaml (stage_b.motion_detection.t_mask), typically 0.5 s.
    use_amp_only : bool, optional
        If True, skip the STDEVthresh criterion (amplitude only).
        If False (Homer3 default), use dual criterion (amp OR stdev).
        Default from config.yaml (stage_b.motion_detection.use_amp_only).
    verbose : bool, optional
        Print detection summary to stdout.

    Returns
    -------
    motion_results : dict
        'masks'        : dict[str → np.ndarray bool (n_times,)]
                         Per-channel signal-domain motion masks.  True = motion.
        'params'       : dict
                         Threshold values actually used during detection.
        'sfreq'        : float
                         Sampling frequency of the input data.
        'n_times'      : int
                         Number of time points in the input data.
        'bad_channels' : list[str]
                         Channels pre-marked bad by Stage A (excluded from B4).
    """
    sfreq = float(raw_od_filt.info['sfreq'])
    data = raw_od_filt.get_data()          # (n_channels, n_times)
    n_times = data.shape[1]
    bad_set = set(raw_od_filt.info['bads'])

    if verbose:
        n_good = len(raw_od_filt.ch_names) - len(bad_set)
        print("\n" + "=" * 70)
        print("MOTION ARTIFACT DETECTION — HOMER3 ALGORITHM (STEP B3)")
        print("  Audit fix applied: max_diff zeros-padding at END (MATLAB parity)")
        print("=" * 70)
        criterion = ("amplitude only  [use_amp_only=True]"
                     if use_amp_only else
                     "dual: amp + stdev  [use_amp_only=False]")
        print(f"\nParameters (from config.yaml stage_b.motion_detection):")
        print(f"  AMPthresh    : {amp_thresh} OD units")
        print(f"  Criterion    : {criterion}")
        if not use_amp_only:
            print(f"  STDEVthresh  : {stdev_thresh}× derivative std")
        print(f"  tMotion      : {t_motion} s  (detection window)")
        print(f"  t_mask       : {t_mask} s  (buffer around events)")
        print(f"  sfreq        : {sfreq:.2f} Hz")
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
            'amp_thresh':    amp_thresh,
            'stdev_thresh':  stdev_thresh,
            't_motion':      t_motion,
            't_mask':        t_mask,
            'use_amp_only':  use_amp_only,
        },
        'sfreq':        sfreq,
        'n_times':      n_times,
        'bad_channels': list(bad_set),
    }


# ===========================================================================
# Task 3.3.2 — Per-channel motion statistics
# ===========================================================================

def compute_motion_stats(
    motion_results: dict,
    raw_od_filt,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Task 3.3.2: Compute per-channel motion statistics from the mask dict.

    Summarises the output of detect_motion_artifacts() into a DataFrame with
    one row per channel, labelling each channel as 'ok', 'elevated_motion',
    'high_motion', or 'bad_channel'.

    Parameters
    ----------
    motion_results : dict
        Output of detect_motion_artifacts().  Must contain keys 'masks',
        'n_times', and 'bad_channels'.
    raw_od_filt : mne.io.Raw
        Used to access channel names and types.
    verbose : bool, optional
        Print summary table and aggregate statistics to stdout.

    Returns
    -------
    stats_df : pd.DataFrame
        One row per channel with columns:
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
            'channel':        ch_name,
            'wavelength':     _wavelength(ch_name),
            'channel_type':   ch_types[i],
            'n_total':        n_times,
            'n_flagged':      n_flagged,
            'motion_ratio':   ratio,
            'pct_corrupted':  round(ratio * 100, 2),
            'motion_quality': quality,
        })

    stats_df = pd.DataFrame(records)

    if verbose:
        good = stats_df[stats_df['motion_quality'] != 'bad_channel']
        n_ok   = int((good['motion_quality'] == 'ok').sum())
        n_elev = int((good['motion_quality'] == 'elevated_motion').sum())
        n_high = int((good['motion_quality'] == 'high_motion').sum())
        n_bad  = int((stats_df['motion_quality'] == 'bad_channel').sum())
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
    """
    Save per-channel motion statistics to a CSV file.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of compute_motion_stats().
    subject_id : str
        Subject identifier used in the output filename.
    output_dir : Path
        Directory where the CSV will be written (created if absent).

    Returns
    -------
    out_path : Path
        Full path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}_motion_stats.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Saved motion stats: {out_path}")
    return out_path


# ===========================================================================
# Task 3.3.3 — Visualise motion artifacts on OD traces
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
    Task 3.3.3: Visualise OD traces with motion-flagged regions highlighted.

    Produces a multi-panel figure showing n_good low-motion channels (blue)
    and n_flagged highest-motion channels (orange/red).  Flagged segments are
    shaded in red.

    Parameters
    ----------
    raw_od_filt : mne.io.Raw
        Filtered OD data (Step B2 output).
    motion_results : dict
        Output of detect_motion_artifacts().
    stats_df : pd.DataFrame
        Output of compute_motion_stats().
    subject_id : str
        Subject identifier used in the figure title and output filename.
    output_dir : Path
        Directory where the PNG will be written (created if absent).
    n_good : int, optional
        Number of low-motion (ok) channels to display.  Default 2.
    n_flagged : int, optional
        Number of highest-motion channels to display.  Default 2.
    t_window : float, optional
        Duration of the displayed signal segment in seconds.  Default 300 s.
    verbose : bool, optional
        Print the saved figure path to stdout.

    Returns
    -------
    fig_path : Path
        Full path to the saved PNG file.
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

    good_pool = sorted_by_ratio[sorted_by_ratio['motion_quality'] == 'ok']['channel'].tolist()
    step = max(1, len(good_pool) // n_good)
    sample_good = good_pool[::step][:n_good]

    high_motion = sorted_by_ratio.sort_values('motion_ratio', ascending=False)
    sample_high = high_motion[high_motion['n_flagged'] > 0]['channel'].tolist()[:n_flagged]
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

        is_bad  = ch_name in bad_set
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
    """
    Shade contiguous True blocks in a boolean mask as filled regions on ax.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    times : np.ndarray
        Time axis (seconds), same length as mask.
    mask : np.ndarray, bool
        Per-sample boolean mask.  True = shade this sample.
    color : str, optional
        Fill colour.  Default 'red'.
    alpha : float, optional
        Fill transparency.  Default 0.25.
    label : str, optional
        Legend label for the first shaded region.
    """
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
    Run the Step B3 motion detection pipeline for one subject.

    Executes Tasks 3.3.1–3.3.3 in sequence:
    1. Runs Step B1 to obtain the OD object (via run_intensity_to_od).
    2. Detects motion artifacts channel-by-channel (3.3.1).
    3. Computes motion statistics and mask summary (3.3.2).
    4. Visualises motion artifacts on OD traces (3.3.3).

    All detection parameters are loaded from config.yaml
    (stage_b.motion_detection).  Editing config.yaml and re-running will
    automatically apply new threshold values.

    Parameters
    ----------
    dataset_name : str
        Dataset key as defined in config.yaml (e.g. 'expert_novice').
    subject_id : str
        Subject identifier (e.g. 'A1', 'R3').
    verbose : bool, optional
        Print step-by-step summaries to stdout.

    Returns
    -------
    results : dict
        'raw_od_filt'    : mne.io.Raw
                           OD data (Step B1 output).
        'motion_results' : dict
                           Per-channel signal-domain motion masks and detection
                           parameters.  Passed directly to Step B4 rPCA.
        'stats_df'       : pd.DataFrame
                           Per-channel motion statistics (motion_ratio, quality).
        'fig_path'       : Path
                           Path to the saved motion visualisation PNG.
        'csv_path'       : Path
                           Path to the saved motion statistics CSV.
    """
    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B3: MOTION DETECTION  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Step B1: OD --------------------------------------------------------
    b1_results = run_intensity_to_od(dataset_name, subject_id, verbose=False)
    raw_od_filt = b1_results['raw_od']

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

    # ---- 3.3.3 Visualise ---------------------------------------------------
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
        'raw_od_filt':    raw_od_filt,
        'motion_results': motion_results,
        'stats_df':       stats_df,
        'fig_path':       fig_path,
        'csv_path':       csv_path,
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
