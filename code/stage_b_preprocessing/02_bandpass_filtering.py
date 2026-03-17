"""
Stage B - Step B2: Bandpass Filtering
Purpose: Apply Homer3-equivalent Butterworth filter to OD data, visualise the
         before/after time series, and verify effectiveness via PSD plots.

Homer3 reference: hmrR_BandpassFilt.m
  - Low-pass:  3rd-order Butterworth, zero-phase filtfilt  (always applied)
  - High-pass: 5th-order Butterworth, zero-phase filtfilt  (only if hpf > 0)
  - Paper/Homer3 default: hpf=0.000, lpf=0.500
    → pure 3rd-order 0.5 Hz low-pass only; NO high-pass applied

Task coverage in this file:
  3.2.1 - Implement Homer3-equivalent filter (3rd-order LPF only by default)
  3.2.2 - Create before/after filtering comparison visualisation
  3.2.3 - Verify filtering effectiveness with PSD plots
  3.2.4 - Test filtering on subject A1

Reference: PROJECT.md Section 5, Stage B Step B2; config.yaml stage_b.filter
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, zpk2sos

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_output_path, get_stage_params, load_config

# Import Step B1 pipeline to get the OD object
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "intensity_to_od",
    Path(__file__).parent / "01_intensity_to_od.py",
)
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
run_intensity_to_od = _b1.run_intensity_to_od


# ---------------------------------------------------------------------------
# Homer3-equivalent core filter (matches hmrR_BandpassFilt.m line-for-line)
# ---------------------------------------------------------------------------

def _homer3_bandpass(data: np.ndarray, fs: float, hpf: float, lpf: float) -> np.ndarray:
    """
    Zero-phase Butterworth filter replicating hmrR_BandpassFilt.m exactly.

    MATLAB pipeline reproduced:
      1. Normalize cutoffs to [0, 1] Nyquist fraction.
        - the nyquist frequency is fs / 2 (half the sampling rate)
        - digital filters like Butterworth are designed using normalized frequency where 1.0 represents the highest frequency.
        - the nyquist frequency 
      2. Low-pass (3rd-order): [z,p,k] = butter(3, lpf_norm, 'low')
                                [sos,g] = zp2sos(z,p,k)
                                y2     = filtfilt(sos, g, y)
      3. High-pass (5th-order): same pattern — applied ONLY if hpf_norm > 0.
         Homer3 default hpf=0.000 → high-pass is skipped entirely.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Input signal; channels on axis 0, time on axis 1.
    fs   : float
        Sampling rate in Hz.
    hpf  : float
        High-pass cutoff frequency in Hz. 0.0 = skip (Homer3 default).
    lpf  : float
        Low-pass cutoff frequency in Hz.

    Returns
    -------
    y2 : np.ndarray, shape (n_channels, n_times)
        Filtered data.
    """
    nyq = fs / 2.0

    if lpf / nyq > 1 or hpf / nyq > 1:
        raise ValueError(
            f"Filter cutoff exceeds Nyquist ({nyq:.2f} Hz). hpf={hpf}, lpf={lpf}"
        )

    y2 = data.copy()

    # Step 1 — Low-pass: 3rd-order Butterworth (always applied when lpf > 0)
    lpf_norm = lpf / nyq
    if lpf_norm > 0:
        #z = zeroes; frequencies where the filter is zero
        #p = poles; where the filter reonates or pushes the signal up
        #k = gain; a simple multiplier/scalar that scales the output after the zeroes and poles have shaped the frequency response
        #butter designs the butterworth filter by calculating z,p,k with the given order nad normalized frequency
        z, p, k = butter(3, lpf_norm, btype='low', output='zpk')
        #zpk2sos converts the z,p,k to a second order sections (sos) format that is more efficient for digital filtering\
        #2nd order sections are much more stable and accurate that higher order sections
        sos = zpk2sos(z, p, k)
        #sosfiltfilt is a zero-phase filter that applies the filter to the data in both directions (forward and backward) to avoid phase distortion
        #the forward pass cleans the signal but slightly shifts the timing of the signal
        #the backward pass flips the data and runs the filter again to cancel out the phase shift
        y2 = sosfiltfilt(sos, y2, axis=-1)

    # Step 2 — High-pass: 5th-order Butterworth (only if hpf > 0)
    hpf_norm = hpf / nyq
    if hpf_norm > 0:
        z, p, k = butter(5, hpf_norm, btype='high', output='zpk')
        sos = zpk2sos(z, p, k)
        y2 = sosfiltfilt(sos, y2, axis=-1)

    return y2


# ---------------------------------------------------------------------------
# Task 3.2.1 — Apply Homer3-equivalent filter to MNE Raw OD object
# ---------------------------------------------------------------------------

def apply_bandpass_filter(
    raw_od,
    hpf: float = 0.0,
    lpf: float = 0.5,
    verbose: bool = True,
):
    """
    Task 3.2.1: Apply Homer3-equivalent Butterworth filter to OD data.

    Reproduces hmrR_BandpassFilt.m:
      - Low-pass:  3rd-order Butterworth, zero-phase (always applied)
      - High-pass: 5th-order Butterworth, zero-phase (skipped when hpf=0)
      - Homer3/paper default: hpf=0.000, lpf=0.500 Hz

    Bad channels (raw_od.info['bads']) are excluded from filtering.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical density data (output of Step B1). Must be preloaded.
    hpf : float
        High-pass cutoff in Hz. 0.0 = no high-pass (Homer3 default).
    lpf : float
        Low-pass cutoff in Hz. Default 0.5 Hz.
    verbose : bool

    Returns
    -------
    raw_od_filt : mne.io.Raw
        Filtered OD object (copy of input; original is unchanged).
    filter_info : dict
        {'hpf', 'lpf', 'hpf_applied', 'lp_order', 'hp_order',
         'method', 'sfreq', 'n_channels_filtered'}
    """
    fs = raw_od.info['sfreq']
    bad_set = set(raw_od.info['bads'])
    good_idx = [i for i, ch in enumerate(raw_od.ch_names) if ch not in bad_set]
    n_bad = len(bad_set)
    n_filt = len(good_idx)

    if verbose:
        hpf_str = f"{hpf} Hz  (5th-order HPF)" if hpf > 0 else "none — hpf=0.0 (Homer3 default)"
        print("\n" + "=" * 70)
        print("BANDPASS FILTER APPLICATION (TASK 3.2.1) — Homer3-equivalent")
        print("=" * 70)
        print(f"  Low-pass cutoff:   {lpf} Hz  (3rd-order Butterworth)")
        print(f"  High-pass cutoff:  {hpf_str}")
        print(f"  Method:            zero-phase SOS filtfilt (mirrors MATLAB filtfilt)")
        print(f"  Sampling rate:     {fs:.2f} Hz")
        print(f"  Channels filtered: {n_filt}  (bad channels skipped: {n_bad})")

    raw_od_filt = raw_od.copy()
    data = raw_od_filt.get_data()                          # shape: (n_ch, n_times)
    data[good_idx] = _homer3_bandpass(data[good_idx], fs, hpf, lpf)
    raw_od_filt._data[:] = data

    filter_info = {
        'hpf': hpf,
        'lpf': lpf,
        'hpf_applied': hpf > 0,
        'lp_order': 3,
        'hp_order': 5 if hpf > 0 else None,
        'method': 'zero-phase SOS filtfilt (Homer3-equivalent)',
        'sfreq': float(fs),
        'n_channels_filtered': n_filt,
    }

    if verbose:
        passband_str = f"0–{lpf} Hz" if hpf == 0.0 else f"{hpf}–{lpf} Hz"
        print(f"\n  Effective passband: {passband_str}")
        print("✓ Filter applied successfully.\n")

    return raw_od_filt, filter_info


# ---------------------------------------------------------------------------
# Task 3.2.2 — Before / after time-series comparison
# ---------------------------------------------------------------------------

def plot_filter_comparison(
    raw_od,
    raw_od_filt,
    subject_id: str,
    output_dir: Path,
    filter_info: dict = None,
    n_channels: int = 4,
    t_start: float = 0.0,
    t_window: float = 120.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.2.2: Side-by-side time-series showing unfiltered vs filtered OD
    for a representative set of good channels.

    Each row is one channel; left panel = raw OD, right panel = filtered OD.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Unfiltered OD data.
    raw_od_filt : mne.io.Raw
        Filtered OD data.
    subject_id : str
        Used for plot title and filename.
    output_dir : Path
        Save directory.
    filter_info : dict, optional
        Output of apply_bandpass_filter(); used to build axis labels.
    n_channels : int
        Number of good channels to plot (spread evenly across channel list).
    t_start : float
        Start time for the displayed window (seconds).
    t_window : float
        Length of the displayed window (seconds).
    verbose : bool
        Print save path.

    Returns
    -------
    fig_path : Path
    """
    sfreq = raw_od.info['sfreq']
    bad_set = set(raw_od.info['bads'])
    good_channels = [ch for ch in raw_od.ch_names if ch not in bad_set]

    step = max(1, len(good_channels) // n_channels)
    sample_chs = good_channels[::step][:n_channels]

    i_start = int(t_start * sfreq)
    i_end = int((t_start + t_window) * sfreq)
    times = raw_od.times[i_start:i_end]

    data_raw = raw_od.get_data()
    data_filt = raw_od_filt.get_data()
    ch_names = raw_od.ch_names

    # Build filter label from filter_info
    if filter_info is not None:
        lpf = filter_info['lpf']
        hpf = filter_info['hpf']
        filt_label = f"0–{lpf} Hz LPF" if hpf == 0.0 else f"{hpf}–{lpf} Hz BPF"
    else:
        filt_label = "filtered"

    fig, axes = plt.subplots(
        n_channels, 2,
        figsize=(16, 2.8 * n_channels),
        sharex=True,
    )
    if n_channels == 1:
        axes = axes[np.newaxis, :]

    for row, ch_name in enumerate(sample_chs):
        idx = ch_names.index(ch_name)
        seg_raw = data_raw[idx, i_start:i_end]
        seg_filt = data_filt[idx, i_start:i_end]

        axes[row, 0].plot(times, seg_raw, color='#888888', lw=0.7)
        axes[row, 0].axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        axes[row, 0].set_ylabel('OD', fontsize=8)
        axes[row, 0].set_title(f'{ch_name}  [unfiltered]', fontsize=8, loc='left')

        axes[row, 1].plot(times, seg_filt, color='#1f77b4', lw=0.8)
        axes[row, 1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        axes[row, 1].set_ylabel('OD', fontsize=8)
        axes[row, 1].set_title(f'{ch_name}  [{filt_label}]', fontsize=8, loc='left')

    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'Subject {subject_id} — OD Before vs After Filter ({filt_label}, '
        f'first {t_window:.0f} s)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_filter_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved filter comparison plot: {fig_path}")

    return fig_path


# ---------------------------------------------------------------------------
# Task 3.2.3 — PSD verification
# ---------------------------------------------------------------------------

def plot_psd_comparison(
    raw_od,
    raw_od_filt,
    subject_id: str,
    output_dir: Path,
    filter_info: dict = None,
    n_channels: int = 6,
    fmax: float = 2.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.2.3: Power spectral density overlay (unfiltered vs filtered) to
    verify that the filter attenuates out-of-band frequencies.

    Uses Welch's method on a subset of good channels.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Unfiltered OD data.
    raw_od_filt : mne.io.Raw
        Filtered OD data.
    subject_id : str
    output_dir : Path
    filter_info : dict, optional
        Output of apply_bandpass_filter(); used for cutoff annotations.
    n_channels : int
        Number of good channels whose PSDs are averaged.
    fmax : float
        Maximum frequency on x-axis (Hz).
    verbose : bool

    Returns
    -------
    fig_path : Path
    """
    bad_set = set(raw_od.info['bads'])
    good_channels = [ch for ch in raw_od.ch_names if ch not in bad_set]
    step = max(1, len(good_channels) // n_channels)
    sample_chs = good_channels[::step][:n_channels]
    sample_idx = [raw_od.ch_names.index(ch) for ch in sample_chs]

    sfreq = raw_od.info['sfreq']
    nperseg = int(sfreq * 60)

    def _welch_mean(raw, indices):
        from scipy.signal import welch
        data = raw.get_data()
        psds = []
        for i in indices:
            freqs, psd = welch(data[i], fs=sfreq, nperseg=nperseg)
            psds.append(psd)
        return freqs, np.mean(psds, axis=0)

    freqs_raw, psd_raw = _welch_mean(raw_od, sample_idx)
    freqs_filt, psd_filt = _welch_mean(raw_od_filt, sample_idx)
    freq_mask = freqs_raw <= fmax

    # Determine labels from filter_info
    if filter_info is not None:
        lpf = filter_info['lpf']
        hpf = filter_info['hpf']
        filt_label = f"0–{lpf} Hz LPF" if hpf == 0.0 else f"{hpf}–{lpf} Hz BPF"
        hpf_applied = filter_info['hpf_applied']
    else:
        lpf, hpf, filt_label, hpf_applied = 0.5, 0.0, "0–0.5 Hz LPF", False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Linear scale ---
    ax = axes[0]
    ax.plot(freqs_raw[freq_mask], psd_raw[freq_mask],
            color='#888888', lw=1.2, label='Unfiltered OD', alpha=0.85)
    ax.plot(freqs_filt[freq_mask], psd_filt[freq_mask],
            color='#1f77b4', lw=1.5, label=f'Filtered OD ({filt_label})', alpha=0.9)
    if hpf_applied:
        ax.axvspan(0, hpf, alpha=0.08, color='red', label=f'< {hpf} Hz (attenuated)')
    ax.axvspan(lpf, fmax, alpha=0.08, color='orange', label=f'> {lpf} Hz (attenuated)')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (OD²/Hz)', fontsize=10)
    ax.set_title('PSD — Linear scale', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fmax)

    # --- Log scale ---
    ax = axes[1]
    psd_raw_db = 10 * np.log10(psd_raw[freq_mask] + 1e-30)
    psd_filt_db = 10 * np.log10(psd_filt[freq_mask] + 1e-30)
    ax.plot(freqs_raw[freq_mask], psd_raw_db,
            color='#888888', lw=1.2, label='Unfiltered OD', alpha=0.85)
    ax.plot(freqs_filt[freq_mask], psd_filt_db,
            color='#1f77b4', lw=1.5, label=f'Filtered OD ({filt_label})', alpha=0.9)
    if hpf_applied:
        ax.axvline(hpf, color='red', ls='--', lw=1.0, alpha=0.7, label=f'{hpf} Hz HPF cutoff')
    ax.axvline(lpf, color='orange', ls='--', lw=1.0, alpha=0.7, label=f'{lpf} Hz LPF cutoff')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (dB re OD²/Hz)', fontsize=10)
    ax.set_title('PSD — Log scale (dB)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fmax)

    fig.suptitle(
        f'Subject {subject_id} — PSD Before vs After Filter ({filt_label})\n'
        f'(mean of {len(sample_chs)} good channels, Welch method)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_psd_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        hf_mask = (freqs_raw > lpf) & (freqs_raw <= fmax)
        if hf_mask.any():
            hf_ratio = psd_filt[hf_mask].mean() / (psd_raw[hf_mask].mean() + 1e-30)
            print(f"PSD verification — High-frequency power ratio "
                  f"(filtered/unfiltered, >{lpf} Hz): {hf_ratio:.4f}  "
                  f"{'✓ attenuated' if hf_ratio < 0.1 else '⚠ check roll-off'}")
        print(f"Saved PSD comparison plot: {fig_path}")

    return fig_path


# ---------------------------------------------------------------------------
# End-to-end pipeline — Tasks 3.2.1–3.2.4
# ---------------------------------------------------------------------------

def run_bandpass_filter(
    dataset_name: str,
    subject_id: str,
    verbose: bool = True,
) -> dict:
    """
    Run the full Step B2 pipeline (Tasks 3.2.1–3.2.4) for one subject.

    Steps:
    1. Run Step B1 to obtain OD object (with bad channels already marked)
    2. Apply Homer3-equivalent filter (3.2.1)
    3. Save before/after time-series plot (3.2.2)
    4. Save PSD comparison plot (3.2.3)
    5. Print summary (3.2.4)

    Parameters
    ----------
    dataset_name : str
        Dataset key (e.g., 'expert_novice')
    subject_id : str
        Subject identifier (e.g., 'A1')
    verbose : bool

    Returns
    -------
    results : dict
        {
          'raw_od':         mne.io.Raw (unfiltered),
          'raw_od_filt':    mne.io.Raw (filtered),
          'filter_info':    dict,
          'fig_comparison': Path,
          'fig_psd':        Path,
        }
    """
    config = load_config()
    filt_params = get_stage_params('stage_b', config)['filter']

    print("\n" + "=" * 70)
    print(f"STAGE B — STEP B2: BANDPASS FILTER  |  Subject: {subject_id}")
    print("=" * 70)

    # ---- Step B1: get OD object -------------------------------------------
    b1_results = run_intensity_to_od(dataset_name, subject_id, verbose=False)
    raw_od = b1_results['raw_od']

    # ---- 3.2.1 Apply Homer3-equivalent filter -------------------------------
    raw_od_filt, filter_info = apply_bandpass_filter(
        raw_od,
        hpf=filt_params['hpf'],
        lpf=filt_params['lpf'],
        verbose=verbose,
    )

    # ---- Outputs directory --------------------------------------------------
    output_dir = get_output_path('stage_b', f'bandpass_filter/{subject_id}')

    # ---- 3.2.2 Before/after time-series plot --------------------------------
    fig_comparison = plot_filter_comparison(
        raw_od, raw_od_filt,
        subject_id=subject_id,
        output_dir=output_dir,
        filter_info=filter_info,
        verbose=verbose,
    )

    # ---- 3.2.3 PSD comparison plot ------------------------------------------
    fig_psd = plot_psd_comparison(
        raw_od, raw_od_filt,
        subject_id=subject_id,
        output_dir=output_dir,
        filter_info=filter_info,
        verbose=verbose,
    )

    # ---- 3.2.4 Summary ------------------------------------------------------
    passband_str = (
        f"0–{filter_info['lpf']} Hz (LPF only, hpf=0)"
        if not filter_info['hpf_applied']
        else f"{filter_info['hpf']}–{filter_info['lpf']} Hz"
    )
    print("\n" + "=" * 70)
    print(f"✅ Step B2 complete for subject {subject_id}")
    print(f"  Effective passband: {passband_str}")
    print(f"  LP order: {filter_info['lp_order']}  |  "
          f"HP order: {filter_info['hp_order']}  |  "
          f"Method: {filter_info['method']}")
    print(f"  Channels filtered: {filter_info['n_channels_filtered']}  "
          f"(bad channels skipped)")
    print(f"  Outputs saved: {output_dir}")
    print("=" * 70 + "\n")

    return {
        'raw_od': raw_od,
        'raw_od_filt': raw_od_filt,
        'filter_info': filter_info,
        'fig_comparison': fig_comparison,
        'fig_psd': fig_psd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_bandpass_filter(
        dataset_name='expert_novice',
        subject_id='A1',
        verbose=True,
    )
