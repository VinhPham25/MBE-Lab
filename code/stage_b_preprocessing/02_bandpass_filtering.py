"""
Stage B - Step B2: Bandpass Filtering
Purpose: Apply 0.01–0.5 Hz IIR bandpass filter to OD data, visualise the
         before/after time series, and verify effectiveness via PSD plots.

Task coverage in this file:
  3.2.1 - Implement bandpass filter application (0.01-0.5 Hz, IIR method)
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
# Task 3.2.1 — Apply bandpass filter to OD data
# ---------------------------------------------------------------------------

def apply_bandpass_filter(
    raw_od,
    l_freq: float = 0.01,
    h_freq: float = 0.5,
    method: str = 'iir',
    verbose: bool = True,
):
    """
    Task 3.2.1: Apply a bandpass filter to optical density data.

    Rationale
    ---------
    - Low cutoff  0.01 Hz: removes very slow drift (respiration overlap at
      ~0.2 Hz is intentionally kept for later regression steps)
    - High cutoff 0.5 Hz: removes cardiac (~1 Hz) and higher-frequency noise
    - IIR method: phase-preserving zero-phase filter via MNE's default
      Butterworth design (``filtfilt`` internally), appropriate for continuous
      fNIRS recording

    Bad channels (in raw_od.info['bads']) are skipped automatically by MNE.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical density data (output of Step B1).  Must be preloaded.
    l_freq : float
        Low cutoff frequency (Hz). Default 0.01.
    h_freq : float
        High cutoff frequency (Hz). Default 0.5.
    method : str
        Filter method passed to mne.io.Raw.filter(). Default 'iir'.
    verbose : bool
        Print progress and a brief summary.

    Returns
    -------
    raw_od_filt : mne.io.Raw
        Filtered OD object (copy of input; original is unchanged).
    filter_info : dict
        {'l_freq': float, 'h_freq': float, 'method': str,
         'sfreq': float, 'n_channels_filtered': int}
    """
    if verbose:
        n_bad = len(raw_od.info['bads'])
        n_filt = len(raw_od.ch_names) - n_bad
        print("\n" + "=" * 70)
        print("BANDPASS FILTER APPLICATION (TASK 3.2.1)")
        print("=" * 70)
        print(f"  Passband:          {l_freq} – {h_freq} Hz")
        print(f"  Method:            {method} (zero-phase Butterworth via filtfilt)")
        print(f"  Sampling rate:     {raw_od.info['sfreq']:.2f} Hz")
        print(f"  Channels filtered: {n_filt}  (bad channels skipped: {n_bad})")

    # Work on a copy so the input OD object remains available for comparison
    raw_od_filt = raw_od.copy()
    raw_od_filt.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        verbose=False,
    )

    filter_info = {
        'l_freq': l_freq,
        'h_freq': h_freq,
        'method': method,
        'sfreq': float(raw_od.info['sfreq']),
        'n_channels_filtered': len(raw_od.ch_names) - len(raw_od.info['bads']),
    }

    if verbose:
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
        axes[row, 1].set_title(f'{ch_name}  [0.01–0.5 Hz filtered]', fontsize=8, loc='left')

    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'Subject {subject_id} — OD Before vs After Bandpass Filter '
        f'(first {t_window:.0f} s)',
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
    n_channels: int = 6,
    fmax: float = 2.0,
    verbose: bool = True,
) -> Path:
    """
    Task 3.2.3: Power spectral density overlay (unfiltered vs filtered) to
    verify that the bandpass filter attenuates out-of-band frequencies.

    Uses Welch's method (via mne.io.Raw.compute_psd) on a subset of good
    channels.  The expected result is:
    - Flat / reduced power below 0.01 Hz (DC drift removed)
    - Preserved power in 0.01–0.5 Hz (hemodynamic band)
    - Attenuation above 0.5 Hz (cardiac artefact at ~1 Hz removed)

    Parameters
    ----------
    raw_od : mne.io.Raw
        Unfiltered OD data.
    raw_od_filt : mne.io.Raw
        Filtered OD data.
    subject_id : str
    output_dir : Path
    n_channels : int
        Number of good channels whose PSDs are averaged for the overlay.
    fmax : float
        Maximum frequency shown on the x-axis (Hz). Default 2.0 Hz
        (captures the cardiac band to show the filter roll-off clearly).
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
    nperseg = int(sfreq * 60)  # 60-second Welch segments for good freq resolution

    def _welch_mean(raw, indices):
        """Return mean PSD (µV²/Hz equivalent in OD²/Hz) over selected channels."""
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Linear scale ---
    ax = axes[0]
    ax.plot(freqs_raw[freq_mask], psd_raw[freq_mask],
            color='#888888', lw=1.2, label='Unfiltered OD', alpha=0.85)
    ax.plot(freqs_filt[freq_mask], psd_filt[freq_mask],
            color='#1f77b4', lw=1.5, label='Filtered OD (0.01–0.5 Hz)', alpha=0.9)
    ax.axvspan(0, 0.01, alpha=0.08, color='red', label='< 0.01 Hz (attenuated)')
    ax.axvspan(0.5, fmax, alpha=0.08, color='orange', label='> 0.5 Hz (attenuated)')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (OD²/Hz)', fontsize=10)
    ax.set_title('PSD — Linear scale', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fmax)

    # --- Log scale (better for seeing filter roll-off) ---
    ax = axes[1]
    psd_raw_db = 10 * np.log10(psd_raw[freq_mask] + 1e-30)
    psd_filt_db = 10 * np.log10(psd_filt[freq_mask] + 1e-30)
    ax.plot(freqs_raw[freq_mask], psd_raw_db,
            color='#888888', lw=1.2, label='Unfiltered OD', alpha=0.85)
    ax.plot(freqs_filt[freq_mask], psd_filt_db,
            color='#1f77b4', lw=1.5, label='Filtered OD (0.01–0.5 Hz)', alpha=0.9)
    ax.axvline(0.01, color='red', ls='--', lw=1.0, alpha=0.7, label='0.01 Hz cutoff')
    ax.axvline(0.5, color='orange', ls='--', lw=1.0, alpha=0.7, label='0.5 Hz cutoff')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (dB re OD²/Hz)', fontsize=10)
    ax.set_title('PSD — Log scale (dB)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fmax)

    fig.suptitle(
        f'Subject {subject_id} — PSD Before vs After Bandpass Filter\n'
        f'(mean of {len(sample_chs)} good channels, Welch method)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f'{subject_id}_psd_comparison.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        # Simple pass/fail check: power above 0.5 Hz should drop post-filter
        hf_mask = (freqs_raw > 0.5) & (freqs_raw <= fmax)
        if hf_mask.any():
            hf_ratio = (psd_filt[hf_mask].mean() / (psd_raw[hf_mask].mean() + 1e-30))
            print(f"PSD verification — High-frequency power ratio "
                  f"(filtered/unfiltered, >0.5 Hz): {hf_ratio:.4f}  "
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
    2. Apply bandpass filter (3.2.1)
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
          'raw_od':        mne.io.Raw (unfiltered),
          'raw_od_filt':   mne.io.Raw (filtered),
          'filter_info':   dict,
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

    # ---- 3.2.1 Apply filter -------------------------------------------------
    raw_od_filt, filter_info = apply_bandpass_filter(
        raw_od,
        l_freq=filt_params['l_freq'],
        h_freq=filt_params['h_freq'],
        method=filt_params['method'],
        verbose=verbose,
    )

    # ---- Outputs directory --------------------------------------------------
    output_dir = get_output_path('stage_b', f'bandpass_filter/{subject_id}')

    # ---- 3.2.2 Before/after time-series plot --------------------------------
    fig_comparison = plot_filter_comparison(
        raw_od, raw_od_filt,
        subject_id=subject_id,
        output_dir=output_dir,
        verbose=verbose,
    )

    # ---- 3.2.3 PSD comparison plot ------------------------------------------
    fig_psd = plot_psd_comparison(
        raw_od, raw_od_filt,
        subject_id=subject_id,
        output_dir=output_dir,
        verbose=verbose,
    )

    # ---- 3.2.4 Summary ------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"✅ Step B2 complete for subject {subject_id}")
    print(f"  Passband:    {filter_info['l_freq']} – {filter_info['h_freq']} Hz  "
          f"({filter_info['method'].upper()} zero-phase)")
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
