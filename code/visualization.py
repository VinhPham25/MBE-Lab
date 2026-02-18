#!/usr/bin/env python3
"""Visualize raw SNIRF time series for Subject A1 (Expert-Novice study).
   This is independent of the preprocessing pipeline, and was utilized as a figure
   in the Week 0b report.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_channels_tsv(path: Path):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r") as handle:
        return json.load(handle)


def decode_one(dataset):
    arr = dataset[:]
    if len(arr) == 0:
        return ""
    value = arr[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def load_snirf(path: Path):
    with h5py.File(path, "r") as handle:
        time = handle["nirs/data1/time"][:]
        data = handle["nirs/data1/dataTimeSeries"][:]
        wavelengths = None
        if "probe" in handle["nirs"]:
            wavelengths = handle["nirs/probe/wavelengths"][:]

        stims = []
        for key in handle["nirs"].keys():
            if key.startswith("stim"):
                stim = {
                    "key": key,
                    "name": decode_one(handle[f"nirs/{key}/name"]) or key,
                    "data": handle[f"nirs/{key}/data"][:],
                }
                stims.append(stim)

        aux = None
        if "aux1" in handle["nirs"]:
            aux = {
                "name": decode_one(handle["nirs/aux1/name"]) or "aux1",
                "time": handle["nirs/aux1/time"][:],
                "data": handle["nirs/aux1/dataTimeSeries"][:],
            }

    return time, data, stims, aux, wavelengths


def select_channels(rows, data_cols, per_wavelength=3):
    nirs_rows = [row for row in rows if row.get("type", "").upper() != "AUX"]
    if len(nirs_rows) != data_cols:
        print(
            "Warning: channels.tsv NIRS rows ({}) != data columns ({}).".format(
                len(nirs_rows), data_cols
            )
        )

    selected = []
    for wavelength in ("690", "830"):
        picks = [
            idx
            for idx, row in enumerate(nirs_rows)
            if row.get("wavelength_nominal") == wavelength
        ][:per_wavelength]
        selected.extend(picks)

    if not selected:
        selected = list(range(min(6, data_cols)))

    labels = []
    for idx in selected:
        if idx < len(nirs_rows):
            labels.append(nirs_rows[idx].get("name", f"ch{idx+1}"))
        else:
            labels.append(f"ch{idx+1}")
    return selected, labels


def plot_data(time, data, stims, aux, channel_rows, out_path=None, raw=False):
    fs = float("nan")
    if len(time) > 1:
        fs = 1.0 / np.median(np.diff(time))

    selected, labels = select_channels(channel_rows, data.shape[1])

    rows = 2 if aux is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 6 + 3 * (rows - 1)), sharex=True)
    if rows == 1:
        axes = [axes]

    ax = axes[0]
    offsets = 0.0 if raw else 4.0
    colors = plt.cm.tab10.colors
    for idx, ch_idx in enumerate(selected):
        series = data[:, ch_idx]
        if not raw:
            std = np.std(series)
            if std == 0:
                std = 1.0
            series = (series - np.mean(series)) / std
        ax.plot(
            time,
            series + idx * offsets,
            lw=0.8,
            color=colors[idx % len(colors)],
            label=labels[idx],
        )

    stim_colors = ["#fdae61", "#2b83ba", "#abdda4", "#f46d43"]
    stim_labels = set()
    for stim_idx, stim in enumerate(stims):
        color = stim_colors[stim_idx % len(stim_colors)]
        for onset, duration, _amp in stim["data"]:
            label = None
            if stim["name"] not in stim_labels:
                label = f"Stim {stim['name']}"
                stim_labels.add(stim["name"])
            ax.axvspan(onset, onset + duration, color=color, alpha=0.15, label=label)

    if raw:
        ax.set_title(f"Selected raw intensity channels | fs ~ {fs:.2f} Hz")
        ax.set_ylabel("Intensity (a.u.)")
    else:
        ax.set_title(
            f"Selected raw intensity channels (z-scored + offset) | fs ~ {fs:.2f} Hz"
        )
        ax.set_ylabel("a.u. (offset)")
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    if aux is not None:
        ax_aux = axes[1]
        aux_series = aux["data"].astype(float)
        if not raw:
            aux_std = np.std(aux_series)
            if aux_std == 0:
                aux_std = 1.0
            aux_series = (aux_series - np.mean(aux_series)) / aux_std
        ax_aux.plot(aux["time"], aux_series, color="#555", lw=0.8)
        ax_aux.set_title(f"Aux channel ({aux['name']})")
        ax_aux.set_ylabel("Intensity (a.u.)" if raw else "a.u. (z-score)")
        ax_aux.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    plt.show()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_snirf = (
        repo_root
        / "data/BIDS/Expert-Novice/sub-A1/nirs/sub-A1_task-ExpNovPatternCutting_nirs.snirf"
    )
    default_channels = (
        repo_root
        / "data/BIDS/Expert-Novice/sub-A1/nirs/sub-A1_task-ExpNovPatternCutting_channels.tsv"
    )
    default_json = (
        repo_root
        / "data/BIDS/Expert-Novice/sub-A1/nirs/sub-A1_task-ExpNovPatternCutting_nirs.json"
    )

    parser = argparse.ArgumentParser(
        description="Visualize SNIRF data (default: Subject A1, Expert-Novice)."
    )
    parser.add_argument("--snirf", type=Path, default=default_snirf)
    parser.add_argument("--channels", type=Path, default=default_channels)
    parser.add_argument("--json", type=Path, default=default_json)
    parser.add_argument("--out", type=Path, default=None, help="Optional PNG output")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw intensities (no z-score, no vertical offsets)",
    )
    args = parser.parse_args()

    time, data, stims, aux, wavelengths = load_snirf(args.snirf)
    channel_rows = load_channels_tsv(args.channels)
    meta = load_json(args.json)

    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    fs = 1.0 / np.median(np.diff(time)) if len(time) > 1 else float("nan")

    print(f"SNIRF file: {args.snirf}")
    print(f"Samples: {data.shape[0]} | Channels: {data.shape[1]}")
    print(f"Duration: {duration:.2f} s | fs (from time): {fs:.2f} Hz")
    if meta:
        print(
            "Sidecar JSON: SamplingFrequency={} Hz | RecordingDuration={} s".format(
                meta.get("SamplingFrequency"), meta.get("RecordingDuration")
            )
        )
    if wavelengths is not None:
        print(f"Wavelengths: {wavelengths}")
    if stims:
        for stim in stims:
            print(f"Stim {stim['name']}: {stim['data'].shape[0]} events")

    plot_data(time, data, stims, aux, channel_rows, args.out, raw=args.raw)


if __name__ == "__main__":
    main()
