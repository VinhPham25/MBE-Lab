#!/usr/bin/env python3
"""Plot raw Expert-Novice SNIRF time series before preprocessing."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_channels_tsv(path: Path):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


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

        stims = []
        for key in handle["nirs"].keys():
            if key.startswith("stim") and "data" in handle[f"nirs/{key}"]:
                stim = {
                    "name": decode_one(handle[f"nirs/{key}/name"]) or key,
                    "data": handle[f"nirs/{key}/data"][:],
                }
                stims.append(stim)
    return time, data, stims


def select_channels(channel_rows, data_cols, per_wavelength=1):
    nirs_rows = [row for row in channel_rows if row.get("type", "").upper() != "AUX"]
    selected = []

    for wavelength in ("690", "830"):
        picks = [
            idx
            for idx, row in enumerate(nirs_rows)
            if row.get("wavelength_nominal") == wavelength
        ][:per_wavelength]
        selected.extend(picks)

    if not selected:
        selected = list(range(min(2, data_cols)))

    labels = []
    for idx in selected:
        if idx < len(nirs_rows):
            labels.append(nirs_rows[idx].get("name", f"ch{idx+1}"))
        else:
            labels.append(f"ch{idx+1}")
    return selected, labels


def classify_subject(subject_id: str):
    if subject_id.startswith("sub-A"):
        return "expert"
    if subject_id.startswith("sub-R"):
        return "novice"
    return "unknown"


def check_rawness(data: np.ndarray):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    positive_fraction = float(np.mean(data > 0.0))
    centered_channels = float(np.mean(np.abs(means) < 1e-2))

    likely_raw = positive_fraction > 0.95 and centered_channels < 0.2 and max_val > 100
    reason = (
        "positive DC-shifted intensities"
        if likely_raw
        else "values look centered/scaled in many channels"
    )
    return {
        "min": min_val,
        "max": max_val,
        "mean_of_means": float(np.mean(means)),
        "median_std": float(np.median(stds)),
        "positive_fraction": positive_fraction,
        "centered_channel_fraction": centered_channels,
        "likely_raw": likely_raw,
        "reason": reason,
    }


def list_subjects(study_root: Path):
    ids = sorted(p.name for p in study_root.glob("sub-*") if p.is_dir())
    experts = [sid for sid in ids if classify_subject(sid) == "expert"]
    novices = [sid for sid in ids if classify_subject(sid) == "novice"]
    return experts, novices


def add_stim_spans(ax, stims):
    stim_colors = ["#fdae61", "#2b83ba", "#abdda4", "#f46d43"]
    seen = set()
    for stim_idx, stim in enumerate(stims):
        color = stim_colors[stim_idx % len(stim_colors)]
        for onset, duration, _amp in stim["data"]:
            label = None
            if stim["name"] not in seen:
                label = f"stim: {stim['name']}"
                seen.add(stim["name"])
            ax.axvspan(onset, onset + duration, color=color, alpha=0.10, label=label)


def plot_group(
    study_root: Path,
    subject_ids,
    group_name,
    channels_per_wavelength,
    max_subjects,
    out_path=None,
):
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]
    if not subject_ids:
        print(f"No subjects found for group: {group_name}")
        return

    n = len(subject_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i, sid in enumerate(subject_ids):
        snirf = study_root / sid / "nirs" / f"{sid}_task-ExpNovPatternCutting_nirs.snirf"
        channels = (
            study_root / sid / "nirs" / f"{sid}_task-ExpNovPatternCutting_channels.tsv"
        )
        ax = axes[i]

        if not snirf.exists():
            ax.set_title(f"{sid} (missing .snirf)")
            ax.axis("off")
            continue

        time, data, stims = load_snirf(snirf)
        channel_rows = load_channels_tsv(channels)
        selected, labels = select_channels(
            channel_rows, data.shape[1], per_wavelength=channels_per_wavelength
        )

        colors = plt.cm.tab10.colors
        for j, ch_idx in enumerate(selected):
            ax.plot(
                time,
                data[:, ch_idx],
                lw=0.7,
                color=colors[j % len(colors)],
                label=labels[j],
            )
        add_stim_spans(ax, stims)

        fs = 1.0 / np.median(np.diff(time)) if len(time) > 1 else float("nan")
        raw_stats = check_rawness(data)
        status = "LIKELY RAW" if raw_stats["likely_raw"] else "CHECK"
        ax.set_title(f"{sid} | fs~{fs:.2f}Hz | {status}", fontsize=9)
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.2)

        print(
            f"[{group_name}] {sid}: min={raw_stats['min']:.2f}, "
            f"max={raw_stats['max']:.2f}, mean={raw_stats['mean_of_means']:.2f}, "
            f"med_std={raw_stats['median_std']:.2f}, positive={raw_stats['positive_fraction']:.3f}, "
            f"centered_channels={raw_stats['centered_channel_fraction']:.3f} -> {raw_stats['reason']}"
        )

        if i == 0:
            handles, legends = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, legends, loc="upper right", fontsize=7, ncol=2)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    for ax in axes[max(0, n - cols) : n]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Expert-Novice raw time series | Group: {group_name} | Subjects: {n}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.text(
            0.995,
            0.005,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
            alpha=0.85,
            transform=fig.transFigure,
        )
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    plt.show()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_study_root = repo_root / "data/BIDS/Expert-Novice"

    parser = argparse.ArgumentParser(
        description="Plot raw SNIRF time-series for Expert-Novice study (experts vs novices)."
    )
    parser.add_argument("--study-root", type=Path, default=default_study_root)
    parser.add_argument(
        "--group",
        choices=["expert", "novice", "both"],
        default="both",
        help="Which participant group to plot.",
    )
    parser.add_argument(
        "--channels-per-wavelength",
        type=int,
        default=1,
        help="Number of channels per wavelength to plot per subject.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap on subjects per group (for faster inspection).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output folder for PNG figures.",
    )
    args = parser.parse_args()

    experts, novices = list_subjects(args.study_root)
    print(f"Study root: {args.study_root}")
    print(f"Experts ({len(experts)}): {experts}")
    print(f"Novices ({len(novices)}): {novices}")
    print(
        "Rawness heuristic notes: raw intensity is usually mostly positive with non-zero DC offset."
    )

    if args.group in ("expert", "both"):
        out_path = None
        if args.out_dir is not None:
            out_path = args.out_dir / "expert_novice_raw_experts.png"
        plot_group(
            args.study_root,
            experts,
            "expert",
            args.channels_per_wavelength,
            args.max_subjects,
            out_path=out_path,
        )

    if args.group in ("novice", "both"):
        out_path = None
        if args.out_dir is not None:
            out_path = args.out_dir / "expert_novice_raw_novices.png"
        plot_group(
            args.study_root,
            novices,
            "novice",
            args.channels_per_wavelength,
            args.max_subjects,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
