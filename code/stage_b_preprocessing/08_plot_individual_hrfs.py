"""Plot post-preprocessing GLM HRFs for each subject separately.

This is the subject-level counterpart to Stage C's group HRF plot.  It reads
the GLM HRF evokeds created by ``07_preprocess_study.py`` and averages the
long-separation HbO channels into the five ASTaUND regions for *one subject at
a time*.  No trace is averaged across subjects.

Example
-------
python3 code/stage_b_preprocessing/08_plot_individual_hrfs.py
python3 code/stage_b_preprocessing/08_plot_individual_hrfs.py --subjects A1 R1

Outputs are written to::

    outputs/stage_b_processed/individual_hrf/expert-novice/
        sub-A1_hrf_panel.png
        ...
        all_subjects_hrf_grid.png
        individual_hrf_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config


# Source numbers follow the five-region mapping used by the Stage C HRF plot.
M1_SOURCES = {4, 5, 6, 7}
SHORT_SEP_DETECTORS = {f"D{detector}" for detector in range(17, 25)}
SHORT_SEP_SPECIAL = {"S8_D9"}
REGIONS = (
    "Left Lateral PFC",
    "Central PFC",
    "Right Lateral PFC",
    "Left Medial M1",
    "SMA",
)


def _is_short_separation(channel_name: str) -> bool:
    """Return whether an fNIRS channel is a short-separation channel."""
    base = channel_name.split(" ", maxsplit=1)[0]
    if base in SHORT_SEP_SPECIAL:
        return True
    parts = base.split("_")
    return len(parts) > 1 and parts[1] in SHORT_SEP_DETECTORS


def _source_number(channel_name: str) -> int | None:
    """Extract the numeric source identifier from e.g. ``S3_D12 hbo``."""
    source = channel_name.split("_", maxsplit=1)[0]
    if not source.startswith("S") or not source[1:].isdigit():
        return None
    return int(source[1:])


def build_region_map_5(channel_names: list[str]) -> dict[str, list[str]]:
    """Map long-separation HbO channels to the five analysis regions."""
    region_map = {region: [] for region in REGIONS}
    for name in channel_names:
        if " hbo" not in name.lower() or _is_short_separation(name):
            continue

        source = _source_number(name)
        if source == 1:
            region_map["Left Lateral PFC"].append(name)
        elif source == 2:
            region_map["Central PFC"].append(name)
        elif source == 3:
            region_map["Right Lateral PFC"].append(name)
        elif source in M1_SOURCES:
            region_map["Left Medial M1"].append(name)
        elif source == 8:
            region_map["SMA"].append(name)
    return region_map


def _load_glm_hrf(subject_id: str, input_dir: Path, task_name: str) -> mne.Evoked:
    """Load a subject's Stage B GLM HRF, allowing an older task-label sidecar."""
    preferred = input_dir / (
        f"sub-{subject_id}_task-{task_name}_desc-glmhrf_nirs-ave.fif"
    )
    if preferred.exists():
        return mne.read_evokeds(preferred, verbose=False)[0]

    candidates = sorted(input_dir.glob(f"sub-{subject_id}_*_desc-glmhrf_nirs-ave.fif"))
    if not candidates:
        raise FileNotFoundError(f"No GLM HRF evoked found for subject {subject_id}")
    return mne.read_evokeds(candidates[0], verbose=False)[0]


def _subject_region_traces(evoked: mne.Evoked) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return one micromolar HbO trace per region and its channel count."""
    traces: dict[str, np.ndarray] = {}
    channel_counts: dict[str, int] = {}
    region_map = build_region_map_5(evoked.ch_names)

    for region, channels in region_map.items():
        present = [name for name in channels if name in evoked.ch_names]
        finite = [
            name
            for name in present
            if np.isfinite(evoked.data[evoked.ch_names.index(name)]).all()
        ]
        channel_counts[region] = len(finite)
        if finite:
            indices = [evoked.ch_names.index(name) for name in finite]
            traces[region] = evoked.data[indices].mean(axis=0) * 1e6
    return traces, channel_counts


def plot_subject_hrf(
    subject_id: str,
    group_name: str,
    evoked: mne.Evoked,
    output_path: Path,
) -> dict[str, int]:
    """Save a five-panel, single-subject HbO HRF figure."""
    traces, channel_counts = _subject_region_traces(evoked)
    mosaic = [["A", "B", "C"], ["D", "E", "legend"]]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.34, "hspace": 0.50},
    )
    panel_keys = dict(zip(REGIONS, ("A", "B", "C", "D", "E"), strict=True))
    color = "#8B1A1A" if group_name == "Expert" else "#1A6B1A"

    fig.suptitle(
        f"Subject {subject_id} ({group_name}) HRF - dHbO2\n"
        "post-preprocessing GLM estimate; regional channel mean",
        fontsize=13,
        fontweight="bold",
    )

    for region in REGIONS:
        ax = axes[panel_keys[region]]
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.axvline(0, color="black", lw=1.5, ls="--", zorder=2)

        trace = traces.get(region)
        if trace is None:
            ax.text(
                0.5,
                0.5,
                "No finite long-separation\nHbO channels",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
        else:
            ax.plot(evoked.times, trace, color=color, lw=2.0, zorder=3)

        ax.set_title(f"{region} (n={channel_counts[region]} channels)", fontweight="bold")
        ax.set_xlim(evoked.times[0], evoked.times[-1])
        ax.set_xlabel("Time (s)")
        if region in ("Left Lateral PFC", "Left Medial M1"):
            ax.set_ylabel("dHbO2 conc. (uM)")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes["legend"].axis("off")
    axes["legend"].legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=2.0, label=f"{group_name} subject"),
            plt.Line2D([0], [0], color="black", lw=1.5, ls="--", label="FLS task start"),
        ],
        loc="center",
        framealpha=0.9,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return channel_counts


def plot_all_subject_hrfs(
    subject_records: list[dict], output_path: Path
) -> None:
    """Save a row-per-subject, column-per-region overview figure.

    Every column uses one shared y-scale across subjects.  This makes the
    between-subject amplitude differences visible while retaining a separate
    scale for each region.
    """
    n_subjects = len(subject_records)
    fig, axes = plt.subplots(
        n_subjects,
        len(REGIONS),
        figsize=(18, max(8, 2.15 * n_subjects + 1.7)),
        sharex="col",
        squeeze=False,
    )

    y_limits: dict[str, tuple[float, float]] = {}
    for region in REGIONS:
        values = [
            record["traces"][region]
            for record in subject_records
            if region in record["traces"]
        ]
        if not values:
            continue
        lower = min(float(np.min(trace)) for trace in values)
        upper = max(float(np.max(trace)) for trace in values)
        padding = max((upper - lower) * 0.08, 0.05)
        y_limits[region] = (lower - padding, upper + padding)

    for row, record in enumerate(subject_records):
        subject_id = record["subject_id"]
        group_name = record["group"]
        evoked = record["evoked"]
        color = "#8B1A1A" if group_name == "Expert" else "#1A6B1A"

        for column, region in enumerate(REGIONS):
            ax = axes[row, column]
            ax.axhline(0, color="black", lw=0.55, zorder=1)
            ax.axvline(0, color="black", lw=0.8, ls="--", zorder=2)
            trace = record["traces"].get(region)
            if trace is None:
                ax.text(
                    0.5,
                    0.5,
                    "No finite\nHbO channels",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.35",
                )
            else:
                ax.plot(evoked.times, trace, color=color, lw=1.25, zorder=3)

            if region in y_limits:
                ax.set_ylim(y_limits[region])
            ax.set_xlim(evoked.times[0], evoked.times[-1])
            ax.grid(True, alpha=0.20, lw=0.4)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

            if row == 0:
                ax.set_title(region, fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{subject_id}\n({group_name})", fontsize=8, fontweight="bold")
            if row == n_subjects - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(
        "Individual Subject HRFs by Region - dHbO2\n"
        "post-preprocessing GLM estimates; each row is one subject; y-scale shared within region",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#8B1A1A", lw=1.5, label="Expert"),
            plt.Line2D([0], [0], color="#1A6B1A", lw=1.5, label="Novice"),
            plt.Line2D([0], [0], color="black", lw=1.0, ls="--", label="FLS task start"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        ncol=3,
        fontsize=9,
        framealpha=0.9,
    )
    fig.text(0.005, 0.5, "dHbO2 conc. (uM)", va="center", rotation="vertical", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.945, bottom=0.045, hspace=0.42, wspace=0.22)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_individual_hrf_plots(study: str, requested_subjects: list[str] | None = None) -> dict:
    """Generate one five-region HRF panel per configured study subject."""
    config = load_config()
    try:
        dataset = config["datasets"][study]
    except KeyError as exc:
        raise ValueError(f"Unknown study {study!r}; see config.yaml datasets.") from exc

    groups = dataset.get("groups")
    if not groups:
        raise ValueError(f"Study {study!r} has no subject groups configured.")
    subject_to_group = {
        subject: group for group, subjects in groups.items() for subject in subjects
    }
    if requested_subjects is None:
        subject_ids = list(subject_to_group)
    else:
        unknown = sorted(set(requested_subjects) - set(subject_to_group))
        if unknown:
            raise ValueError(
                "Requested subject(s) are not in the configured groups: " + ", ".join(unknown)
            )
        subject_ids = requested_subjects

    cohort = dataset.get("output_folder", study.replace("_", "-"))
    stage_b_root = Path(config["paths"]["stage_b_outputs"])
    input_dir = stage_b_root / "processed_data" / cohort
    output_dir = stage_b_root / "individual_hrf" / cohort
    if not input_dir.exists():
        raise FileNotFoundError(f"Stage B processed-data directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = dataset["bids_task"]
    manifest: dict[str, object] = {
        "study": study,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "subjects": {},
        "skipped": {},
    }
    overview_records: list[dict] = []

    print("\n" + "=" * 64)
    print(" Stage B - Individual five-region GLM HRF plots")
    print(f" Input : {input_dir}")
    print(f" Output: {output_dir}")
    print(f" Subjects: {len(subject_ids)}")
    print("=" * 64)

    for subject_id in subject_ids:
        try:
            evoked = _load_glm_hrf(subject_id, input_dir, task_name)
            output_path = output_dir / f"sub-{subject_id}_hrf_panel.png"
            counts = plot_subject_hrf(
                subject_id, subject_to_group[subject_id], evoked, output_path
            )
            manifest["subjects"][subject_id] = {
                "group": subject_to_group[subject_id],
                "figure": output_path.name,
                "finite_long_hbo_channels_by_region": counts,
            }
            traces, _ = _subject_region_traces(evoked)
            overview_records.append(
                {
                    "subject_id": subject_id,
                    "group": subject_to_group[subject_id],
                    "evoked": evoked,
                    "traces": traces,
                }
            )
            print(f" [OK] {subject_id}: {output_path.name}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            manifest["skipped"][subject_id] = str(exc)
            print(f" [SKIP] {subject_id}: {exc}")

    if overview_records:
        overview_path = output_dir / "all_subjects_hrf_grid.png"
        plot_all_subject_hrfs(overview_records, overview_path)
        manifest["all_subjects_figure"] = overview_path.name
        print(f" [OK] all subjects: {overview_path.name}")

    manifest_path = output_dir / "individual_hrf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f" Manifest: {manifest_path.name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot five-region, subject-level post-preprocessing GLM HbO HRFs."
    )
    parser.add_argument("--study", default="expert_novice", help="Dataset key in config.yaml.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        metavar="SUBJECT",
        help="Optional configured subject IDs to plot (default: all grouped subjects).",
    )
    args = parser.parse_args()
    run_individual_hrf_plots(args.study, args.subjects)
