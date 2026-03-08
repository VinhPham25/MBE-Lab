"""
Stage C - HRF Visualization (GLM-first)

Purpose
-------
Aggregate and visualize study-level HRFs directly from Stage B GLM outputs:
  sub-{subject}_task-{task}_desc-glmhrf_nirs-ave.fif

This script is independent of 01_hrf_extraction.py / 02_group_hrf_analysis.py.
Run sequence for this branch:
  1) python3 code/stage_b_preprocessing/07_preprocess_study.py --study expert_novice
  2) python3 code/stage_c_validation/03_hrf_visualization.py --study expert_novice
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config


# ---------------------------------------------------------------------------
# Region helpers (copied from Stage C extraction logic to keep this file standalone)
# ---------------------------------------------------------------------------
SHORT_SEP_SUFFIX = tuple(f"D{d}" for d in range(17, 25))
SHORT_SEP_SPECIAL = {"S8_D9"}
PFC_SOURCES = {1, 2, 3}
M1_SOURCES = {4, 5, 6, 7}
SMA_SOURCES = {8}


def _is_short_sep(ch_name: str) -> bool:
    base = ch_name.split(" ")[0]
    if base in SHORT_SEP_SPECIAL:
        return True
    detector = base.split("_")[1]
    return detector in SHORT_SEP_SUFFIX


def _source_number(ch_name: str) -> int | None:
    base = ch_name.split(" ")[0]
    src = base.split("_")[0]
    if not src.startswith("S"):
        return None
    sid = src[1:]
    return int(sid) if sid.isdigit() else None


def build_region_map(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    region_map: dict[str, list[str]] = {"PFC": [], "M1": [], "SMA": []}
    for ch in raw.info["chs"]:
        name = ch["ch_name"]
        if "hbo" not in name or "hbr" in name:
            continue
        if _is_short_sep(name):
            continue

        base = name.split(" ")[0]
        src_n = _source_number(name)
        if src_n in PFC_SOURCES:
            region_map["PFC"].append(base)
        elif src_n in M1_SOURCES:
            region_map["M1"].append(base)
        elif src_n in SMA_SOURCES:
            region_map["SMA"].append(base)
    return region_map


def build_region_map_5(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    region_map: dict[str, list[str]] = {
        "Left Lateral PFC": [],
        "Central PFC": [],
        "Right Lateral PFC": [],
        "Left Medial M1": [],
        "SMA": [],
    }
    for ch in raw.info["chs"]:
        name = ch["ch_name"]
        if "hbo" not in name or "hbr" in name:
            continue
        if _is_short_sep(name):
            continue

        base = name.split(" ")[0]
        src_n = _source_number(name)
        if src_n == 1:
            region_map["Left Lateral PFC"].append(base)
        elif src_n == 2:
            region_map["Central PFC"].append(base)
        elif src_n == 3:
            region_map["Right Lateral PFC"].append(base)
        elif src_n in M1_SOURCES:
            region_map["Left Medial M1"].append(base)
        elif src_n in SMA_SOURCES:
            region_map["SMA"].append(base)

    return region_map


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
GROUP_STYLE: dict[str, dict] = {
    "Expert": {
        "mean_color": "#8B1A1A",
        "band_color": "#FFAAAA",
        "band_alpha": 0.45,
        "lw": 2.0,
        "label": "Expert surgeon",
    },
    "Novice": {
        "mean_color": "#1A6B1A",
        "band_color": "#A8E6A8",
        "band_alpha": 0.40,
        "lw": 2.0,
        "label": "Novice surgeon",
    },
}


def _load_glm_evoked(
    subject_id: str,
    stage_b_dir: Path,
    task_name: str,
) -> mne.Evoked | None:
    preferred = stage_b_dir / f"sub-{subject_id}_task-{task_name}_desc-glmhrf_nirs-ave.fif"
    if preferred.exists():
        return mne.read_evokeds(preferred, verbose=False)[0]

    fallback = sorted(stage_b_dir.glob(f"sub-{subject_id}_*_desc-glmhrf_*ave.fif"))
    if fallback:
        return mne.read_evokeds(fallback[0], verbose=False)[0]

    return None


def _harmonise_evoked(evoked: mne.Evoked, target_sfreq: float) -> mne.Evoked:
    nan_channels = [
        ch for ch, row in zip(evoked.ch_names, evoked.data)
        if not np.isfinite(row).all()
    ]
    if nan_channels:
        evoked = evoked.copy().drop_channels(nan_channels)

    if float(evoked.info["sfreq"]) != float(target_sfreq):
        evoked = evoked.copy().resample(target_sfreq)

    return evoked


def collect_group_traces(
    subjects: list[str],
    stage_b_dir: Path,
    task_name: str,
    region_map: dict[str, list[str]],
    chromophore: str,
    target_sfreq: float,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    region_mats: dict[str, list[np.ndarray]] = {r: [] for r in region_map}
    times_ref: np.ndarray | None = None

    for sid in subjects:
        ev = _load_glm_evoked(sid, stage_b_dir, task_name)
        if ev is None:
            print(f"    [{sid}] missing GLM HRF sidecar, skipped")
            continue

        ev = _harmonise_evoked(ev, target_sfreq)
        if times_ref is None:
            times_ref = ev.times

        for region, bases in region_map.items():
            targets = [f"{b} {chromophore}" for b in bases]
            present = [c for c in targets if c in ev.ch_names]
            if not present:
                continue
            idx = [ev.ch_names.index(c) for c in present]
            trace_um = ev.data[idx, :].mean(axis=0) * 1e6
            region_mats[region].append(trace_um)

    out = {
        r: np.vstack(mats) if mats else np.empty((0, 0))
        for r, mats in region_mats.items()
    }
    return out, times_ref


def plot_hrf_panel(
    ax: plt.Axes,
    times: np.ndarray,
    group_traces: dict[str, np.ndarray],
    region: str,
    chromophore: str,
    show_legend: bool = False,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> dict:
    peak_window = (5.0, 15.0)
    t_mask = (times >= peak_window[0]) & (times <= peak_window[1])
    stats = {}

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(0, color="black", lw=1.5, ls="--", zorder=2, label="FLS task start")

    for group_name, mat in group_traces.items():
        if mat.size == 0:
            continue
        style = GROUP_STYLE[group_name]
        mean = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

        ax.fill_between(times, mean - sd, mean + sd,
                        color=style["band_color"], alpha=style["band_alpha"], zorder=3)
        ax.plot(times, mean, color=style["mean_color"], lw=style["lw"],
                label=f"{style['label']} mean  (n={mat.shape[0]})", zorder=4)

        if chromophore == "hbo":
            pidx = int(np.argmax(mean[t_mask]))
        else:
            pidx = int(np.argmin(mean[t_mask]))
        pt = float(times[t_mask][pidx])
        pa = float(mean[t_mask][pidx])
        stats[group_name] = {"peak_time_s": round(pt, 2), "peak_amp_uM": round(pa, 3)}

    chrom_label = "HbO2" if chromophore == "hbo" else "HbR"
    ax.set_title(f"{region} - d{chrom_label}", fontsize=11, fontweight="bold")
    ax.set_xlim(times[0], times[-1])

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(f"d{chrom_label} conc. (uM)", fontsize=10)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        task_h = [h for h, l in zip(handles, labels) if "FLS" in l]
        data_h = [h for h, l in zip(handles, labels) if "FLS" not in l]
        task_l = [l for l in labels if "FLS" in l]
        data_l = [l for l in labels if "FLS" not in l]
        ax.legend(data_h + task_h, data_l + task_l, fontsize=8, loc="upper right", framealpha=0.85)

    ax.grid(True, alpha=0.25, lw=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    return stats


def generate_hrf_plots(
    groups: dict[str, list[str]],
    stage_b_dir: Path,
    task_name: str,
    region_map: dict[str, list[str]],
    out_dir: Path,
    target_sfreq: float,
) -> dict:
    regions = list(region_map.keys())
    all_stats = {}
    saved = []

    for chromophore in ("hbo", "hbr"):
        chrom_label = "HbO" if chromophore == "hbo" else "HbR"
        print(f"\n  [Plotting {chrom_label}]")

        group_region_traces = {}
        times_ref = None
        for group_name, subjects in groups.items():
            traces, t = collect_group_traces(
                subjects, stage_b_dir, task_name, region_map,
                chromophore=chromophore, target_sfreq=target_sfreq,
            )
            group_region_traces[group_name] = traces
            if times_ref is None and t is not None:
                times_ref = t

        if times_ref is None:
            print(f"    no traces found for {chrom_label}, skipped")
            continue

        fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5), sharey=False)
        if len(regions) == 1:
            axes = [axes]

        chrom_display = "HbO2" if chromophore == "hbo" else "HbR"
        fig.suptitle(
            f"Expert vs Novice Group HRF - d{chrom_display}\n(mean +- 1 SD across subjects)",
            fontsize=13,
            fontweight="bold",
        )

        for col, region in enumerate(regions):
            ax = axes[col]
            panel_traces = {
                g: group_region_traces[g][region]
                for g in groups
                if group_region_traces[g][region].size > 0
            }
            panel_stats = plot_hrf_panel(
                ax, times_ref, panel_traces,
                region=region,
                chromophore=chromophore,
                show_legend=(col == len(regions) - 1),
                show_xlabel=True,
                show_ylabel=(col == 0),
            )
            all_stats[f"{chrom_label}_{region}"] = panel_stats

        plt.tight_layout()
        out_path = out_dir / f"hrf_plot_{chrom_label}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path.name)
        print(f"  Saved: {out_path.name}")

    return {"peak_stats": all_stats, "saved_files": saved}


def generate_hrf_plots_5region(
    groups: dict[str, list[str]],
    stage_b_dir: Path,
    task_name: str,
    region_map_5: dict[str, list[str]],
    out_dir: Path,
    target_sfreq: float,
) -> dict:
    top_regions = ["Left Lateral PFC", "Central PFC", "Right Lateral PFC"]
    bottom_regions = ["Left Medial M1", "SMA"]
    all_regions = top_regions + bottom_regions

    all_stats = {}
    saved = []

    for chromophore in ("hbo", "hbr"):
        chrom_label = "HbO" if chromophore == "hbo" else "HbR"
        chrom_display = "HbO2" if chromophore == "hbo" else "HbR"
        print(f"\n  [Plotting 5-region {chrom_label}]")

        group_region_traces = {}
        times_ref = None
        for group_name, subjects in groups.items():
            traces, t = collect_group_traces(
                subjects, stage_b_dir, task_name, region_map_5,
                chromophore=chromophore, target_sfreq=target_sfreq,
            )
            group_region_traces[group_name] = traces
            if times_ref is None and t is not None:
                times_ref = t

        if times_ref is None:
            print(f"    no traces found for 5-region {chrom_label}, skipped")
            continue

        mosaic = [["A", "B", "C"], ["D", "E", "legend"]]
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 8),
                                       gridspec_kw={"wspace": 0.35, "hspace": 0.50})
        panel_keys = {
            "Left Lateral PFC": "A",
            "Central PFC": "B",
            "Right Lateral PFC": "C",
            "Left Medial M1": "D",
            "SMA": "E",
        }

        fig.suptitle(
            f"Expert vs Novice Group HRF - d{chrom_display} (Figure 3c reproduction)\n"
            f"mean +- 1 SD across subjects",
            fontsize=11,
            fontweight="bold",
        )

        for region in all_regions:
            ax = axes[panel_keys[region]]
            panel_traces = {
                g: group_region_traces[g][region]
                for g in groups
                if group_region_traces[g].get(region, np.empty((0,))).size > 0
            }
            panel_stats = plot_hrf_panel(
                ax, times_ref, panel_traces,
                region=region,
                chromophore=chromophore,
                show_legend=False,
                show_xlabel=True,
                show_ylabel=(region in ("Left Lateral PFC", "Left Medial M1")),
            )
            all_stats[f"{chrom_label}_{region}"] = panel_stats

        ax_leg = axes["legend"]
        ax_leg.axis("off")
        legend_elements = []
        for group_name in groups:
            s = GROUP_STYLE[group_name]
            legend_elements.append(
                plt.Line2D([0], [0], color=s["mean_color"], lw=s["lw"], label=f"{s['label']} mean")
            )
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, fc=s["band_color"], alpha=s["band_alpha"],
                              label=f"{s['label']} SD")
            )
        legend_elements.append(
            plt.Line2D([0], [0], color="black", lw=1.5, ls="--", label="FLS task start")
        )
        ax_leg.legend(handles=legend_elements, loc="center", fontsize=9,
                      framealpha=0.9, handlelength=2.0)

        fig.subplots_adjust(top=0.88)
        out_path = out_dir / f"hrf_plot_5region_{chrom_label}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path.name)
        print(f"  Saved: {out_path.name}")

    return {"peak_stats": all_stats, "saved_files": saved}


def verify_physiology(all_stats: dict) -> dict:
    checks = {}
    print("\n  --- Physiological Verification ---")
    for key, group_stats in all_stats.items():
        chrom, region = key.split("_", 1)
        for group_name, ps in group_stats.items():
            amp = ps["peak_amp_uM"]
            check_key = f"{group_name}_{region}_{chrom}"
            if chrom == "HbO":
                ok = amp > 0
                label = f"HbO up ({amp:+.3f} uM)"
            else:
                ok = amp < 0
                label = f"HbR down ({amp:+.3f} uM)"
            checks[check_key] = {"pass": ok, "peak_amp_uM": amp, "peak_time_s": ps["peak_time_s"]}
            print(f"  {'[OK]' if ok else '[FAIL]'} {check_key:35s}: {label} @ {ps['peak_time_s']:.1f} s")

    n_pass = sum(1 for v in checks.values() if v["pass"])
    n_total = len(checks)
    print(f"\n  Physiology check: {n_pass}/{n_total} regions pass")
    return checks


def run_hrf_visualization(study: str = "expert_novice") -> dict:
    config = load_config()
    if study not in config.get("datasets", {}):
        raise ValueError(f"Unknown study '{study}'")

    ds_cfg = config["datasets"][study]
    groups = ds_cfg.get("groups", {})
    if not groups:
        raise RuntimeError(f"No groups defined in config for study '{study}'")

    task_name = ds_cfg.get("bids_task", "task")
    target_sfreq = float(ds_cfg.get("sfreq", 25.0))
    study_display_name = Path(ds_cfg["data_path"]).name

    stage_b_dir = (Path(config["paths"]["stage_b_outputs"])
                   / "processed_data" / study.replace("_", "-"))
    if not stage_b_dir.exists():
        raise FileNotFoundError(f"Stage B processed dir not found: {stage_b_dir}")

    # Reference raw for channel geometry
    first_group = next(iter(groups.keys()))
    first_subject = groups[first_group][0]
    raw_candidates = sorted(stage_b_dir.glob(f"sub-{first_subject}_*_desc-preprocessed_nirs_raw.fif"))
    if not raw_candidates:
        raw_candidates = sorted(stage_b_dir.glob(f"sub-{first_subject}_*_raw.fif"))
    if not raw_candidates:
        raise FileNotFoundError(f"No preprocessed raw fif for reference subject {first_subject}")

    raw_ref = mne.io.read_raw_fif(raw_candidates[0], preload=False, verbose=False)
    region_map = build_region_map(raw_ref)
    region_map_5 = build_region_map_5(raw_ref)

    out_base = Path(config["paths"]["stage_c_outputs"]) / "hrf" / study_display_name
    out_dir = out_base / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Stage C - HRF Visualization (GLM-first)")
    print(f"  Study  : {study_display_name}")
    print(f"  Input  : {stage_b_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Groups : {', '.join(f'{g} (n={len(s)})' for g, s in groups.items())}")
    print("=" * 60)

    print("\n--- 3-Region Plots ---")
    r3 = generate_hrf_plots(
        groups=groups,
        stage_b_dir=stage_b_dir,
        task_name=task_name,
        region_map=region_map,
        out_dir=out_dir,
        target_sfreq=target_sfreq,
    )

    print("\n--- 5-Region Plots ---")
    r5 = generate_hrf_plots_5region(
        groups=groups,
        stage_b_dir=stage_b_dir,
        task_name=task_name,
        region_map_5=region_map_5,
        out_dir=out_dir,
        target_sfreq=target_sfreq,
    )

    combined_stats = {**r3["peak_stats"], **r5["peak_stats"]}
    phys = verify_physiology(combined_stats)

    all_saved = r3["saved_files"] + r5["saved_files"]
    n_pass = sum(1 for v in phys.values() if v["pass"])
    n_total = len(phys)

    print("\n" + "=" * 60)
    print("  Visualization complete")
    print(f"  Saved files: {', '.join(all_saved)}")
    print(f"  Physiology : {n_pass}/{n_total} pass")
    print("=" * 60 + "\n")

    return {
        "saved_files": all_saved,
        "peak_stats": combined_stats,
        "physiology_checks": phys,
        "physiology_pass": f"{n_pass}/{n_total}",
        "region_map_3": region_map,
        "region_map_5": region_map_5,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage C HRF visualization (GLM-first)")
    parser.add_argument("--study", default="expert_novice", help="Study key in config.yaml")
    args = parser.parse_args()
    run_hrf_visualization(study=args.study)
