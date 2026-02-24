"""
Stage C — Step C1 (Visualization): HRF Visualization
Purpose: Replicate paper Figure 3c/3d style — per-region HRF curves for Expert
         and Novice groups, with mean line + ±1 SD shading for both HbO and HbR.

Task coverage:
  4.3.1 — HRF plotting function (paper-style: Expert dark-red, Novice green,
           SD bands, vertical task-onset line, horizontal zero line)
  4.3.2 — Generate plots for all regions (PFC, M1, SMA) × both groups
  4.3.3 — Save plots and compare to paper Figure 3c, 3d
  4.3.4 — Verify HRF characteristics match expected physiology

Figure layout
-------------
One figure per chromophore (HbO / HbR), each with 1×3 subplots (PFC | M1 | SMA).
Saves two files:
  hrf_plot_HbO.png   — ΔHbO mean ± SD for Expert (red) and Novice (green)
  hrf_plot_HbR.png   — ΔHbR mean ± SD for Expert (red) and Novice (green)

Per-subject SD
--------------
SD bands are computed from per-subject region-mean traces (not across channels):
  1. Load each subject's -ave.fif.
  2. Harmonise (resample if needed, drop NaN channels — same as Section 4.2).
  3. For each region × chromophore, average across that region's channels.
  4. Stack per-subject traces → array (n_subjects × n_times).
  5. Compute group mean and ±1 SD.

Visual style (matches paper Figure 3c)
  Expert   : mean line  #8B1A1A (dark red)  |  SD band  #FFAAAA (light red/pink)
  Novice   : mean line  #1A6B1A (dark green) |  SD band  #A8E6A8 (light green)
  Task start: vertical dashed black line at t = 0
  Zero line : thin grey horizontal line at y = 0

Outputs (outputs/stage_c_validation/hrf/{study_name}/group/)
  hrf_plot_HbO.png
  hrf_plot_HbR.png

Upstream : 01_hrf_extraction.py, 02_group_hrf_analysis.py
Reference : PROGRESS.md Section 4.3; paper Figure 3c, 3d
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config

# ---------------------------------------------------------------------------
# Import Section 4.1 / 4.2 helpers
# ---------------------------------------------------------------------------
_spec_01 = importlib.util.spec_from_file_location(
    "hrf_extraction", Path(__file__).parent / "01_hrf_extraction.py"
)
_hrf = importlib.util.module_from_spec(_spec_01)
_spec_01.loader.exec_module(_hrf)

_spec_02 = importlib.util.spec_from_file_location(
    "group_hrf", Path(__file__).parent / "02_group_hrf_analysis.py"
)
_grp = importlib.util.module_from_spec(_spec_02)
_spec_02.loader.exec_module(_grp)

build_region_map               = _hrf.build_region_map
_load_or_compute_subject_evoked = _grp._load_or_compute_subject_evoked
_harmonise_evoked              = _grp._harmonise_evoked

# ---------------------------------------------------------------------------
# Groups and study constants
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "Expert": ["A1", "A2", "R7", "R8", "R9", "R10", "R11", "R12"],
    "Novice": ["R1", "R2", "R3", "R4", "R5", "R6"],
}
STUDY = "expert_novice"
TARGET_SFREQ = 25.0

# Visual style (paper Figure 3c palette)
GROUP_STYLE: dict[str, dict] = {
    "Expert": {
        "mean_color":  "#8B1A1A",   # dark red
        "band_color":  "#FFAAAA",   # light pink
        "band_alpha":  0.45,
        "lw": 2.0,
        "label": "Expert surgeon",
    },
    "Novice": {
        "mean_color":  "#1A6B1A",   # dark green
        "band_color":  "#A8E6A8",   # light green
        "band_alpha":  0.40,
        "lw": 2.0,
        "label": "Novice surgeon",
    },
}


# ===========================================================================
# 4.3.1 — Per-subject region-mean trace extraction
# ===========================================================================

def collect_group_traces(
    subjects: list[str],
    out_base: Path,
    study: str,
    region_map: dict[str, list[str]],
    chromophore: str,
    target_sfreq: float = TARGET_SFREQ,
) -> dict[str, np.ndarray]:
    """
    For each subject in `subjects`, load the per-subject -ave.fif, harmonise,
    and compute the region-mean HbO or HbR trace.

    Returns
    -------
    traces : dict  {region: ndarray (n_subjects × n_times)}
        Only includes subjects whose -ave.fif exists and has data for the region.
    """
    region_matrices: dict[str, list[np.ndarray]] = {r: [] for r in region_map}
    times_ref: np.ndarray | None = None

    for sid in subjects:
        ev = _load_or_compute_subject_evoked(sid, out_base, study)
        if ev is None:
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
            # region-mean trace for this subject (μM)
            trace = ev.data[idx, :].mean(axis=0) * 1e6
            region_matrices[region].append(trace)

    return {
        r: np.vstack(mats) if mats else np.empty((0, 0))
        for r, mats in region_matrices.items()
    }, times_ref


# ===========================================================================
# 4.3.1 — Core plotting function (paper-style)
# ===========================================================================

def plot_hrf_panel(
    ax: plt.Axes,
    times: np.ndarray,
    group_traces: dict[str, np.ndarray],   # {group_name: (n_subj × n_times)}
    region: str,
    chromophore: str,
    show_legend: bool = False,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> dict:
    """
    Draw one HRF panel on `ax`:
      - Group mean ± 1 SD shading for Expert (red) and Novice (green)
      - Vertical dashed line at t = 0 (task onset)
      - Horizontal grey line at y = 0

    Returns per-group peak stats for the panel.
    """
    peak_window = (5.0, 15.0)
    t_mask = (times >= peak_window[0]) & (times <= peak_window[1])
    stats: dict = {}

    # Draw zero and onset lines first (behind data)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(0, color="black", lw=1.5, ls="--", zorder=2, label="FLS task start")

    for group_name, mat in group_traces.items():
        if mat.size == 0:
            continue
        style = GROUP_STYLE[group_name]
        mean  = mat.mean(axis=0)
        sd    = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

        # SD band
        ax.fill_between(
            times, mean - sd, mean + sd,
            color=style["band_color"], alpha=style["band_alpha"], zorder=3,
        )
        # Mean line
        ax.plot(
            times, mean,
            color=style["mean_color"], lw=style["lw"],
            label=f"{style['label']} mean  (n={mat.shape[0]})",
            zorder=4,
        )

        # Peak stats
        if chromophore == "hbo":
            pidx = np.argmax(mean[t_mask])
        else:
            pidx = np.argmin(mean[t_mask])
        pt = float(times[t_mask][pidx])
        pa = float(mean[t_mask][pidx])
        stats[group_name] = {"peak_time_s": round(pt, 2), "peak_amp_uM": round(pa, 3)}

    # Formatting
    chrom_label = "HbO₂" if chromophore == "hbo" else "HbR"
    ax.set_title(f"{region} — Δ{chrom_label}", fontsize=11, fontweight="bold")
    ax.set_xlim(times[0], times[-1])

    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=10)
    if show_ylabel:
        chrom_unit = "HbO₂" if chromophore == "hbo" else "HbR"
        ax.set_ylabel(f"Δ{chrom_unit} conc. (μM)", fontsize=10)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Put task-start line last
        task_h = [h for h, l in zip(handles, labels) if "FLS" in l]
        data_h = [h for h, l in zip(handles, labels) if "FLS" not in l]
        task_l = [l for l in labels if "FLS" in l]
        data_l = [l for l in labels if "FLS" not in l]
        ax.legend(
            data_h + task_h, data_l + task_l,
            fontsize=8, loc="upper right", framealpha=0.85,
        )

    ax.grid(True, alpha=0.25, lw=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    return stats


# ===========================================================================
# 4.3.2 — Generate all 6 plots (2 chromophores × 3 regions)
# ===========================================================================

def generate_hrf_plots(
    groups: dict[str, list[str]],
    out_base: Path,
    study: str,
    region_map: dict[str, list[str]],
    out_dir: Path,
) -> dict:
    """
    Build two figures (one per chromophore), each with 1×3 region subplots.
    Saves hrf_plot_HbO.png and hrf_plot_HbR.png to out_dir.

    Returns peak stats dict.
    """
    regions   = list(region_map.keys())
    all_stats: dict = {}
    saved:    list[str] = []

    for chromophore in ("hbo", "hbr"):
        chrom_label = "HbO" if chromophore == "hbo" else "HbR"
        print(f"\n  [Plotting {chrom_label}]")

        # ── Collect per-subject traces for each group ──────────────────────
        group_region_traces: dict[str, dict[str, np.ndarray]] = {}
        times_ref: np.ndarray | None = None

        for group_name, subjects in groups.items():
            traces, t = collect_group_traces(
                subjects, out_base, study, region_map,
                chromophore=chromophore,
            )
            group_region_traces[group_name] = traces
            if times_ref is None and t is not None:
                times_ref = t

        if times_ref is None:
            print(f"    ⚠ No time array found, skipping {chrom_label}")
            continue

        # ── Build figure ───────────────────────────────────────────────────
        n_regions = len(regions)
        fig, axes = plt.subplots(
            1, n_regions,
            figsize=(5 * n_regions, 5),
            sharey=False,
        )
        chrom_display = "HbO₂" if chromophore == "hbo" else "HbR"
        fig.suptitle(
            f"Expert vs. Novice Group HRF — Δ{chrom_display}\n"
            f"(mean ± 1 SD across subjects)",
            fontsize=13, fontweight="bold",
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
                show_legend=(col == n_regions - 1),
                show_xlabel=True,
                show_ylabel=(col == 0),
            )
            all_stats[f"{chrom_label}_{region}"] = panel_stats

            n_expert = panel_traces.get("Expert", np.empty((0,))).shape[0]
            n_novice = panel_traces.get("Novice", np.empty((0,))).shape[0]
            print(f"    {region}: Expert n={n_expert}, Novice n={n_novice}")
            for grp, ps in panel_stats.items():
                print(f"      {grp}: peak={ps['peak_amp_uM']:+.3f} μM "
                      f"@ {ps['peak_time_s']:.1f} s")

        plt.tight_layout()
        out_path = out_dir / f"hrf_plot_{chrom_label}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path.name)
        print(f"  Saved: {out_path.name}")

    return {"peak_stats": all_stats, "saved_files": saved}


# ===========================================================================
# 4.3.4 — Physiological verification
# ===========================================================================

def verify_physiology(all_stats: dict) -> dict:
    """
    Check that HbO peaks are positive and HbR troughs are negative in M1 and SMA,
    where activation is strongest for a motor task.
    """
    checks: dict = {}
    print("\n  --- Physiological Verification ---")

    for key, group_stats in all_stats.items():
        chrom, region = key.split("_", 1)
        for group_name, ps in group_stats.items():
            amp   = ps["peak_amp_uM"]
            check_key = f"{group_name}_{region}_{chrom}"

            if chrom == "HbO":
                ok = amp > 0
                label = f"HbO↑ ({amp:+.3f} μM)"
            else:
                ok = amp < 0
                label = f"HbR↓ ({amp:+.3f} μM)"

            checks[check_key] = {"pass": ok, "peak_amp_uM": amp,
                                  "peak_time_s": ps["peak_time_s"]}
            flag = "✓" if ok else "✗"
            print(f"  {flag} {check_key:35s}: {label} @ {ps['peak_time_s']:.1f} s")

    n_pass = sum(1 for v in checks.values() if v["pass"])
    n_total = len(checks)
    print(f"\n  Physiology check: {n_pass}/{n_total} regions pass")
    return checks


# ===========================================================================
# Top-level pipeline
# ===========================================================================

def run_hrf_visualization(
    groups: dict[str, list[str]] = GROUPS,
    study: str = STUDY,
) -> dict:
    """
    Full Section 4.3 pipeline.
    """
    config = load_config()
    study_display_name = Path(config["datasets"][study]["data_path"]).name
    out_base = Path(config["paths"]["stage_c_outputs"]) / "hrf" / study_display_name
    out_dir  = out_base / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Region map (from reference subject A1) ────────────────────────────
    stage_b_dir = (Path(config["paths"]["stage_b_outputs"])
                   / "processed_data"
                   / study.replace("_", "-"))
    raw_ref = mne.io.read_raw_fif(
        list(stage_b_dir.glob("sub-A1_*_raw.fif"))[0],
        preload=False, verbose=False,
    )
    region_map = build_region_map(raw_ref)

    print(f"\n{'='*60}")
    print(f"  Section 4.3 — HRF Visualization")
    print(f"  Study  : {study_display_name}")
    print(f"  Groups : {', '.join(f'{g} (n={len(s)})' for g, s in groups.items())}")
    print(f"  Regions: {', '.join(f'{r}={len(v)}ch' for r, v in region_map.items())}")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}")

    # ── 4.3.1 + 4.3.2 + 4.3.3 — Plot both chromophores ──────────────────
    plot_results = generate_hrf_plots(
        groups, out_base, study, region_map, out_dir
    )

    # ── 4.3.4 — Physiology check ─────────────────────────────────────────
    physio_checks = verify_physiology(plot_results["peak_stats"])

    n_pass  = sum(1 for v in physio_checks.values() if v["pass"])
    n_total = len(physio_checks)

    print(f"\n{'='*60}")
    print(f"  Section 4.3 complete")
    print(f"  Saved: {', '.join(plot_results['saved_files'])}")
    print(f"  Physiology: {n_pass}/{n_total} pass")
    print(f"{'='*60}\n")

    return {
        "saved_files": plot_results["saved_files"],
        "peak_stats": plot_results["peak_stats"],
        "physiology_checks": physio_checks,
        "physiology_pass": f"{n_pass}/{n_total}",
    }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_hrf_visualization()
