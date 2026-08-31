"""
Stage E — Section 10.11-10.12: Cross-Method Significance Comparison.

Answers the researcher's actual question for Stage D/E as a whole: **which
of GC / Phase 2 / Phase 4 (raw and ETE) show significant Expert-vs-Novice
results, and do they agree or contrast?** Every registered AI track
(``significance_methods`` entries with ``ai_track: true`` — currently
``gc``, ``phase_2``, ``phase_4_raw``, ``phase_4_ete``; ``phase_2_ete`` and a
Geweke-equipped ``gc`` will appear automatically once 10.4b/10.7 are
activated per the Section 10.13 runbook) is compared on:

  (a) H1 raw-magnitude significance — is *any* connection FDR-significant
      in this method's own group-level H1 test (Stage D's ANOVA-based test
      for GC, the project's MWU-based test for the symbolic-TE methods)?
  (b) AI significance — H_AI1 (per-pair MWU+FDR) and H_AI3 (pre-specified
      pooled PFC-LM1 permutation test) results, from ``12_asymmetry_index_core.py``.
  (c) For ETE-equipped tracks: mean noise-floor-fraction (how much of the
      raw signal is surrogate floor vs. genuine coupling).
  (d) **Directional agreement** — for every connection, the sign of
      mean(Expert) − mean(Novice) is computed generically via
      ``12_asymmetry_index_core.load_method_arrays`` (the *same* arrays
      each method's own H1/AI tests ran on, so directions are directly
      comparable) — flagged wherever ≥2 methods are independently
      significant on the same connection to check if they agree or
      contradict.

Registry-driven, no method hardcoded
-------------------------------------
Every data source is resolved through ``config.yaml``'s
``significance_methods`` registry:
  - ``h1_loader`` (new, Section 10.11): ``{kind: "stage_d_csv"|"output_subdir_csv",
    dir_key, filename}`` — resolves each method's own H1 CSV. GC's on-disk
    location differs from the symbolic methods' (Stage D vs. Stage E output
    roots), the one place a bespoke ``kind`` is needed.
  - ``ai_stats_dir`` (Section 10.9-10.10): reused here to find
    ``ai_group_comparison.csv``/``ai_lm1_focused.csv``/``ete_floor_fraction.csv``.
  - ``ai_loader`` (Section 10.9): reused via
    ``12_asymmetry_index_core.load_method_arrays`` for directional signs.

Adding a 5th method (Section 10.12's extensibility proof, or a real future
method like Phase 2 ETE) requires **zero changes to this file** — only a
new registry entry with these three loader blocks.

Outputs
-------
``cross_method_significance/``
  significance_summary.csv     one row per (method, connection): H1/AI
                                significance, p-values, direction sign.
  method_overview.csv          one row per method: counts + floor fraction.
  significance_comparison.png  method x connection matrix (direction color,
                                significance asterisk).

CLI
---
python3 16_cross_method_significance.py
python3 16_cross_method_significance.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).parent.parent          # code/
_STAGE_E_ROOT = Path(__file__).parent              # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_PFC_LM1_PAIR_LABELS = ["LPFC\u2194LM1", "CPFC\u2194LM1", "RPFC\u2194LM1"]


def _import_ai_core():
    """Load ``12_asymmetry_index_core.py`` (leading-digit filename -> importlib)."""
    p = _STAGE_E_ROOT / "12_asymmetry_index_core.py"
    spec = importlib.util.spec_from_file_location("asymmetry_index_core", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_registry_entry(method_key: str, config: dict) -> dict:
    return config["stage_e_transfer_entropy"]["significance_methods"][method_key]


# ---------------------------------------------------------------------------
# (a) H1 raw-magnitude significance
# ---------------------------------------------------------------------------

def load_h1_dataframe(method_key: str, config: dict) -> Optional[pd.DataFrame]:
    """Load a method's own H1 (Expert-vs-Novice) test CSV via its ``h1_loader``."""
    entry = get_registry_entry(method_key, config)
    loader = entry.get("h1_loader")
    if loader is None:
        return None

    kind = loader.get("kind")
    if kind == "stage_d_csv":
        from stage_d_granger.utils_gc import get_gc_output_dir
        d = get_gc_output_dir(loader["dir_key"], config=config)
    elif kind == "output_subdir_csv":
        from utils_te import get_te_output_dir
        d = get_te_output_dir(loader["dir_key"], config=config)
    else:
        logger.warning("load_h1_dataframe [%s]: unknown h1_loader.kind=%r.", method_key, kind)
        return None

    p = d / loader["filename"]
    if not p.exists():
        logger.warning("load_h1_dataframe [%s]: file not found: %s", method_key, p)
        return None
    return pd.read_csv(p)


def summarize_h1(method_key: str, config: dict, connection_labels: list[str]) -> dict:
    """Per-connection H1 significance summary for one method."""
    df = load_h1_dataframe(method_key, config)
    if df is None:
        return {"status": "missing", "per_connection": {c: None for c in connection_labels}}

    per_conn: dict[str, Optional[dict]] = {}
    for conn in connection_labels:
        sub = df[df["connection"] == conn]
        if sub.empty:
            per_conn[conn] = None
            continue
        is_sig = bool(sub["significant"].fillna(False).any())
        if is_sig:
            min_p = float(sub.loc[sub["significant"] == True, "p_fdr"].min())
        else:
            min_p = float(sub["p_fdr"].min()) if "p_fdr" in sub.columns and sub["p_fdr"].notna().any() else np.nan
        per_conn[conn] = {"significant": is_sig, "p_fdr": min_p, "n_rows": int(len(sub))}

    n_sig_conn = sum(1 for v in per_conn.values() if v is not None and v["significant"])
    return {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_significant_connections": n_sig_conn,
        "per_connection": per_conn,
    }


# ---------------------------------------------------------------------------
# (b) AI significance (H_AI1 per-pair, H_AI3 pooled PFC-LM1)
# ---------------------------------------------------------------------------

def summarize_ai(method_key: str, config: dict) -> dict:
    entry = get_registry_entry(method_key, config)
    stats_dir_key = entry.get("ai_stats_dir")
    if stats_dir_key is None:
        return {"status": "missing", "per_pair": {}}

    from utils_te import get_te_output_dir
    stats_dir = get_te_output_dir(stats_dir_key, config=config)

    per_pair: dict[str, dict] = {}
    h_ai1_path = stats_dir / "ai_group_comparison.csv"
    if h_ai1_path.exists():
        df1 = pd.read_csv(h_ai1_path)
        for _, row in df1.iterrows():
            per_pair[str(row["pair"])] = {
                "significant": bool(row.get("mwu_significant", False)),
                "p_fdr": float(row.get("mwu_p_fdr")) if pd.notna(row.get("mwu_p_fdr")) else np.nan,
                "ai_diff": float(row.get("ai_diff")) if pd.notna(row.get("ai_diff")) else np.nan,
            }

    out: dict = {
        "status": "ok" if h_ai1_path.exists() else "missing",
        "per_pair": per_pair,
        "h_ai1_n_significant": sum(1 for v in per_pair.values() if v["significant"]),
    }

    lm1_path = stats_dir / "ai_lm1_focused.csv"
    if lm1_path.exists():
        df_lm1 = pd.read_csv(lm1_path)
        pooled = df_lm1[df_lm1["pair"] == "PFC-LM1 - 3 pairs pooled (primary)"]
        if not pooled.empty:
            row = pooled.iloc[0]
            diff = row.get("ai_diff_exp_minus_nov", np.nan)
            out["h_ai3_pooled_significant"] = bool(row.get("perm_significant_1s", False))
            out["h_ai3_pooled_p"] = float(row.get("perm_p_onesided_greater")) if pd.notna(row.get("perm_p_onesided_greater")) else np.nan
            out["h_ai3_pooled_direction"] = (
                "Expert>Novice" if pd.notna(diff) and diff > 0 else
                "Novice>Expert" if pd.notna(diff) and diff < 0 else "n/a"
            )
    return out


# ---------------------------------------------------------------------------
# (c) ETE floor-fraction diagnostic
# ---------------------------------------------------------------------------

def load_floor_fraction(method_key: str, config: dict) -> Optional[pd.DataFrame]:
    entry = get_registry_entry(method_key, config)
    stats_dir_key = entry.get("ai_stats_dir")
    if stats_dir_key is None:
        return None
    from utils_te import get_te_output_dir
    p = get_te_output_dir(stats_dir_key, config=config) / "ete_floor_fraction.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# (d) Directional sign per connection (generic, via ai_loader arrays)
# ---------------------------------------------------------------------------

def compute_direction_map(method_key: str, config: dict, ai_core, connection_labels: list[str]) -> dict[str, float]:
    """connection_label -> sign(mean(Expert) - mean(Novice)), NaN if unavailable."""
    B_Exp, B_Nov = ai_core.load_method_arrays(method_key, config)
    if B_Exp is None or B_Nov is None:
        return {c: np.nan for c in connection_labels}
    mean_exp = np.nanmean(B_Exp, axis=(0, 2))
    mean_nov = np.nanmean(B_Nov, axis=(0, 2))
    diff = mean_exp - mean_nov
    return {connection_labels[i]: float(diff[i]) for i in range(len(connection_labels))}


# ---------------------------------------------------------------------------
# Master builder (Section 10.11)
# ---------------------------------------------------------------------------

def compare_significance_across_methods(
    config: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """
    Build the full cross-method significance comparison (Section 10.11).

    Returns
    -------
    dict
        Keys: ``significance_summary`` (long, method x connection),
        ``method_overview`` (one row per method), ``figure`` (Path).
    """
    if config is None:
        config = load_config()

    from utils_te import get_connection_labels, get_te_output_dir

    ai_core = _import_ai_core()
    connection_labels = get_connection_labels()

    out_dir = get_te_output_dir("cross_method_significance", config=config)
    summary_path = out_dir / "significance_summary.csv"
    overview_path = out_dir / "method_overview.csv"

    if not overwrite and summary_path.exists() and overview_path.exists():
        logger.info("compare_significance_across_methods: cache hit.")
        return {
            "significance_summary": pd.read_csv(summary_path),
            "method_overview": pd.read_csv(overview_path),
            "figure": out_dir / "significance_comparison.png",
        }

    tracks = ai_core.registered_ai_tracks(config)
    logger.info("compare_significance_across_methods: comparing tracks = %s", tracks)

    long_rows = []
    overview_rows = []
    direction_by_method: dict[str, dict[str, float]] = {}
    h1_by_method: dict[str, dict] = {}
    ai_by_method: dict[str, dict] = {}

    for mk in tracks:
        entry = get_registry_entry(mk, config)
        h1 = summarize_h1(mk, config, connection_labels)
        ai = summarize_ai(mk, config)
        floor_df = load_floor_fraction(mk, config)
        direction = compute_direction_map(mk, config, ai_core, connection_labels)

        h1_by_method[mk] = h1
        ai_by_method[mk] = ai
        direction_by_method[mk] = direction

        mean_floor = float(floor_df["floor_fraction"].mean()) if floor_df is not None else np.nan

        # Map connections -> undirected pair label for AI lookup.
        for conn in connection_labels:
            roi_a, roi_b = conn.split("\u2192")
            pair_lbl_1 = f"{roi_a}\u2194{roi_b}"
            pair_lbl_2 = f"{roi_b}\u2194{roi_a}"
            ai_entry = ai["per_pair"].get(pair_lbl_1) or ai["per_pair"].get(pair_lbl_2)

            h1_entry = h1["per_connection"].get(conn)
            long_rows.append({
                "method_key": mk,
                "method_label": entry.get("label", mk),
                "kind": "ETE (floor-corrected)" if "ete" in mk else "raw magnitude",
                "connection": conn,
                "is_pfc_lm1_pair": (pair_lbl_1 in _PFC_LM1_PAIR_LABELS or pair_lbl_2 in _PFC_LM1_PAIR_LABELS),
                "direction_sign": direction.get(conn, np.nan),
                "h1_significant": h1_entry["significant"] if h1_entry else False,
                "h1_p_fdr": h1_entry["p_fdr"] if h1_entry else np.nan,
                "ai1_significant": ai_entry["significant"] if ai_entry else False,
                "ai1_p_fdr": ai_entry["p_fdr"] if ai_entry else np.nan,
            })

        overview_rows.append({
            "method_key": mk,
            "method_label": entry.get("label", mk),
            "kind": "ETE (floor-corrected)" if "ete" in mk else "raw magnitude",
            "h1_status": h1["status"],
            "h1_n_significant_connections": h1.get("n_significant_connections", 0),
            "ai1_n_significant_pairs": ai.get("h_ai1_n_significant", 0),
            "ai3_pooled_significant": ai.get("h_ai3_pooled_significant", False),
            "ai3_pooled_p": ai.get("h_ai3_pooled_p", np.nan),
            "ai3_pooled_direction": ai.get("h_ai3_pooled_direction", "n/a"),
            "mean_floor_fraction": round(mean_floor, 4) if np.isfinite(mean_floor) else np.nan,
            "any_significant": bool(
                h1.get("n_significant_connections", 0) > 0
                or ai.get("h_ai1_n_significant", 0) > 0
                or ai.get("h_ai3_pooled_significant", False)
            ),
        })

    df_long = pd.DataFrame(long_rows)
    df_overview = pd.DataFrame(overview_rows)

    # --- (d) Directional agreement check: connections where >=2 methods are
    # independently significant (H1 or AI1) -- do their direction signs agree? ---
    agreement_rows = []
    for conn in connection_labels:
        sub = df_long[df_long["connection"] == conn]
        sig_sub = sub[sub["h1_significant"] | sub["ai1_significant"]]
        if len(sig_sub) >= 2:
            signs = np.sign(sig_sub["direction_sign"].values)
            agree = bool(np.all(signs == signs[0]))
            agreement_rows.append({
                "connection": conn,
                "n_methods_significant": len(sig_sub),
                "methods": ";".join(sig_sub["method_key"].tolist()),
                "signs": ";".join(f"{s:+.0f}" for s in signs),
                "agree": agree,
            })
    if agreement_rows:
        pd.DataFrame(agreement_rows).to_csv(out_dir / "direction_agreement.csv", index=False)
        logger.info("compare_significance_across_methods: %d connection(s) with >=2 significant methods -> see direction_agreement.csv", len(agreement_rows))
    else:
        logger.info(
            "compare_significance_across_methods: no connection has >=2 independently-significant "
            "methods this pass (expected with n=13 -- see method_overview.csv per-method significance instead)."
        )

    df_long.to_csv(summary_path, index=False)
    df_overview.to_csv(overview_path, index=False)
    logger.info("compare_significance_across_methods: saved %s (%d rows), %s (%d rows)",
                summary_path.name, len(df_long), overview_path.name, len(df_overview))

    fig_path = plot_significance_comparison(
        df_long, tracks, connection_labels, out_dir, config=config,
    )

    return {"significance_summary": df_long, "method_overview": df_overview, "figure": fig_path}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_significance_comparison(
    df_long: pd.DataFrame,
    tracks: list[str],
    connection_labels: list[str],
    out_dir: Path,
    config: Optional[dict] = None,
) -> Path:
    """
    Method x connection matrix: color = direction sign (diverging, clipped),
    asterisk = significant (H1 or AI1) on that connection/pair.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if config is None:
        config = load_config()

    out_path = out_dir / "significance_comparison.png"
    n_methods = len(tracks)
    n_conn = len(connection_labels)

    matrix = np.full((n_methods, n_conn), np.nan)
    sig_mask = np.zeros((n_methods, n_conn), dtype=bool)
    labels_y = []

    for mi, mk in enumerate(tracks):
        entry = get_registry_entry(mk, config)
        labels_y.append(entry.get("label", mk))
        sub = df_long[df_long["method_key"] == mk].set_index("connection")
        for ci, conn in enumerate(connection_labels):
            if conn not in sub.index:
                continue
            row = sub.loc[conn]
            matrix[mi, ci] = row["direction_sign"]
            sig_mask[mi, ci] = bool(row["h1_significant"] or row["ai1_significant"])

    # Clip / normalise per-row (methods have very different raw-value scales)
    # so the diverging colormap is comparable across rows -- only the SIGN
    # and relative magnitude within a method's own row matter here.
    norm_matrix = np.full_like(matrix, np.nan)
    for mi in range(n_methods):
        row = matrix[mi]
        max_abs = np.nanmax(np.abs(row)) if np.any(np.isfinite(row)) else 1.0
        norm_matrix[mi] = row / max_abs if max_abs > 0 else row

    fig, ax = plt.subplots(figsize=(15, 2.2 + 0.9 * n_methods))
    im = ax.imshow(norm_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_conn))
    ax.set_xticklabels(connection_labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(labels_y, fontsize=9)

    for mi in range(n_methods):
        for ci in range(n_conn):
            if sig_mask[mi, ci]:
                ax.text(ci, mi, "*", ha="center", va="center", fontsize=14, color="black", fontweight="bold")

    pfc_lm1_conns = {c for c in connection_labels if "LM1" in c and ("LPFC" in c or "CPFC" in c or "RPFC" in c)}
    for ci, conn in enumerate(connection_labels):
        if conn in pfc_lm1_conns:
            ax.axvspan(ci - 0.5, ci + 0.5, color="gold", alpha=0.12, zorder=-1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Direction (Expert \u2212 Novice), row-normalised sign/magnitude", fontsize=8)

    ax.set_title(
        "Cross-Method Significance Comparison \u2014 Expert vs Novice Directed Connectivity\n"
        "Red = Expert > Novice  |  Blue = Novice > Expert  |  * = significant (H1 or AI, own test)  |  "
        "gold columns = pre-specified PFC\u2194LM1 pairs",
        fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_significance_comparison: saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Stage E Section 10.11 -- Cross-Method Significance Comparison")
    parser.set_defaults(overwrite=False)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    cfg = load_config()
    out = compare_significance_across_methods(config=cfg, overwrite=args.overwrite)
    df_overview = out["method_overview"]
    logger.info("=== Method Overview ===")
    for _, row in df_overview.iterrows():
        logger.info(
            "[%s] H1_sig=%d  AI1_sig=%d  AI3_pooled_sig=%s (%s)  any_significant=%s",
            row["method_key"], row["h1_n_significant_connections"], row["ai1_n_significant_pairs"],
            row["ai3_pooled_significant"], row["ai3_pooled_direction"], row["any_significant"],
        )
    logger.info("Figure: %s", out["figure"])


if __name__ == "__main__":
    _cli_main()
