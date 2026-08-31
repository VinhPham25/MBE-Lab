"""
Stage E — Phase 4: Asymmetry Index Statistics (Section 9.9) — legacy CLI shim.

**This file is now a thin backward-compatible wrapper.** The full
statistical core (per-subject AI computation, H_AI1/H_AI2/H_AI3 tests,
visualizations) was refactored into a method-agnostic module,
``12_asymmetry_index_core.py`` (Section 10.9), so the same code can run for
any registered connectivity method (GC, Phase 2 dSPTE, Phase 4 raw dSTE,
Phase 4 ETE) instead of only Phase 4 TE-vs-GC.

This shim exists purely so any pre-existing tooling/muscle-memory that runs
``python3 18_asymmetry_index_statistics.py`` keeps working — it calls
``run_ai_for_method("phase_4", ...)`` in the core, which resolves to the
same raw amplitude-dSTE data this file always used
(``phase_4_group/B_{Exp,Nov}_amplitude_mean.npz``) and writes to the same
output directories (``phase_4_statistics/``, ``phase_4_viz/``) as before.

**Schema change note:** the original version of this file computed TE and
GC asymmetry indices side-by-side in every CSV (``te_mwu_p_raw``,
``gc_mwu_p_raw``, ...) for direct eyeball comparison. The generic core
computes one method per run, so the CSVs produced by this shim now have
single, unprefixed columns (``mwu_p_raw``, ``ai_diff``, ...) rather than
``te_*``/``gc_*`` pairs. GC is now its own registered AI track
(``method_key="gc"``, see the core's CLI or ``run_ai_all_methods()``) and
cross-method agreement/comparison is handled by
``16_cross_method_significance.py`` (Section 10.11), which reads all
tracks' per-method CSVs back in — this is a deliberate architectural
change, not a regression: it lets a 5th/6th method be compared without
rewriting this file's statistics again (Section 10.12).

CLI (unchanged)
---------------
python3 18_asymmetry_index_statistics.py
python3 18_asymmetry_index_statistics.py --test all
python3 18_asymmetry_index_statistics.py --test per-connection
python3 18_asymmetry_index_statistics.py --test cross-pair
python3 18_asymmetry_index_statistics.py --test lm1
python3 18_asymmetry_index_statistics.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).parent.parent.parent        # code/
_STAGE_E_ROOT = Path(__file__).parent.parent             # stage_e_transfer_entropy/

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_METHOD_KEY = "phase_4"   # backward-compat alias registered in config.yaml


def _import_ai_core():
    """Load ``12_asymmetry_index_core.py`` (leading-digit filename -> importlib)."""
    p = _STAGE_E_ROOT / "12_asymmetry_index_core.py"
    spec = importlib.util.spec_from_file_location("asymmetry_index_core", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_all_ai_statistics(config=None, overwrite: bool = False) -> dict:
    """Legacy entry point — delegates to ``run_ai_for_method("phase_4", ...)``."""
    core = _import_ai_core()
    if config is None:
        config = load_config()
    result = core.run_ai_for_method(_METHOD_KEY, config=config, overwrite=overwrite)
    if result is None:
        raise RuntimeError(
            "run_all_ai_statistics: phase_4 raw amplitude-dSTE group arrays not found. "
            "Run 15_amplitude_dste_group.py first."
        )
    return result


def _cli_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Stage E Phase 4 — Asymmetry Index Statistics (Section 9.9, legacy shim over 12_asymmetry_index_core.py)"
    )
    parser.add_argument("--test", choices=["all", "per-connection", "cross-pair", "lm1"], default="all")
    parser.set_defaults(overwrite=False)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    args = parser.parse_args()

    core = _import_ai_core()
    cfg = load_config()

    if args.test == "all":
        out = core.run_ai_for_method(_METHOD_KEY, config=cfg, overwrite=args.overwrite)
        if out is None:
            logger.warning("phase_4 AI data not available — run the Phase 4 group builder first.")
            return
        df_per = out["per_connection"]
        n_sig_mwu = int(df_per["mwu_significant"].fillna(False).sum())
        n_sig_t = int(df_per["t_significant"].fillna(False).sum())
        logger.info("H_AI1 per-connection: %d/%d pairs significant (MWU FDR), %d (Welch FDR)",
                    n_sig_mwu, len(df_per), n_sig_t)

        df_cross = out["cross_pair"]
        cross_all = df_cross[df_cross["label"] == "All 10 pairs"]
        if not cross_all.empty:
            row = cross_all.iloc[0]
            logger.info("H_AI2 cross-pair Wilcoxon (all 10): p_2s=%.4f  p_1s=%.4f  sig=%s",
                        row.get("wilcoxon_p_twosided", float("nan")),
                        row.get("wilcoxon_p_onesided_greater", float("nan")),
                        row.get("wilcoxon_significant_onesided", False))

        df_lm1 = out["lm1"]
        pooled = df_lm1[df_lm1["pair"] == "PFC-LM1 - 3 pairs pooled (primary)"]
        if not pooled.empty:
            row = pooled.iloc[0]
            logger.info("H_AI3 LM1 pooled perm: p_2s=%.4f  p_1s=%.4f  sig_1s=%s  d=%.3f",
                        row.get("perm_p_twosided", float("nan")),
                        row.get("perm_p_onesided_greater", float("nan")),
                        row.get("perm_significant_1s", False),
                        row.get("cohens_d", float("nan")))
        logger.info("Figures: %s  %s", out["fig_grouped"], out["fig_lm1"])
        return

    core.build_ai_per_subject(_METHOD_KEY, config=cfg, overwrite=args.overwrite)

    if args.test == "per-connection":
        df = core.test_per_connection_ai(_METHOD_KEY, config=cfg, overwrite=args.overwrite)
        n_sig = int(df["mwu_significant"].fillna(False).sum())
        logger.info("H_AI1: %d/%d significant (MWU FDR)", n_sig, len(df))
    elif args.test == "cross-pair":
        df = core.test_cross_pair_ai(_METHOD_KEY, config=cfg, overwrite=args.overwrite)
        logger.info("H_AI2: %d rows", len(df))
    elif args.test == "lm1":
        df = core.test_lm1_focused_ai(_METHOD_KEY, config=cfg, overwrite=args.overwrite,
                                       n_permutations=args.n_permutations)
        logger.info("H_AI3 LM1: %d rows", len(df))

    core.write_ai_statistics_summary(_METHOD_KEY, config=cfg)


if __name__ == "__main__":
    _cli_main()
