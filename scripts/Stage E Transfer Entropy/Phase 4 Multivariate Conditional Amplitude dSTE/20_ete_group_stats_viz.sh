#!/usr/bin/env bash
# Stage E / Phase 4 — ETE group arrays, statistics, and diagnostic plots.
# Wraps the method-parameterized
#   code/stage_e_transfer_entropy/15_ete_group_stats_viz.py
# with --method phase_4. Numbered 20 here (pipeline order after 19_ete_process)
# rather than 15, because 15_group.sh already wraps the raw-dSTE group builder.
#
# Defaults to `--method phase_4 --step all --overwrite` when run with no
# arguments: ETE group/stats/viz caches from a prior surrogate design are
# stale after a 19_ete_process.sh re-run, and the Python default is
# --no-overwrite. Pass --no-overwrite explicitly to reuse caches.
#
# Usage:
#   ./20_ete_group_stats_viz.sh                         # group + stats + plots
#   ./20_ete_group_stats_viz.sh --step visualizations
#   ./20_ete_group_stats_viz.sh --step statistics --overwrite
#   ./20_ete_group_stats_viz.sh --method phase_2 --step all --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --method phase_4 --step all --overwrite
fi

python3 code/stage_e_transfer_entropy/15_ete_group_stats_viz.py "$@"
