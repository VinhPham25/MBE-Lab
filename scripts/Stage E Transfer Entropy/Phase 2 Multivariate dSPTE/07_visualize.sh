#!/usr/bin/env bash
# Stage E / Phase 2 — Visualization (group matrix, group bar chart,
# Phase 1 vs Phase 2 comparison, lag profiles).
# Wraps: code/stage_e_transfer_entropy/phase_2/07_hybrid_dspte_visualization.py
#
# Defaults to --batch (all Section 7 plots) when run with no arguments.
#
# Usage:
#   ./07_visualize.sh                      # all plots (default)
#   ./07_visualize.sh --comparison         # Phase 1 vs Phase 2 comparison only
#   ./07_visualize.sh --lag-profiles --subject A1 --trial Tr_1
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_2/07_hybrid_dspte_visualization.py "$@"
