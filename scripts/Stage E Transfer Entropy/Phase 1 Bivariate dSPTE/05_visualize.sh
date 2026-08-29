#!/usr/bin/env bash
# Stage E / Phase 1 — Visualization (heatmaps, delay/lag profiles, GC-TE
# comparison, topology).
# Wraps: code/stage_e_transfer_entropy/phase_1/05_dspte_visualization.py
#
# Usage:
#   ./05_visualize.sh                      # all plots (default)
#   ./05_visualize.sh --plot lag --subject A1
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_1/05_dspte_visualization.py "$@"
