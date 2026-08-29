#!/usr/bin/env bash
# Stage E / Phase 3 — Visualization (overall + grouped bimodal TE plots).
# Wraps: code/stage_e_transfer_entropy/phase_3/13_bimodal_dspte_visualization.py
#
# Usage:
#   ./13_visualize.sh                      # all plots (default)
#   ./13_visualize.sh --plot grouped --subjects A1 R7
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_3/13_bimodal_dspte_visualization.py "$@"
