#!/usr/bin/env bash
# Stage E / Phase 4 — Visualization: connectivity heatmaps, Expert-vs-Novice
# delta bars, bivariate-vs-multivariate suppression comparison.
# Wraps: code/stage_e_transfer_entropy/phase_4/16_amplitude_dste_visualization.py
#
# Usage:
#   ./16_visualize.sh                      # all plots (default)
#   ./16_visualize.sh --plot delta
#   ./16_visualize.sh --plot heatmaps --no-save
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_4/16_amplitude_dste_visualization.py "$@"
