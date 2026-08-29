#!/usr/bin/env bash
# Stage D — Generate all GC visualizations (connectivity heatmaps, group bar
# charts, brain topology diagrams).
# Wraps: code/stage_d_granger/06_gc_visualization.py
#
# Usage:
#   ./06_run_visualization.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_d_granger/06_gc_visualization.py "$@"
