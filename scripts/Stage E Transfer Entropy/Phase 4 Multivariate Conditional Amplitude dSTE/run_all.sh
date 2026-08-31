#!/usr/bin/env bash
# Stage E / Phase 4 (current primary method) — Run the full sequence:
# raw dSTE process -> group -> visualize -> statistics -> raw AI
# -> ETE process -> ETE group/stats/viz -> ETE AI.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/8] Phase 4 — Process (batch) ==="
"$SCRIPT_DIR/14_process.sh"

echo "=== [2/8] Phase 4 — Group aggregation ==="
"$SCRIPT_DIR/15_group.sh"

echo "=== [3/8] Phase 4 — Visualization ==="
"$SCRIPT_DIR/16_visualize.sh"

echo "=== [4/8] Phase 4 — Statistics ==="
"$SCRIPT_DIR/17_statistics.sh"

echo "=== [5/8] Phase 4 — Asymmetry index statistics (raw dSTE) ==="
"$SCRIPT_DIR/18_asymmetry.sh"

echo "=== [6/8] Phase 4 — ETE process (batch) ==="
"$SCRIPT_DIR/19_ete_process.sh"

echo "=== [7/8] Phase 4 — ETE group / statistics / visualizations ==="
"$SCRIPT_DIR/20_ete_group_stats_viz.sh"

echo "=== [8/8] Phase 4 — Asymmetry index statistics (ETE) ==="
"$SCRIPT_DIR/21_ete_asymmetry.sh"

echo "=== Phase 4 complete (raw dSTE + ETE) ==="
