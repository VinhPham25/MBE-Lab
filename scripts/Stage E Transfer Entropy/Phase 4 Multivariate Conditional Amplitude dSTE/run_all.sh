#!/usr/bin/env bash
# Stage E / Phase 4 (current primary method) — Run the full sequence:
# process -> group aggregation -> visualize -> statistics -> asymmetry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/5] Phase 4 — Process (batch) ==="
"$SCRIPT_DIR/14_process.sh"

echo "=== [2/5] Phase 4 — Group aggregation ==="
"$SCRIPT_DIR/15_group.sh"

echo "=== [3/5] Phase 4 — Visualization ==="
"$SCRIPT_DIR/16_visualize.sh"

echo "=== [4/5] Phase 4 — Statistics ==="
"$SCRIPT_DIR/17_statistics.sh"

echo "=== [5/5] Phase 4 — Asymmetry index statistics ==="
"$SCRIPT_DIR/18_asymmetry.sh"

echo "=== Phase 4 complete ==="
