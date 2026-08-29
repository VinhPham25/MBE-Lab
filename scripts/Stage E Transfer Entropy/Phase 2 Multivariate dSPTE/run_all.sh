#!/usr/bin/env bash
# Stage E / Phase 2 — Run process -> statistics -> visualize in sequence.
# (Does not run the IDTxl validation or diagnostic sweep — those are
# standalone cross-checks; run them individually.)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Phase 2 — Process (batch) ==="
"$SCRIPT_DIR/06_process.sh"

echo "=== [2/3] Phase 2 — Statistics ==="
"$SCRIPT_DIR/08_statistics.sh"

echo "=== [3/3] Phase 2 — Visualization ==="
"$SCRIPT_DIR/07_visualize.sh"

echo "=== Phase 2 complete ==="
