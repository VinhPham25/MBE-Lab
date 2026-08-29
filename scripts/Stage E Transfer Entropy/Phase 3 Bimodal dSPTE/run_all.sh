#!/usr/bin/env bash
# Stage E / Phase 3 — Run process -> statistics -> visualize in sequence.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Phase 3 — Process (batch) ==="
"$SCRIPT_DIR/11_process.sh"

echo "=== [2/3] Phase 3 — Statistics ==="
"$SCRIPT_DIR/12_statistics.sh"

echo "=== [3/3] Phase 3 — Visualization ==="
"$SCRIPT_DIR/13_visualize.sh"

echo "=== Phase 3 complete ==="
