#!/usr/bin/env bash
# Stage E / Phase 1 — Run process -> statistics -> visualize in sequence.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Phase 1 — Process (batch) ==="
"$SCRIPT_DIR/03_process.sh"

echo "=== [2/3] Phase 1 — Statistics ==="
"$SCRIPT_DIR/04_statistics.sh"

echo "=== [3/3] Phase 1 — Visualization ==="
"$SCRIPT_DIR/05_visualize.sh"

echo "=== Phase 1 complete ==="
