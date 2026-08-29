#!/usr/bin/env bash
# Stage A — Run the full Raw QA sequence: QA report, then signal exclusion check.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/2] Stage A — QA report ==="
"$SCRIPT_DIR/07_run_qa_report.sh"

echo "=== [2/2] Stage A — Signal amplitude exclusion check ==="
"$SCRIPT_DIR/08_run_signal_exclusion.sh"

echo "=== Stage A complete ==="
