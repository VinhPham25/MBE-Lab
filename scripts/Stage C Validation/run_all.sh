#!/usr/bin/env bash
# Stage C — Run the full validation sequence: group HRF analysis, HRF
# visualization, then both CNR analyses. (Skips 01_run_hrf_extraction.sh,
# which is a single-subject debug tool, and run_hyperparam_sweep.sh, which
# is a standalone diagnostic — run those individually if needed.)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Stage C — Group HRF analysis ==="
"$SCRIPT_DIR/02_run_group_hrf_analysis.sh"

echo "=== [2/4] Stage C — HRF visualization ==="
"$SCRIPT_DIR/03_run_hrf_visualization.sh"

echo "=== [3/4] Stage C — CNR analysis ==="
"$SCRIPT_DIR/04_run_cnr_analysis.sh"

echo "=== [4/4] Stage C — Technical validation CNR ==="
"$SCRIPT_DIR/04_run_technical_validation_cnr.sh"

echo "=== Stage C complete ==="
