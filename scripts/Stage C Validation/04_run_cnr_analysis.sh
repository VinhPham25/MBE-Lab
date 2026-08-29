#!/usr/bin/env bash
# Stage C — Main channel-wise Contrast-to-Noise Ratio (CNR) analysis, Expert vs Novice.
# Wraps: code/stage_c_validation/04_calculate_cnr.py
#
# Usage:
#   ./04_run_cnr_analysis.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/04_calculate_cnr.py "$@"
