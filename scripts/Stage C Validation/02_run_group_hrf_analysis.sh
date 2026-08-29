#!/usr/bin/env bash
# Stage C — Group-level HRF analysis (grand averages for Expert vs Novice).
# Computes/caches per-subject HRFs as needed, so this is the de facto
# "run everything" driver for Stage C's HRF extraction step.
# Wraps: code/stage_c_validation/02_group_hrf_analysis.py
#
# Usage:
#   ./02_run_group_hrf_analysis.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/02_group_hrf_analysis.py "$@"
