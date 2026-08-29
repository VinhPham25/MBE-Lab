#!/usr/bin/env bash
# Stage C — HRF visualization (GLM-first evoked plots).
# Wraps: code/stage_c_validation/03_hrf_visualization.py
#
# Usage:
#   ./03_run_hrf_visualization.sh
#   ./03_run_hrf_visualization.sh --study expert_novice
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/03_hrf_visualization.py "$@"
