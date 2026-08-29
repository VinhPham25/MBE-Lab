#!/usr/bin/env bash
# Stage E / Phase 4 — Group array aggregation (Expert vs Novice arrays,
# 3 strategies: all-windows, subject-mean, first-window).
# Wraps: code/stage_e_transfer_entropy/phase_4/15_amplitude_dste_group.py
#
# Usage:
#   ./15_group.sh                          # all strategies (default)
#   ./15_group.sh --strategy subject_mean
#   ./15_group.sh --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_4/15_amplitude_dste_group.py "$@"
