#!/usr/bin/env bash
# Stage E / Phase 2 — Diagnostic sweep (e.g. PCA-conditioning component count)
# investigating why most connections still had flat/uninformative lag profiles.
# Wraps: code/stage_e_transfer_entropy/phase_2/09_phase2_diagnostic_sweep.py
#
# Usage:
#   ./09_diagnostic_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_2/09_phase2_diagnostic_sweep.py "$@"
