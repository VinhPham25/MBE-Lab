#!/usr/bin/env bash
# Stage E / Phase 4 — NG convergence diagnostic for the amplitude embedding
# space (confirms amplitude data clusters non-uniformly, unlike phase).
# Wraps: code/stage_e_transfer_entropy/phase_4/phase_4_debug.py
#
# Usage:
#   ./debug.sh                             # subject A1, Tr_1, window 0 (defaults)
#   ./debug.sh --sub R7 --trial Tr_2 --win 1 --k_max 25
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_4/phase_4_debug.py "$@"
