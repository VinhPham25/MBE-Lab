#!/usr/bin/env bash
# Stage E / Phase 4 — Asymmetry Index (AI = TE(X->Y) - TE(Y->X)) diagnostics,
# comparing Phase 4 TE asymmetry against GC directionality. Part of the
# current interpretive focus (why Novice > Expert dominates raw TE magnitude).
# Wraps: code/stage_e_transfer_entropy/phase_4/18_asymmetry_index_statistics.py
#
# Usage:
#   ./18_asymmetry.sh                      # all tests (default)
#   ./18_asymmetry.sh --test per-connection
#   ./18_asymmetry.sh --n-permutations 5000
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_4/18_asymmetry_index_statistics.py "$@"
