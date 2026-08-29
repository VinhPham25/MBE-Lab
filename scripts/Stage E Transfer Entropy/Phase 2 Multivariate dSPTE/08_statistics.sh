#!/usr/bin/env bash
# Stage E / Phase 2 — Statistical tests for the hybrid (conditioned) dSPTE.
# Wraps: code/stage_e_transfer_entropy/phase_2/08_hybrid_dspte_statistics.py
#
# Usage:
#   ./08_statistics.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_2/08_hybrid_dspte_statistics.py "$@"
