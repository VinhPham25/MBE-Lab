#!/usr/bin/env bash
# Stage E / Phase 3 — Statistical tests for the bimodal (phase+amplitude) dSPTE.
# Wraps: code/stage_e_transfer_entropy/phase_3/12_bimodal_dspte_statistics.py
#
# Usage:
#   ./12_statistics.sh                     # all tests (default)
#   ./12_statistics.sh --test h1
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_3/12_bimodal_dspte_statistics.py "$@"
