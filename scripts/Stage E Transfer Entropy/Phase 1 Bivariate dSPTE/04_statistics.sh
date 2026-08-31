#!/usr/bin/env bash
# Stage E / Phase 1 — Statistical tests (normality, H1 Expert-vs-Novice, H0, HIV).
# Wraps: code/stage_e_transfer_entropy/phase_1/04_bivariate_dspte_statistics.py
#
# Usage:
#   ./04_statistics.sh                     # all tests (default)
#   ./04_statistics.sh --test h1
#   ./04_statistics.sh --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_1/04_bivariate_dspte_statistics.py "$@"
