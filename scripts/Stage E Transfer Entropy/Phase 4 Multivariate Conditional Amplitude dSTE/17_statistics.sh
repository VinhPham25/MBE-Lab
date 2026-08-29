#!/usr/bin/env bash
# Stage E / Phase 4 — Statistical tests: normality (Shapiro-Wilk), H1 Expert
# vs Novice (Mann-Whitney U + FDR), H0 within-trial stability, HIV
# across-trial stability.
# Wraps: code/stage_e_transfer_entropy/phase_4/17_amplitude_dste_statistics.py
#
# Usage:
#   ./17_statistics.sh                     # all tests (default)
#   ./17_statistics.sh --test h1
#   ./17_statistics.sh --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_4/17_amplitude_dste_statistics.py "$@"
