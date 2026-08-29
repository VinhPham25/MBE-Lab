#!/usr/bin/env bash
# Stage E — Synthetic benchmark validation against Wang (2020) Eqs. 26-28
# (linear AR, nonlinear, and multivariate simulation models). Validates the
# dSPTE/dSTE machinery itself against ground-truth synthetic data, independent
# of any real fNIRS subject.
# Wraps: code/stage_e_transfer_entropy/wang2020_simulation_validation.py
#
# Usage:
#   ./run_wang2020_validation.sh                # all 3 models (default)
#   ./run_wang2020_validation.sh --model 1       # linear AR model only
#   ./run_wang2020_validation.sh --N 50000 --quiet
#
# NOT YET CREATED (see CLAUDE.md Section 3, "What's NOT done yet in Stage E"):
#   - 08_cross_phase_comparison.py (Section 10 — GC + Phase 1-4 comparison table)
#   - a unified batch runner (Section 11)
# Once those exist, add wrapper scripts here alongside this one.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/wang2020_simulation_validation.py "$@"
