#!/usr/bin/env bash
# Stage C — Hyperparameter sweep runner (sweeps preprocessing params against
# HRF/CNR quality metrics).
# Wraps: code/stage_c_validation/hyperparam_sweep_runner.py
#
# Usage:
#   ./run_hyperparam_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/hyperparam_sweep_runner.py "$@"
