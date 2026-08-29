#!/usr/bin/env bash
# Stage D — Run all four GC statistical tests (normality, H1 Expert-vs-Novice,
# H0 within-trial stability, HIV across-trial stability).
# Wraps: code/stage_d_granger/05_gc_statistics.py
#
# Usage:
#   ./05_run_statistics.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_d_granger/05_gc_statistics.py "$@"
