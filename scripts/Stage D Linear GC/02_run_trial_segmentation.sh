#!/usr/bin/env bash
# Stage D — Segment ROI signals into trials/windows for GC computation.
# Wraps: code/stage_d_granger/02_trial_segmentation.py
#
# Defaults to --all (all 13 GC-eligible subjects) when run with no arguments,
# since the underlying script's own default is just A1.
#
# Usage:
#   ./02_run_trial_segmentation.sh                 # all 13 subjects (default)
#   ./02_run_trial_segmentation.sh --subjects A1 R7
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --all
fi

python3 code/stage_d_granger/02_trial_segmentation.py "$@"
