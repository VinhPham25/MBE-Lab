#!/usr/bin/env bash
# Stage D — Extract 5-ROI (LPFC, CPFC, RPFC, LM1, SMA) averaged signals.
# Wraps: code/stage_d_granger/01_extract_roi_signals.py
#
# Defaults to --all (all 13 GC-eligible subjects, R10 excluded) when run with
# no arguments, since the underlying script's own default is just A1.
#
# Usage:
#   ./01_run_roi_extraction.sh                     # all 13 subjects (default)
#   ./01_run_roi_extraction.sh --subjects A1 R7     # only specific subjects
#   ./01_run_roi_extraction.sh --all --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --all
fi

python3 code/stage_d_granger/01_extract_roi_signals.py "$@"
