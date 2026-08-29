#!/usr/bin/env bash
# Stage B — Quick sanity check that event markers (1=start, 2=end) survive
# preprocessing and that rest/task periods are still mixed (not segmented).
# Wraps: code/stage_b_preprocessing/verify_event_markers.py
#
# Note: this script is hardcoded to subject A1 — it's a spot-check utility,
# not a batch driver.
#
# Usage:
#   ./run_verify_event_markers.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_b_preprocessing/verify_event_markers.py "$@"
