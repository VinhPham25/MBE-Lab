#!/usr/bin/env bash
# Stage A — Signal amplitude exclusion check (trailing-zero / anomalous-variance flags,
# e.g. the R10 exclusion documented in CLAUDE.md)
# Wraps: code/stage_a_raw_qa/08_signal_amplitude_exclusion.py
#
# Usage:
#   ./08_run_signal_exclusion.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_a_raw_qa/08_signal_amplitude_exclusion.py "$@"
