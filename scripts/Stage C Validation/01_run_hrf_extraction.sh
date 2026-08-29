#!/usr/bin/env bash
# Stage C — HRF epoch extraction + morphology verification for ONE subject.
# Wraps: code/stage_c_validation/01_hrf_extraction.py
#
# Note: this is a single-subject debugging/QA tool (default subject: A1).
# The group-level driver (02_run_group_hrf_analysis.sh) computes/caches HRFs
# for every subject automatically — you usually don't need to run this first.
#
# Usage:
#   ./01_run_hrf_extraction.sh                    # subject A1, expert_novice
#   ./01_run_hrf_extraction.sh --subject R7
#   ./01_run_hrf_extraction.sh --subject R7 --study expert_novice
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/01_hrf_extraction.py "$@"
