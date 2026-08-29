#!/usr/bin/env bash
# Stage A — Raw QA report (SNR + motion ratio + cumulative study report)
# Wraps: code/stage_a_raw_qa/07_generate_qa_report.py
#
# Usage:
#   ./07_run_qa_report.sh                          # full report, expert_novice (default)
#   ./07_run_qa_report.sh --dataset learning_curve  # another dataset key from config.yaml
#   ./07_run_qa_report.sh --skip-batch              # skip per-subject batch, cumulative only
#   ./07_run_qa_report.sh --quiet                   # less console output
#   ./07_run_qa_report.sh --cleanup-subject-files    # remove intermediate per-subject files after run
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_a_raw_qa/07_generate_qa_report.py "$@"
