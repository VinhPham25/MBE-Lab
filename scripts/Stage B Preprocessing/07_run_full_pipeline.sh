#!/usr/bin/env bash
# Stage B — Full batch preprocessing (B1 OD -> B2 bandpass -> B3 motion detect ->
# B4 rPCA -> B5 MBLL -> B6 OLS/GLM) for every subject in a study.
# Wraps: code/stage_b_preprocessing/07_preprocess_study.py
#
# Usage:
#   ./07_run_full_pipeline.sh                              # expert_novice (default)
#   ./07_run_full_pipeline.sh --study learning_curve
#   ./07_run_full_pipeline.sh --subjects A1 R10             # only re-run specific subjects
#   ./07_run_full_pipeline.sh --n_components 0.99 --min_channels 10
#   ./07_run_full_pipeline.sh --verbose
#
# Note: learning_curve / short_term_tes / long_term_tes are not yet the active
# focus of this project (see CLAUDE.md Section 1) — short/long-term tES studies
# also need Section 3.9 adaptation before results are physiologically valid.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_b_preprocessing/07_preprocess_study.py "$@"
