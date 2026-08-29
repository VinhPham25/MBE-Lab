#!/usr/bin/env bash
# Stage D — Main Linear GC driver: batch-processes all 13 GC-eligible subjects
# (VAR(20) -> frequency-domain GW causality -> 5x5 connectivity matrix) and
# builds the Expert/Novice group arrays. Includes a built-in A1 sanity check.
# Wraps: code/stage_d_granger/04_subject_gc_pipeline.py
#
# Usage:
#   ./04_run_subject_pipeline.sh                 # full batch, overwrite=True (default)
#   ./04_run_subject_pipeline.sh --no-overwrite   # skip subjects already cached
#
# Note: currently only supports --study expert_novice (see CLAUDE.md Section 1).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_d_granger/04_subject_gc_pipeline.py "$@"
