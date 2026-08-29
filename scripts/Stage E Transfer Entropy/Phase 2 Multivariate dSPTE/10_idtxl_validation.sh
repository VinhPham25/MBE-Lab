#!/usr/bin/env bash
# Stage E / Phase 2 — IDTxl (KSG estimator) parallel validation: a model-free
# cross-check against the NG-symbolic hybrid-dSPTE approach.
# Wraps: code/stage_e_transfer_entropy/phase_2/10_idtxl_validation.py
#
# This script has many performance-tuning flags (see the Python file's
# --help); a few common recipes below. For HPC/PACE cluster batch submission,
# see pace_idtxl_job.sh (repo root) instead.
#
# Usage:
#   ./10_idtxl_validation.sh --smoke-test
#   ./10_idtxl_validation.sh --batch --first-window-only    # fast validation (~5-10 min)
#   ./10_idtxl_validation.sh --compare                       # compare vs symbolic (after --batch)
#   ./10_idtxl_validation.sh --batch                          # full batch (~52 min estimate)
#   ./10_idtxl_validation.sh --eval-md                        # regenerate IDTXL_EVAL.md
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_2/10_idtxl_validation.py "$@"
