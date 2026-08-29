#!/usr/bin/env bash
# Stage E / Phase 1 — Bivariate dSPTE: compute connectivity for all subjects
# and build group arrays. (Superseded by Phase 4 amplitude dSTE — kept for
# comparison; see CLAUDE.md Section 3.)
# Wraps: code/stage_e_transfer_entropy/phase_1/03_subject_dspte_pipeline.py
#
# Defaults to --batch when run with no arguments (the underlying script
# defaults to a single-subject A1 smoke test otherwise).
#
# Usage:
#   ./03_process.sh                        # full batch (default)
#   ./03_process.sh --subject A1           # single subject
#   ./03_process.sh --group-only           # rebuild group arrays from existing .npz only
#   ./03_process.sh --batch --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_1/03_subject_dspte_pipeline.py "$@"
