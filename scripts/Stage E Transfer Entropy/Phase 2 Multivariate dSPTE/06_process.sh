#!/usr/bin/env bash
# Stage E / Phase 2 — Multivariate dSPTE: PCA-conditioned phase TE for
# all subjects. (Superseded by Phase 4 amplitude dSTE — kept for comparison.)
# Wraps: code/stage_e_transfer_entropy/phase_2/06_multivariate_dspte.py
#
# Defaults to --batch when run with no arguments (the underlying script
# defaults to synthetic smoke tests otherwise).
#
# Usage:
#   ./06_process.sh                        # full batch (default)
#   ./06_process.sh --subject A1
#   ./06_process.sh --batch --first-window-only   # fast mode, matches IDTxl compare
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_2/06_multivariate_dspte.py "$@"
