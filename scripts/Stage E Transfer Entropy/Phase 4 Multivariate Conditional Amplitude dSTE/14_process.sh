#!/usr/bin/env bash
# Stage E / Phase 4 — CURRENT PRIMARY METHOD. Multivariate conditional
# amplitude dSTE: Hilbert amplitude envelope of broadband HbO, NG-symbolized,
# PCA-conditioned on the other 3 ROIs. No phase, no narrowband filtering.
# Wraps: code/stage_e_transfer_entropy/phase_4/14_amplitude_dste.py
#
# Defaults to --batch when run with no arguments (the underlying script
# requires one of --smoke-test / --subject / --batch).
#
# Usage:
#   ./14_process.sh                        # full batch (default)
#   ./14_process.sh --subject A1
#   ./14_process.sh --smoke-test
#   ./14_process.sh --batch --overwrite
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_4/14_amplitude_dste.py "$@"
