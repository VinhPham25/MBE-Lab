#!/usr/bin/env bash
# Stage E / Phase 3 — Bimodal dSPTE: phase + amplitude weighted fusion, for
# all subjects. (Superseded by Phase 4 amplitude-only dSTE, but this is where
# it became clear amplitude carried most of the real signal — see CLAUDE.md.)
# Wraps: code/stage_e_transfer_entropy/phase_3/11_bimodal_dspte.py
#
# Defaults to --batch when run with no arguments (the underlying script
# requires one of --smoke-test / --subject / --batch).
#
# Usage:
#   ./11_process.sh                        # full batch (default)
#   ./11_process.sh --subject A1
#   ./11_process.sh --smoke-test
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_3/11_bimodal_dspte.py "$@"
