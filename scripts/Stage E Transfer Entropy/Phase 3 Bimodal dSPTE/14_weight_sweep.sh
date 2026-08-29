#!/usr/bin/env bash
# Stage E / Phase 3 — Sweep the phase/amplitude fusion weight (w_phase vs
# w_amplitude) to see how group separation changes as amplitude is upweighted.
# This is the diagnostic that motivated moving to amplitude-only TE in Phase 4.
# Wraps: code/stage_e_transfer_entropy/phase_3/14_bimodal_weight_sweep.py
#
# Usage:
#   ./14_weight_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_3/14_bimodal_weight_sweep.py "$@"
