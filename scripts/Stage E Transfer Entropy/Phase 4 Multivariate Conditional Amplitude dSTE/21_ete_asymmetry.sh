#!/usr/bin/env bash
# Stage E / Phase 4 — Asymmetry Index on floor-corrected ETE (not raw dSTE).
# Wraps the method-parameterized
#   code/stage_e_transfer_entropy/12_asymmetry_index_core.py
# with --method phase_4_ete. Distinct from 18_asymmetry.sh, which runs the
# raw-magnitude Phase 4 AI track via 18_asymmetry_index_statistics.py.
#
# Defaults to `--method phase_4_ete --overwrite` when run with no arguments
# (must follow 20_ete_group_stats_viz.sh so the ETE subject-mean group arrays
# exist). Pass extra flags through as usual.
#
# Usage:
#   ./21_ete_asymmetry.sh                          # Phase 4 ETE AI, rebuild
#   ./21_ete_asymmetry.sh --no-overwrite           # use cached AI outputs
#   ./21_ete_asymmetry.sh --all --overwrite        # every registered AI track
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --method phase_4_ete --overwrite
fi

python3 code/stage_e_transfer_entropy/12_asymmetry_index_core.py "$@"
