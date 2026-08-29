#!/usr/bin/env bash
# Stage C — Alternate CNR pipeline using a minimal/independent preprocessing
# branch, used as a technical-validation cross-check against 04_run_cnr_analysis.sh.
# Wraps: code/stage_c_validation/04_technical_validation_cnr.py
#
# Usage:
#   ./04_run_technical_validation_cnr.sh
#   ./04_run_technical_validation_cnr.sh --force   # force recompute cached .fif/evoked files
#                                                   # (needed after changing config.yaml, e.g. bandpass lpf)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_c_validation/04_technical_validation_cnr.py "$@"
