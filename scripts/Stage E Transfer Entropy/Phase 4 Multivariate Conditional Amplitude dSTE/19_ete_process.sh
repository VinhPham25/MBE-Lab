#!/usr/bin/env bash
# Stage E / Phase 4 — Effective Transfer Entropy (ETE): circular-shift
# surrogate floor-correction of multivariate conditional amplitude dSTE.
# Surrogates are conditioned on the same PCA composite the raw pipeline used
# (not bivariate). Wraps: code/stage_e_transfer_entropy/phase_4/19_amplitude_ete.py
#
# Defaults to --batch when run with no arguments (the underlying script
# requires one of --benchmark / --subject / --batch). Reprocessing is the
# Python default (pass --no-overwrite to reuse cached per-subject .npz).
#
# Previous ~4 min full-cohort runtime (13 subjects, 120 windows, n_surrogates=500)
# is a reasonable estimate; --benchmark A1 first if you change n_surrogates.
#
# Usage:
#   ./19_ete_process.sh                         # full batch, reprocesses (default)
#   ./19_ete_process.sh --subject A1
#   ./19_ete_process.sh --benchmark A1 --n-windows 5
#   ./19_ete_process.sh --batch --no-overwrite  # use cached per-subject .npz
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  set -- --batch
fi

python3 code/stage_e_transfer_entropy/phase_4/19_amplitude_ete.py "$@"
