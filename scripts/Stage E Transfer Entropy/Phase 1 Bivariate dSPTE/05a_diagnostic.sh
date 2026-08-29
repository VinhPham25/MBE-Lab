#!/usr/bin/env bash
# Stage E / Phase 1 — Phase-entropy diagnostic: the investigation that showed
# broadband Hilbert phase is quasi-uniformly distributed (root cause of
# Phase 1's flat lag profiles / Novice > Expert on all 20 connections).
# See CLAUDE.md Section 3 for the full narrative.
# Wraps: code/stage_e_transfer_entropy/phase_1/05a_phase_entropy_diagnostic.py
#
# Usage:
#   ./05a_diagnostic.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 code/stage_e_transfer_entropy/phase_1/05a_phase_entropy_diagnostic.py "$@"
