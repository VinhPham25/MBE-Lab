#!/usr/bin/env bash
# Stage D — Run the full Linear GC pipeline end-to-end: ROI extraction ->
# trial segmentation -> subject GC pipeline -> statistics -> visualization.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/5] Stage D — ROI extraction ==="
"$SCRIPT_DIR/01_run_roi_extraction.sh"

echo "=== [2/5] Stage D — Trial segmentation ==="
"$SCRIPT_DIR/02_run_trial_segmentation.sh"

echo "=== [3/5] Stage D — Subject GC pipeline + group arrays ==="
"$SCRIPT_DIR/04_run_subject_pipeline.sh"

echo "=== [4/5] Stage D — Statistics ==="
"$SCRIPT_DIR/05_run_statistics.sh"

echo "=== [5/5] Stage D — Visualization ==="
"$SCRIPT_DIR/06_run_visualization.sh"

echo "=== Stage D complete ==="
