#!/bin/bash
# SLURM job script — IDTxl Parallel Validation (Section 7.4)
# Submit with: sbatch pace_idtxl_job.sh
# Monitor with: squeue -u vpham62

#SBATCH -J idtxl_validation          # Job name
#SBATCH -A gts-mnezafati3            # Charge account (from pace-quota)
#SBATCH -q embers                    # Queue (embers has no per-job walltime cap)
#SBATCH -N 1                         # 1 node
#SBATCH --ntasks-per-node=1          # 1 task
#SBATCH --cpus-per-task=8            # 8 CPU cores
#SBATCH --mem=32G                    # Memory per node
#SBATCH -t 2:00:00                   # Walltime: 2 h (full batch ~52 min; first-window ~10 min)
#SBATCH -o logs/idtxl_%j.out         # stdout  (%j = job ID)
#SBATCH -e logs/idtxl_%j.err         # stderr

# ── Environment ──────────────────────────────────────────────────────────────
module load anaconda3/2022.05
conda activate mbelab

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT=/storage/scratch1/3/vpham62/MBE-Lab
cd "$PROJECT"

mkdir -p logs

# ── Threading guard (avoid BLAS oversubscription) ───────────────────────────
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Run mode ─────────────────────────────────────────────────────────────────
# Choose ONE of the three lines below by commenting/uncommenting:

# Fast mode — first window only (~5–10 min):
python3 code/stage_e_transfer_entropy/10_idtxl_validation.py \
        --batch --first-window-only --overwrite \
        --jobs 4 --idtxl-num-threads 2 --idtxl-max-lag 25 --idtxl-fast

# Full batch — all windows (~52 min):
# python3 code/stage_e_transfer_entropy/10_idtxl_validation.py \
#         --batch --overwrite \
#         --jobs 4 --idtxl-num-threads 2 --idtxl-max-lag 25

# Comparison only (run after batch completes):
# python3 code/stage_e_transfer_entropy/10_idtxl_validation.py --compare
