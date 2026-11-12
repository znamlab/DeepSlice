#!/bin/bash
# Generate the fixed synthetic validation set. Submit from the repository root:
#   sbatch training/submit_make_val.sh
# Optional: --export=ALL,N=5000,SEED=12345,OUT=training/data/val_5k.npz
#SBATCH --job-name=ds-make-val
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=training/runs/slurm-%x-%j.out
set -euo pipefail
N=${N:-5000}
SEED=${SEED:-12345}
OUT=${OUT:-training/data/val_5k.npz}
CONFIG=${CONFIG:-default}
.venv/bin/python -m training.dataset --out "$OUT" --n "$N" --seed "$SEED" --workers "$SLURM_CPUS_PER_TASK" --config "$CONFIG"
