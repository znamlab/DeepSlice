#!/bin/bash
# Pre-generate training shards on CPU nodes (one array task per shard). From the repo root:
#   sbatch --array=0-49 training/submit_make_shards.sh        # 50 x 20k = 1M samples (~90 GB)
# Optional: --export=ALL,N=20000,SEED=1,OUT=training/data/shards
#SBATCH --job-name=ds-shards
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=training/runs/slurm-%x-%A_%a.out
set -euo pipefail
N=${N:-20000}
SEED=${SEED:-1}
OUT=${OUT:-training/data/shards}
CONFIG=${CONFIG:-default}  # "canvas": fixed 14 mm field of view, any rotation
.venv/bin/python -m training.dataset --shard-dir "$OUT" --shard "$SLURM_ARRAY_TASK_ID" --n "$N" \
    --seed "$SEED" --workers "$SLURM_CPUS_PER_TASK" --config "$CONFIG"
