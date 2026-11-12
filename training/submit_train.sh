#!/bin/bash
# Fine-tune DeepSlice on synthetic horizontal sections on one GPU. Submit from the
# repository root (check partitions with `sinfo -s`):
#   sbatch --export=ALL,RUN=h01,SHARDS=training/data/shards training/submit_train.sh
#   sbatch --export=ALL,RUN=smoke,TRAIN_ARGS="--steps-per-epoch 30 --head-epochs 1 --epochs 1" training/submit_train.sh
# SHARDS: pre-generated shards (training/submit_make_shards.sh); without it, samples are
# generated on the fly, which is CPU-bound (~60 samples/s on 32 CPUs vs ~350 on the GPU).
# LOCAL_COPY=1 copies the shards to node-local disk first.
# The environment is the repository's .venv plus the cluster's CUDA module.
#SBATCH --job-name=ds-train
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=training/runs/slurm-%x-%j.out
set -euo pipefail
RUN=${RUN:-h01}
VAL=${VAL:-training/data/val_5k.npz}
SHARDS=${SHARDS:-}
LOCAL_COPY=${LOCAL_COPY:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}
nvidia-smi -L
# the pip CUDA libraries miss some that TensorFlow needs (e.g. cuSOLVER); the module has them
ml CUDA
export TF_CPP_MIN_LOG_LEVEL=1
if [ -n "$SHARDS" ]; then
    if [ "$LOCAL_COPY" = "1" ]; then
        LOCAL=${TMPDIR:-/tmp}/ds_shards_$SLURM_JOB_ID
        mkdir -p "$LOCAL"
        trap 'rm -rf "$LOCAL"' EXIT
        echo "copying shards to $LOCAL"; time cp "$SHARDS"/shard_* "$LOCAL"/
        SHARDS=$LOCAL
    fi
    TRAIN_ARGS="--train-shards $SHARDS $TRAIN_ARGS"
fi
# leave a couple of cores for the training process itself
.venv/bin/python -m training.train --out-dir "training/runs/$RUN" --val-set "$VAL" \
    --workers $((SLURM_CPUS_PER_TASK - 2)) $TRAIN_ARGS
