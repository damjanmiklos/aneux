#!/bin/bash
# Rate-target (R*) sweep: four independent 1-GPU runs, one R* each.
# Submit with hpc/submit_rate_sweep.sh, which builds the tube cache first.
#
# Why these settings:
#   batch 16 on one GPU = the production global batch (4 GPUs x 4), so the
#     R* this picks carries over to hpc/train_stage2.sbatch unchanged
#   R* in {0.25, 0.5, 1, 2} nats per valid token, log-spaced: the 2026-09-22
#     run's val recon had flattened by KL ~ 0.2-1, and above that the latent
#     only memorised (train KL rose to 12 while val stayed put)
#   weight decay 0.01, EMA 0.996, LR 2e-4: fixed for all four runs
#   150 epochs, cosine schedule over those 150 (the old run's val peaked at
#     epoch ~110 of 500)
# Time: ~16 min an epoch on one A100 (the old job's 223 s/epoch on 4 GPUs,
# x4 for one GPU, x1.08 for the current model, measured on the same cases),
# so ~41 h; 54 h leaves room. Memory: ~22 GiB peak at batch 16 of 40.
#
# Compare the runs by best val recon (EMA mu path) against val KL and the
# active-unit count in data/epoch_metrics.csv; prefer the smallest R* whose
# val recon is within noise of the best.

#SBATCH --job-name=aneux-rsweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2-06:00:00
#SBATCH --array=0-3
#SBATCH --no-requeue
# One epoch is ~16 min on one GPU: TERM 20 min early lets the handler finish
# the epoch, write last.pt and copy the run off scratch.
#SBATCH --signal=TERM@1200
#SBATCH --output=aneux_rsweep_%A_%a.out
#SBATCH --error=aneux_rsweep_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
R_VALUES=(0.25 0.5 1 2)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
R="${R_VALUES[$TASK]}"
echo "=== aneux R* sweep $(date -Is) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${TASK} R*=${R} node=$(hostname) ==="

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/hpc/common.sh"

export ANEUX_RATE_TARGET="$R"
export ANEUX_RUN_TAG="rstar${R}"
export ANEUX_BATCH_SIZE="${ANEUX_BATCH_SIZE:-16}"
export ANEUX_ACCUM_STEPS=1
export ANEUX_EPOCHS="${ANEUX_EPOCHS:-150}"
export ANEUX_WEIGHT_DECAY="${ANEUX_WEIGHT_DECAY:-0.01}"
export ANEUX_EMA_DECAY="${ANEUX_EMA_DECAY:-0.996}"
export ANEUX_LR="${ANEUX_LR:-2e-4}"
# the GPU needs ~27 s per batch-16 step, so 8 loaders keep up; the default 30
# per 16 cores only costs RAM. Cache stragglers build with 8 too.
export ANEUX_NUM_WORKERS="${ANEUX_NUM_WORKERS:-8}"
export ANEUX_CACHE_WORKERS="${ANEUX_CACHE_WORKERS:-8}"

srun --cpu-bind=none --kill-on-bad-exit=1 \
  python "${REPO}/1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py"
echo "=== finished $(date -Is) ==="

shopt -s nullglob
_cands=("${ANEUX_OUTPUT}/runs/"*"_job${SLURM_JOB_ID}_${ANEUX_RUN_TAG}")
if ((${#_cands[@]})); then
  aneux_write_job_stats "${_cands[-1]}/data"
else
  aneux_write_job_stats "${ANEUX_OUTPUT}/runs/stats_job${SLURM_JOB_ID}"
fi
