#!/bin/bash
# Rate-target (R*) sweep in ONE job on ONE 4-GPU node: build the tube cache,
# stage the inputs to scratch once, then train four independent runs side by
# side, one R* per GPU. One queue wait for the whole sweep.
#
#   cd /project/<account>/aneux && git pull
#   (upload cleandata/{uniformly_remeshed,template_mesh,original_centerline},
#    the *.ostium_frames.npz next to the .vtp files included)
#   sbatch --account=<account> --mail-user=YOU@email hpc/sweep_rate_target.sbatch
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
# Time: ~16 min an epoch per run (the old job's 223 s/epoch on 4 GPUs, x4 for
# one GPU, x1.08 for the current model), so ~41 h; the cache build and staging
# add well under an hour. Memory: ~22 GiB peak per GPU at batch 16 of 40.
#
# Compare the runs by best val recon (EMA mu path) against val KL and the
# active-unit count in data/epoch_metrics.csv; prefer the smallest R* whose
# val recon is within noise of the best.

#SBATCH --job-name=aneux-rsweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-04:00:00
#SBATCH --no-requeue
# TERM goes to this shell (B:), which forwards it to the four runs: 20 min is
# enough to finish the current epoch and write last.pt.
#SBATCH --signal=B:TERM@1200
#SBATCH --output=aneux_rsweep_%j.out
#SBATCH --error=aneux_rsweep_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
echo "=== aneux R* sweep $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/hpc/common.sh"
TP="${REPO}/1test_encoder_decoder_only/train_pipeline"
R_VALUES=(0.25 0.5 1 2)

# 1. Tube cache, built in place on project disk (existing files are kept).
echo "--- cache build $(date -Is)"
ANEUX_CACHE_ONLY=1 ANEUX_CACHE_WORKERS="${ANEUX_CACHE_WORKERS:-48}" python "${TP}/aneuxai_HPC.py"

# 2. Stage once: the three cleandata folders and the current cache version.
ver=$(cd "$TP" && python -c 'from config import CACHE_VERSION; print(CACHE_VERSION)')
WS="/scratch/${SLURM_JOB_ACCOUNT:?}/${USER}/aneux/${SLURM_JOB_ID}"
echo "--- staging to ${WS} $(date -Is)"
mkdir -p "${WS}/cleandata" "${WS}/tube_cache"
for d in uniformly_remeshed template_mesh original_centerline; do
  rsync -a "${ANEUX_CLEANDATA}/${d}" "${WS}/cleandata/"
done
rsync -a --include "*_v${ver}_*" --exclude '*' "${ANEUX_CACHE}/" "${WS}/tube_cache/"
echo "staged $(find "${WS}/tube_cache" -maxdepth 1 -name "*_v${ver}_*.pt" | wc -l) v${ver} cache files $(date -Is)"

# 3. Four runs, one GPU each. Runs write straight to project output.
export ANEUX_SKIP_STAGE=1
export ANEUX_CLEANDATA="${WS}/cleandata"
export ANEUX_CACHE="${WS}/tube_cache"
export ANEUX_BATCH_SIZE="${ANEUX_BATCH_SIZE:-16}"
export ANEUX_ACCUM_STEPS=1
export ANEUX_EPOCHS="${ANEUX_EPOCHS:-150}"
export ANEUX_WEIGHT_DECAY="${ANEUX_WEIGHT_DECAY:-0.01}"
export ANEUX_EMA_DECAY="${ANEUX_EMA_DECAY:-0.996}"
export ANEUX_LR="${ANEUX_LR:-2e-4}"
# 16 cores per run; the GPU needs ~27 s per batch-16 step, so 8 loaders keep up
export ANEUX_NUM_WORKERS="${ANEUX_NUM_WORKERS:-8}"
export ANEUX_CACHE_WORKERS="${ANEUX_CACHE_WORKERS_RUN:-8}"
export SLURM_CPUS_PER_TASK=16

pids=()
for i in "${!R_VALUES[@]}"; do
  R="${R_VALUES[$i]}"
  log="${REPO}/aneux_rsweep_${SLURM_JOB_ID}_rstar${R}.log"
  echo "--- run R*=${R} on GPU ${i} -> ${log}"
  CUDA_VISIBLE_DEVICES="$i" ANEUX_RATE_TARGET="$R" ANEUX_RUN_TAG="rstar${R}" \
    python "${TP}/aneuxai_HPC.py" > "$log" 2>&1 &
  pids+=($!)
done
trap 'echo "TERM: forwarding to runs $(date -Is)"; kill -TERM "${pids[@]}" 2>/dev/null || true' TERM

rc=0
for i in "${!pids[@]}"; do
  # wait returns early when the trap fires; wait again for the real exit
  while true; do
    set +e; wait "${pids[$i]}"; code=$?; set -e
    kill -0 "${pids[$i]}" 2>/dev/null || break
  done
  echo "run R*=${R_VALUES[$i]} exited ${code} $(date -Is)"
  ((code == 0)) || rc=1
done

aneux_write_job_stats "${ANEUX_OUTPUT}/runs/rsweep_job${SLURM_JOB_ID}_stats"
rm -rf "${WS}"
echo "=== finished $(date -Is) rc=${rc} ==="
exit "$rc"
