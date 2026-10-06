#!/bin/bash
# Replay one training step dumped by hpc/train_stage2_diag.sbatch.
#
# The diagnostic job writes crash_rank<R>_e<EPOCH>_s<STEP>.pt (the CPU batch and
# the RNG states of the step that crashed, and of the step before it) under
#   1test_encoder_decoder_only/output/diag/job<ID>/
# This runs that step again on ONE GPU with CUDA_LAUNCH_BLOCKING=1, and then under
# compute-sanitizer (memcheck) when the node has it. The sanitizer reports the
# kernel, thread and address of an invalid access even if it would not crash.
# One GPU is fine here: this is a post-mortem on a dumped step, not the repro.
#
#   sbatch --account=nr_hemo_ai1 \
#     --export=ALL,CRASH_DUMP=1test_encoder_decoder_only/output/diag/job<ID>/crash_rank1_e9_s31.pt,\
# CKPT=1test_encoder_decoder_only/output/diag/job<ID>/run/<stamp>_job<ID>/checkpoints/last.pt \
#     hpc/replay_crash.sbatch
#
# CKPT is optional (the weights from the end of the previous epoch, which the
# diagnostic job copies off scratch when it fails). Replay both dumps (the failing
# step and the one before) if the first completes cleanly.

#SBATCH --job-name=aneux-replay
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=aneux_replay_%j.out
#SBATCH --error=aneux_replay_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/1test_encoder_decoder_only/train_pipeline" ]]; then
  REPO="${ANEUX_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
: "${CRASH_DUMP:?set CRASH_DUMP=path/to/crash_rank*_e*_s*.pt with --export=ALL,CRASH_DUMP=...}"
if [[ ! -f "$CRASH_DUMP" ]]; then echo "no such dump: $CRASH_DUMP" >&2; exit 1; fi

source "${HOME}/aneuxai_env/bin/activate"
export ANEUX_REPO_ROOT="$REPO"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
# expandable_segments is OFF on Komondor since the 2026-10-05 driver/kernel update (see
# hpc/train_stage2_diag.sbatch). Re-enable with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

REPLAY="${REPO}/1test_encoder_decoder_only/train_pipeline/diag_replay.py"
ARGS=("$CRASH_DUMP")
if [[ -n "${CKPT:-}" ]]; then ARGS+=(--ckpt "$CKPT"); fi

echo "=== replay 1: launch-blocking, plain python ($(hostname)) ==="
set +e
python "$REPLAY" "${ARGS[@]}" --repeat "${REPEAT:-3}"
echo "plain replay exit code: $?"

SANITIZER="$(command -v compute-sanitizer || true)"
if [[ -z "$SANITIZER" ]]; then
  for m in cuda CUDA; do
    module load "$m" >/dev/null 2>&1 && SANITIZER="$(command -v compute-sanitizer || true)"
    [[ -n "$SANITIZER" ]] && break
  done
fi
if [[ -n "$SANITIZER" ]]; then
  echo "=== replay 2: $SANITIZER --tool memcheck ==="
  "$SANITIZER" --tool memcheck --launch-timeout 0 --print-limit 40 \
    python "$REPLAY" "${ARGS[@]}"
  echo "sanitizer replay exit code: $?"
else
  echo "compute-sanitizer is not on this node (try: module avail cuda). Plain replay only."
fi
