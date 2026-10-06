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
#
# Four stages, cheapest first, each logged with its exit code:
#   1  augmentation only (mirror, pose jitter, x_true resampling), exact replay x AUG_REPEAT
#      then AUG_SWEEP other RNG draws on the same batch, launch-blocking. Seconds per pass.
#   2  the same under compute-sanitizer memcheck
#   3  the full step, launch-blocking
#   4  the full step under memcheck
# Jobs 14511899 and 14514130 faulted at the same instruction (identical warp-exception
# registers) in the torch.cat of _hybrid_subset_indices at epoch 9 step 30, with
# expandable_segments both on and off, and the same step replays clean on an RTX 3080 Ti.
# If stage 1 faults on the A100 the fault is a function of this batch; if all four pass,
# it depends on the state of the live process (allocator layout, earlier corruption).

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

# compute-sanitizer: on PATH, from a cuda module, from a pip nvidia-cuda-sanitizer wheel
# inside the venv, or from a toolkit under /usr/local or /opt.
find_sanitizer() {
  local s
  s="$(command -v compute-sanitizer || true)"
  if [[ -z "$s" ]]; then
    for m in cuda CUDA; do
      module load "$m" >/dev/null 2>&1 && s="$(command -v compute-sanitizer || true)"
      [[ -n "$s" ]] && break
    done
  fi
  if [[ -z "$s" ]]; then
    s="$(ls -1 "${VIRTUAL_ENV:-/nonexistent}"/lib/python*/site-packages/nvidia/*/bin/compute-sanitizer \
            /usr/local/cuda*/bin/compute-sanitizer /opt/nvidia/*/bin/compute-sanitizer \
            /opt/nvidia/hpc_sdk/*/*/cuda/*/bin/compute-sanitizer 2>/dev/null | tail -1 || true)"
  fi
  echo "$s"
}

set +e
echo "=== node: $(hostname)  $(uname -r)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) ==="

# 1. The pre-model part of the step only (mirror, pose jitter, x_true resampling): seconds per
#    pass, so repeat it and then sweep other RNG draws over the same batch. If the A100 faults
#    here, the fault is a function of this data and the first failing pass names the kernel.
echo "=== replay 1: augmentation only, launch-blocking, exact replay x${AUG_REPEAT:-20} + ${AUG_SWEEP:-200} RNG sweeps ==="
python "$REPLAY" "${ARGS[@]}" --aug-only --repeat "${AUG_REPEAT:-20}" --sweep "${AUG_SWEEP:-200}"
echo "augmentation replay exit code: $?"

SANITIZER="$(find_sanitizer)"
if [[ -n "$SANITIZER" ]]; then
  echo "=== sanitizer: $SANITIZER ==="
  "$SANITIZER" --version 2>&1 | head -3
  echo "=== replay 2: augmentation only under memcheck (shared/local/global out-of-range accesses) ==="
  "$SANITIZER" --tool memcheck --launch-timeout 0 --print-limit 40 \
    python "$REPLAY" "${ARGS[@]}" --aug-only --repeat 1 --sweep 2
  echo "sanitizer augmentation exit code: $?"
else
  echo "compute-sanitizer not found (module avail cuda; pip install nvidia-cuda-sanitizer-api-cu13 into the venv)."
fi

# 3. The whole step (model forward/backward), plain, then under memcheck.
echo "=== replay 3: full step, launch-blocking, plain python ==="
python "$REPLAY" "${ARGS[@]}" --repeat "${REPEAT:-3}"
echo "plain replay exit code: $?"
if [[ -n "$SANITIZER" ]]; then
  echo "=== replay 4: full step under memcheck ==="
  "$SANITIZER" --tool memcheck --launch-timeout 0 --print-limit 40 python "$REPLAY" "${ARGS[@]}"
  echo "sanitizer replay exit code: $?"
fi
