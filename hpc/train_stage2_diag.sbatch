#!/bin/bash
# Komondor Stage-2 DIAGNOSTIC run: the same 4-GPU DDP job as train_stage2.sbatch,
# plus everything needed to name the cause if it hits CUDA error 700 again
# (jobs 14506605 and 14509551: one rank "illegal memory access", the other three
# stuck in an all-reduce until the 10-minute NCCL watchdog).
#
# It must run the SAME code the crashed jobs ran (commit 3a88db5 plus the opt-in
# diagnostics). The two attempted fixes are reverted on purpose: a fix that hides
# the crash cannot tell us what caused it.
#
# What it adds over train_stage2.sbatch:
#   CUDA_LAUNCH_BLOCKING=1   the Python traceback names the kernel that faulted,
#                            not the next synchronizing call. It slows the step;
#                            ANEUX_LAUNCH_BLOCKING=0 turns it off (the breadcrumbs
#                            below still give the stage and the batch).
#   ANEUX_DIAG=1             crash_diag.py: per-sample scan (non-finite values,
#                            out-of-range indices, spline pseudo-coordinates),
#                            a per-step log (sample ids, sizes, loss, grad norm),
#                            a crash report, the failing batch and RNG state for
#                            replay, a stall watchdog, and an immediate hard exit
#                            so torchrun stops the job.
#   NCCL flight recorder     which collective each rank was in.
#   GPU health before/after  ECC, retired pages, Xid, plus the exact code state.
#
# Everything lands in
#   1test_encoder_decoder_only/output/diag/job<ID>/
# on project disk (a killed rank never runs the scratch copy-back).
#
# Submit from the repo root on the login node, after `git pull`:
#   sbatch --account=nr_hemo_ai1 --mail-user=YOU@email hpc/train_stage2_diag.sbatch
#
# Read first when it fails:
#   diag/job<ID>/summary.txt         what the job collected after the crash
#   diag/job<ID>/crash_rank*.txt     stage, sample ids, traceback, GPU state
#   diag/job<ID>/rank*.log           one JSON line per step; grep WARN
# Then replay the dumped step with hpc/replay_crash.sbatch.
#
# 250 epochs on purpose: the LR / KL schedule has to match the crashed jobs. The
# 12 h limit ends the job if nothing crashes, which is also informative (see the
# note printed at the end).

#SBATCH --job-name=aneux-s2-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --signal=TERM@1800
#SBATCH --output=aneux_s2_diag_%j.out
#SBATCH --error=aneux_s2_diag_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miklos.damjan@gmail.com

set -euo pipefail

echo "=== aneux Stage-2 $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
echo "account=${SLURM_JOB_ACCOUNT:-unset} partition=${SLURM_JOB_PARTITION:-unset}"

# Repo: sbatch from the git checkout, or set ANEUX_REPO_ROOT.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/1test_encoder_decoder_only/train_pipeline" ]]; then
  REPO="${ANEUX_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
elif [[ -n "${ANEUX_REPO_ROOT:-}" ]]; then
  REPO="$ANEUX_REPO_ROOT"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"
export ANEUX_REPO_ROOT="$REPO"
export ANEUX_CLEANDATA="${ANEUX_CLEANDATA:-$REPO/cleandata}"
export ANEUX_CACHE="${ANEUX_CACHE:-$REPO/1test_encoder_decoder_only/tube_cache}"
export ANEUX_OUTPUT="${ANEUX_OUTPUT:-$REPO/1test_encoder_decoder_only/output}"
# 4 graphs per GPU and no gradient accumulation: 16 on the 4-GPU allocation.
export ANEUX_BATCH_SIZE="${ANEUX_BATCH_SIZE:-4}"
export ANEUX_ACCUM_STEPS="${ANEUX_ACCUM_STEPS:-1}"
export ANEUX_EPOCHS="${ANEUX_EPOCHS:-250}"
echo "repo=$REPO"
echo "batch_per_gpu=${ANEUX_BATCH_SIZE} accum=${ANEUX_ACCUM_STEPS} epochs=${ANEUX_EPOCHS}"

_source_conda() {
  local cand
  for cand in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/.conda/etc/profile.d/conda.sh"
  do
    if [[ -f "$cand" ]]; then
      # shellcheck disable=SC1090
      source "$cand"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return 0
  fi
  return 1
}

# Python: the interactive env, ~/aneuxai_env. aneurysmgnn is only a fallback.
# Do not `module load cuda` — that wheel already bundles CUDA
# (https://docs.hpc.dkf.hu/AI/pytorch.html).
# torchrun below is the same aneuxai_HPC.py, one process per GPU.
if [[ -f "${HOME}/aneuxai_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/aneuxai_env/bin/activate"
  echo "env=aneuxai_env"
elif _source_conda; then
  _ANEurySMGNN=""
  for _cand in \
    "${HOME}/.conda/envs/aneurysmgnn" \
    "${HOME}/miniconda3/envs/aneurysmgnn" \
    "${HOME}/anaconda3/envs/aneurysmgnn"
  do
    if [[ -x "${_cand}/bin/python" ]]; then
      _ANEurySMGNN="${_cand}"
      break
    fi
  done
  if [[ -n "$_ANEurySMGNN" ]]; then
    conda activate aneurysmgnn
    echo "env=aneurysmgnn"
  fi
fi
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no ~/aneuxai_env and no conda env aneurysmgnn" >&2
  exit 1
fi
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "exe", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
PY

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Single-node DDP as in the Komondor PyTorch example.
# --cpu-bind=none: DataLoader workers must not be pinned to one core.
export RDZV_HOST
RDZV_HOST="$(hostname)"
export RDZV_PORT="${RDZV_PORT:-29400}"
NPROC="$(python -c 'import torch; print(max(1, int(torch.cuda.device_count())))')"
echo "nproc_per_node=${NPROC} cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"


# ---------------------------------------------------------------- diagnostics
DIAG_DIR="${ANEUX_DIAG_DIR:-${ANEUX_OUTPUT}/diag/job${SLURM_JOB_ID:-local}}"
mkdir -p "$DIAG_DIR"
export ANEUX_DIAG=1
export ANEUX_DIAG_DIR="$DIAG_DIR"
# A rank that makes no progress for this long dumps all of its stacks.
export ANEUX_DIAG_STALL_S="${ANEUX_DIAG_STALL_S:-1200}"

export CUDA_LAUNCH_BLOCKING="${ANEUX_LAUNCH_BLOCKING:-1}"
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${DIAG_DIR}/nccl_trace_rank"
echo "diag dir=$DIAG_DIR  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<default: expandable_segments from config.py>}"

{
  echo "=== code $(date -Is) ==="
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" log -1 --format='%h %s'
  echo "--- git status (uncommitted changes):"
  git -C "$REPO" status --short
  echo "--- packages:"
  python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
for m in ("torch_geometric", "pyg_lib", "pytorch3d", "numpy", "scipy", "vtk", "pyvista"):
    try:
        print(m, getattr(importlib.import_module(m), "__version__", "ok"))
    except Exception as exc:
        print(m, "MISSING", exc)
PY
  echo "--- node: $(hostname)"
  nvidia-smi -L
} > "${DIAG_DIR}/env.txt" 2>&1 || true
nvidia-smi -q -d ECC,PAGE_RETIREMENT > "${DIAG_DIR}/gpu_before.txt" 2>&1 || true

# ---------------------------------------------------------------- run
# A failing srun must not skip the post-mortem below.
set +e
if [[ "${NPROC}" -eq 1 ]]; then
  srun --cpu-bind=none --kill-on-bad-exit=1 \
    python "${REPO}/1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py"
else
  srun --cpu-bind=none --kill-on-bad-exit=1 torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node="${NPROC}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
    "${REPO}/1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py"
fi
RC=$?
set -e
echo "=== srun exit code ${RC} at $(date -Is) ==="

# ---------------------------------------------------------------- post-mortem
{
  echo "=== summary job ${SLURM_JOB_ID} rc=${RC} node=$(hostname) $(date -Is) ==="
  echo "--- crash reports:"
  ls -1 "${DIAG_DIR}"/crash_rank*.txt 2>/dev/null || echo "none (no rank raised a Python exception)"
  for f in "${DIAG_DIR}"/crash_rank*.txt; do
    [[ -f "$f" ]] || continue
    echo; echo "##### $f"; sed -n '1,60p' "$f"
  done
  echo; echo "--- sample / device warnings from the step logs:"
  grep -h '"kind": "WARN' "${DIAG_DIR}"/rank*.log 2>/dev/null | cut -c1-400 | head -60 || true
  echo; echo "--- last step line of every rank:"
  for f in "${DIAG_DIR}"/rank*.log; do
    [[ -f "$f" ]] || continue
    echo "$f"
    grep -h '"kind": "step"' "$f" | tail -1 | cut -c1-300 || true
    grep -h '"kind": "done"' "$f" | tail -1 | cut -c1-300 || true
  done
  echo; echo "--- stall dumps (non-empty stacks files):"
  find "${DIAG_DIR}" -name 'rank*.stacks.txt' -size +0 2>/dev/null
  echo; echo "--- Xid / NVRM in the kernel log (empty if not readable):"
  (dmesg 2>/dev/null | grep -i -E 'xid|nvrm' | tail -20) || true
} > "${DIAG_DIR}/summary.txt" 2>&1 || true
nvidia-smi -q -d ECC,PAGE_RETIREMENT > "${DIAG_DIR}/gpu_after.txt" 2>&1 || true
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,retired_pages.pending,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv >> "${DIAG_DIR}/gpu_after.txt" 2>&1 || true
sacct -j "${SLURM_JOB_ID}" -o JobID,State,ExitCode,Elapsed,MaxRSS,NodeList --parsable2 > "${DIAG_DIR}/sacct.txt" 2>&1 || true

if [[ "$RC" -ne 0 ]]; then
  # The crashing rank exits at once, so rank 0's own scratch -> project copy
  # never runs. Take the run folder (metrics, last.pt for the replay) here.
  ACCT="${SLURM_JOB_ACCOUNT:-nr_hemo_ai1}"
  WS="/scratch/${ACCT}/${USER}/aneux/${SLURM_JOB_ID}"
  if [[ -d "${WS}/output/runs" ]]; then
    mkdir -p "${DIAG_DIR}/run"
    rsync -a "${WS}/output/runs/" "${DIAG_DIR}/run/" || echo "rsync of ${WS}/output/runs failed"
  else
    echo "no ${WS}/output/runs to copy"
  fi
fi

echo
echo "diagnostics: ${DIAG_DIR}"
echo "  cat ${DIAG_DIR}/summary.txt"
if [[ "$RC" -eq 0 ]]; then
  echo "NOTE: the job ended without a CUDA error. With CUDA_LAUNCH_BLOCKING=1 that is"
  echo "      itself a result (a timing-dependent fault); rerun with ANEUX_LAUNCH_BLOCKING=0."
fi
exit "$RC"
