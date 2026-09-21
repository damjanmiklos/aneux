#!/bin/bash
# Komondor GPU-partition Stage-2 training.
# Docs: https://docs.hpc.dkf.hu/tasks/sbatch.html
#        https://docs.hpc.dkf.hu/AI/pytorch.html
#        https://docs.hpc.dkf.hu/hardware/gpu.html
#        https://docs.hpc.dkf.hu/storage/overview.html
#
# Workflow (login node):
#   cd /project/<account>/aneux
#   git pull
#   # If a previous job already built tube_cache on scratch, copy it first:
#   # rsync -a /scratch/<account>/$USER/aneux/<oldjob>/tube_cache/ \
#   #   $PWD/1test_encoder_decoder_only/tube_cache/
#   sbatch --account=<account> --mail-user=YOU@email hpc/train_stage2.sbatch
#
# Interactive 1-GPU smoke test (1 hour, 16 cores — do not use this sbatch):
#   srun -p gpu -A nr_hemo_ai1 --gres=gpu:1 -c 16 --mem-per-cpu=4000 \
#     --time=01:00:00 --pty bash
#   source ~/aneuxai_env/bin/activate
#   export ANEUX_EPOCHS=1
#   python 1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py
# Faster queue, 1h cap: srun -p test -A nr_hemo_ai1 --gres=gpu:1 -c 16 --time=01:00:00 --pty bash
#
# Optional env (sbatch --export=ALL,ANEUX_EPOCHS=2 ...):
#   ANEUX_BATCH_SIZE  ANEUX_NUM_WORKERS  ANEUX_EPOCHS  ANEUX_SKIP_STAGE=1
#
# Logs land in the submit directory. The timestamped run folder is written
# on /scratch during the job and copied back to
# 1test_encoder_decoder_only/output/runs/<stamp>_job<id>/.

#SBATCH --job-name=aneux-s2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --mem=0
#SBATCH --no-requeue
#SBATCH --signal=TERM@180
#SBATCH --output=aneux_s2_%j.out
#SBATCH --error=aneux_s2_%j.err
#SBATCH --mail-type=END,FAIL
# Default QOS is "normal" (not preempted). Max walltime on gpu is 7 days.
# --exclusive does not imply all RAM; --mem=0 does (docs: tasks/sbatch.html).

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
echo "repo=$REPO"

# Python: aneurysmgnn (Python 3.11, already on this cluster), then ~/aneuxai_env.
# Do not `module load cuda` — a custom torch wheel already bundles it
# (https://docs.hpc.dkf.hu/AI/pytorch.html).
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

if [[ -n "$_ANEurySMGNN" ]] && _source_conda; then
  conda activate aneurysmgnn
  echo "env=aneurysmgnn"
elif [[ -f "${HOME}/aneuxai_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/aneuxai_env/bin/activate"
  echo "env=aneuxai_env"
else
  echo "ERROR: no conda env aneurysmgnn and no ~/aneuxai_env" >&2
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

echo "=== finished $(date -Is) ==="

# Utilization for dialing workers / batch / staging. seff is incomplete while
# the allocation is still held (https://docs.hpc.dkf.hu/tasks/efficiency.html);
# jobstats works for running and completed jobs. Re-run seff after the job
# leaves the queue for the billed CPU/mem efficiency.
STATS_DIR=""
shopt -s nullglob
_cands=("${ANEUX_OUTPUT}/runs/"*"_job${SLURM_JOB_ID}")
if ((${#_cands[@]})); then
  STATS_DIR="${_cands[-1]}/data"
fi
if [[ -z "$STATS_DIR" || ! -d "$STATS_DIR" ]]; then
  STATS_DIR="${ANEUX_OUTPUT}/runs"
  mkdir -p "$STATS_DIR"
fi
{
  echo "=== seff ${SLURM_JOB_ID} $(date -Is) ==="
  seff "${SLURM_JOB_ID}" || true
} > "${STATS_DIR}/seff.txt" 2>&1 || true
{
  echo "=== jobstats ${SLURM_JOB_ID} $(date -Is) ==="
  jobstats "${SLURM_JOB_ID}" || true
} > "${STATS_DIR}/jobstats.txt" 2>&1 || true
{
  echo "=== sacct ${SLURM_JOB_ID} $(date -Is) ==="
  sacct -j "${SLURM_JOB_ID}" -o JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES,TotalCPU,MaxDiskRead,MaxDiskWrite --parsable2 || true
} > "${STATS_DIR}/sacct.txt" 2>&1 || true
{
  echo "=== sstat ${SLURM_JOB_ID} $(date -Is) ==="
  sstat --allsteps -j "${SLURM_JOB_ID}" -o JobID,MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite || true
} > "${STATS_DIR}/sstat.txt" 2>&1 || true
{
  echo "=== nvidia-smi $(date -Is) ==="
  nvidia-smi || true
} > "${STATS_DIR}/nvidia_smi_end.txt" 2>&1 || true

echo "Stats written under ${STATS_DIR}"
echo "After the job leaves the queue, re-run: seff ${SLURM_JOB_ID}"
echo "GPU/CPU time series: jobstats ${SLURM_JOB_ID}"
echo "Web: https://jobstats.komondor.hinfra.hu"
echo "Run folder: ${ANEUX_OUTPUT}/runs/"
