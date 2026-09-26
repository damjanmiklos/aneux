#!/bin/bash
# Shared setup for the Komondor sbatch scripts in hpc/. Source it from a job
# that was submitted from the repo root (SLURM_SUBMIT_DIR = the git checkout):
#   source "${SLURM_SUBMIT_DIR}/hpc/common.sh"
# Sets REPO and the ANEUX_* paths, activates the Python env, and defines
# aneux_write_job_stats. Same env rules as train_stage2.sbatch.

if [[ -n "${ANEUX_REPO_ROOT:-}" ]]; then
  REPO="$ANEUX_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/1test_encoder_decoder_only/train_pipeline" ]]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  echo "ERROR: submit from the aneux checkout or set ANEUX_REPO_ROOT" >&2
  exit 1
fi
cd "$REPO"
export ANEUX_REPO_ROOT="$REPO"
export ANEUX_CLEANDATA="${ANEUX_CLEANDATA:-$REPO/cleandata}"
export ANEUX_CACHE="${ANEUX_CACHE:-$REPO/1test_encoder_decoder_only/tube_cache}"
export ANEUX_OUTPUT="${ANEUX_OUTPUT:-$REPO/1test_encoder_decoder_only/output}"
echo "repo=$REPO"

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

# Python: ~/aneuxai_env, else conda env aneurysmgnn. Do not `module load cuda`
# -- the wheel bundles CUDA (https://docs.hpc.dkf.hu/AI/pytorch.html).
if [[ -f "${HOME}/aneuxai_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/aneuxai_env/bin/activate"
  echo "env=aneuxai_env"
elif _source_conda; then
  for _cand in \
    "${HOME}/.conda/envs/aneurysmgnn" \
    "${HOME}/miniconda3/envs/aneurysmgnn" \
    "${HOME}/anaconda3/envs/aneurysmgnn"
  do
    if [[ -x "${_cand}/bin/python" ]]; then
      conda activate aneurysmgnn
      echo "env=aneurysmgnn"
      break
    fi
  done
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Job statistics next to the run's data/. seff and jobstats are not on the
# compute nodes' PATH (the 2026-09-22 job logged "command not found"), and
# sacct cannot reach slurmdbd from there: run them on the login node later.
aneux_write_job_stats() {
  local stats_dir="$1"
  mkdir -p "$stats_dir"
  local tool
  for tool in seff jobstats; do
    if command -v "$tool" >/dev/null 2>&1; then
      { echo "=== $tool ${SLURM_JOB_ID} $(date -Is) ==="; "$tool" "${SLURM_JOB_ID}" || true; } \
        > "${stats_dir}/${tool}.txt" 2>&1 || true
    else
      echo "$tool is not on the compute node; on the login node run: $tool ${SLURM_JOB_ID}" \
        > "${stats_dir}/${tool}.txt"
    fi
  done
  { echo "=== sstat ${SLURM_JOB_ID} $(date -Is) ==="
    sstat --allsteps -j "${SLURM_JOB_ID}" -o JobID,MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite || true
  } > "${stats_dir}/sstat.txt" 2>&1 || true
  { echo "=== nvidia-smi $(date -Is) ==="; nvidia-smi || true; } > "${stats_dir}/nvidia_smi_end.txt" 2>&1 || true
  echo "Stats written under ${stats_dir}"
}
