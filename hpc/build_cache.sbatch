#!/bin/bash
# Build the Stage-2 tube cache once, into the project copy, before a set of
# training jobs stages it (hpc/submit_rate_sweep.sh chains this in front of
# the sweep). Cache v12 needs the 2026-09-25/26 cleandata: upload
#   cleandata/uniformly_remeshed  cleandata/template_mesh  cleandata/original_centerline
# first. Existing v12 files are kept; only missing ones are built.
#
#   sbatch --account=<account> hpc/build_cache.sbatch
#
# A full node for a short time: ~742 cases x 15-40 s each over 48 workers is
# ~10 min. Each worker builds one case at a time (VTK + raycast, a few GB of
# RSS on the largest cases), which is why the cache is not built inside four
# 1-GPU jobs at once.

#SBATCH --job-name=aneux-cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=aneux_cache_%j.out
#SBATCH --error=aneux_cache_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
echo "=== aneux tube cache $(date -Is) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/hpc/common.sh"

export ANEUX_CACHE_ONLY=1
export ANEUX_CACHE_WORKERS="${ANEUX_CACHE_WORKERS:-48}"
srun --cpu-bind=none --kill-on-bad-exit=1 \
  python "${REPO}/1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py"

ver=$(cd "${REPO}/1test_encoder_decoder_only/train_pipeline" && python -c 'from config import CACHE_VERSION; print(CACHE_VERSION)')
n_ready=$(find "$ANEUX_CACHE" -maxdepth 1 -name "*_v${ver}_*.pt" | wc -l)
echo "v${ver} cache files in ${ANEUX_CACHE}: ${n_ready}"
echo "=== finished $(date -Is) ==="
