#!/bin/bash
# Submit the R* sweep: one cache build, then the 4-run array once it succeeds.
#
# On the login node, from the aneux checkout, after `git pull` and after
# uploading cleandata/{uniformly_remeshed,template_mesh,original_centerline}:
#   bash hpc/submit_rate_sweep.sh <account> [mail]
# e.g. bash hpc/submit_rate_sweep.sh nr_hemo_ai1 you@example.com
#
# Old tube_cache versions (v10/v11) can stay; staging copies only the current
# version. Delete them to free project space: rm tube_cache/*_v1[01]_*.pt
set -euo pipefail
ACCOUNT="${1:?usage: bash hpc/submit_rate_sweep.sh <account> [mail]}"
MAIL="${2:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -d 1test_encoder_decoder_only/train_pipeline ]] || { echo "run from the aneux checkout" >&2; exit 1; }

mail_args=()
[[ -n "$MAIL" ]] && mail_args=(--mail-user="$MAIL")

cache_job=$(sbatch --parsable --account="$ACCOUNT" "${mail_args[@]}" hpc/build_cache.sbatch)
echo "cache build: job ${cache_job}"
sweep_job=$(sbatch --parsable --account="$ACCOUNT" "${mail_args[@]}" \
  --dependency=afterok:"${cache_job}" --kill-on-invalid-dep=yes hpc/sweep_rate_target.sbatch)
echo "R* sweep:    array job ${sweep_job} (tasks 0-3 = R* 0.25 0.5 1 2), starts after ${cache_job}"
echo "watch:       squeue -u \$USER ; tail -f aneux_cache_${cache_job}.out"
echo "runs land in 1test_encoder_decoder_only/output/runs/*_job<id>_rstar<R>/"
