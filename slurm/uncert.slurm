#!/bin/bash
# Uncertainty and grade of every E1 A-value and lifetime, after tuning (both stages).
# Give the same deck options as for the tuning (for example --closed-core), so the deck is the same.
#   sbatch slurm/uncert.slurm "Ni XV"
#   sbatch slurm/uncert.slurm "Ni XV" --closed-core
# Writes work/<ion>/uncert.txt. About 17 full runs (the λ ensemble and λ = 1), run in parallel.
#SBATCH --job-name=ssopt-unc
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=ssopt_uncert_%j.log
set -euo pipefail
ION="${1:-Si I}"
cd "$SLURM_SUBMIT_DIR"
if [ -n "${SSOPT_MODULES:-}" ]; then module load $SSOPT_MODULES; fi   # e.g. export SSOPT_MODULES="python"
python -m ssopt.cli --offline --binary ./structx --data data --workdir "${WORKDIR:-work}" \
    uncert "$ION" --mode orbital --workers "${SLURM_CPUS_PER_TASK:-4}" "${@:2}"
