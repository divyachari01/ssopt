#!/bin/bash
# Full structx run (radiative data ON) at tuned λ + A-value check.
#   sbatch slurm/full.slurm "Si I" l 0.98,1.02,1.1,0.95
#SBATCH --job-name=ssopt-full
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=ssopt_full_%j.log
#SBATCH --account=PAS0578
#SBATCH --cluster=cardinal
set -euo pipefail
ION="${1:-Si I}"; MODE="${2:-l}"; LAMS="${3:-best}"; MATCH="${4:-j}"
cd "$SLURM_SUBMIT_DIR"
module load intel/2021.10.0
module load python/3.12
python -m ssopt.cli --offline --binary ./structx --data data --workdir "${WORKDIR:-work}" \
    full "$ION" --mode "$MODE" --lambdas "$LAMS" --match-key "$MATCH"
# NORAD-style tables (en.ls, en.fs, f.ls, f.fs) from the finished ssout.full
python -m ssopt.export "$ION" --workdir "${WORKDIR:-work}" --data data --out norad_ss || true
