#!/bin/bash
# Full structx run (radiative data ON) at tuned λ + A-value check.
# Arguments after the 4th go to ssopt, e.g. --doubles complex (use the same as for tuning).
#   sbatch slurm/full.slurm "Si I" l 0.98,1.02,1.1,0.95
#SBATCH --job-name=ssopt-full
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=ssopt_full_%j.log
set -euo pipefail
ION="${1:-Si I}"; MODE="${2:-l}"; LAMS="${3:-best}"; MATCH="${4:-j}"
cd "$SLURM_SUBMIT_DIR"
if [ -n "${SSOPT_MODULES:-}" ]; then module load $SSOPT_MODULES; fi   # e.g. export SSOPT_MODULES="python"
python -m ssopt.cli --offline --binary ./structx --data data --workdir "${WORKDIR:-work}" \
    full "$ION" --mode "$MODE" --lambdas "$LAMS" --match-key "$MATCH" "${@:5}"
# NIST-consistent labels in ssout.full (original kept as ssout.full.ss), then NORAD-style tables
python -m ssopt.export "$ION" --workdir "${WORKDIR:-work}" --data data --out norad_ss --relabel-ssout || true
