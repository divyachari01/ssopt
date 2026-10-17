#!/bin/bash
# =============================================================================
# One ion, the whole tuning pipeline, one job:
#     per-l tune  ->  per-orbital tune (seeded)
#   sbatch slurm/ion.slurm "Fe XVII"
# A-values need the (slow, single-core) full run: sbatch slurm/full.slurm "Fe XVII" orbital best
# =============================================================================
#SBATCH --job-name=ssopt-ion
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=ssopt_ion_%j.log
set -euo pipefail
ION="${1:?ion, e.g. 'Fe XVII'}"
NMAX="${2:-4}"
cd "$SLURM_SUBMIT_DIR"
if [ -n "${SSOPT_MODULES:-}" ]; then module load $SSOPT_MODULES; fi   # e.g. export SSOPT_MODULES="python"
SCRATCH="${TMPDIR:-/tmp}/ssopt_$SLURM_JOB_ID"
WORKDIR="${WORKDIR:-work}"
COMMON="--offline --binary ./structx --data data --workdir $WORKDIR"
TUNE="--workers $SLURM_CPUS_PER_TASK --scratch $SCRATCH --n-init 48 --rounds 5 --batch 12 --polish 2 --resume --n-max $NMAX ${*:3}"
#   any further arguments are passed to both tune stages, e.g.
#   sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6
echo "ion=$ION n_max=$NMAX host=$(hostname) start=$(date)"

python -m ssopt.cli $COMMON tune "$ION" --mode l $TUNE

TAG=$(python -c "from ssopt.ions import build_ion; print(build_ion('$ION').tag)")
SEED=$(python - <<PY
from ssopt.surrogate import History
from ssopt.deck import orbitals_used
import json
h = History.load("$WORKDIR/$TAG/history_l.json"); b = h.best()
d = json.load(open("$WORKDIR/$TAG/deck.json"))
cfgs = [{tuple(o): n for o, n in c} for c in d["configs"]]
print(",".join(f"{b.lam[l]:.4f}" for (n, l) in orbitals_used(cfgs)))
PY
)
python -m ssopt.cli $COMMON tune "$ION" --mode orbital $TUNE --init-lambdas "$SEED"

# optional check of which correlation groups matter (one batch of full-cost runs):
# python -m ssopt.cli $COMMON prune "$ION" --mode orbital --lambdas best --workers $SLURM_CPUS_PER_TASK --n-max $NMAX
echo "end=$(date)"
