#!/bin/bash
# =============================================================================
# ssopt: λ tuning with the energy surrogate (any Slurm cluster)
#
#   sbatch slurm/optimize.slurm "Si I"            # per-l stage (4 λ)
#   sbatch slurm/optimize.slurm "Si I" orbital    # per-orbital refinement
#
# Before the FIRST run of an ion, on a LOGIN node (compute nodes have no
# internet):   python -m ssopt.cli fetch "Si I"
# =============================================================================
#SBATCH --job-name=ssopt
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=ssopt_%j.log
set -euo pipefail

ION="${1:-Si I}"
MODE="${2:-l}"
MATCH="${3:-j}"      # level matching: j (default, any ion) or ls
cd "$SLURM_SUBMIT_DIR"

if [ -n "${SSOPT_MODULES:-}" ]; then module load $SSOPT_MODULES; fi   # e.g. export SSOPT_MODULES="python"

echo "ion=$ION mode=$MODE host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
[ -x ./structx ] || { echo "no ./structx — build on a login node: ifort -O2 -o structx struct.f"; exit 2; }

EXTRA=""
if [ "$MODE" = "orbital" ]; then
  # seed the per-orbital stage from the per-l result if present
  TAG=$(python -c "from ssopt.ions import build_ion; print(build_ion('$ION').tag)")
  if [ -f "work/$TAG/history_l.json" ]; then
    SEED=$(python - <<PY
from ssopt.surrogate import History
from ssopt.deck import orbitals_used
import json
h = History.load("work/$TAG/history_l.json"); b = h.best()
d = json.load(open("work/$TAG/deck.json"))
cfgs = [{tuple(o): n for o, n in c} for c in d["configs"]]
print(",".join(f"{b.lam[l]:.4f}" for (n, l) in orbitals_used(cfgs)))
PY
)
    EXTRA="--init-lambdas $SEED"
    echo "seeding orbital stage from per-l best: $SEED"
  fi
fi

python -m ssopt.cli --offline --binary ./structx --data data --workdir work \
    tune "$ION" --mode "$MODE" --workers "$SLURM_CPUS_PER_TASK" \
    --scratch "${TMPDIR:-/tmp}/ssopt_$SLURM_JOB_ID" --n-init 48 --rounds 5 --batch 12 \
    --polish 2 --resume --match-key "$MATCH" $EXTRA
# --resume: re-submitting the same ion/mode continues from work/<tag>/history_<mode>.json

echo "end=$(date)"
