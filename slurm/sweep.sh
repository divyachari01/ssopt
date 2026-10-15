#!/bin/bash
# Run the pipeline for every ion in a list file (one spectroscopic name per line).
#   bash slurm/sweep.sh ions_fe.txt                          # per ion: tune + prune, then the full
#                                                            # structx run (ssout.full, avalues.txt)
#   bash slurm/sweep.sh ions_fe.txt --core-bounds 0.8,1.6    # extra flags -> tune stages
# Runs `fetch` on this (login) node first so compute nodes never need internet.
# Ions that already have work/<tag>/levels_orbital.txt are skipped (delete it to redo).
set -euo pipefail
LIST="${1:?ion list file}"; shift
EXTRA=("$@")
export WORKDIR="${WORKDIR:-work}"          # WORKDIR=work_rel bash slurm/sweep.sh ...
if [ -n "${SSOPT_MODULES:-}" ]; then module load $SSOPT_MODULES; fi   # e.g. export SSOPT_MODULES="python"
while IFS= read -r ION; do
  [ -z "$ION" ] && continue; case "$ION" in \#*) continue;; esac
  echo "== $ION"
  TAG=$(python -c "from ssopt.ions import build_ion; print(build_ion('$ION').tag)")
  python -m ssopt.cli fetch "$ION" | tail -2
  if [ -f "$WORKDIR/$TAG/levels_orbital.txt" ]; then
    echo "   already tuned ($WORKDIR/$TAG/levels_orbital.txt) — skipping tuning"
    J=""
  else
    J=$(sbatch --parsable slurm/ion.slurm "$ION" 4 "${EXTRA[@]}" | cut -d';' -f1)
    echo "   tuning job $J"
  fi
  if [ ! -f "$WORKDIR/$TAG/ssout.full" ]; then
    DEP=""; [ -n "$J" ] && DEP="--dependency=afterok:$J"
    sbatch $DEP slurm/full.slurm "$ION" orbital best | sed 's/^/   A-values: /'
  fi
done < "$LIST"
