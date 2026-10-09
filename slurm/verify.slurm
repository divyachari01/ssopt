#!/bin/bash
# λ-predictor test: rebuild the dataset, fit, then ONE structx run per tuned ion at its
# leave-one-ion-out predicted λ (gp and seq models).  Cheap: ~2 runs per ion on 48 cores.
#   sbatch slurm/verify.slurm
#SBATCH --job-name=ssopt-verify
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --output=ssopt_verify_%j.log
#SBATCH --account=PAS0578
#SBATCH --cluster=cardinal
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load intel/2021.10.0
module load python/3.12
SCRATCH="${TMPDIR:-/tmp}/ssopt_verify_$SLURM_JOB_ID"
WORKDIR="${WORKDIR:-work}"; DATASET="${DATASET:-dataset}"   # WORKDIR=work_rel DATASET=dataset_rel sbatch ...
python -m ssopt.dataset --workdir $WORKDIR --out $DATASET
for MODEL in gp seq; do
  python -m ssopt.predict --dataset $DATASET --model $MODEL fit
  python -m ssopt.predict --dataset $DATASET --model $MODEL verify --workdir $WORKDIR --data data \
      --binary ./structx --workers $SLURM_CPUS_PER_TASK --scratch $SCRATCH/$MODEL
done
python -m ssopt.figures --dataset $DATASET
echo "end=$(date)"
