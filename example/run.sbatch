#!/bin/bash
#SBATCH --job-name=run-chromatic-shear-bias
#SBATCH --partition=kipac
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --error=run-%j.log
#SBATCH --output=run-%j.out
#SBATCH --time=00:10:00

module load openmpi

source ~/setup.sh
conda activate chromatic-shear-bias

srun run-chromatic-shear-bias \
    --config config.yaml \
    --sim test_sagittarius.yaml \
    --output ${SCRATCH}/out \
    --n_sims 25

# out-${SLURM_JOB_ID}
