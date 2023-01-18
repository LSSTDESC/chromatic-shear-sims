#!/bin/bash
#SBATCH --job-name=run-chromatic-shear-bias
#SBATCH --partition=kipac
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#SBATCH --error=run-%j.log
#SBATCH --output=run-%j.out
#SBATCH --time=00:15:00

module load openmpi

source ~/setup.sh
conda activate chromatic-shear-bias

srun run-chromatic-shear-bias \
    --config config.yaml \
    --sim baseline.yaml \
    --output ${SCRATCH}/out-${SLURM_JOB_ID}.fits \
    --n_sims 100
