#!/bin/bash
#SBATCH --job-name=meas-chromatic-shear-bias
#SBATCH --partition=kipac
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#SBATCH --error=meas-%j.log
#SBATCH --output=meas-%j.out
#SBATCH --time=00:15:00

source ~/setup.sh
conda activate chromatic-shear-bias

meas-chromatic-shear-bias \
    --config config.yaml \
    --s2n-cut 10 \
    --mfrac-cut 100 \
    --method jackknife \
    --n_resample 2 \
    --output ${SCRATCH}/out-*.fits
