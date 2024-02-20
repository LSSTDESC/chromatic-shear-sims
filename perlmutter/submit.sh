#!/bin/bash
#SBATCH -J chromatic-shear-sims
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --output=logs/slurm-%J.out
#SBATCH --error=logs/slurm-%J.log

source setup.sh

# python run.py \
run-chromatic-shear-sims
    --config config.yaml \
    --output /pscratch/sd/s/smau/out \
    --seed $RANDOM \
    --n_sims 1000 \
    --n_jobs 96
