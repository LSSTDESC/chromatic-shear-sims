#!/bin/bash
#SBATCH -J chromatic-shear-sims
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.log

echo "task: $SLURM_ARRAY_TASK_ID"

while getopts 'c:s:' opt; do
	case $opt in
		c) config=$OPTARG;;
		s) seed=$OPTARG;;
	esac
done

if [[ ! $config ]]; then
	printf '%s\n' "No config specified. Exiting.">&2
	exit 1
fi
echo "config: $config"

if [[ ! $seed ]]; then
    printf '%s\n' "No seed specified. Exiting.">&2
    exit 1
fi
echo "seed: $seed"

seed=$(($seed + ${SLURM_ARRAY_TASK_ID}))
echo "task seed: $seed"

source setup.sh

run-chromatic-shear-sims \
    --config $config \
    --output /pscratch/sd/s/smau/out \
    --seed $seed \
    --n_sims 1000 \
    --n_jobs 64 \
    --log_level 1
