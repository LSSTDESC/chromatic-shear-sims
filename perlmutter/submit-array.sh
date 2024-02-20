#!/bin/bash

while getopts 'c:s:n:' opt; do
    case $opt in
        c) config=$OPTARG;;
        s) seed=$OPTARG;;
        n) njobs=$OPTARG;;
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
RANDOM=$seed

if [[ ! $njobs ]]; then
	printf '%s\n' "No njobs specified. Exiting.">&2
	exit 1
fi
echo "njobs: $njobs"

echo "sbatch --array=1-$njobs array-task.sh -c $config -s $seed"
sbatch --array=1-$njobs array-task.sh -c $config -s $seed
