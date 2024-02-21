#!/bin/bash

module load system
module load texlive

source ~/setup_mamba.sh
conda activate /scratch/users/smau/conda/chromatic-shear-sims

# https://joblib.readthedocs.io/en/latest/parallel.html#avoiding-over-subscription-of-cpu-resources
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# https://arrow.apache.org/docs/cpp/env_vars.html
export ARROW_IO_THREADS=1

# https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
export JAX_PLATFORMS=cpu

# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# https://github.com/google/jax/issues/743
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

export THROUGHPUT_DIR=/scratch/users/smau/baseline/
export SED_DIR=/scratch/users/smau/starSED/
export SSP_TEMPLATES=/scratch/users/smau/dsps_ssp_data_singlemet.h5
