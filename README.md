# chromatic-shear-sims

Chromatic image simulations for shear testing

## environment

```
conda create --prefix /path/to/prefix --file requirements.txt
```

## scripts

`plot-scene-chromatic-shear-sim`
```
usage: plot-scene-chromatic-shear-sim [-h] [--seed SEED] [--n_sims N_SIMS] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]

options:
  -h, --help            show this help message and exit
  --seed SEED           RNG seed [int]
  --n_sims N_SIMS       Number of sims to run [int; 1]
  --log_level LOG_LEVEL
                        logging level [int; 2]
```

`plot-obs-chromatic-shear-sim`
```
usage: plot-obs-chromatic-shear-sim [-h] [--seed SEED] [--n_sims N_SIMS] [--n_jobs N_JOBS] [--detect] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]

options:
  -h, --help            show this help message and exit
  --seed SEED           RNG seed [int]
  --n_sims N_SIMS       Number of sims to run [int; 1]
  --n_jobs N_JOBS       Number of parallel jobs to run [int; 1]
  --detect              run detection
  --log_level LOG_LEVEL
                        logging level [int; 2]
```

`measure-chromatic-shear-sim`
```
usage: measure-chromatic-shear-sim [-h] [--seed SEED] [--n_sims N_SIMS] [--n_jobs N_JOBS] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]

options:
  -h, --help            show this help message and exit
  --seed SEED           RNG seed [int]
  --n_sims N_SIMS       Number of sims to run [int; 1]
  --n_jobs N_JOBS       Number of parallel jobs to run [int; 1]
  --log_level LOG_LEVEL
                        logging level [int; 2]
```
