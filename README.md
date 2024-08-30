# chromatic-shear-sims

Chromatic image simulations for shear testing

## environment

```
conda create --prefix /path/to/prefix --file requirements.txt
pip install -e .
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

`run-chromatic-shear-sim`
```
usage: run-chromatic-shear-sim [-h] [--seed SEED] [--n_sims N_SIMS] [--n_jobs N_JOBS] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]
  output                output directory

options:
  -h, --help            show this help message and exit
  --seed SEED           RNG seed [int]
  --n_sims N_SIMS       Number of sims to run [int; 1]
  --n_jobs N_JOBS       Number of parallel jobs to run [int; 1]
  --log_level LOG_LEVEL
                        logging level [int; 2]
```


`aggregate-chromatic-shear-sim`
```
usage: aggregate-chromatic-shear-sim [-h] [--s2n-cut S2N_CUT] [--ormask-cut ORMASK_CUT] [--mfrac-cut MFRAC_CUT] [--n_jobs N_JOBS] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]
  output                output directory

options:
  -h, --help            show this help message and exit
  --s2n-cut S2N_CUT     Signal/noise cut [int; 10]
  --ormask-cut ORMASK_CUT
                        Cut to make on ormask. 0 indicates make a cut, 1 indicates no cut.
  --mfrac-cut MFRAC_CUT
                        Cut to make on mfrac. Given in percentages and comma separated. Cut keeps all objects less than the given value.
  --n_jobs N_JOBS       Number of parallel jobs to run [int; 1]
  --log_level LOG_LEVEL
                        logging level [int; 2]
```

`measure-chromatic-shear-sim`
```
usage: measure-chromatic-shear-sim [-h] [--seed SEED] [--n_resample N_RESAMPLE] [--n_jobs N_JOBS] [--log_level LOG_LEVEL] config

positional arguments:
  config                configuration file [yaml]
  output                output directory

options:
  -h, --help            show this help message and exit
  --seed SEED           RNG seed [int]
  --n_resample N_RESAMPLE
                        Number of resample iterations
  --n_jobs N_JOBS       Number of jobs to run [int; 1]
  --log_level LOG_LEVEL
                        logging level [int; 2]
```
