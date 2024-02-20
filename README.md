# chromatic-shear-sims

Chromatic image simulations for shear testing

## environment

```
conda create --prefix /path/to/prefix --file requirements.txt
```

## bin

- `prepare-chromatic-shear-sims`: prepare a config for analysis
- `plot-chromatic-shear-sims`: make an example plot of a config
- `run-chromatic-shear-sims`: run simulations
- `aggregate-chromatic-shear-sims`: aggregate measurement results
- `meas-chromatic-shear-sims`: make final measurements
- `plot-drdc-chromatic-shear-sims`: make a plot of the chromatic response

## usage

```
prepare-chromatic-shear-sims --config config.yaml
```

```
plot-chromatic-shear-sims --config config.yaml --seed $RANDOM --detect
 ```


```
run-chromatic-shear-sims --config config.yaml --seed $RANDOM --n_sims 1000 --n_jos 96 --output $SCRATCH/out
```

```
aggregate-chromatic-shear-sims --config config.yaml --seed $RANDOM --s2n-cut 10 --ormask-cut 1 --n_jobs 16 --output /pscratch/sd/s/smau/out
```

```
meas-chromatic-shear-sims.py --config config.yaml --seed $RANDOM --n_resample 1000 --n_jobs 16 --output /pscratch/sd/s/smau/out/
```
