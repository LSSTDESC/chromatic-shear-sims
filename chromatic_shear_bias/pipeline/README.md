# Chromatic Shear Sims

1. `prepare.py`

```
python prepare.py --config config.yaml
```

1.1 `plot.py`
```
 python plot.py --config config.yaml --seed $RANDOM --detect True
 ```

2. `run.py`

```
python run.py --config config.yaml --seed $RANDOM --n_sims 1000 --n_jos 96 --output $SCRATCH/out
```

3. `aggregate.py`
```
python aggregate.py --config config.yaml --seed $RANDOM --s2n-cut 10 --ormask-cut 1 --n_jobs 16 --output /pscratch/sd/s/smau/out
```

4. `meas.py`
```
python meas.py --config config.yaml --seed $RANDOM --n_resample 1000 --n_jobs 16 --output /pscratch/sd/s/smau/out/
```
