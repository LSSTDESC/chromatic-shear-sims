# chromatic-shear-bias
Simulations to validate calibration of shear bias

## bin

- `viz-chromatic-shear-bias`: visualize a simulation
- `run-chromatic-shear-bias`: run simulation and shear measurement
- `meas-chromatic-shear-bias`: measure multiplicative and additive bias of shear

## note about `psutil`

While not used directly in this code, it is imoprtant that `psutil` is available so that the loky backend to joblib can [protect against memory leeks](https://joblib.readthedocs.io/en/latest/developing.html#release-0-12-2).

