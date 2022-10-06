# shear-bias-sims
Simulations to validate calibration of shear bias

## bin

- `viz-shear-bias-sims`: visualize a simulation
- `run-shear-bias-sims`: run simulation and shear measurement
- `meas-shear-bias-sims`: measure multiplicative and additive bias of shear

## note about `psutil`

While not used directly in this code, it is imoprtant that `psutil` is available so that the loky backend to joblib can [protect against memory leeks](https://joblib.readthedocs.io/en/latest/developing.html#release-0-12-2).

