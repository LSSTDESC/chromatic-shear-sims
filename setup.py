import os
from setuptools import setup, find_packages

scripts = [
    "bin/aggregate-chromatic-shear-sims",
    "bin/meas-chromatic-shear-sims",
    "bin/plot-chromatic-shear-sims",
    "bin/plot-drdc-chromatic-shear-sims",
    "bin/prepare-chromatic-shear-sims",
    "bin/run-chromatic-shear-sims",
]


setup(
    name="chromatic-shear-sims",
    version=0.3,
    description="Studies of the contribution to shear calibration bias from chromatic effects ",
    author="smau",
    packages=find_packages(),
    scripts=scripts,
)
