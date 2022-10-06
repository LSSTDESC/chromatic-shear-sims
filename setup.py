import os
from setuptools import setup, find_packages

scripts = [
    "bin/run-shear-bias-sims",
    "bin/meas-shear-bias-sims",
    "bin/viz-shear-bias-sims",
]

setup(
    name="shear-bias-sims",
    version=0.1,
    description="sims for testing shear estimation bias",
    author="smau",
    packages=find_packages(),
    scripts=scripts,
)
