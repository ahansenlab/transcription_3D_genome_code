## Overview
This directory contains code to run 1D-3D coupled polymer simulations, analyze them, and visualize the data.

## Dependencies
To run the code in this directory, create a Python environment with the following packages:
- polychrom (see https://polychrom.readthedocs.io/en/latest/ for installation instructions)
- cython
- bayesmsd
- noctiluca
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- cooltools
- cooler

## Instructions
`compSim.py` is the main script used to perform polymer simulations. This script is run from the command line. After running, it is recommended to rename the output directory to something descriptive.

The `analysis.ipynb` notebook is used to perform analyses and create visualizations on the outputs of `compSim.py`.
