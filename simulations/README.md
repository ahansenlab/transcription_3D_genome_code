## Overview
This directory contains code to run 1D-3D coupled polymer simulations, analyze them, and visualize the data.

## Dependencies
To run the code in this directory, create a Python environment with the following packages:
- polychrom 0.1.1 (see https://polychrom.readthedocs.io/en/latest/ for installation instructions)
- cython 3.2.4
- bayesmsd 0.1.7
- noctiluca 0.1.4
- numpy 1.24.4
- matplotlib 3.10.9
- seaborn 0.13.2
- pandas 3.0.1
- scipy 1.15.2
- cooltools 0.7.1
- cooler 0.10.2

## Instructions
`compSim.py` is the main script used to perform polymer simulations. This script is run from the command line. After running, it is recommended to rename the output directory to something descriptive.

The `analysis.ipynb` notebook is used to perform analyses and create visualizations on the outputs of `compSim.py`.
