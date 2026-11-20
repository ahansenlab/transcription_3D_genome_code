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

Installation should take a few minutes with a package manager like conda.

## Instructions
`compSim.py` is the main script used to perform polymer simulations. This script is run from the command line. After running, it is recommended to rename the output directory (created within the directory specified by the `outpath` argument) to something descriptive.

Example usage (for a local decompaction model):

```python compSim.py numsave=2000 outpath=./sim_outputs epsB=0.2 epsC=0.2 epsBC=0.2 epsA=0 comppath=/path/to/repo/simulations/setup_files/comps_sticky_except_upstream_and_Fbn2.dat ctcfpath=/path/to/repo/simulations/setup_files/ctcf.dat microcomppath=/path/to/repo/simulations/setup_files/microcomps.dat polpath=/path/to/repo/simulations/setup_files/pols.dat polpermL=1 polpermR=1```

The `analysis.ipynb` notebook is used to perform analyses and create visualizations on the outputs of `compSim.py`.
