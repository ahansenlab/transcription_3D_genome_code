## Overview
This directory contains code to extract 3D distance and MS2 intensity trajectories from live-cell movies, use Bayesian Inference of Looping Dynamics (BILD, Gabriele et al. 2022) to infer looping events from the distance trajectories, and visualize the resulting data.

## Installation and dependencies
To use the code in this directory, create a Python environment with the following packages:
- bayesmsd
- noctiluca
- bild

Then, activate the environment, navigate to the `pipeline` directory, and follow the instructions in the README there to install the Fyrtarn pipeline and remaining dependencies.

## Instructions
Use `multirun.py` in the `pipeline` subdirectory to run the Fyrtarn pipeline on a set of live-cell movies. This script requires a TSV file containing the movies to process and the directories to create for the results. After processing, use the `evaluate.ipynb` notebook to perform manual quality control. Use `ExportQCtracks_clean.py` in the `BILD` subdirectory to extract manual-QC-approved tracks from the Fyrtarn output file structure, then run `load_and_filter_tracks.py` to perform automated filtering of outlier points and jumps.

To run BILD, start by running `run_calibration.py` in the `BILD` subdirectory. This script calibrates the model parameters using the dRAD21 and dCTCF conditions. Run `view_calibration_results.py` to verify that the calibration looks good, then run `run_bild.py` to perform inference on the main conditions.

The `plotting.ipynb` notebook in the main SRLCI directory contains functions to reproduce most data visualizations for these data. BILD-related analyses can be found in `analysis.ipynb` within the `BILD` subdirectory.
