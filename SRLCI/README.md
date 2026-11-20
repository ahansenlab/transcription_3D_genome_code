## Overview
This directory contains code to extract 3D distance and MS2 intensity trajectories from live-cell movies, use Bayesian Inference of Looping Dynamics (BILD, Gabriele et al. 2022) to infer looping events from the distance trajectories, and visualize the resulting data.

## Installation and dependencies
To use the code in this directory, create a Python environment with the following packages:
- bayesmsd 0.1.7
- noctiluca 0.1.4
- bild 0.0.5

Then, activate the environment, navigate to the `pipeline` directory, and follow the instructions in the README there to install the Fyrtarn pipeline and remaining dependencies. Installation should take a few minutes.

## Instructions
Use `multirun.py` in the `pipeline` subdirectory to run the Fyrtarn pipeline on a set of live-cell movies. This script requires a TSV file listing the movies to process and the directories to create for the results. Another TSV file with the locations of bead movies to recompute PSFs can be provided, or the PSFs provided in this repo can be used. 

Example usage:

```python multirun.py -f /path/to/movie/file.tsv -p /path/to/PSF.h5 -c /path/to/config.yaml -d ./pipeline_outputs/ -s 1 -n 8 -t 16 -g 1 -F```

After processing, use the `evaluate.ipynb` notebook to perform manual quality control. Use `ExportQCtracks_clean.py` in the `BILD` subdirectory to extract manual-QC-approved tracks from the Fyrtarn output file structure, then run `load_and_filter_tracks.py` to perform automated filtering of outlier points and jumps.

Example usage:

```python ExportQCtracks_clean.py```

```python load_and_filter_tracks.py```

To run BILD, start by running `run_calibration.py` in the `BILD` subdirectory. This script calibrates the model parameters using the dRAD21 and dCTCF conditions. Run `view_calibration_results.py` to verify that the calibration looks good, then run `run_bild.py` to perform inference on the main conditions. These scripts do not require command-line arguments and should be updated with correct paths to relevant files before running.

Example usage:

```python run_calibration.py```

```python view_calibration_results.py```

```python run_bild.py```


The `plotting.ipynb` notebook in the main SRLCI directory contains functions to reproduce most data visualizations for these data. BILD-related analyses can be found in `analysis.ipynb` within the `BILD` subdirectory.
