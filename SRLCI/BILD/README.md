# Running BILD

## Environment
* python 3.9.23
* bild 0.0.5
* noctiluca 0.1.4
* bayesmsd 0.1.7

## Data processing
For each condition, run `ExportQCtracks_clean.py` on the outputs from Fyrtarn to consolidate the raw tracks. (You do not need to do this if working with the tracks from the Zenodo repository for this manuscript). Next, run `load_and_filter_tracks.py` to process and filter the data.

**Unfiltered tracks:** The first part of the script `load_and_filter_tracks.py` takes the raw tracks and consolidates them into one `.npz` file per condition, stored in `data_consolidated_npz/unfiltered_data`.

**Filtered tracks:** The second part of the script `load_and_filter_tracks.py` takes the unfiltered tracks, performs outlier filtering, and saves the results in the same `.npz` format, stored in `data_consolidated_npz/filtered_data`.

## Calibrate BILD
The script `run_calibration.py` performs MSD fitting to calibrate the underlying Rouse model parameters (this differs from other MSD fits in the manuscript, as $\alpha$ is fixed at 0.5 for BILD). We use the ∆CTCF condition as the unlooped state and rescale the parameters of the ∆RAD21 condition to obtain the looped state. This script also plots the raw MSDs.

The script `view_calibration_results.py` calculates the BILD model parameters ($L$, $L_\text{looped}$, $k$, and $D$) from the Rouse model parameters ($\Gamma$, $J$, and localization error for each condition). It also plots the localization error-corrected experimental MSDs overlaid with the MSDs of the model's looped and unloooped states.

## Run BILD
The script `run_bild.py` runs BILD on the trajectory data.

```python run_bild.py --condition 340kb_Ce_Cp_None --nproc 12 --dE 2 --traj_len 100```
```python run_bild.py --condition 340kb_Ce_Cp_IAA --nproc 12 --dE 2 --traj_len 100```