import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import set_start_method, Pool
from pipeline.methods.measuring import to_curve, bleaching_proc, measuring_proc
from pipeline.methods import read_options, to_directory, configure_plots, \
     bar_format, read_yaml, read_kernel, read_shape, save_config, mkdir, where

set_start_method("spawn", force=True)  # helps with GPUs
configure_plots()  # matplotlib config


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the measuring module to measure features
    within nuclei.

    Parameters
    ----------
    file : str, optional
        Path to CZI file.
    beads : str, optional
        Path to TXT file of bead files.
    psf : str, optional
        Path to H5 file of PSFs.
    directory : str, optional
        Path for pipeline outputs.
    config : str, optional
        Path to YAML file.
    start : int, optional
        Starting frame (python indexed).
    end : int, optional
        Ending frame (python indexed).
    nproc : int, optional
        Number of processes used.
    gpus : int, optional
        Number of GPUs to used.
    mode : str, optional
        Module specific mode.
    other : any, optional
        For pipeline testing.
    force: bool, optional
        If forcing overwrite.
    """
    # argument handling
    if file is None:
        print("File is required for this module!") 
        exit()
    if psf is None:
        print("PSF(s) are required for this module!") 
        exit()
    if directory is None:
        directory = to_directory(file)

    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    corrections_path = os.path.join(directory, "corrections")

    # handle for missing frames/nuclei
    if (not os.path.exists(frames_path)) or (not os.path.exists(nuclei_path)):
        print("Nuclei must be segmented and tracked before this module!")
        exit()

    # get nuclei to process
    nuclei, overwrite = [], force
    for item in os.listdir(nuclei_path):
        if item.isnumeric():
            # get path to nucleus
            nucleus_path = os.path.join(nuclei_path, item)
            nuclei.append(int(item))

            # get path and handle existing
            tracks_path = os.path.join(nucleus_path, "tracks")
            if not overwrite and os.path.exists(tracks_path):
                ans = input("Features already measured! Do you want to overwrite? (y/n): ")
                if ans == "y":
                    overwrite = True
                else:
                    exit()

    # handle config
    if config is None:
        c = {"measuring": {}}
    else:
        c = read_yaml(config)
        if "measuring" not in c:
            c["measuring"] = {}
        save_config(config, directory)
    n = len(nuclei)  # number of nuclei
        
    # -------------- estimate MS2 photobleaching --------------
    
    # make jobs and use process pool
    args = list(zip(nuclei, n*[directory]))
    with Pool(min(n,nproc), maxtasksperchild=5) as pool:
        curves = pool.map(bleaching_proc, args)
    
    # get frames and make path
    frames = []
    for item in os.listdir(frames_path):
        if item.isnumeric():
            frames.append(int(item))
    frames = np.asarray(sorted(frames))
    bleaching_path = mkdir(os.path.join(corrections_path, "photobleaching"))
    
    # get estimated curve and save as PNG
    curve, ensemble = to_curve(curves, frames)
    plt.figure(figsize=(5,3))
    plt.plot(frames, ensemble, color=(0,0.8,0.8), linewidth=2)
    plt.plot(frames, curve, color="k", linewidth=1)
    plt.xlabel("Frames")
    plt.ylabel("Norm. Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(bleaching_path, "001.png"))
    plt.close()

    # save as CSV
    fmt, header = ["%d"] + ["%.6f"], "frame,intensity (au)" 
    info = np.stack((frames, curve), axis=1)
    np.savetxt(os.path.join(bleaching_path, "001.csv"), info,
               delimiter=",", header=header, fmt=fmt, comments="")

    curve = dict(zip(frames, curve))  # make as lookup

    # -------------- make E-P tracks --------------

    # get path to chromatic aberrations and hold E-P offsets
    aberrations_path = os.path.join(corrections_path, "aberrations")
    offsets = np.empty((0,3))

    # read PSF mask shapes(s)
    shape = None if psf is None else read_shape(psf, c["general"]["lasers"])
    
    # make jobs and use process pool
    devices = np.arange(n) % gpus
    args = list(zip(nuclei, n*[curve], n*[shape], n*[directory], n*[c], devices))
    with Pool(min(n,nproc), maxtasksperchild=5) as pool:
        for tracks in tqdm(pool.imap_unordered(measuring_proc, args), desc="Measuring",
                      bar_format=bar_format("nuclei"), total=n):
            # iterate through tracks in nucleus
            for track in tracks:
                # remove frames marked replicated
                cut = where(track[:,16])
                if cut is not None:
                    track = track[cut:]

                # append to ensemble offsets
                track = track[~np.isnan(track[:,17])]
                offsets = np.concatenate((offsets, track[:,12:15] - track[:,6:9]), axis=0)
    
    # plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(12,3))
    labels, medians = ["Z", "Y", "X"], np.median(offsets, axis=0)
    for i in range(3):
        axes[i].hist(offsets[:,i], color="darkblue", linewidth=2, histtype="step")
        axes[i].set_title(f"{labels[i]} median: {medians[i]:.2f} nm", fontsize=15)
        axes[i].set_xlabel("E-P offsets (nm)")

    # save as PNG
    plt.tight_layout()
    plt.savefig(os.path.join(aberrations_path, "002-000.png"))
    plt.close()


if __name__ == "__main__":
    # read args from command-line and run
    kwargs = read_options()
    run(**kwargs)
