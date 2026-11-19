import os
import warnings
import itertools
import cupy as cp
import numpy as np

from numba import jit
from scipy.ndimage import convolve
from scipy.optimize import minimize
from .utilities import proc, read_h5, where, mkdir, to_intensity, \
     to_fmt, to_shift, mapping
from .detection import detect

warnings.filterwarnings("ignore", message="loadtxt: input contained no data")


@proc
def measuring_proc(nucleus, curve, shape, directory, c, device):
    """
    Process job for measuring features in nuclei.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    curve : dict
        Estimated photobleaching curve.
    shape : numpy.ndarray
        Shape of PSF masks for each channel.
    directory : str
        Path to the directory for input/outputs.
    c : dict
        Dictionary for config.
    device : int
        GPU device to use.
    """
    # set CSV format
    fmt = ["%d"] + ["%.3f"] + ["%d"] + 18*["%.3f"] + 2*["%d"] + 3*["%.3f"]
    header = "frame,time (s),label,ms2_z_sub (px),ms2_y_sub (px),ms2_x_sub (px)," + \
             "ms2_z (nm),ms2_y (nm),ms2_x (nm),right_z_sub (px),right_y_sub (px)," + \
             "right_x_sub (px),right_z (nm),right_y (nm),right_x (nm),left_z_sub (px)," + \
             "left_y_sub (px),left_x_sub (px),left_z (nm),left_y (nm),left_x (nm)," + \
             "sisters,replicated,distance (nm),background (au),intensity (au)"
    
    # store tracks and get pixel size
    tracks = []  
    pixel = c["deskewing"]["pixel"]
    
    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    corrections_path = os.path.join(directory, "corrections")
    tracks_path = mkdir(os.path.join(nuclei_path, to_fmt(nucleus), "tracks"))

    # read aberration weights and bead intensities
    weights = np.loadtxt(os.path.join(corrections_path, "aberrations",
                         "002-001.csv"), delimiter=",", ndmin=2)
    bead = np.loadtxt(os.path.join(corrections_path, "intensities",
                      "000.csv"), delimiter=",")

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # read channel tracks
    chnls, chnl_tracks = [1,2], []
    chnls_path = os.path.join(nucleus_path, "channels")
    for i in range(2):
        chnl = chnls[i]
        chnl_tracks.append([])

        # find in channel directory
        chnl_tracks_path = os.path.join(chnls_path, to_fmt(chnl), "tracks")
        if os.path.exists(chnl_tracks_path):
            for item in os.listdir(chnl_tracks_path):
                if item[:-4].isnumeric():
                    track = np.loadtxt(os.path.join(chnl_tracks_path, item),
                                       delimiter=",", skiprows=1, ndmin=2)
                    chnl_tracks[i].append(track)

    # get number of tracks in each channel
    counts = [len(chnl_tracks[0]), len(chnl_tracks[1])]
    if min(counts) == 0:
        return tracks  # if none in one channel
    scores = np.zeros(counts, dtype=int)
    for ij in itertools.product(range(counts[0]), range(counts[1])):
        # make pair
        i, j = ij
        pair = [chnl_tracks[0][i], chnl_tracks[1][j]]
                
        # get start and end of pair
        start = int(min(pair[0][0,0], pair[1][0,0]))
        end = int(max(pair[0][-1,0], pair[1][-1,0]))
    
        # compute score for pair
        positions = np.empty((2, 3))  
        indices, exists = np.zeros(2, dtype=int), np.zeros(2, dtype=bool)
        for frame in range(start, end+1):
            # update positions at frame (if exist)
            for k in range(2):
                if (indices[k] < len(pair[k])) and (int(pair[k][indices[k],0]) == frame):
                    positions[k] = pair[k][indices[k],2:5]
                    indices[k] += 1
                    exists[k] = True

            # increase if frames match and in radius
            search = c["measuring"]["search"]
            if np.all(exists) and (np.linalg.norm(positions[0] - positions[1]) < search):
                scores[i,j] += 1
            exists[:] = 0

    # make groups of channel track "pairs"
    groups, place = [], [[],[]]  # hold indices
    for i in range(2):
        for j in range(counts[i]):
            group = [[],[]]
            group[i].append(chnl_tracks[i][j])
            groups.append(group)
            place[i].append(j + i*counts[0])

    while True:
        # get best score
        if np.max(scores) < c["measuring"]["overlap"]:
            break  # break on minimum overlap
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        scores[i,j] = 0
        
        # arrange indices for placing channel tracks
        after, before = place[0][i], place[1][j]
        if before == after:
            continue  # already in same group
        if after > before:
            before, after = after, before

        # merge channel tracks group
        group = groups.pop(before)
        for k in range(2):
            # update indices then place in new group
            for l in range(len(place[k])):
                if place[k][l] == before:
                    place[k][l] = after
                elif place[k][l] > before:
                    place[k][l] -= 1
                else:
                    pass
            for l in range(len(group[k])):
                groups[after][k].append(group[k][l])

    # make paired tracks by piecing groups together
    for group in groups:
        # ignore if no pairs
        if len(group[0])*len(group[1]) == 0:
            continue

        # find the start/end in each group
        start, end = None, None
        for i in range(2):
            # sort for later
            group[i].sort(key=len, reverse=True)
            for track in group[i]:
                if start is None:
                    start = int(track[0,0])
                else:
                    if track[0,0] < start:
                        start = int(track[0,0])
                if end is None:
                    end = int(track[-1,0])
                else:
                    if track[-1,0] > end:
                        end = int(track[-1,0])

        # build lines for each track
        info, exists = [], np.zeros(2, dtype=bool)
        for frame in range(start, end+1):
            # get subset lower bound for frame coordinates
            nucleus_idx = where(nucleus_track[:,0] == frame)
            label = int(nucleus_track[nucleus_idx, 1])
            lower = nucleus_track[nucleus_idx, 5:8]
            
            # initialize line
            line = np.empty(26)
            line[0], line[1] = frame, c["general"]["time"]*frame  # time
            line[2] = label
            line[3:] = np.nan
            line[21:23] = 0  # sisters/replicated

            # fill in line
            for i in range(2):
                # use longest tracks last
                for track in group[i]:
                    idx = where(track[:,0] == frame)
                    if (idx is not None) and (not np.isnan(track[idx,5:8]).any()):
                        # mark that localization exists for frame
                        exists[i] = True
                        
                        # add positions and sisters
                        line[9+i*6:12+i*6] = track[idx,5:8]
                        line[12+i*6:15+i*6] = lower + track[idx,5:8]
                        line[21] += track[idx,8]  # sisters
                        break
            
            # right-side localization exists
            if exists[0]:
                line[12:15] *= pixel  # convert

            # promoter localization exists
            if exists[1]:
                # get position (in nm) of promoter in enhancer channel
                chromatic = mapping(line[18:21]) @ weights
                line[18:21] = pixel * (line[18:21] - chromatic)

                # add distance
                if exists[0]:
                    line[23] = np.linalg.norm(line[12:15] - line[18:21])

                # read mask/deskew for nucleus
                lbl = to_fmt(label)
                frame_path = os.path.join(frames_path, to_fmt(frame))
                mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
                dsk = read_h5(os.path.join(frame_path, "deskews.h5"), lbl)
                dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
                
                # estimate MS2 position with camera shift
                with cp.cuda.Device(device):
                    camera = to_shift(dsk[1], dsk[0])
                line[3:6] = line[15:18] - chromatic - camera

                # attempt to detect any MS2 dots
                with cp.cuda.Device(device):
                    # use enhancer/promoter detection method
                    # TODO: maybe tune these parameters
                    positions, _ = detect(dcv[0], dsk[0], mask, shape[0],
                                          radius={"lower": 0.75, "upper": 1.50},
                                          response=250)
                if len(positions) > 0:
                    # compute distances to detected dots
                    distances = np.linalg.norm(positions - line[3:6], axis=1)
                    
                    # try to find nearby
                    idx = np.argmin(distances)
                    if distances[idx] < 5:
                        line[3:6] = positions[idx]
                
                # get position (in nm) and as integer
                line[6:9] = pixel * (lower + line[3:6])
                p = np.rint(line[3:6]).astype(int)  # indices in subset

                # check if in mask then measure
                if (p >= 0).all() and (p < mask.shape).all() and (mask[p[0],p[1],p[2]]):                    
                    intensity, background = to_intensity(dsk[0], line[3:6], shape[0],
                                                         mask=mask, radius=7)
                    # apply intensity corrections
                    const = bead * curve[frame]
                    line[24], line[25] = background / const, intensity / const
                
            # clear and append
            exists[:] = 0
            info.append(line)

        # get track as array and esimate replication start with window
        info = np.asarray(info)
        idx = where(convolve(info[:,21], np.ones(5), mode="nearest") >= 3)
        if idx is not None:
            info[idx:,22] = 1
        
        # save as CSV
        np.savetxt(os.path.join(tracks_path, to_fmt(len(tracks)) + ".csv"), info,
                   fmt=fmt, header=header, delimiter=",", comments="")
        tracks.append(info)  # add to tracks
    
    return tracks


@proc
def bleaching_proc(nucleus, directory):
    """
    Process job to measure photobleaching within a nucleus.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    directory : str
        Path for pipeline outputs.
    """
    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)      
    # iterate through track
    intensities = np.empty(len(nucleus_track))
    for i in range(len(nucleus_track)):
        # get frame/label and path
        frame, label = int(nucleus_track[i,0]), int(nucleus_track[i,1])
        frame_path = os.path.join(frames_path, to_fmt(frame))
        
        # read mask/deskew for nucleus and get mean
        lbl = to_fmt(label)
        mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
        dsk = read_h5(os.path.join(frame_path, "deskews.h5"), lbl)
        intensities[i] = np.mean(dsk[1][mask])

    return intensities


def to_curve(curves, frames):
    """
    Fits a double exponential curve
    to photobleaching data.

    Parameters
    ----------
    curves : list
        List of nuclei photobleaching curves
    frames : numpy.ndarray
        Time steps as indicies.
    
    Returns
    -------
    numpy.ndarray
        The fitted curve.
    numpy.ndarray
        The nuclei ensemble estimate.
    """
    # TODO: get correction from Jack that ensures fit sums to 1 at first frame
    # make ensemble curve
    m = len(frames)
    ensemble, samples = np.zeros(m), np.zeros(m)
    for curve in curves:
        # filter
        first, last = np.median(curve[:5]), np.median(curve[-5])
        if np.all(curve != 0) and (first > last):
            a = 0.9 < first/np.max(curve) < 1.1
            b = 0.9 < last/np.min(curve) < 1.1
            if a and b:
                curve /= curve[0]  # normalize
                for i in range(len(curve)):
                    ensemble[i] += curve[i]
                    samples[i] += 1
    
    # make mask based on samples and fit
    mask = samples >= 5
    ensemble[mask] /= samples[mask]
    ensemble[~mask] = np.nan
    x, y = frames[mask], ensemble[mask]
    
    # loss function
    loss = lambda u, x: np.sum((model(np.exp(u), x) - y)**2)
   
    # fit in log-space
    u_init = np.array([0.5, 0.001, 0.5, 0.0001, 0.01])
    u_opt = np.exp(minimize(loss, np.log(u_init), args=(x,),
                   method="BFGS", options={"maxiter": 10000}).x)

    return model(u_opt, frames), ensemble


@jit(nopython=True)
def model(u, x):
    """
    Double exponential model used for 
    photobleaching curves.

    Parameters
    ----------
    u : numpy.ndarray
        Parameters of the fit.
    x : numpy.ndarray
        Time steps as indicies.
        
    Returns
    -------
    float
    """
    a, b, c, d, e = u
    return a*np.exp(-b*x) + c*np.exp(-d*x) + e
