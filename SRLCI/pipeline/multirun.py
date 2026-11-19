from pipeline import beads, segmentation, detection, tracking, measuring, movies
import argparse
import pandas as pd
import os
import copy
import gc

def read_options():
    """
    Reads command-line arguments for modules.
    
    Returns
    -------
    dict
        Dictionary containing command-line options.
    """ 
    # make parser and add options
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help='Tab-separated file containing list of movies to process and directory names to put results in')
    parser.add_argument("-b", "--beads")
    parser.add_argument("-p", "--psf")
    parser.add_argument("-c", "--config")
    parser.add_argument("-d", "--directory", help='overarching directory in which to place subdirectories for each movie')
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int)
    parser.add_argument("-n", "--nproc", type=int, default=8, help='Number of threads to use for segmentation')
    parser.add_argument("-t", "--trackproc", type=int, default=8, help='Number of threads to use for pipeline steps after segmentation')
    parser.add_argument("-g", "--gpus", type=int, default=1)
    parser.add_argument("-o", "--other")
    parser.add_argument("-F", "--force", action="store_true")
    
    # read and convert to dictionary
    options = vars(parser.parse_args())

    return options

def multirun():
    """
    Run processing pipeline for multiple movies in sequence
    """
    # read command-line options (as dict)
    main_args = read_options()
    
    to_process = pd.read_csv(main_args['file'],sep='\t',header=None,names=['file','directory'])
    for index, row in to_process.iterrows():
        kwargs = copy.deepcopy(main_args)
        kwargs.pop('trackproc')
        kwargs['file'] = row['file']
        outdir = os.path.join(main_args['directory'],row['directory'])
        kwargs['directory']=outdir
        print(f"Now processing {row['file']}")

        # # Run pipeline modules
        segmentation.run(**kwargs)
        kwargs['nproc']=main_args['trackproc'] # up the number of threads for everything post-segmentation?
        tracking.run(**kwargs, mode="nuclei")
        detection.run(**kwargs)
        tracking.run(**kwargs, mode="features")
        beads.run(**kwargs, mode="fitting")
        measuring.run(**kwargs)

        # make outputs
        movies.run(**kwargs, mode="labels")
        # movies.run(**kwargs, mode="channels")
        # movies.run(**kwargs, mode="nuclei")
        movies.run(**kwargs, mode="evaluate")
        gc.collect()
    
    print("Done!")

        
if __name__ == "__main__":
    multirun()
