## Overview
This directory contains code to analyze and visualize genomics data.
## Dependencies
To use the code in this directory, create a Python environment with the following packages:
- cooler
- cooltools
- CoolBoxMod (see https://github.com/varshiniramanathan/CoolBoxMod)
- macs3
- bwa-mem2
- bowtie2
- sambamba
- astropy
- pandas
- scipy
- numpy
- matplotlib
- h5py
- pygenometracks
- statannotations
- bioframe
- xlrd
- samtools
You will also need to download the wigToBigWig utility from UCSC (get it here: https://hgdownload.gi.ucsc.edu/admin/exe/)
## Instructions
To create .cool and .mcool files from raw RCMC reads, run microcprocessing.py.

To create bigWig files from ChIP-seq data, do the following:
- run `spikeinChIP_PE_alignment.py` on IP and input reads
- compute downsampling factors to normalize read counts based on spike-in and make a .tsv file with the results
- run `bam_downsample.py`
- run `makebigwig_macs3.py`
- if you want to make merged files (e.g. combining multiple replicates of the same condition), use `bammerger.py` to combine the BAM files, then run `makebigwig_macs3.py`

The notebook `plotting.ipynb` contains functions and sample code to make the types of plots shown in the manuscript.
