## Overview
This directory contains code to analyze and visualize genomics data.
## Dependencies
To use the code in this directory, create a Python environment with the following packages:
- cooler 0.10.4
- cooltools 0.7.1
- CoolBoxMod (see https://github.com/varshiniramanathan/CoolBoxMod)
- macs3 3.0.4
- bwa-mem2 2.2.1
- bowtie2 2.5.4
- sambamba 1.0.1
- astropy 6.1.2
- pandas 1.5.3
- scipy 1.11.4
- numpy 1.23.5
- matplotlib 3.6.2
- h5py 3.10.0
- pygenometracks 3.8
- statannotations 0.7.2
- bioframe 0.5.1
- xlrd 2.0.2
- samtools 1.19
Installation should take a few minutes with a package manager like conda. You will also need to download the wigToBigWig utility from UCSC (get it here: https://hgdownload.gi.ucsc.edu/admin/exe/).
## Instructions
To create .cool and .mcool files from raw RCMC reads, run microcprocessing.py. This script can take many hours to run on large datasets.

Example usage:

```python microcprocessing.py -1 /path/to/read1.fastq.gz -2 /path/to/read2.fastq.gz -g mm39 -t 16 -r 10000000 5000000 2500000 1000000 500000 250000 100000 50000 25000 10000 5000 2000 1000 500 200 100 -o output_name -a bwa -w expand -c chr18:58032072-59034072```

To create bigWig files from ChIP-seq data, do the following:
- run `spikeinChIP_PE_alignment.py` on IP and input reads. This script can take many hours to run on large datasets. For spike-in ChIP-seq, this script requires a concatenated genome build of mm39 and the spike-in genome (dm6 or hg38 for the datasets in this study).
- compute downsampling factors to normalize read counts based on spike-in and make a .tsv file with the results
- run `bam_downsample.py`
- run `makebigwig_macs3.py`
- if you want to make merged files (e.g. combining multiple replicates of the same condition), use `bammerger.py` to combine the BAM files, then run `makebigwig_macs3.py`

Example usage:

```python spikeinChIP_PE_alignment.py -f /path/to/list/of/fastqs.tsv -g mm39 -s hg38 -t 16```

```python bam_downsample.py -f /path/to/downsample/factors.tsv -t 16 -o downsample_read_counts```

```python bammerger.py -f /path/to/list/of/files/to/merge.tsv -t 16 -n list of names for merged files -o merged_counts```

```python makebigwig_macs3.py /path/to/files/to/process.tsv paired mm39 -t 16```

The notebook `plotting.ipynb` contains functions and sample code to make the types of plots shown in the manuscript.
