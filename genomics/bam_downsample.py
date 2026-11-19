#For downsampling bams for spikein using a list of input files and downsampling factors

from sys import exit
import subprocess as sp
import argparse
import multiprocessing
import pandas as pd

parser = argparse.ArgumentParser(description = "downsample bams based on file with filenames and downsample factors - NB this uses a fixed sampling seed")
parser.add_argument("--filename", "-f", help = "a tab-separated file two columns: path to file to downsample, and downsampling factor")
parser.add_argument("--threads", "-t", help = "number of threads to use - default is 1", default = "1")
parser.add_argument("--outname", "-o", help = "name for the table to store counts of merged files")
parser.add_argument("--outdir", help = "a directory to store output files - default is current directory", default = "./")
args = parser.parse_args()

#####################################################################################
#Prep inputs
filename = args.filename
threads = args.threads
outname = args.outname
outdir = args.outdir

#Check requirements are fulfilled:
condapacks = sp.run("conda list".split(), capture_output=True)
condapacksstr = str(condapacks.stdout)

if "sambamba" not in condapacksstr:
    print("Please make sure sambamba is installed in your current conda environment (check conda list)")
    exit()

#Check that outdir ends with a /, add one if it doesn't
if args.outdir is not None and not outdir.endswith("/"):
    outdir = outdir + "/"

#Check that a sensible number of threads has been requested - more protections here are possible - at the moment users are trusted to be sensible
cpucount = multiprocessing.cpu_count()
if args.threads is None:
    print("Defaulting to one thread")
    threads = 1
elif int(args.threads) >= cpucount:
    print("Too many threads requested, resetting to default")
    threads = 1
else:
    print(f"Downsampling with {threads} threads...")

################################################################################
#Run the downsample

#Get the files to downsample as lists
files = pd.read_csv(args.filename, sep='\t', header=None, names = ['file', 'dsfactor'])
filepaths = files.file.values.tolist()
dsfactors = files.dsfactor.values.tolist()

#Also generate filenames for the output files - remove path and extension
# names = [bams.rsplit("/", 1)[1].rsplit(".", 1)[0] for bams in filepaths]
names = [None] * len(filepaths)
for count, bams in enumerate(filepaths):
    if '/' in bams:
        # path = bams.rsplit("/",1)[0]
        name = bams.rsplit("/",1)[1].rsplit(".",1)[0]
    else:
        # path = "./"
        name = bams.rsplit(".",1)[0]
    names[count] = name

#Make empty list for counts
allcountslist = []

#Loop through each one and downsample
for i, bam in enumerate(filepaths):
    dsfactor = dsfactors[i]
    name = names[i]
    #Downsample
    sp.run(f"sambamba view -h -t {threads} -f bam --subsampling-seed=123 -s {dsfactor} {bam} -o {outdir}{name}_downsampled.bam", shell=True, executable="/bin/bash")
    #Index
    sp.run(f"sambamba index -t {threads} {outdir}{name}_downsampled.bam", shell=True, executable="/bin/bash")
    #Count how many reads:
    dsCount = sp.run(f"sambamba view -c -t {threads} {outdir}{name}_downsampled.bam", capture_output=True, shell=True, executable="/bin/bash")
    countsList = [name, dsCount.stdout.decode('ascii').strip()]
    allcountslist.append(countsList)

countstable = pd.DataFrame(allcountslist, columns = ['name', 'allcounts'])
countstable.to_csv(outdir + outname + '.tsv', sep = '\t', index = False, header = True)
