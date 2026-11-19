#For merging aligned bam files from sequencing experiments - can include duplicate removal (for bams from the same library) or exclude (for merging eg replicates)

from sys import exit
import subprocess as sp
import argparse
import multiprocessing
import uuid
import pandas as pd

parser = argparse.ArgumentParser(description = "merge lists of bams to make merged files, optionally remove duplicates, and count merged reads")
parser.add_argument("--filename", "-f", help = "a tab-separated file with paths to files to merge together on each line of the file")
parser.add_argument("--threads", "-t", help = "number of threads to use - default is 1", default = "1")
parser.add_argument("--names", "-n", help = "list of names for merged bams - required and must be same length as the number of lines in the supplied tsv", nargs = "*")
parser.add_argument("--outname", "-o", help = "name for the table to store counts of merged files")
parser.add_argument("--outdir", help = "a directory to store output files - default is current directory", default = "./")
parser.add_argument("--rmdup", "-d", help = "flag for whether or not to remove PCR duplicates after merging - set to remove if merging e.g. samples from different sequencing runs on the same library - default: not set", action = "store_true")
args = parser.parse_args()

#####################################################################################
#Prep inputs
filename = args.filename
threads = args.threads
outname = args.outname
outdir = args.outdir
names = args.names

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
    print(f"Merging with {threads} threads...")

#Process ID (used to make unique sorttemp, so these are not overlapping for multiple processes in the same outdir)
uniqueid = str(uuid.uuid4())

################################################################################
#Run the merging

#Read in file to process
bams_to_process_file = open(filename, "r")
#Split by lines
bamfiles_to_process = bams_to_process_file.read().splitlines()
#Check that there are enough filenames for the number of merged files
if len(bamfiles_to_process) != len(names):
    print("Mismatch between number of filenames and number of merged files to produce")
    exit()
#Split each line up by tab characters
bamfiles_to_process_split = [bams.split("\t") for bams in bamfiles_to_process]

#Make empty list for counts
allcountslist = []

for i, bams in enumerate(bamfiles_to_process_split):
    bamstring = " ".join(bams)
    name = names[i]
    #Merge
    sp.run(f"sambamba merge -t {threads} {outdir}{name}_MERGED.bam {bamstring}", shell=True, executable="/bin/bash")
    if args.rmdup:
        print(f"Removing duplicates for {name}")
        #Remove duplicates
        sp.run(f"sambamba markdup --tmpdir {outdir}{uniqueid}/ -r -t {threads} {outdir}{name}_MERGED.bam {outdir}{name}_MERGED_rmdup.bam", shell=True, executable="/bin/bash")
        #Index merged, deduped bam
        sp.run(f"sambamba index -t {threads} {outdir}{name}_MERGED_rmdup.bam", shell=True, executable="/bin/bash")
        #Count reads
        mergedCount = sp.run(f"sambamba view -c -t {threads} {outdir}{name}_MERGED_rmdup.bam", capture_output=True, shell=True, executable="/bin/bash")
        #And remove non deduped file:
        sp.run(f"rm {outdir}{name}_MERGED.bam", shell=True, executable="/bin/bash")
    else:
        #Index merged bam
        sp.run(f"sambamba index -t {threads} {outdir}{name}_MERGED.bam", shell=True, executable="/bin/bash")
        #Count reads
        mergedCount = sp.run(f"sambamba view -c -t {threads} {outdir}{name}_MERGED.bam", capture_output=True, shell=True, executable="/bin/bash")
    #Finally, make a list of name and counts
    countsList = [name, mergedCount.stdout.decode('ascii').strip()]
    allcountslist.append(countsList)

#Remove the temp directory for duplicate removal
if args.rmdup:
    sp.run(f"rm {outdir}{uniqueid}/ -r", shell=True, executable="/bin/bash")
countstable = pd.DataFrame(allcountslist, columns = ['name', 'mergedcounts'])
countstable.to_csv(outdir + outname + '.tsv', sep = '\t', index = False, header = True)
