import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
import subprocess
import itertools
import matplotlib.pyplot as plt
import logomaker
import sys
import pickle
import argparse

def motif_to_fasta(bf, kernel, outFasta):
    tmp_df = df[df["kernel"] == kernel].reset_index()
    with open(outFasta, "w") as f:
        for i, row in tmp_df.iterrows():
            chrom, start, end, seq = row.chrom, row.start, row.end, row.seq
            print(">{}:{}-{}".format(chrom, start, end), file=f)
            print(seq, file=f)

def parse_streme_out(streme_out):
    with open(streme_out) as f:
        lines = [line.strip().split() for line in f.readlines()]
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if line[0].replace(".","").isdigit()]
    ppm = np.array(lines, dtype=float)
    return(ppm)

def get_information_content(x):
    ic = x * np.log2((x + .001) / .25)
    if ic > 0:
        return(ic)
    else:
        return(0.0)
    
parser = argparse.ArgumentParser(description='PWM-G2A-ZMotif')
parser.add_argument('-i', '--motifs_file', 
                    help='Motifs file to refine', type=str, required=True)

parser.add_argument('-o', '--output', 
                    help='Output motifs (in pickle format)', type=str, required=True)

parser.add_argument('-prefix', '--prefix',
                    help='Prefix for temporary files', type=str, required=True)

parser.add_argument('-n_cpus', '--n_cpus',
                    help='Number of parallel processes', type=int, required=False, default=4)

parser.add_argument('-logos', '--logos',
                    help='File to plot logos to', type=str, required=False, default=None)


args = parser.parse_args()
motifs_file = args.motifs_file
prefix = args.prefix
tmpDir = "/tmp/" + prefix
outPickle = args.output
threads = args.n_cpus
logos = args.logos
    

WIDTHS = [None,8,10,12,14,16,18,24,30]
df = pd.read_csv(motifs_file, 
                 sep="\t", 
                 header=0)



AUCs = dict(zip(df.kernel, df.auc))

motifs_to_refine = [max(AUCs, key=AUCs.get)]
og_ppm = logomaker.alignment_to_matrix(df[df["kernel"] == motifs_to_refine[0]].seq, to_type="counts")
og_ppm = og_ppm[["A", "C", "G", "T"]]
og_ppm = og_ppm.div(og_ppm.sum(axis=1), axis=0)

if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

baseDir = os.getcwd()
os.chdir(tmpDir)
print(os.getcwd())
for k in motifs_to_refine:
    motif_to_fasta(df, k, "{}.tmp.fa".format(k))
    
streme_cmds = []
for k in motifs_to_refine:
    for w in WIDTHS:
        if w is None:
            cmd = "streme --nmotifs 1 --order 2 --text -p {0}.tmp.fa > {0}-{1}.streme.out 2> {0}-{1}.streme.err".format(k, "None")
        else:
            cmd = "streme --nmotifs 1 --order 2 --text -w {1} -p {0}.tmp.fa > {0}-{1}.streme.out 2> {0}-{1}.streme.err".format(k, w)
        streme_cmds.append(cmd)
        
print("Running STREME")
def run(cmd):
    subprocess.run(cmd, shell=True)
    
with Pool(threads) as p:
    p.map(run, streme_cmds)

PPMs = {}
for k, w in itertools.product(motifs_to_refine, WIDTHS):
    streme_out = "{}-{}.streme.out".format(k,w)
    ppm = parse_streme_out(streme_out)
    PPMs[str(k) + "-" + str(w)] = ppm
    

print("Deleting temporary files")
os.chdir(baseDir)
run("rm -r {}".format(tmpDir))

if logos is not None:
    fig, axes = plt.subplots(4,4, figsize=(20,8), tight_layout=True)
    
    #plot original ppm
    ax = axes.flatten()[0]
    w = og_ppm.shape[0]
    logomaker.Logo(og_ppm.applymap(get_information_content), ax=ax)
    ax.set_ylim([0,2])
    ax.set_xlim([0-0.5, w-0.5])
    ax.set_xticks([0, w-1])
    ax.set_title("CNN learned motif")
    for i, (width, ppm) in enumerate(zip(WIDTHS, list(PPMs.values()))):
        ax = axes.flatten()[i+1]
        ppm = pd.DataFrame(ppm, columns=["A", "C", "G", "T"])
        w = ppm.shape[0]
        logomaker.Logo(ppm.applymap(get_information_content), ax=ax)
        ax.set_ylim([0,2])
        ax.set_xlim([0-0.5, w-0.5])
        ax.set_xticks([0, w-1])
        if width is not None:
            ax.set_title("Width = {}".format(width))
        else:
            ax.set_title("Default parameters")
            
    for i in range(len(PPMs)+1, 16):
        fig.delaxes(axes.flatten()[i])
    plt.savefig(logos)

print("Saving results")
with open(outPickle, "wb") as f:
    pickle.dump(PPMs, f)
    
