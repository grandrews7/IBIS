import os
import itertools
import glob

if not os.path.exists("Results"):
    os.mkdir("Results")
    
# G2A_TFs = os.listdir("./data/CHS")
G2A_TFs = ["GABPA"]
G2A_assays = ["CHS", "GHTS"]
peaks = ["shared", "all"]
n_motifs = [1, 16]
reps = [i for i in range(20)]

# For debugging and testing
# G2A_TFs = ["GABPA"]
# G2A_assays = ["CHS"]
# peaks = ["shared"]
# n_motifs = [1]
# reps = [i for i in range(1)]


# HTS_TFs = os.listdir("/home/gregory.andrews-umw/IBIS/data/HTS")
# HTS_fastqs = glob.glob("/home/gregory.andrews-umw/IBIS/data/HTS/*/*fastq.gz")
# HTS_TFs_cycles = []
# for fastq in HTS_fastqs:
#     split = os.path.basename(fastq).split("_")
#     HTS_TFs_cycles.append(split[0] + "-" + split[2])
# HTS_TFs_cycles = list(set(HTS_TFs_cycles))
#HTS_TFs_cycles = HTS_TFs_cycles[:1]

rule all:
    input: 
        expand("Results/G2A/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps),
        #expand("Results/Genomic/{tf}-{assay}-{peaks}-{n}-{l}-{rep}/streme.pkl", tf=Genomic_TFs, assay=Genomic_assays, peaks=peaks, n=n_motifs, l=lengths, rep=reps),
        #expand("Results/HTS/{tf_cycle}-{rep}/motifs.txt.gz", tf_cycle = HTS_TFs_cycles, rep=reps)
        # expand("/home/gregory.andrews-umw/IBIS/Results/AAA-Arrays/{partition}-{assay}.npy", partition = ["Leaderboard", "Final"], assay=["CHS", "GHTS", "HTS", "PBM", "SMS"])
        #expand("/home/gregory.andrews-umw/IBIS/Results/AAA-Arrays/{partition}-{assay}.npy", partition = ["Final"], assay=["HTS"])

rule PWM_G2A_ZMotif:
    output: "Results/G2A/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        "logs/PWM_G2A_ZMotif/{tf}-{assay}-{peaks}-{n}-{rep}.log"
    shell:
        """
        python3 scripts/PWM-G2A-ZMotif.py \
        -tf {wildcards.tf} -assay {wildcards.assay} -peaks {wildcards.peaks} -n {wildcards.n}  \
        -o Results/G2A/{wildcards.tf}-{wildcards.assay}-{wildcards.peaks}-{wildcards.n}-{wildcards.rep} -g ./data/hg38.fa -d ./data
        rm Results/G2A/{wildcards.tf}-{wildcards.assay}-{wildcards.peaks}-{wildcards.n}-{wildcards.rep}/motifs.bed
        """
        
# rule STREME:
#     input: "Results/Genomic/{tf}-{assay}-{peaks}-{n}-{l}-{rep}/motifs.txt.gz"
#     output: "Results/Genomic/{tf}-{assay}-{peaks}-{n}-{l}-{rep}/streme.pkl"
#     threads: 8
#     singularity:
#         "docker://andrewsg/ibis:1.15"
#     log:
#         "logs/STREME/{tf}-{assay}-{peaks}-{n}-{l}-{rep}.log"
#     shell:
#         """
#         python3 scripts/STREME.py {input} {wildcards.tf}-{wildcards.assay}-{wildcards.peaks}-{wildcards.n}-{wildcards.l}-{wildcards.rep} {output} {threads}
#         """
        
# rule ZMotif_HTS:
#     output: "Results/HTS/{tf}-{cycle}-{rep}/motifs.txt.gz"
#     threads: 4
#     singularity:
#         "docker://andrewsg/ibis:1.15"
#     log:
#         "logs/ZMotif-HTS/{tf}-{cycle}-{rep}.log"
#     shell:
#         """
#         python3 scripts/HTS.py {wildcards.tf} {wildcards.cycle} Results/HTS/{wildcards.tf}-{wildcards.cycle}-{wildcards.rep}
#         rm Results/HTS/{wildcards.tf}-{wildcards.cycle}-{wildcards.rep}/motifs.bed
#         """
        
# rule Get_Array:
#     output: "/home/gregory.andrews-umw/IBIS/Results/AAA-Arrays/{partition}-{assay}.npy"
#     threads: 10
#     singularity:
#         "docker://andrewsg/ibis:1.15"
#     log:
#         "logs/Get-Array/{partition}-{assay}.log"
#     shell:
#         """
#         python3 scripts/Get-Array.py {wildcards.partition} {wildcards.assay}
#         """