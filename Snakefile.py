import os
import itertools
import glob
import sys

TFs_of_interest = os.listdir("data/CHS") + os.listdir("data/HTS")
nReps = 20
# Please place the TFs you'd like to run here!
# Can be any combination of genomic / artifical and leaderboard / final TFs 
TFs_of_interest = ["NFKB1", "LEF1", "TIGD3", "SP140L", "USF3", "CAMTA1"]
#nReps = 1

print("TFs of interest:", *TFs_of_interest)
reps = [i for i in range(nReps)]

if not os.path.exists("Results"):
    os.mkdir("Results")
    
data_dir = "./data/"
    
G2A_TFs = [_ for _ in os.listdir("./data/CHS") if _ in TFs_of_interest]
print("Genomic TFs:", *G2A_TFs)

G2A_assays = ["CHS", "GHTS"]
peaks = ["shared", "all"]
n_motifs = [1, 16]

HTS_TFs = [_ for _ in os.listdir("./data/HTS") if _ in TFs_of_interest]
print("HTS TFs:", *HTS_TFs)
HTS_fastqs = []
for _ in HTS_TFs:
    HTS_fastqs += glob.glob("./data/HTS/{}/*fastq.gz".format(_))
    
HTS_TFs_cycles = []
for fastq in HTS_fastqs:
    split = os.path.basename(fastq).split("_")
    HTS_TFs_cycles.append(split[0] + "-" + split[2])
HTS_TFs_cycles = list(set(HTS_TFs_cycles))

rule all:
    input: 
        expand("Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps),
        expand("Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/streme.pkl", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps),
        expand("Results/PWM-G2A-CV/{tf}-{train_assay}-{test_assay}.pkl", tf=G2A_TFs, train_assay=G2A_assays, test_assay=G2A_assays),
        "Results/PWM-G2A-Final/PWM-G2A-Leaderboard.txt", 
        "Results/PWM-G2A-Final/PWM-G2A-Final.txt",
        expand("Results/PWM-A2G-ZMotif/{tf_cycle}-{rep}/motifs.txt.gz", tf_cycle = HTS_TFs_cycles, rep=reps),
        "Results/PWM-A2G-Final/PWM-A2G-Leaderboard.txt", 
        "Results/PWM-A2G-Final/PWM-A2G-Final.txt",
        
rule PWM_G2A_ZMotif:
    output: "Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_G2A_ZMotif/{tf}-{assay}-{peaks}-{n}-{rep}.out",
        err = "logs/PWM_G2A_ZMotif/{tf}-{assay}-{peaks}-{n}-{rep}.err"
    shell:
        """
        python3 scripts/PWM-G2A-ZMotif.py -tf {wildcards.tf} -assay {wildcards.assay} -peaks {wildcards.peaks} -n {wildcards.n} -o Results/PWM-G2A-ZMotif-STREME/{wildcards.tf}-{wildcards.assay}-{wildcards.peaks}-{wildcards.n}-{wildcards.rep} -g ./data/hg38.fa -d ./data 2> {log.err} 1> {log.out}
        """
        
rule PWM_G2A_STREME:
    input: "Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz"
    output: 
        "Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/streme.pkl", 
        "Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/streme-logos.png"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_G2A_STREME/{tf}-{assay}-{peaks}-{n}-{rep}.out",
        err = "logs/PWM_G2A_STREME/{tf}-{assay}-{peaks}-{n}-{rep}.err"
    shell:
        """
        python3 scripts/PWM-G2A-STREME.py -i {input} -o {output[0]} -prefix {wildcards.tf}-{wildcards.assay}-{wildcards.peaks}-{wildcards.n}-{wildcards.rep}  -logos {output[1]} 2> {log.err} 1> {log.out}
        """
        
rule PWM_G2A_CV:
    input: expand("Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/streme.pkl", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps)
    output: 
        pickle = "Results/PWM-G2A-CV/{tf}-{train_assay}-{test_assay}.pkl",
        plot = "Results/PWM-G2A-CV/{tf}-{train_assay}-{test_assay}.png"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_G2A_CV/{tf}-{train_assay}-{test_assay}.out",
        err = "logs/PWM_G2A_CV{tf}-{train_assay}-{test_assay}.err"
    shell:
        """
        python3 scripts/PWM-G2A-CV.py -TF {wildcards.tf} -train {wildcards.train_assay} -test {wildcards.test_assay} -n_cpus {threads} --data_dir data --results_dir Results/PWM-G2A-ZMotif-STREME/ --output_pickle {output.pickle} --output_plot {output.plot}
        """
        
rule PWM_G2A_Plot_Save:
    input: expand("Results/PWM-G2A-CV/{tf}-{train_assay}-{test_assay}.pkl", tf=G2A_TFs, train_assay=G2A_assays, test_assay=G2A_assays)
    output: 
        l_motifs = "Results/PWM-G2A-Final/PWM-G2A-Leaderboard.txt",
        l_logos = "Results/PWM-G2A-Final/PWM-G2A-Leaderboard.png",
        f_motifs = "Results/PWM-G2A-Final/PWM-G2A-Final.txt",
        f_logos = "Results/PWM-G2A-Final/PWM-G2A-Final.png"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_G2A_Plot_Save/log.out",
        err = "logs/PWM_G2A_Plot_Save/log.err"
    params:
        TFs = ",".join(G2A_TFs)
    shell:
        """
        python scripts/PWM-G2A-Plot-Best-Motifs.py -TFs {params.TFs} -d data -r Results/PWM-G2A-CV/ -l_motifs {output.l_motifs} -f_motifs {output.f_motifs} -l_logos {output.l_logos} -f_logos {output.f_logos}
        """
        
        
rule PWM_A2G_ZMotif:
    output: "Results/PWM-A2G-ZMotif/{tf}-{cycle}-{rep}/motifs.txt.gz"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_A2G_ZMotif/{tf}-{cycle}-{rep}.out",
        err = "logs/PWM_A2G_ZMotif/{tf}-{cycle}-{rep}.err"
    shell:
        """
        python3 scripts/PWM-A2G-ZMotif.py -TF {wildcards.tf} -cycle {wildcards.cycle} -o Results/PWM-A2G-ZMotif/{wildcards.tf}-{wildcards.cycle}-{wildcards.rep} -d data 2> {log.err} 1> {log.out}
        """
        
rule PWM_A2G_Plot_Save:
    input: expand("Results/PWM-A2G-ZMotif/{tf_cycle}-{rep}/motifs.txt.gz", tf_cycle = HTS_TFs_cycles, rep=reps)
    output: 
        l_motifs = "Results/PWM-A2G-Final/PWM-A2G-Leaderboard.txt",
        l_logos = "Results/PWM-A2G-Final/PWM-A2G-Leaderboard.png",
        f_motifs = "Results/PWM-A2G-Final/PWM-A2G-Final.txt",
        f_logos = "Results/PWM-A2G-Final/PWM-A2G-Final.png"
    threads: 4
    singularity:
        "docker://andrewsg/ibis:1.15"
    conda:
        "ibis"
    log:
        out = "logs/PWM_A2G_Plot_Save/log.out",
        err = "logs/PWM_A2G_Plot_Save/log.err"
    params:
        TFs = ",".join(HTS_TFs)
    shell:
        """
        python scripts/PWM-A2G-Plot-Best-Motifs.py -TFs {params.TFs} -d data -r Results/PWM-A2G-ZMotif -l_motifs {output.l_motifs} -f_motifs {output.f_motifs} -l_logos {output.l_logos} -f_logos {output.f_logos}
        """
        