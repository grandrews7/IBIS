import os
import itertools
import glob

if not os.path.exists("Results"):
    os.mkdir("Results")
    
data_dir = "./data/"
    
G2A_TFs = os.listdir("./data/CHS")
G2A_assays = ["CHS", "GHTS"]
peaks = ["shared", "all"]
n_motifs = [1, 16]
reps = [i for i in range(20)]

G2A_assays = ["CHS", "GHTS"]
#For debugging and testing
# G2A_TFs = ["GABPA"]
# G2A_assays = ["CHS"]
# peaks = ["shared"]
# n_motifs = [1]
# reps = [i for i in range(1)]


# To be completed for PWM-A2G
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
        expand("Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/motifs.txt.gz", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps),
        expand("Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/streme.pkl", tf=G2A_TFs, assay=G2A_assays, peaks=peaks, n=n_motifs, rep=reps),
        expand("Results/PWM-G2A-CV/{tf}-{train_assay}-{test_assay}.pkl", tf=G2A_TFs, train_assay=G2A_assays, test_assay=G2A_assays),
        "Results/PWM-G2A-Final/PWM-G2A-Leaderboard.txt", 
        "Results/PWM-G2A-Final/PWM-G2A-Final.txt"
        #expand("Results/HTS/{tf_cycle}-{rep}/motifs.txt.gz", tf_cycle = HTS_TFs_cycles, rep=reps)
        # expand("/home/gregory.andrews-umw/IBIS/Results/AAA-Arrays/{partition}-{assay}.npy", partition = ["Leaderboard", "Final"], assay=["CHS", "GHTS", "HTS", "PBM", "SMS"])
        #expand("/home/gregory.andrews-umw/IBIS/Results/AAA-Arrays/{partition}-{assay}.npy", partition = ["Final"], assay=["HTS"])

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
        "Results/PWM-G2A-ZMotif-STREME/{tf}-{assay}-{peaks}-{n}-{rep}/steme-logos.png"
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
    shell:
        """
        python scripts/PWM-G2A-Plot-Best-Motifs.py -d data -r Results/PWM-G2A-CV/ -l_motifs {output.l_motifs} -f_motifs {output.f_motifs} -l_logos {output.l_logos} -f_logos {output.f_logos}
        """
        
        
# rule PWM_A2G_ZMotif:
#     output: "Results/PWM-A2G-ZMotif/{tf}-{cycle}-{rep}/motifs.txt.gz"
#     threads: 4
#     singularity:
#         "docker://andrewsg/ibis:1.15"
#     log:
#         "logs/ZMotif-HTS/{tf}-{cycle}-{rep}.log"
#     shell:
#         """
#         python3 scripts/PWM-A2G-ZMotif.py {wildcards.tf} {wildcards.cycle} Results/HTS/{wildcards.tf}-{wildcards.cycle}-{wildcards.rep}
#         rm Results/HTS/{wildcards.tf}-{wildcards.cycle}-{wildcards.rep}/motifs.bed
#         """
        
        