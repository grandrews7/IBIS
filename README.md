# IBIS
Code and instructions to replicate our results for the PWM-G2A and PWM-A2G tasks of the IBIS challenge. The steps of the pipeline are implemented in a Snakemake workflow. We executed the workflow using singularity, but we provide working conda / mamba environments and provide commands to execute the workflow both ways.

## Installation instructions
Clone repo
```
git clone https://github.com/grandrews7/IBIS
cd IBIS
```
Download and extract the nececessary data. The workflow is executed on all TFs, both leaderboard and final, simultaneously. This allowed us to ensure the identical workflow was exeuted and we could use the plots the motifs generated for the leaderboard TFs as a pseudo QC. For this reason, we slightly modified the structure of the data directory tree to put leaderboard and final datasets for a given assay in the same directory. Additionally, we provide the the FASTA file for human genome assembly hg38 (GRCh38) where the Snakemake workflow expects it
```
wget -O data.tar.gz  https://www.dropbox.com/s/p97mdpesn196lay/data.tar.gz?dl=0
tar -xzvf data.tar.gz
```
Verify data integrity
`ls data`

`CHS  GHTS  hg38.fa  hg38.fa.fai  HTS  PBM  SMS`

Each assay folder has all TFs, leaderboard and final

`ls data/CHS`

`CAMTA1  LEUTX  PRDM13  SALL3  USF3   ZBED5  ZNF251  ZNF367  ZNF407  ZNF518B
GABPA   MYF6   PRDM5   SP140  ZBED2  ZNF20  ZNF362  ZNF395  ZNF493  ZNF648`

Install Mamba
```
conda install -y -n base -c conda-forge mamba
```
Create mamba environment for jobs to be executed in
```
mamba env create -f env/ibis.yml
```
Create mamba environment to execute Snakemake workflow
```
mamba create -c conda-forge -c bioconda -n snakemake snakemake
```
You may need to install one of the several snakemake plug-ins to execute workflow on a high-performance computing cluster, but this workflow was developed and tested on a cluster running LSF and the generic cluster plug in
```
mamba activate snakemake
pip install snakemake-executor-plugin-cluster-generic
```

## Running the pipeline
```
mamba activate snakemake
snakemake --profile profiles/lsf/ -p
```
Use `-n` to do a dry-run to ensure there are no issues with the workflow itself. The above directs snakemake to use the profile we used to execute the workflow on our HPC cluster with Singularity installed. You can run it on your local machine with conda with
```
snakemake --use-conda --cores 80
```
### Final outputs of pipeline
`Results/PWM-G2A-Final/PWM-G2A-Leaderboard.txt` PWM-G2A output motifs for Leaderboard phase formatted for submission

`Results/PWM-G2A-Final/PWM-G2A-Leaderboard.png` PWM-G2A ouptut motifs for Leaderboard phase plotted

`Results/PWM-G2A-Final/PWM-G2A-Final.txt` PWM-G2A output motifs for Final phase formatted for submission

`Results/PWM-G2A-Final/PWM-G2A-Final.png` PWM-G2A output motifs for Final phase plotted

`Results/PWM-G2A-Final/PWM-A2G-Leaderboard.txt` PWM-A2G output motifs for Leaderboard phase formatted for submission

`Results/PWM-G2A-Final/PWM-A2G-Leaderboard.png` PWM-A2G ouptut motifs for Leaderboard phase plotted

`Results/PWM-G2A-Final/PWM-A2G-Final.txt` PWM-A2G output motifs for Final phase formatted for submission

`Results/PWM-G2A-Final/PWM-A2G-Final.png` PWM-A2G output motifs for Final phase plotted



## Notes
1. The pipeline as we ran it to generate the final submissions performs 20 replicates for each TF (for both PWM-G2A and PWM-A2G). This was to overcome the inherent randomness in training neural networks. This takes about 24 hours on our HPC environment limited to 1,600 cores in use at once. If necessary, change `nReps` in `Snakefile` to `1` in order to run the pipeline in a fraction of the time. The results should be similar, 20 replicates was most likely overkill but we were fortunate enough to have the resources available to us 
2. We process the input peaks for the PWM-G2A data in 2 ways, designated `all` and `shared`. `all` concatnates all provided peaks across datasets while `shared` calculates a new set of merged, shared peaks. The original intent was to do this for both CHS and GHTS data, but at least 1 TF in the final set did not have any peaks shared between datasets so we pivoted to only do this for CHS data. Therefore, GHTS processed with `shared` peaks are simply more replicates of `all`.


## Training the models
Below we provide the notebooks and corresponding scripts exectured by the pipeline to train the CNN models and obtain the deep learned motifs. 

The scripts can be executed in the provided Docker container or conda environment. To start jupyter-lab server in our pre-built docker image (must be done in docker, jupyter-lab installation is broken in conda environment)
```
docker container run -it --rm -p 8888:8888 andrewsg/ibis:1.15 jupyter-lab --port=8888 --ip=* --no-browser --allow-root
```
### ZMotif PWM-G2A
```
python3 scripts/PWM-G2A-ZMotif.py -tf {Genomic TF} -assay {CHS|GHTS} -peaks {all|shared} -n {# motifs / kernels} -o {Results Directory} -g ./data/hg38.fa -d ./data 2> log.err 1> log.out
```

#### Inputs
- `tf`: Transcription factor of interest. To see the list of available TFs run `ls data/CHS` or `ls data/GHTS`

- `assay`: `CHS` or `GHTS`

- `peaks`: `all` or `shared`. For `CHS` only. `GHTS` will use `all` regardless of setting. `all` will concatenate all peaks to use as input and motif construction. `shared` will construct a set of shared, merged peaks

-  `n`: Number of motifs to find. This will be the number of convolution kernels of the trained CNN. 1 works well in motifs cases to find the motif of the TF of interest

-  `o`: Output directory to save results to

-  `g`: Path to hg38 genome fasta

-  `d`: Directory where challenge data is stored.

#### Outputs
- `motifs.txt.gz` Technically a BED file. Locations of motif sites within the input peaks. Columns are...
  1. `chrom` chromosome of motif site
  2. `start` start position of motif site
  3. `end` end position of motif site
  4. `kernel` convolution kernel motif site is from
  5. `score` score of convolution kernel over motif site
  6. `strand` strand of motif site
  7. `seq` sequence of motif site
  8. `chrom_peak` peak of chromosome motif site orignated from
  9. `start_peak` start position of peak
  10. `end_peak` end position of peak
  11. `abs_summit_peak` position of summit of peak of motif site
  12. `signal` peak signal
  13. `seq_peak` sequence of peak
  14. `distance` from motif site to peak (should all be 0)
  15. `auc` auROC of corresponding kernel / motif. Used for ranking significance of motifs

### ZMotif PWM-A2G
```
python3 scripts/PWM-A2G-ZMotif.py -TF {wildcards.tf} -cycle {wildcards.cycle} -o {Results Directory} -d data 2> log.err 1> log.out
```

#### Inputs
- `tf`: Transcription factor of interest. To see the list of available TFs run `ls data/HTS`

- `cycle`: Cycle to pull reads from. FASTQS from multiple replicates within a cycle will be concatenated

-  `o`: Output directory to save results to

-  `d`: Directory where challenge data is stored.

#### Outputs
- `motifs.txt.gz` Technically a BED file. Locations of motif sites within the input peaks. Columns are...
  1. `chrom` chromosome of motif site
  2. `start` start position of motif site
  3. `stop` end position of motif site
  4. `kernel` convolution kernel motif site is from
  5. `score` score of convolution kernel over motif site
  6. `strand` strand of motif site
  7. `seq` sequence of motif site
  8. `og_seq` original read
  9. `auc` auROC of corresponding kernel / motif. Used for ranking significance of motifs