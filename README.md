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


## PWM G2A
The following 4 notebooks walk you through the 3 major steps (plus 1 additional notebook for ploting and saving motifs) of the PWM-G2A workflow.

Start a jupyter-lab server in our pre-built docker image (must be done in docker, jupyter-lab installation is broken in conda environment)
```
docker container run -it --rm -p 8888:8888 andrewsg/ibis:1.15 jupyter-lab --port=8888 --ip=* --no-browser --allow-root
```

### Relevant notebooks / walkthrough
1. `PWM-G2A-ZMotif.ipynb` Train a kmer-initialized, single kernel, CNN on GABPA CHS data and obtain the motif from the multiple sequence alignment of all seqlets that pass through the convolution kernel
2. `PWM-G2A-STREME.ipynb` Run STREME on the MSE
3. `PWM-G2A-CV.ipynb`
4. `PWM-G2A-Plot-Best-Motifs.ipynb`

