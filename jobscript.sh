#!/bin/bash
#SBATCH --mail-user=newton.ollengo@students.unibe.ch ## your email address
#SBATCH --mail-type=ALL ## send job status through email
#SBATCH --job-name=MIALAB ## job name
#SBATCH --output=MIALAB_RUN.txt ## job output location, default is in this_file.sh folder
#SBATCH --mail-type=ALL
#SBATCH --error=%x_%j.err
​#SBATCH --partition=epyc2 ## default cpu clusters
#SBATCH --qos=job_epyc2 ## default (running time limit 4 days), if requires longer running time use --qos=job_epyc2_long (15 days)
#SBATCH --cpus-per-task=8  ## CPU allocation
#SBATCH --mem-per-cpu=8G ## RAM allocation
#SBATCH --time=24:00:00 ## job running time limit (user setting)
module load Anaconda3 ## load available module, show modules -> module avail
eval "$(conda shell.bash hook)" ## init anaconda in shell
conda activate mialab ## activate environment
​python3 main.py