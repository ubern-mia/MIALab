#!/bin/bash

# SLURM Settings
#SBATCH --job-name="mialab"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2_short
#SBATCH --mail-user=bryan.perdrizat@students.unibe.ch
#SBATCH --mail-type=none
#SBATCH --output=output/_%x_%j.out
#SBATCH --error=output/_%x_%j.err

# Load Anaconda3
module load Anaconda3
eval "$(conda shell.bash hook)"

# Load your environment
conda create --name mialab --file env.txt
conda activate mialab

# Run your code
srun python3 MIALab/bin/main.py --result_dir ../output/%x_%j/