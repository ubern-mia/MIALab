#!/bin/bash

# SLURM Settings
#SBATCH --job-name="mialab"
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2
#SBATCH --mail-user=pascal.zingg@students.unibe.ch
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Load Anaconda3
module load Anaconda3
eval "$(conda shell.bash hook)"

# Load your environment
conda activate mialab

# Run your code
srun python3 exercise/exercise_rf.py --num_trees=1 --tree_depth=1
srun python3 exercise/exercise_rf.py --num_trees=10 --tree_depth=1
srun python3 exercise/exercise_rf.py --num_trees=50 --tree_depth=1
srun python3 exercise/exercise_rf.py --num_trees=100 --tree_depth=1
srun python3 exercise/exercise_rf.py --num_trees=1000 --tree_depth=1

srun python3 exercise/exercise_rf.py --num_trees=1 --tree_depth=10
srun python3 exercise/exercise_rf.py --num_trees=1 --tree_depth=50
srun python3 exercise/exercise_rf.py --num_trees=1 --tree_depth=100
srun python3 exercise/exercise_rf.py --num_trees=1 --tree_depth=1000

srun python3 exercise/exercise_rf.py --num_trees=10 --tree_depth=10
srun python3 exercise/exercise_rf.py --num_trees=50 --tree_depth=50
srun python3 exercise/exercise_rf.py --num_trees=100 --tree_depth=100
srun python3 exercise/exercise_rf.py --num_trees=1000 --tree_depth=1000