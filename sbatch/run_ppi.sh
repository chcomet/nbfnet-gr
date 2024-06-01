#!/bin/bash
#SBATCH --job-name=nbfnet-lnctardppi
#SBATCH --output=slurm_out/job_out_%j.txt
#SBATCH --error=slurm_out/job_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=32Gb
#SBATCH -c 4
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Load Conda environment
CONDA_DIR=/home/icb/hui.cheng/miniconda3
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate nbfnet

# Print conda environment and installed packages for debugging
echo "Conda environment:"
conda info --envs

# Print Python interpreter path and version
which python
python --version

# Set the working directory
REPO_DIR=/lustre/groups/crna01/workspace/hui/nbfnet-gr
cd $REPO_DIR

# Run the script
python script/run.py -c config/knowledge_graph/lnctardppi.yaml --gpus [0] --version v1
#python script/run.py -c config/knowledge_graph/lnctardppi.yaml -s 42 --gpus [0] --version v1
