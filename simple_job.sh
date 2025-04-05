#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gpus-per-node=1
#SBATCH --mem=127000M               # memory per node
#SBATCH --time=0-30:00

source ~/py37/bin/activate
cd ~/scratch/DyGLib
bash run_experiments.sh