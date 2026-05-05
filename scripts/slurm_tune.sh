#!/bin/bash
#SBATCH --job-name=ml_tune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/tune_%j.out

source activate ml_env_2

python scripts/tune.py