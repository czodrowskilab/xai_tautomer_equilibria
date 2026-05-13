#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out

source ~/ml_env_2/bin/activate
pip install -e .

python scripts/train.py