#!/bin/bash
#SBATCH --job-name=optuna_tune
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/tune_%A_%a.out

module load python
source ~/ml_env_2/bin/activate
pip install -e .

export N_TRIALS_PER_JOB=20
export OPTUNA_DB="postgresql://user:pass@10.12.3.45:5432/optuna"   # replace with current postgres
python scripts/tune.py