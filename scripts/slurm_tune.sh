#!/bin/bash

#SBATCH --job-name=optuna_tune
#SBATCH -A ki-mlaatp  # Your account: alter account: m2_jgu-fsl-cp
#SBATCH -p ki-parallel         # Queue name
#SBATCH -M mogonki
####### Parallelism #######
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4

####### Resources #######
#SBATCH --mem-per-cpu=500MB  # Memory per CPU (in MB)
#SBATCH -t 06:35:00          # Run time (hh:mm:ss)

#SBATCH --output=logs/tune_%A_%a.out






module load python
#source ~/ml_env_2/bin/activate
#pip install -e .

export N_TRIALS_PER_JOB=20
export OPTUNA_DB="postgresql://user:pass@10.12.3.45:5432/optuna"   # replace with current postgres
python scripts/tune.py