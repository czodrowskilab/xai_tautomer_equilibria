#!/bin/bash
#SBATCH --job-name=optuna_demo
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki

# Fewer workers -> much less DB contention
#SBATCH --array=0-5%3

#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -t 04:00:00

#SBATCH --output=logs/tune_%A_%a.out

set -e

export PYTHONDONTWRITEBYTECODE=1

export BASE_DIR=/fshpc/limmiche/projects/xai_tautomer_equilibria

mkdir -p logs

echo "Running on $(hostname)"
echo "Worker ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
# DATABASE
# =========================================================
DB_HOST="cpu0011"
DB_PORT=5432

echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."

until pg_isready -h $DB_HOST -p $DB_PORT -U $USER; do
    sleep 2
done

# =========================================================
# OPTUNA STORAGE
# =========================================================
#export OPTUNA_STORAGE="postgresql://$USER@$DB_HOST:$DB_PORT/optuna_db"
export OPTUNA_STORAGE="sqlite:///optuna_demo.db"

# =========================================================
# FAST DEMO SETTINGS
# =========================================================

# Very small studies
export N_TRIALS_PER_JOB=2

# Fewer datasets/models if your script supports it
export DEMO_MODE=1

# Faster sklearn behavior
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Optuna DB: $OPTUNA_STORAGE"

python -B $BASE_DIR/scripts/tune_cluster_demo.py