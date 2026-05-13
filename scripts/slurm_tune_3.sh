#!/bin/bash
#SBATCH --job-name=optuna_tune
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500MB
#SBATCH -t 35:50:00
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
    sleep 3
done

# =========================================================
# OPTUNA -> PostgreSQL ONLY
# =========================================================
export OPTUNA_STORAGE="postgresql://$USER@$DB_HOST:$DB_PORT/optuna_db"

# =========================================================
# REDUCE DB CONNECTION PRESSURE
# =========================================================
export SQLALCHEMY_ENGINE_OPTIONS='{"pool_size":1,"max_overflow":0,"pool_pre_ping":true}'

# =========================================================
# TUNING
# =========================================================
export TOTAL_TRIALS=40
export N_TRIALS_PER_JOB=3

echo "Optuna DB: $OPTUNA_STORAGE"

python -B $BASE_DIR/scripts/tune_cluster_3.py