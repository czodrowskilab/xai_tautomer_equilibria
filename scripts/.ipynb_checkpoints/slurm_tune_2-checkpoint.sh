#!/bin/bash
#SBATCH --job-name=optuna_tune
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki
#SBATCH --array=0-19%14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500MB
#SBATCH -t 00:50:00
#SBATCH --output=logs/tune_%A_%a.out

set -e
export PYTHONDONTWRITEBYTECODE=1
export BASE_DIR=/fshpc/limmiche/projects/xai_tautomer_equilibria
BASE_DIR=/fshpc/limmiche/projects/xai_tautomer_equilibria

mkdir -p logs
mkdir -p mlruns_backup

echo "Running on $(hostname)"
echo "Worker ID: $SLURM_ARRAY_TASK_ID"

# -------------------------
# DB CONFIG
# -------------------------
DB_HOST="cpu0018"
DB_PORT=5432

echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."

until pg_isready -h $DB_HOST -p $DB_PORT -U $USER; do
    sleep 3
done

# -------------------------
# ENV
# -------------------------
export MLFLOW_TRACKING_URI="postgresql+psycopg2://$USER@$DB_HOST:$DB_PORT/mlflow_db"
export OPTUNA_STORAGE="postgresql://$USER@$DB_HOST:$DB_PORT/optuna_db"

export MLFLOW_LOCAL_DIR="$BASE_DIR/mlruns_backup"
export MLFLOW_ARTIFACT_ROOT="$BASE_DIR/mlruns_backup"

export SQLALCHEMY_ENGINE_OPTIONS='{"pool_size":2,"max_overflow":2,"pool_pre_ping":true}'

export TOTAL_TRIALS=40
export N_TRIALS_PER_JOB=4

echo "Optuna DB: $OPTUNA_STORAGE"

python -B $BASE_DIR/scripts/tune_cluster_2.py