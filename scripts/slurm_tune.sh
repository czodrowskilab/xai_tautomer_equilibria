#!/bin/bash
#SBATCH --job-name=optuna_tune
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki

#SBATCH --array=0-19%4     # LIMIT CONCURRENT JOBS 

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500MB
#SBATCH -t 00:50:00

#SBATCH --output=logs/tune_%A_%a.out


set -e
export PYTHONDONTWRITEBYTECODE=1

BASE_DIR=/fshpc/limmiche/projects/xai_tautomer_equilibria

mkdir -p logs

echo "Running on $(hostname)"
echo "Worker ID: $SLURM_ARRAY_TASK_ID"

# -------------------------
# DATABASE CONFIG
# -------------------------
DB_HOST="cpu0018"
DB_PORT=5432

echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."

until pg_isready -h $DB_HOST -p $DB_PORT -U $USER; do
    echo "PostgreSQL not ready, waiting..."
    sleep 3
done

# -------------------------
# ENV VARIABLES
# -------------------------
export MLFLOW_TRACKING_URI="postgresql+psycopg2://$USER@$DB_HOST:$DB_PORT/mlflow_db"
export OPTUNA_STORAGE="postgresql://$USER@$DB_HOST:$DB_PORT/optuna_db"

#  lower connection pressure per process
export SQLALCHEMY_ENGINE_OPTIONS='{"pool_size":2,"max_overflow":2,"pool_pre_ping":true}'

echo "MLflow URI: $MLFLOW_TRACKING_URI"
echo "Optuna DB: $OPTUNA_STORAGE"

# -------------------------
# GLOBAL TRIAL BUDGET CONTROL
# -------------------------
export TOTAL_TRIALS=40
export N_TRIALS_PER_JOB=2

echo "SCRIPT PATH:"
ls -l $BASE_DIR/scripts/tune_cluster_1.py

# -------------------------
# RUN
# -------------------------
python -B $BASE_DIR/scripts/tune_cluster_1.py