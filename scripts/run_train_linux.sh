#!/bin/bash

set -euo pipefail

# =========================================================
# ENV
# =========================================================

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PGGSSENCMODE=disable

export BASE_DIR="$HOME/projects/xai_tautomer_equilibria"

# =========================================================
# POSTGRESQL 
# =========================================================

export DB_HOST="127.0.0.1"
export DB_PORT=5455

# Create mlflow_db if it does not aleady exist
psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='mlflow_db'" \
    | grep -q 1 \
    || psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE DATABASE mlflow_db"

export OPTUNA_STORAGE="postgresql://postgres@$DB_HOST:$DB_PORT/optuna_db?gssencmode=disable"

# =========================================================
# MLFLOW
# =========================================================

export MLFLOW_TRACKING_URI="postgresql://postgres@$DB_HOST:$DB_PORT/mlflow_db?gssencmode=disable"

# =========================================================
# CPU / THREAD CONTROL
# =========================================================

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# =========================================================
# LOGGING
# =========================================================

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/mlruns"

echo "=============================================="
echo "Starting local training (no SLURM)"
echo "Node:            $(hostname)"
echo "Base dir:        $BASE_DIR"
echo "PostgreSQL:      $DB_HOST:$DB_PORT"
echo "Optuna storage:  $OPTUNA_STORAGE"
echo "MLflow URI:      $MLFLOW_TRACKING_URI"
echo "=============================================="

# =========================================================
# RUN
# =========================================================

python -B "$BASE_DIR/scripts/run_train_linux.py" \
    2>&1 | tee "$BASE_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Training finished."