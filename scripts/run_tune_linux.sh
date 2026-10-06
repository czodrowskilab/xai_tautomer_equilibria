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
# POSTGRESQL (private cluster on /var/tmp, port 5455)
# =========================================================

export PGDATA="/var/tmp/$USER/postgres_data"
export PGPORT=5455
export PATH="/usr/lib/postgresql/16/bin:$PATH"

# start postgres server
"$BASE_DIR/scripts/start_postgres_linux.sh"

export DB_HOST="127.0.0.1"
export DB_PORT="$PGPORT"

until pg_isready -h "$DB_HOST" -p "$DB_PORT"; do
    sleep 2
done

export OPTUNA_STORAGE="postgresql://postgres@$DB_HOST:$DB_PORT/optuna_db?gssencmode=disable"

# =========================================================
# OPTUNA
# =========================================================

export TOTAL_TRIALS=500
export N_TRIALS_PER_JOB=25

# =========================================================
# CPU CONFIGURATION
# =========================================================

export NUM_WORKERS=20
export OPTUNA_N_JOBS=1
export MODEL_N_JOBS=6

# =========================================================
# THREAD CONTROL
# =========================================================

export OMP_NUM_THREADS=$MODEL_N_JOBS
export MKL_NUM_THREADS=$MODEL_N_JOBS
export OPENBLAS_NUM_THREADS=$MODEL_N_JOBS
export NUMEXPR_NUM_THREADS=$MODEL_N_JOBS

# =========================================================
# LOGGING
# =========================================================

mkdir -p "$BASE_DIR/logs"

echo "=============================================="
echo "Starting Optuna optimization (local, no SLURM)"
echo "Node:              $(hostname)"
echo "PostgreSQL:        $DB_HOST:$DB_PORT  (data: $PGDATA)"
echo "Optuna storage:    $OPTUNA_STORAGE"
echo "Workers:           $NUM_WORKERS"
echo "Model threads:     $MODEL_N_JOBS"
echo "Total CPUs target: $((NUM_WORKERS * MODEL_N_JOBS))"
echo "=============================================="

# =========================================================
# START WORKERS
# =========================================================

for worker in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Starting worker $worker"

    WORKER_ID=$worker \
    python -B "$BASE_DIR/scripts/tune_cluster_linux.py" \
        > "$BASE_DIR/logs/worker_${worker}.log" 2>&1 &
done

# =========================================================
# WAIT
# =========================================================

wait

echo "All optimization workers finished."