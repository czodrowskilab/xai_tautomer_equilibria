#!/bin/bash
#SBATCH --job-name=postgres
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki
#SBATCH -t 35:50:00
#SBATCH --output=postgres_%j.log

set -e

DB_DIR=/fshpc/limmiche/projects/xai_tautomer_equilibria/postgres_data
PORT=5432

echo "Starting on $(hostname)"
echo "PostgreSQL will run on: $(hostname):$PORT"

# Initialize database if not exists
if [ ! -d "$DB_DIR" ]; then
    echo "Initializing PostgreSQL database at $DB_DIR"
    initdb -D $DB_DIR
    echo "host all all 0.0.0.0/0 trust" >> $DB_DIR/pg_hba.conf
    echo "listen_addresses='*'" >> $DB_DIR/postgresql.conf
    echo "port=$PORT" >> $DB_DIR/postgresql.conf
fi

# Start PostgreSQL
pg_ctl -D $DB_DIR \
  -o "-p $PORT -c listen_addresses='*'" \
  -l $DB_DIR/postgres.log start

sleep 5

# Create databases if they don't exist
createdb -h localhost -p $PORT optuna_db 2>/dev/null || true
createdb -h localhost -p $PORT mlflow_db 2>/dev/null || true

pg_isready -h localhost -p $PORT

# Keep the job running
echo "PostgreSQL running. Press Ctrl+C to stop."
sleep infinity