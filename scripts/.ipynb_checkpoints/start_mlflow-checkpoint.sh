#!/bin/bash
#SBATCH --job-name=mlflow_server
#SBATCH --clusters=mogonki
#SBATCH --partition=ki-parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=mlflow_server_%j.out
#SBATCH --error=mlflow_server_%j.err

# Activate environment
source /gpfs/fs1/home/limmiche/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate ml_env_2

# Create artifacts directory if it doesn't exist
mkdir -p $HOME/mlflow_artifacts

# Print debug info
echo "========================================="
echo "Starting MLflow server on $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "MLflow: $(which mlflow)"
echo "========================================="

# Test PostgreSQL connection
echo "Testing PostgreSQL connection..."
python -c "
import psycopg2
try:
    conn = psycopg2.connect(host='cpu0011', user='limmiche', dbname='postgres')
    print('✓ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'✗ PostgreSQL connection failed: {e}')
    exit(1)
"

# Create database for MLflow if it doesn't exist
echo "Creating mlflow_db if it doesn't exist..."
psql -h cpu0011 -U limmiche -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow_db'" | grep -q 1 || psql -h cpu0011 -U limmiche -d postgres -c "CREATE DATABASE mlflow_db"

# Start MLflow server
echo "Starting MLflow server on port 5000..."
mlflow server \
    --backend-store-uri postgresql://limmiche@cpu0011:5432/mlflow_db \
    --default-artifact-root $HOME/mlflow_artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1

# Keep running
wait
EOF

# Make it executable
chmod +x start_mlflow.sh