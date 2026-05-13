#!/bin/bash
#SBATCH --job-name=optuna_tune
#SBATCH -A ki-mlaatp
#SBATCH -p ki-parallel
#SBATCH -M mogonki
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500MB
#SBATCH -t 00:10:00
#SBATCH --output=logs_sum/tune_%A_%a.out



python export_mlflow_summary.py