# scripts/tune.py

import os
import optuna
import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import objective_factory
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow
from xai_tautomer_equilibria.ml.search_spaces import MODEL_CONFIG



# Dataset × model × seed space = 1 Optuna study
# Parallelism across Optuna studies (via shared DB): Optuna provides parameter for data base for that purpose

SEEDS = [42, 43, 44, 45, 46]


def get_storage():
    """
    Use Postgres in real cluster setups.
    SQLite is OK for single-node experiments.
    """
    return os.getenv(
        "OPTUNA_DB",
        "sqlite:///optuna.db"
    )


def main():

    setup_mlflow("tautomer_optuna")

    registry = DatasetRegistry(Path("data/processed"))
    datasets = registry.list_datasets()

    storage = get_storage()

    # SLURM gives each job a unique task id
    slurm_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    for dataset_path in datasets:

        X, y, config, _ = registry.load(dataset_path)

        for model_name, search_space in MODEL_CONFIG.items():

            for seed in SEEDS:

                study_name = f"{dataset_path.name}_{model_name}_seed{seed}"

                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=True
                )

                # IMPORTANT: split workers via SLURM
                n_trials = int(os.getenv("N_TRIALS_PER_JOB", 20)) #N_TRIALS_PER_JOB from slurm_tune.sh

                objective = objective_factory(
                    X, y,
                    model_name,
                    search_space,
                    seed
                )

                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=1  # SLURM handles parallelism
                )

                best = study.best_trial

                with mlflow.start_run():

                    mlflow.log_params(best.params)
                    mlflow.log_metric("best_mse", best.value)
                    mlflow.set_tag("dataset", dataset_path.name)
                    mlflow.set_tag("model", model_name)
                    mlflow.set_tag("seed", seed)
                    mlflow.set_tag("slurm_task", slurm_task_id)


if __name__ == "__main__":
    main()