# scripts/tune.py

import os
import optuna
import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import objective_factory
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow
from xai_tautomer_equilibria.ml.search_spaces import MODEL_CONFIG


SEEDS = [42, 43, 44, 45, 46]


def get_storage():
    return os.getenv("OPTUNA_DB", "sqlite:///optuna.db")


def main():

    setup_mlflow("tautomer_optuna")

    registry = DatasetRegistry(Path("data/processed"))
    datasets = registry.list_datasets()

    storage = get_storage()

    slurm_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    for dataset_path in datasets:

        X, y, config, _ = registry.load(dataset_path)

        # -------------------------
        # extract variant
        # -------------------------
        variant = config.get("variant", "unknown")

        print(f"\nTuning: {dataset_path.name} | variant={variant}")

        for model_name, search_space in MODEL_CONFIG.items():

            for seed in SEEDS:

                # -------------------------
                # set study name
                # -------------------------
                study_name = f"{dataset_path.name}_{model_name}_{variant}_seed{seed}"

                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=True
                )

                n_trials = int(os.getenv("N_TRIALS_PER_JOB", 20))

                objective = objective_factory(
                    X, y,
                    model_name,
                    search_space,
                    seed
                )

                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=1
                )

                best = study.best_trial

                # -------------------------
                # log results to MLflow
                # -------------------------
                with mlflow.start_run(
                    run_name=f"{dataset_path.name}_{model_name}_{variant}_seed{seed}"
                ):

                    mlflow.log_params(best.params)
                    mlflow.log_metric("best_mse", best.value)

                    # set tags
                    mlflow.set_tag("dataset", dataset_path.name)
                    mlflow.set_tag("variant", variant)
                    mlflow.set_tag("model", model_name)
                    mlflow.set_tag("seed", seed)
                    mlflow.set_tag("slurm_task", slurm_task_id)


if __name__ == "__main__":
    main()