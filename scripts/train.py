import os
import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.models import get_model
from xai_tautomer_equilibria.ml.training import train_model
from xai_tautomer_equilibria.ml.mlflow_utils import (
    setup_mlflow,
    log_model_and_metrics,
    get_best_params
)



# -------------------------
# set tracking for mlflow
# -------------------------
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

SEEDS = [42, 43, 44, 45, 46]


def main():

    setup_mlflow("tautomer_equilibria_training", tracking_uri)

    registry = DatasetRegistry(Path("data/processed"))
    datasets = registry.list_datasets()

    for dataset_path in datasets:

        X, y, config, features = registry.load(dataset_path)

        # -------------------------
        # extract variant explicitly
        # -------------------------
        variant = config.get("variant", "unknown")

        print(f"\nTraining on: {dataset_path.name} | variant={variant}")

        for seed in SEEDS:

            best_params = get_best_params(
                experiment_name="tautomer_optuna",
                dataset=dataset_path.name,
                model=config["model"],
                seed=seed,
                tracking_uri = tracking_uri
            )

            if best_params is None:
                print(f"No tuned params found → using defaults for {dataset_path.name}")
                best_params = {}

            model = get_model(
                config["model"],
                best_params,
                seed
            )

            # -------------------------
            # MLflow run per (dataset + variant + seed)
            # -------------------------
            with mlflow.start_run(run_name=f"{dataset_path.name}_seed{seed}"):

                model, metrics = train_model(model, X, y)

                log_model_and_metrics(
                    model,
                    metrics,
                    {
                        "seed": seed,
                        "variant": variant, 
                        "dataset_name": dataset_path.name,
                        **config,
                        **best_params
                    },
                    config
                )

                mlflow.sklearn.log_model(model, "model")
if __name__ == "__main__":
    main()