import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import tune
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow
from xai_tautomer_equilibria.ml.search_spaces import MODEL_CONFIG


# todo: increase nr of trials, parallelize models




SEEDS = [42, 43, 44, 45, 46]


def main():

    setup_mlflow("tautomer_optuna")

    registry = DatasetRegistry(Path("data/processed"))
    datasets = registry.list_datasets()

    for dataset_path in datasets:

        X, y, config, _ = registry.load(dataset_path)

        for model_name, search_space in MODEL_CONFIG.items():

            for seed in SEEDS:

                with mlflow.start_run():

                    best_params, best_metrics = tune(
                        X, y,
                        model_name,
                        search_space,
                        seed,
                        n_trials=50
                    )

                    mlflow.log_params(best_params)
                    mlflow.log_metric("best_mse", best_metrics["mse"])
                    mlflow.log_metric("best_mae", best_metrics["mae"])
                    mlflow.set_tag("dataset", dataset_path.name)
                    mlflow.set_tag("model", model_name)
                    mlflow.set_tag("seed", seed)
                    mlflow.set_tag("stage", "tuning") 

if __name__ == "__main__":
    main()