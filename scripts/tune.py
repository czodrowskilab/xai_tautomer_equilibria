import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import tune
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow


# todo: increase nr of trials

MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": (50, 500),
        "max_features": (0.1, 1.0),
        "max_depth": (10, 50),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 4),
        "bootstrap": [True, False]
    },

    "ridge": {
        "alpha": (0.0001, 1.0),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "lasso": {
        "alpha": (0.0001, 1.0),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "elastic_net": {
        "alpha": (0.0001, 1.0),
        "l1_ratio": (0.01, 0.9),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "xgboost": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "min_child_weight": (1, 5),
        "gamma": (0, 0.3),
        "subsample": (0.7, 1.0),
        "colsample_bytree": (0.7, 1.0),
        "reg_alpha": (0.0001, 1),
        "reg_lambda": (0.001, 1)
    }
}


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