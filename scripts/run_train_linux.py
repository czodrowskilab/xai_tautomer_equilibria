import os
from pathlib import Path

import mlflow
import optuna

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.models import get_model
from xai_tautomer_equilibria.ml.training import train_model
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow
from xai_tautomer_equilibria.ml.search_spaces_old_seed_in_lin_models import MODEL_CONFIG

# =========================================================
# ENV
# =========================================================



BASE_DIR = Path(os.environ["BASE_DIR"])
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
optuna_storage = os.environ["OPTUNA_STORAGE"]


SEEDS = [42, 43, 44, 45, 46, 47]


# =========================================================
# LOAD BEST PARAMS FROM OPTUNA (POSTGRESQL)
# =========================================================

def load_best_params(study_name: str):

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna_storage
        )

        # no completed trials safety check
        completed = study.get_trials(
            states=(optuna.trial.TrialState.COMPLETE,)
        )

        if len(completed) == 0:
            return {}

        return study.best_trial.params

    except Exception as e:
        print(f"[WARN] Could not load study {study_name}: {e}")
        return {}


# =========================================================
# MAIN
# =========================================================

def main():

    setup_mlflow(
        "tautomer_equilibria_training",
        tracking_uri
    )

    registry = DatasetRegistry(BASE_DIR / "data/processed")

    datasets = registry.list_datasets()

    for dataset_path in datasets:

        X, y, config, features = registry.load(dataset_path)

        variant = config.get("variant", "unknown")
        subset = config.get("subset", "unknown")
        target = config.get("target", "unknown")


        for model_name in MODEL_CONFIG.keys():
    
            print(
                f"\nTraining: {dataset_path.name} "
                f"| model={model_name} "
                f"| variant={variant}"
            )
    
            for seed in SEEDS:
    
                study_name = (
                    f"{dataset_path.name}_"
                    f"{model_name}_"
                    f"{subset}_"
                    f"seed{seed}_"
                    f"{variant}_"
                    f"{target}_v7"
                )
    
                # =====================================================
                # LOAD BEST PARAMS FROM OPTUNA
                # =====================================================
    
                best_params = load_best_params(study_name)
    
                if best_params:
                    print(f"[OK] Loaded Optuna params: {study_name}")
                else:
                    print(f"[WARN] No Optuna params found: {study_name}")
    
                # =====================================================
                # MODEL
                # =====================================================
    
                model = get_model(
                    model_name,
                    best_params,
                    seed
                )
    
                # =====================================================
                # TRAINING + MLFLOW LOGGING
                # =====================================================
    
                run_name = (
                    f"{dataset_path.name}_"
                    f"{model_name}_seed{seed}"
                )
    
                with mlflow.start_run(run_name=run_name):
    
                    model, metrics = train_model(model, X, y)
    
                    # log metrics
                    mlflow.log_metrics(metrics)
    
                    # log params (clean + explicit)
                    mlflow.log_params(
                        {
                            "seed": seed,
                            "variant": variant,
                            "subset": subset,
                            "target": target,
                            "dataset_name": dataset_path.name,
                            "model_name": model_name,
                            **best_params,
                        }
                    )
    
                    # log model
                    mlflow.sklearn.log_model(model, "model")
    
                    print(f"Finished: {study_name}")
    

# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()