import os
import optuna
from optuna.trial import TrialState
from optuna.storages import RDBStorage
import mlflow
from pathlib import Path

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import objective_factory
from xai_tautomer_equilibria.ml.search_spaces import MODEL_CONFIG
from xai_tautomer_equilibria.ml.mlflow_utils import setup_mlflow

import xai_tautomer_equilibria

print("EXECUTING FILE:", os.path.abspath(__file__), flush=True)
print("PACKAGE PATH:", xai_tautomer_equilibria.__file__)
print("### NEW VERSION LOADED ###")


# -------------------------
# ENV SETUP
# -------------------------
BASE_DIR = Path(os.environ["BASE_DIR"])

optuna_storage = os.environ["OPTUNA_STORAGE"]
storage = RDBStorage(
    url=optuna_storage,
    engine_kwargs={
        "pool_size": 2,
        "max_overflow": 2,
        "pool_pre_ping": True,
    },
)



mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]


setup_mlflow("tautomer_optuna", mlflow_uri)


BASE_DIR = Path(os.environ["BASE_DIR"])

#mlflow.log_artifacts(str(BASE_DIR / "mlruns_backup"))

SEEDS = [42, 43, 44, 45, 46]


def main():

    print(f"Task: {os.getenv('SLURM_ARRAY_TASK_ID', '0')}")
    print(f"Optuna DB: {optuna_storage}")
    print(f"MLflow: {mlflow_uri}")

    registry = DatasetRegistry(BASE_DIR / "data/processed")
    datasets = list(registry.list_datasets())

    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")

    for dataset_path in datasets:

        X, y, config, _ = registry.load(dataset_path)
        variant = config.get("variant", "unknown")
        subset = config.get("subset", "unknown")
        target = config.get("target", "unknown")

        for model_name, search_space in MODEL_CONFIG.items():
            for seed in SEEDS:

                study_name = f"{dataset_path.name}_{model_name}_{subset}_seed{seed}_{variant}_{target}"

                # -------------------------
                # FIX: SAFE PARALLEL STUDY ACCESS
                # -------------------------
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=True  
                )

                study.optimize(
                    objective_factory(X, y, model_name, search_space, seed),
                    n_trials=int(os.getenv("N_TRIALS_PER_JOB", 20)),
                    n_jobs=1
                )

                best = study.best_trial

                # -------------------------
                # FIX: MLflow logging inside run
                # -------------------------
                with mlflow.start_run(run_name=study_name):

                    mlflow.log_params(best.params)
                    mlflow.log_metric("best_mse", best.value)
                    mlflow.log_metric("best_mae", best.user_attrs.get("mae", 0.0))





                    mlflow.set_tags({
                        "dataset": dataset_path.name,
                        "subset": subset,
                        "variant": variant,
                        "model": model_name,
                        "seed": seed,
                        "slurm_task": slurm_task_id,
                        "target": target,
                        "stage": "tuning",
                    })



#                    complete_trials = study.get_trials(states=(TrialState.COMPLETE,))
#
#                    for i, t in enumerate(complete_trials):
#                        mlflow.log_metric("mse_curve", t.value, step=i)
#                        mlflow.log_metric("mae_curve", t.user_attrs.get("mae", float("nan")), step=i)




if __name__ == "__main__":
    main()