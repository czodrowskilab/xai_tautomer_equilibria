import os
import optuna
from optuna.trial import TrialState
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


# keep DB tracking (cluster)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

mlflow.set_experiment("tautomer_optuna")

SEEDS = [42, 43, 44, 45, 46]



def main():



    # -------------------------
    # ENV VALIDATION
    # -------------------------
    optuna_storage = os.environ.get("OPTUNA_STORAGE")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    BASE_DIR = Path("/fshpc/limmiche/projects/xai_tautomer_equilibria")

    if not optuna_storage or not mlflow_uri:
        raise RuntimeError("Missing OPTUNA_STORAGE or MLFLOW_TRACKING_URI")

    # -------------------------
    # MLflow setup
    # -------------------------
    mlflow.set_tracking_uri(mlflow_uri)
    setup_mlflow("tautomer_optuna")

    # -------------------------
    # dataset registry
    # -------------------------
    registry = DatasetRegistry(BASE_DIR / "data/processed")

    datasets = list(registry.list_datasets())   #  no mutation later

    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")

    print(f"Slurm task: {slurm_task_id}")
    print(f"Using Optuna DB: {optuna_storage}")
    print(f"Using MLflow: {mlflow_uri}")

    # -------------------------
    # main loop
    # -------------------------
    for dataset_path in datasets:

        X, y, config, _ = registry.load(dataset_path)
        variant = config.get("variant", "unknown")
        subset = config.get("subset", "unknown")

        print("DATASET:", dataset_path.name, flush=True)
        print("VARIANT:", variant, flush=True)

        for model_name, search_space in MODEL_CONFIG.items():

            for seed in SEEDS:

                study_name = (
                    f"{dataset_path.name}_"
                    f"{model_name}_"
                    f"{subset}_"
                    f"seed{seed}_"
                    f"task{slurm_task_id}"
                )

                print("STUDY:", study_name, flush=True)

                # -------------------------
                # OPTUNA STUDY (FIXED)
                # -------------------------
                study = optuna.create_study(
                    study_name=study_name,
                    storage=optuna_storage,
                    direction="minimize",
                    load_if_exists=False   #
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
                    n_jobs=2
                )

                best = study.best_trial

                # -------------------------
                # MLflow logging (clean)
                # -------------------------
                with mlflow.start_run(
                    run_name=study_name,
                    nested=False
                ):

                    mlflow.log_params(best.params)
                    mlflow.log_metric("best_mse", best.value)
                    mlflow.log_metric("best_mae", best.user_attrs["mae"])
                    mlflow.log_metric("n_trials", len(study.trials))

                    mae_curve = []
                    mse_curve = []

                    for t in study.trials:
                        if t.state == TrialState.COMPLETE:
                            mse_curve.append(t.value)
                            mae_curve.append(t.user_attrs.get("mae"))

                    for i, (mse, mae) in enumerate(zip(mse_curve, mae_curve)):
                        mlflow.log_metric("mse_curve", mse, step=i)
                        mlflow.log_metric("mae_curve", mae, step=i)

                    mlflow.set_tag("dataset", dataset_path.name)
                    mlflow.set_tag("subset", subset)
                    mlflow.set_tag("variant", variant)
                    mlflow.set_tag("model", model_name)
                    mlflow.set_tag("seed", seed)
                    mlflow.set_tag("slurm_task", slurm_task_id)

if __name__ == "__main__":
    main()
