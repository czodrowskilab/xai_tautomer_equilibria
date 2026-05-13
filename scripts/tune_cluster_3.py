import os
import time
from pathlib import Path

import optuna
from sqlalchemy.exc import OperationalError
from optuna.storages import RDBStorage

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import objective_factory
from xai_tautomer_equilibria.ml.search_spaces import MODEL_CONFIG

import xai_tautomer_equilibria


print("EXECUTING FILE:", os.path.abspath(__file__), flush=True)
print("PACKAGE PATH:", xai_tautomer_equilibria.__file__)
print("### NEW VERSION LOADED ###", flush=True)


# =========================================================
# PATHS / ENV
# =========================================================
BASE_DIR = Path(os.environ["BASE_DIR"])

# PostgreSQL backend ONLY for Optuna
optuna_storage_url = os.environ["OPTUNA_STORAGE"]

SEEDS = [42, 43, 44, 45, 46]

# =========================================================
# RETRY SETTINGS
# =========================================================
MAX_RETRIES = 10
RETRY_SLEEP_SECONDS = 30


def build_storage():

    return RDBStorage(
        url=optuna_storage_url,
        engine_kwargs={
            # IMPORTANT FOR HPC STABILITY
            "pool_size": 1,
            "max_overflow": 0,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        },
    )


def run_study(
    study_name,
    storage,
    X,
    y,
    model_name,
    search_space,
    seed,
    slurm_task_id,
):

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"[TASK {slurm_task_id}] "
                f"Attempt {attempt + 1}/{MAX_RETRIES} "
                f"for {study_name}",
                flush=True,
            )

            # =====================================================
            # SAFE PARALLEL OPTUNA STUDY ACCESS
            # =====================================================
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=True,
            )

            # =====================================================
            # OPTIMIZATION
            # =====================================================
            study.optimize(
                objective_factory(
                    X,
                    y,
                    model_name,
                    search_space,
                    seed,
                ),
                n_trials=int(
                    os.getenv("N_TRIALS_PER_JOB", 20)
                ),
                n_jobs=1,
            )

            print(
                f"[TASK {slurm_task_id}] Finished: {study_name}",
                flush=True,
            )

            print(
                f"[TASK {slurm_task_id}] "
                f"Best value: {study.best_value}",
                flush=True,
            )

            return

        except OperationalError as e:

            print(
                f"[TASK {slurm_task_id}] "
                f"Postgres connection failed "
                f"for {study_name}",
                flush=True,
            )

            print(str(e), flush=True)

            if attempt < MAX_RETRIES - 1:

                print(
                    f"[TASK {slurm_task_id}] "
                    f"Sleeping {RETRY_SLEEP_SECONDS}s "
                    f"before retry...",
                    flush=True,
                )

                time.sleep(RETRY_SLEEP_SECONDS)

            else:

                print(
                    f"[TASK {slurm_task_id}] "
                    f"Skipping study after "
                    f"{MAX_RETRIES} failed attempts: "
                    f"{study_name}",
                    flush=True,
                )

        except Exception as e:

            print(
                f"[TASK {slurm_task_id}] "
                f"Unexpected error in {study_name}",
                flush=True,
            )

            print(str(e), flush=True)

            # Continue to next study instead of crashing whole job
            return

        finally:

            # =====================================================
            # IMPORTANT:
            # Dispose SQLAlchemy engine connections
            # to avoid idle connection accumulation
            # =====================================================
            try:
                storage.engine.dispose()
            except Exception:
                pass


def main():

    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")

    print(f"Task: {slurm_task_id}", flush=True)
    print(f"Optuna DB: {optuna_storage_url}", flush=True)

    registry = DatasetRegistry(
        BASE_DIR / "data/processed"
    )

    datasets = list(registry.list_datasets())

    print(
        f"[TASK {slurm_task_id}] "
        f"Found {len(datasets)} datasets",
        flush=True,
    )

    # =========================================================
    # SHARED STORAGE OBJECT
    # =========================================================
    storage = build_storage()

    for dataset_path in datasets:

        print(
            f"[TASK {slurm_task_id}] "
            f"Loading dataset: {dataset_path.name}",
            flush=True,
        )

        X, y, config, _ = registry.load(dataset_path)

        subset = config.get("subset", "unknown")

        for model_name, search_space in MODEL_CONFIG.items():

            for seed in SEEDS:

                study_name = (
                    f"{dataset_path.name}_"
                    f"{model_name}_"
                    f"{subset}_"
                    f"seed{seed}"
                )

                print(
                    f"[TASK {slurm_task_id}] "
                    f"Running study: {study_name}",
                    flush=True,
                )

                run_study(
                    study_name=study_name,
                    storage=storage,
                    X=X,
                    y=y,
                    model_name=model_name,
                    search_space=search_space,
                    seed=seed,
                    slurm_task_id=slurm_task_id,
                )

    print(
        f"[TASK {slurm_task_id}] ALL STUDIES COMPLETED",
        flush=True,
    )


if __name__ == "__main__":
    main()