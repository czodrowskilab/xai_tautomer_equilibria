import os
from pathlib import Path

import optuna
from optuna.trial import TrialState
from optuna.storages import RDBStorage

from xai_tautomer_equilibria.ml.registry import DatasetRegistry
from xai_tautomer_equilibria.ml.tuning import objective_factory
from xai_tautomer_equilibria.ml.search_spaces_old_seed_in_lin_models import MODEL_CONFIG

import xai_tautomer_equilibria


print("EXECUTING FILE:", os.path.abspath(__file__), flush=True)
print("PACKAGE PATH:", xai_tautomer_equilibria.__file__)
print("### NEW HPC-OPTIMIZED VERSION ###")


# =========================================================
# ENV SETUP
# =========================================================

BASE_DIR = Path(os.environ["BASE_DIR"])
optuna_storage = os.environ["OPTUNA_STORAGE"]





# =========================================================
# CPU CONFIG (from run_tune_linux.sh)
# =========================================================

MODEL_N_JOBS = int(os.getenv("MODEL_N_JOBS", "6"))
OPTUNA_N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "1"))

print(f"MODEL_N_JOBS: {MODEL_N_JOBS}")
print(f"OPTUNA_N_JOBS: {OPTUNA_N_JOBS}")







# Threading control 
os.environ["OMP_NUM_THREADS"] = str(MODEL_N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(MODEL_N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MODEL_N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MODEL_N_JOBS)


# =========================================================
# OPTUNA STORAGE
# =========================================================

storage = RDBStorage(
    url=optuna_storage,
    engine_kwargs={"pool_pre_ping": True},
    skip_compatibility_check=False,
    skip_table_creation=False,
)

SEEDS = [42, 43, 44, 45, 46, 47]


# =========================================================
# MAIN
# =========================================================

def main():







    worker_id = os.getenv("WORKER_ID", "0")

    print(f"WORKER ID: {worker_id}")
    print(f"MODEL_N_JOBS: {MODEL_N_JOBS}")
    print(f"OPTUNA_N_JOBS: {OPTUNA_N_JOBS}")
    print(f"OPTUNA DB: {optuna_storage}")



    registry = DatasetRegistry(BASE_DIR / "data/processed")

    datasets = list(registry.list_datasets())
  

    for dataset_path in datasets:

        print(f"\nDATASET: {dataset_path.name}")

        X, y, config, _ = registry.load(dataset_path)

        variant = config.get("variant", "unknown")
        subset = config.get("subset", "unknown")
        target = config.get("target", "unknown")

        for model_name, search_space in MODEL_CONFIG.items():

            print(f"\nMODEL: {model_name}")

            for seed in SEEDS:

                study_name = (
                    f"{dataset_path.name}_"
                    f"{model_name}_"
                    f"{subset}_"
                    f"seed{seed}_"
                    f"{variant}_"
                    f"{target}_v7"
                )

                print(f"STUDY: {study_name}")

                # =====================================================
                # LOAD / CREATE STUDY
                # =====================================================

                study = optuna.create_study(
                
                    # avoid instabilities during optimization
                    sampler=optuna.samplers.TPESampler(
                        seed=42,
                        n_startup_trials=50,
                    ),
                
                    study_name=study_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=True,
                )



                completed_trials = study.get_trials(
                    states=(TrialState.COMPLETE,),
                    deepcopy=False,
                )
                
                completed_count = len(completed_trials)
                
                total_trials = int(os.getenv("TOTAL_TRIALS", "500"))
                remaining_trials = total_trials - completed_count

                print(f"Remaining: {remaining_trials}")
                print(f"completed: {completed_count}")



                
                if remaining_trials <= 0:
                    print("Study complete.")
                    continue
                
                n_trials_this_job = min(
                    int(os.getenv("N_TRIALS_PER_JOB", "25")),
                    remaining_trials,
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
                        n_jobs=MODEL_N_JOBS
                    ),
                    n_trials=n_trials_this_job,
                    n_jobs=OPTUNA_N_JOBS,
                )

                # =====================================================
                # BEST RESULT
                # =====================================================

                best = study.best_trial

                print("\nBEST RESULT")
                print(f"MSE: {best.value}")

                if "mae" in best.user_attrs:
                    print(f"MAE: {best.user_attrs['mae']}")

                print(f"Params: {best.params}")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()