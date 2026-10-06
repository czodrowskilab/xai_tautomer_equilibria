import gc
import warnings

import numpy as np
import optuna

from sklearn.exceptions import ConvergenceWarning

from .models import get_model
from .training import train_model


# =========================================================
# OBJECTIVE FACTORY
# =========================================================

def objective_factory(
    X,
    y,
    model_name,
    search_space,
    seed,
    n_jobs,
):

    def objective(trial):

        # =====================================================
        # BUILD PARAMS
        # =====================================================

        params = {}

        for k, v in search_space.items():

            if isinstance(v, tuple):

                param_type = v[0]

                if param_type == "int":

                    params[k] = trial.suggest_int(
                        k,
                        v[1],
                        v[2],
                    )

                elif param_type == "float":

                    params[k] = trial.suggest_float(
                        k,
                        v[1],
                        v[2],
                    )

                elif param_type == "float_log":

                    params[k] = trial.suggest_float(
                        k,
                        v[1],
                        v[2],
                        log=True,
                    )

            elif isinstance(v, list):

                params[k] = trial.suggest_categorical(
                    k,
                    v,
                )

        # =====================================================
        # MODEL THREADING
        # =====================================================

        if model_name in ["random_forest", "xgboost"]:

            params["n_jobs"] = n_jobs

        # =====================================================
        # TRAIN
        # =====================================================

        try:

            with warnings.catch_warnings():

                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                )





                
                model = get_model(
                    model_name,
                    params,
                    seed,
                )

    

                model, metrics = train_model(
                    model,
                    X,
                    y
                )

        except Exception as e:

            # mark failed trials cleanly
            trial.set_user_attr(
                "failed",
                True,
            )

            trial.set_user_attr(
                "exception",
                str(e),
            )

            raise

        # =====================================================
        # OPTUNA METADATA
        # =====================================================

        trial.set_user_attr(
            "mae",
            float(metrics["mae"]),
        )

        trial.set_user_attr(
            "mse",
            float(metrics["mse"]),
        )

        # =====================================================
        # MEMORY CLEANUP
        # =====================================================

        del model
        gc.collect()

        # =====================================================
        # OBJECTIVE
        # =====================================================

        return float(metrics["mse"])

    return objective