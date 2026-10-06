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
    target=None,
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
                    params[k] = trial.suggest_int(k, v[1], v[2])

                elif param_type == "float":
                    params[k] = trial.suggest_float(k, v[1], v[2])

                elif param_type == "float_log":
                    params[k] = trial.suggest_float(
                        k,
                        v[1],
                        v[2],
                        log=True
                    )

            elif isinstance(v, list):

                params[k] = trial.suggest_categorical(k, v)


        if model_name in ["random_forest", "xgboost"]:
            params["n_jobs"] = n_jobs


        # =====================================================
        # RETRY LOGIC
        # =====================================================

        MAX_RETRIES = 3

        if target == "ddg":
            mse_threshold = 28.5

        elif target == "K":
            mse_threshold = 12.0

        else:
            mse_threshold = None


        last_metrics = None


        for attempt in range(MAX_RETRIES):

            try:

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

                last_metrics = metrics

                mse = metrics["mse"]

                print(
                    f"Trial {trial.number} "
                    f"attempt {attempt+1}/{MAX_RETRIES} "
                    f"MSE={mse}"
                )

                # good enough -> accept
                if (
                    mse_threshold is None
                    or mse <= mse_threshold
                ):
                    break


                # bad MSE but retries remain
                del model
                gc.collect()


            except Exception as e:

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
        # STORE METADATA
        # =====================================================

        trial.set_user_attr(
            "mae",
            float(last_metrics["mae"])
        )

        trial.set_user_attr(
            "mse",
            float(last_metrics["mse"])
        )

        trial.set_user_attr(
            "attempts",
            attempt + 1
        )


        del model
        gc.collect()


        return float(last_metrics["mse"])


    return objective