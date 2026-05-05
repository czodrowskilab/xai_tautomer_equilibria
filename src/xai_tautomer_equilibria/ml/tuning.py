import optuna
import mlflow
from .models import get_model
from .training import train_model


# todo: 

def objective_factory(X, y, model_name, search_space, seed):

    def objective(trial):

        params = {}

        for k, v in search_space.items():

            if isinstance(v, tuple): # if regression task
                if isinstance(v[0], float):
                    params[k] = trial.suggest_float(k, v[0], v[1])
                else:
                    params[k] = trial.suggest_int(k, v[0], v[1])

            elif isinstance(v, list):  # if classification task
                params[k] = trial.suggest_categorical(k, v)

        model = get_model(model_name, params, seed)

        model, metrics = train_model(model, X, y)

        mlflow.log_metrics(metrics)
        # store extra mae in trial
        trial.set_user_attr("mae", metrics["mae"])

        return metrics["mse"]

    return objective


def tune(X, y, model_name, search_space, seed, n_trials=50):

    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective_factory(X, y, model_name, search_space, seed),
        n_trials=n_trials
    )

    best_trial = study.best_trial

    best_metrics = {
        "mse": best_trial.value,
        "mae": best_trial.user_attrs["mae"],
    }

    return study.best_params, best_metrics