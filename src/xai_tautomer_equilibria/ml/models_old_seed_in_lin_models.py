import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor


def get_model(name, params, seed, n_jobs=1):

    params = params.copy()
    params.pop("n_jobs", None)

    if name == "random_forest":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            **params
        )

    if name == "xgboost":
        return XGBRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            **params
        )

    if name == "ridge":
        return Ridge(
            solver="sag",   # or "saga" if you want stronger stochasticity
            random_state=seed,
            **params
        )

    if name == "lasso":
        return Lasso(
            selection="random",
            random_state=seed,
            **params
        )

    if name == "elastic_net":
        return ElasticNet(
            selection="random",
            random_state=seed,
            **params
        )




    raise ValueError(f"Unknown model {name}")