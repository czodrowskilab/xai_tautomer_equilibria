
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd


def permutation_importance_global(
    model,
    X,
    y,
    feature_names,
    n_repeats=20,
    seed=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })

    return df.sort_values("importance", ascending=False)





def linear_model_importance(model, feature_names):
    coef = model.coef_

    if hasattr(coef, "ndim") and coef.ndim > 1:
        coef = np.mean(np.abs(coef), axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(coef),
    })

    return df.sort_values("importance", ascending=False)



def get_importance(model_name):
    """
    Central dispatch so your scripts stay clean.
    """
    if model_name in {
        "lasso_regression",
        "ridge_regression",
        "elastic_net_regression",
    }:
        return "linear"

    if model_name in {
        "random_forest_regressor",
        "xgboost_regressor",
    }:
        return "permutation"

    return "permutation"