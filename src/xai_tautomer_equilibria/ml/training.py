from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


def train_model(model, X, y, seed = 43):

    scoring = {
        "mae": make_scorer(mean_absolute_error),
        "mse": make_scorer(mean_squared_error),
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,  # 
    )

    metrics = {
        "mae": cv_results["test_mae"].mean(),
        "mse": cv_results["test_mse"].mean(),
        "mse_std": cv_results["test_mse"].std(),
    }

    model.fit(X, y)

    return model, metrics