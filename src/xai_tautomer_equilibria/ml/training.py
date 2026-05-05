from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


def train_model(model, X, y):
    scoring = {
        "mae": make_scorer(mean_absolute_error),
        "mse": make_scorer(mean_squared_error),
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring,
        return_train_score=False,
    )

    metrics = {
        "mae": cv_results["test_mae"].mean(),
        "mse": cv_results["test_mse"].mean(),
    }

    # Fit final model on full dataset
    model.fit(X, y)

    return model, metrics