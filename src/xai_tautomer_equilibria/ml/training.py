from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score



# Todo: 5fold cross val, mse,mae instead of r2


def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }

    return model, metrics



