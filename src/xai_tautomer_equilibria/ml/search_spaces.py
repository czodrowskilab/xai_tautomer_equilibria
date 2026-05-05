MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": (50, 500),
        "max_features": (0.1, 1.0),
        "max_depth": (10, 50),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 4),
        "bootstrap": [True, False]
    },

    "ridge": {
        "alpha": (0.0001, 1.0),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "lasso": {
        "alpha": (0.0001, 1.0),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "elastic_net": {
        "alpha": (0.0001, 1.0),
        "l1_ratio": (0.01, 0.9),
        "fit_intercept": [True, False],
        "max_iter": (100, 1000)
    },

    "xgboost": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "min_child_weight": (1, 5),
        "gamma": (0, 0.3),
        "subsample": (0.7, 1.0),
        "colsample_bytree": (0.7, 1.0),
        "reg_alpha": (0.0001, 1),
        "reg_lambda": (0.001, 1)
    }
}