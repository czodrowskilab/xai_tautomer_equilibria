MODEL_CONFIG = {

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    "random_forest": {


        "n_estimators": ("int", 150, 600),


        "max_depth": ("int", 6, 24),


        "min_samples_split": ("int", 2, 16),
        "min_samples_leaf": ("int", 1, 8),

        "max_features": [
            "sqrt",
            "log2",
            0.2,
            0.3,
            0.5,
        ],


        "bootstrap": [True],


        "max_samples": [
            0.5,
            0.7,
            0.9,
            None,
        ],
    },


    # =====================================================
    # RIDGE
    # =====================================================
    "ridge": {


        "alpha": ("float_log", 1e-5, 10.0),

        "fit_intercept": [True, False],


        "max_iter": ("int", 500, 5000),
    },

    # =====================================================
    # LASSO
    # =====================================================
    "lasso": {


        "alpha": ("float_log", 1e-5, 10.0),

        "fit_intercept": [True, False],

        "max_iter": ("int", 1000, 10000),
    },

    # =====================================================
    # ELASTIC NET
    # =====================================================
    "elastic_net": {


        "alpha": ("float_log", 1e-5, 10.0),


        "l1_ratio": ("float", 0.01, 0.99),

        "fit_intercept": [True, False],

        "max_iter": ("int", 1000, 10000),
    },


    # =====================================================
    # XGBOOST
    # =====================================================
    "xgboost": {


        "n_estimators": ("int", 200, 1200),


        "learning_rate": ("float_log", 0.02, 0.3),


        "max_depth": ("int", 3, 10),


        "min_child_weight": ("float_log", 1.0, 20.0),


        "gamma": ("float_log", 1e-4, 1.0),


        "subsample": ("float", 0.6, 1.0),


        "colsample_bytree": ("float", 0.4, 1.0),


        "reg_alpha": ("float_log", 1e-5, 5.0),


        "reg_lambda": ("float_log", 1e-3, 20.0),


        "max_delta_step": ("int", 0, 5),


        "grow_policy": [
            "depthwise",
            "lossguide",
        ],
    },
}