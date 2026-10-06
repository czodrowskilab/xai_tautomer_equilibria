MODEL_CONFIG = {

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    "random_forest": {

        # compute driver
        "n_estimators": ("int", 150, 600),

        # prevents pathological huge trees
        "max_depth": ("int", 6, 24),

        # stabilize trees
        "min_samples_split": ("int", 2, 16),
        "min_samples_leaf": ("int", 1, 8),

        # important for fingerprint spaces
        "max_features": [
            "sqrt",
            "log2",
            0.2,
            0.3,
            0.5,
        ],

        # usually beneficial
        "bootstrap": [True],

        # strong regularizer for RF
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

        # tiny alphas rarely useful after scaling
        "alpha": ("float_log", 1e-3, 20.0),

        "fit_intercept": [True],


        # stabilize convergence
        "tol": ("float_log", 1e-4, 1e-2),
    },


    # =====================================================
    # LASSO
    # =====================================================
    "lasso": {

        # avoid convergence hell
        "alpha": ("float_log", 1e-3, 2.0),

        "fit_intercept": [True],

        # large enough for fingerprints
        "max_iter": ("int", 5000, 20000),

        # convergence-speed tuning
        "tol": ("float_log", 1e-4, 1e-2),


    },


    # =====================================================
    # ELASTIC NET
    # =====================================================
    "elastic_net": {

        "alpha": ("float_log", 1e-3, 2.0),

        # avoid unstable extremes
        "l1_ratio": ("float", 0.1, 0.9),

        "fit_intercept": [True],

        "max_iter": ("int", 5000, 20000),

        "tol": ("float_log", 1e-4, 1e-2),


    },


    # =====================================================
    # XGBOOST
    # =====================================================
    "xgboost": {

        # most important compute parameter
        "n_estimators": ("int", 200, 1200),

        # avoid ultra-slow tiny learning rates
        "learning_rate": ("float_log", 0.02, 0.3),

        # fingerprints usually prefer shallow-medium trees
        "max_depth": ("int", 3, 10),

        # VERY important regularizer
        "min_child_weight": ("float_log", 1.0, 20.0),

        # split regularization
        "gamma": ("float_log", 1e-4, 1.0),

        # row subsampling
        "subsample": ("float", 0.6, 1.0),

        # feature subsampling
        "colsample_bytree": ("float", 0.4, 1.0),

        # L1 regularization
        "reg_alpha": ("float_log", 1e-5, 5.0),

        # L2 regularization
        "reg_lambda": ("float_log", 1e-3, 20.0),

        # prevent giant trees
        "max_delta_step": ("int", 0, 5),

        # histogram growth policy often wins on wide data
        "grow_policy": [
            "depthwise",
            "lossguide",
        ],
    },
}