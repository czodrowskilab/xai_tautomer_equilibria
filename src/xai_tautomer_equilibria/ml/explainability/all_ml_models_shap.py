#!/usr/bin/env python
"""
Compute SHAP feature importances for all trained models in the model registry.

Iterates over ModelRegistry entries, loads the corresponding training data
via the project's DatasetRegistry, computes SHAP importances, and writes
one CSV per model.

Output:
    data_shap/<target>/<model_id>/<family>_shap.csv
    columns: Feature, Mean_SHAP_Abs

Usage:
    python all_ml_models_shap.py \
        --models-root /path/to/models/processed \
        --data-root   /path/to/data/processed \
        --out-dir     ./data_shap \
        --seed        43 \
        [--model-family elastic_net_regression ...] \
        [--target K ddg] \
        [--subset change_in_arom_r no_change_in_arom_r] \
        [--standardize | --no-standardize]

or:
    python all_ml_models_shap.py \
        --models-root /mnt/czolab-storage-01/limmiche/projects/xai_tautomer_equilibria/models/processed \
        --data-root   /mnt/czolab-storage-01/limmiche/projects/xai_tautomer_equilibria/data/processed \
        --out-dir     ./data_shap \
        --seed        43
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from xai_tautomer_equilibria.ml.registry import ModelRegistry, DatasetRegistry


# =========================================================
# PICKLE COMPATIBILITY SHIM
# =========================================================

import sys as _sys

def _install_sklearn_pickle_shims() -> None:
    shims = {
        "sklearn.ensemble.forest":  "sklearn.ensemble._forest",
        "sklearn.tree.tree":        "sklearn.tree._tree",
        "sklearn.externals.joblib": "joblib",
        "sklearn.utils._joblib":    "joblib",
        "sklearn.externals.six":    "six",
    }
    for old, new in shims.items():
        if old in _sys.modules:
            continue
        try:
            mod = __import__(new, fromlist=["*"])
        except ImportError:
            continue
        _sys.modules[old] = mod

_install_sklearn_pickle_shims()


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shap")


# =========================================================
# CONSTANTS
# =========================================================

TARGET_TO_LABEL_COLUMN = {
    "K":   "log_K",
    "ddg": "ddG",
}

LINEAR_FAMILIES = {
    "elastic_net_regression",
    "lasso_regression",
    "ridge_regression",
}
TREE_FAMILIES = {
    "random_forest",
    "xgboost_regressor",
}


SCALING_ALREADY_DONE = {
    "standard_all_features",
    "standard",
    "standardized",
}


# =========================================================
# HELPERS
# =========================================================

def expected_n_features(model) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    if hasattr(model, "coef_"):
        return int(np.asarray(model.coef_).ravel().shape[0])
    return None


def prepare_features(
    X: np.ndarray,
    features: list[str],
    config: dict,
    standardize: bool | None,
) -> pd.DataFrame:
    """
    Wrap X as a DataFrame with feature names, optionally standardizing.

    Standardization policy:
        standardize is None  -> trust config["scaling"];
                                scale only if NOT already standardized
        standardize is True  -> always apply StandardScaler
        standardize is False -> never apply StandardScaler
    """
    scaling_value = str(config.get("scaling", "")).lower()
    already_scaled = scaling_value in SCALING_ALREADY_DONE

    if standardize is None:
        do_scale = not already_scaled
    else:
        do_scale = standardize

    if do_scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    return pd.DataFrame(X, columns=features)


def load_model(entry):
    pkl = entry.path / f"{entry.model_id}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(pkl)

    try:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"legacy pickle incompatibility in {pkl.name}: {exc}. "
            f"Likely saved with scikit-learn <0.22; retrain or re-save."
        ) from exc


# =========================================================
# SHAP
# =========================================================

def compute_shap_importance(
    model,
    X: pd.DataFrame,
    model_family: str,
) -> np.ndarray:
    """
    Returns mean(|SHAP|) per feature, shape (n_features,).
    """
    if model_family in LINEAR_FAMILIES:
        explainer = shap.LinearExplainer(
            model, X, feature_perturbation="interventional",
        )
    elif model_family in TREE_FAMILIES:
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"no SHAP explainer for {model_family}")

    shap_values = explainer.shap_values(X)

    # Some explainers return (n, f) or a list of arrays (multiclass) — handle both.
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values).mean(axis=0)

    return np.abs(shap_values).mean(axis=0)


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--models-root", type=Path, required=True,
                    help="models/processed (registry.json is expected one level up)")
    ap.add_argument("--data-root",   type=Path, required=True,
                    help="data/processed — passed to DatasetRegistry")
    ap.add_argument("--out-dir",     type=Path, default=Path("./data_shap"))
    ap.add_argument("--seed",        type=int,  default=43)

    ap.add_argument("--model-family", nargs="*", default=None,
                    help="restrict to these model families")
    ap.add_argument("--target",       nargs="*", default=None,
                    choices=["K", "ddg"])
    ap.add_argument("--subset",       nargs="*", default=None,
                    choices=["change_in_arom_r", "no_change_in_arom_r"])

    # Tri-state standardization
    ap.add_argument("--standardize", dest="standardize",
                    action="store_const", const=True, default=None,
                    help="Force standardize, ignoring config['scaling']")
    ap.add_argument("--no-standardize", dest="standardize",
                    action="store_const", const=False,
                    help="Force NO standardize, ignoring config['scaling']")

    args = ap.parse_args()

    # ---- model registry --------------------------------------------
    registry_path = args.models_root.parent / "registry.json"
    if not registry_path.exists():
        registry_path = args.models_root / "registry.json"
    if not registry_path.exists():
        sys.exit(f"[ERROR] model registry not found near {args.models_root}")

    log.info(f"Loading model registry: {registry_path}")
    model_registry = ModelRegistry(registry_path.parent)

    # ---- dataset registry ------------------------------------------
    log.info(f"Loading dataset registry from: {args.data_root}")
    data_registry = DatasetRegistry(args.data_root)

    # ---- filter model entries --------------------------------------
    entries = [
        e for e in model_registry.list_models()
        if e.seed == args.seed
        and (args.model_family is None or e.model_family in args.model_family)
        and (args.target       is None or e.target       in args.target)
        and (args.subset       is None or e.subset       in args.subset)
    ]
    log.info(f"Matched {len(entries)} models "
             f"(seed={args.seed}, "
             f"families={args.model_family or 'all'}, "
             f"targets={args.target or 'all'}, "
             f"subsets={args.subset or 'all'})")

    if not entries:
        sys.exit("Nothing to do.")

    # ---- iterate ----------------------------------------------------
    n_ok, n_fail = 0, 0

    for entry in entries:

        log.info("-" * 60)
        log.info(f"{entry.model_id}")

        try:
            # ---- locate dataset via DatasetRegistry ----------------
            try:
                dataset_dir = data_registry.get_by_id(entry.dataset_id)
            except KeyError as exc:
                raise KeyError(
                    f"dataset_id '{entry.dataset_id}' not found under {args.data_root}"
                ) from exc

            X_raw, y_raw, config, features = data_registry.load(dataset_dir)
            X = prepare_features(X_raw, features, config, standardize=args.standardize)
            y = pd.Series(y_raw, name=TARGET_TO_LABEL_COLUMN[entry.target])

            log.info(
                f"  data:  {dataset_dir.name}  X={X.shape}  "
                f"scaling={config.get('scaling')!r}"
            )

            model = load_model(entry)

            n_model_feat = expected_n_features(model)
            n_X_feat     = X.shape[1]
            if n_model_feat is not None and n_model_feat != n_X_feat:
                raise ValueError(
                    f"feature count mismatch: model expects {n_model_feat} "
                    f"but dataset {dataset_dir.name} provides {n_X_feat}"
                )

            importances = compute_shap_importance(model, X, entry.model_family)

            shap_df = (
                pd.DataFrame({
                    "Feature":       list(X.columns),
                    "Mean_SHAP_Abs": importances,
                })
                .sort_values("Mean_SHAP_Abs", ascending=False)
                .reset_index(drop=True)
            )

            out_dir = args.out_dir / entry.target / entry.model_id
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{entry.model_family}_shap.csv"
            shap_df.to_csv(out_path, index=False)

            log.info(f"  → {out_path.relative_to(args.out_dir)}  ({len(shap_df)} features)")
            n_ok += 1

        except Exception as exc:
            log.error(f"  FAILED: {exc}")
            n_fail += 1

    log.info("=" * 60)
    log.info(f"Done.  ok={n_ok}  failed={n_fail}")


if __name__ == "__main__":
    main()