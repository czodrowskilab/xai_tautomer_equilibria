#!/usr/bin/env python
"""
Compute INTRINSIC feature importances for trained models in the model registry

Output:
    data_intrinsic/<target>/<model_id>/<family>_coef.csv
    columns: Feature, Coefficient

Usage:
    python all_ml_models_intrinsic.py \
        --models-root  /path/to/models/processed \
        --data-root    /path/to/data/processed \
        --out-dir      ./data_intrinsic \
        --seed         43

or:
python all_ml_models_intrinsic.py \
    --models-root /mnt/czolab-storage-01/limmiche/projects/xai_tautomer_equilibria/models/processed \
    --data-root   /mnt/czolab-storage-01/limmiche/projects/xai_tautomer_equilibria/data/processed \
    --out-dir     ./data_intrinsic \
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
from sklearn.preprocessing import StandardScaler

from xai_tautomer_equilibria.ml.registry import ModelRegistry, DatasetRegistry


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intrinsic")


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


SCALING_ALREADY_DONE = {
    "standard_all_features",
    "standard",
    "standardized",
}


# =========================================================
# HELPERS
# =========================================================
import sys
import sklearn.ensemble

# Old pickle path → new path shim
if "sklearn.ensemble.forest" not in sys.modules:
    try:
        import sklearn.ensemble._forest as _new_forest
        sys.modules["sklearn.ensemble.forest"] = _new_forest
    except ImportError:
        pass

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

    Standardization:
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
    with open(pkl, "rb") as f:
        return pickle.load(f)


# =========================================================
# INTRINSIC IMPORTANCE 
# =========================================================

def importance_linear(model, X_train: pd.DataFrame) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        "Feature":     X_train.columns,
        "Coefficient": model.coef_,
    })
    coef_df["Coefficient"] = coef_df["Coefficient"].abs()
    return coef_df


def importance_xgboost(model, X_train: pd.DataFrame) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "Feature":    X_train.columns,
        "Importance": model.feature_importances_,
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df.columns = ["Feature", "Coefficient"]
    return importance_df.reset_index(drop=True)


def importance_random_forest(model, X_train, y_train) -> pd.DataFrame:
    from rfpimp import importances as rfpimp_importances
    imp = rfpimp_importances(model, X_train, y_train, n_samples=-1)
    imp_reset = imp.copy().reset_index()
    imp_reset = imp_reset.rename(columns={"Importance": "Coefficient"})
    return imp_reset


# =========================================================
# DISPATCHER
# =========================================================

def compute_intrinsic(model, entry, X, y) -> pd.DataFrame:
    fam = entry.model_family
    if fam in LINEAR_FAMILIES:
        return importance_linear(model, X)
    if fam == "xgboost_regressor":
        return importance_xgboost(model, X)
    if fam == "random_forest":
        return importance_random_forest(model, X, y)
    raise ValueError(f"no intrinsic extractor for {fam}")


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--models-root", type=Path, required=True,
                    help="models/processed (registry.json is expected one level up)")
    ap.add_argument("--data-root",   type=Path, required=True,
                    help="data/processed — passed to DatasetRegistry")
    ap.add_argument("--out-dir",     type=Path, default=Path("./data_intrinsic"))
    ap.add_argument("--seed",        type=int,  default=43)

    ap.add_argument("--model-family", nargs="*", default=None)
    ap.add_argument("--target",       nargs="*", default=None, choices=["K", "ddg"])
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
    log.info(f"Matched {len(entries)} models")

    if not entries:
        sys.exit("Nothing to do.")

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



            df = compute_intrinsic(model, entry, X, y)

            out_dir = args.out_dir / entry.target / entry.model_id
            out_dir.mkdir(parents=True, exist_ok=True)

            family_token = (
                "RF-regression"
                if entry.model_family == "random_forest"
                else f"{entry.model_family}-regression"
            )
            out_path = out_dir / f"{family_token}_coef.csv"
            df.to_csv(out_path, index=False)

            log.info(f"  → {out_path.relative_to(args.out_dir)}  ({len(df)} features)")
            n_ok += 1

        except Exception as exc:
            log.error(f"  FAILED: {exc}")
            n_fail += 1

    log.info("=" * 60)
    log.info(f"Done.  ok={n_ok}  failed={n_fail}")


if __name__ == "__main__":
    main()