import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from xai_tautomer_equilibria.util.preprocess import (
    build_tautomer_pair,
    build_dataset,
    standardize_columns,
    filter_by_row_pickle,
    move_rows,
)

from xai_tautomer_equilibria.util.ring_analysis import (
    extract_relevant_ring_indices,
    extract_ring_bond_types,
    extract_ring_bond_types_neighbor,
)

from xai_tautomer_equilibria.features.features import (
    featurize_pair,
    load_aromaticity_features)

from xai_tautomer_equilibria.config.paths import (
    TAUTOBASE_PATH,
    WIEDER_ROWS_PATH)

from xai_tautomer_equilibria.config.paths import (
    PROCESSED_DIR, 
    NICSZZ_VALUES_PATH,
    HOMA_VALUES_PATH,
    MCI_VALUES_PATH)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)




# -------------------------
# Build feature matrix
# -------------------------
def build_features(
    df: pd.DataFrame,
    radius: int,
    n_bits: int,
    homa_df
) -> Tuple[np.ndarray, pd.DataFrame]:

    features = []
    valid_rows = []

    for row in df.itertuples(index=False):

        # -------------------------
        # load aromaticity per row
        # -------------------------
        arom = load_aromaticity_features(
            row.row,
            homa_df
        )

        if arom is None:
            continue

        # -------------------------
        # molecular features
        # -------------------------
        feat = featurize_pair(
            row.t1_smiles,
            row.t2_smiles,
            aromaticity_features=arom,
            radius=radius,
            n_bits=n_bits
        )

        if feat is None:
            continue

        features.append(feat)
        valid_rows.append(row)

    X = np.array(features)
    df_valid = pd.DataFrame(valid_rows)

    return X, df_valid

def save_dataset(X, y, feature_names, config, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    assert X.shape[1] == len(feature_names), (
        f"Mismatch: X has {X.shape[1]} columns "
        f"but {len(feature_names)} feature names"
    )

    # save arrays
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    # save metadata
    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)




def save_dataset_metadata(
    df_valid: pd.DataFrame,
    target: str,
    out_dir: Path
):
    """
    Save row identifiers, SMILES and target values
    corresponding exactly to X.npy rows.
    """

    cols = [
        c for c in [
            "row",
            "t1_smiles",
            "t2_smiles",
            target,
        ]
        if c in df_valid.columns
    ]

    df_valid[cols].reset_index(drop=True).to_csv(
        out_dir / "dataset_rows.csv",
        index=False
    )


def remove_constant_features(
    X: np.ndarray,
    feature_names: list[str],
):
    """
    Remove features whose value is identical for every row.

    Returns
    -------
    X_filtered : np.ndarray
        Feature matrix with constant columns removed.
    feature_names_filtered : list[str]
        Feature names corresponding to X_filtered.
    n_dropped : int
        Number of removed constant features.
    """
    # Equivalent to pandas:
    # df.loc[:, df.nunique() != 1]

    unique_mask = np.ptp(X, axis=0) != 0

    X_filtered = X[:, unique_mask]

    feature_names_filtered = [
        name
        for name, keep in zip(feature_names, unique_mask)
        if keep
    ]

    n_dropped = (~unique_mask).sum()

    return X_filtered, feature_names_filtered, n_dropped


def main():

    # -------------------------
    # Load raw data
    # -------------------------
    df = pd.read_csv(TAUTOBASE_PATH)
    homa_df = pd.read_csv(HOMA_VALUES_PATH)

    # -------------------------
    # Standardize schema
    # -------------------------
    df = standardize_columns(df)

    # -------------------------
    # Ring analysis
    # -------------------------
    results = []

    for row in df.itertuples(index=False):
        try:
            t1_mol, t2_mol = build_tautomer_pair(
                row.t1_smiles,
                row.t2_smiles
            )

            rel_t1, rel_t2 = extract_relevant_ring_indices(
                t1_mol, t2_mol,
                row.ring_idx_t1_list_rdkit,
                row.ring_idx_t2_list_rdkit,
                row.rxmol_ring_idx_t1,
                row.rxmol_ring_idx_t2,
                row.nicszz_vals_t1,
                row.nicszz_vals_t2
            )

            ring_change = (
                extract_ring_bond_types(t1_mol, rel_t1)
                != extract_ring_bond_types(t2_mol, rel_t2)
            )

            neighbor_change = (
                extract_ring_bond_types_neighbor(t1_mol, rel_t1)
                != extract_ring_bond_types_neighbor(t2_mol, rel_t2)
            )

            results.append({
                "row": row.row,
                "ring_change": ring_change,
                "neighbor_change": neighbor_change
            })

        except Exception as e:
            print(f"Row {row.row} failed: {e}")
            continue

    df = df.merge(pd.DataFrame(results), on="row")


    # -------------------------
    # Subsets
    # -------------------------
    change_in_arom_r = df[df["ring_change"]].copy()

    change_near = df[
        df["neighbor_change"] & ~df["ring_change"]
    ].copy()

    no_change_near = df[
        ~df["neighbor_change"] & ~df["ring_change"]
    ].copy()


    change_in_arom_r, change_near = move_rows(
        change_in_arom_r,
        change_near,
        [227, 525]
    )

    # save subsets
    subsets = {
        "change_in_arom_r": change_in_arom_r,
        "no_change_in_arom_r": pd.concat(
            [change_near, no_change_near],
            ignore_index=True
        ),
    }



    # -------------------------
    # Config
    # -------------------------
    bit_lengths = [1024, 2048, 4096, 8192]
    bit_radii = [2, 3, 4]

    targets = ["ddg", "K"]

    BASE_DIR = PROCESSED_DIR
    registry = []
    dataset_registry_rows = []

    # -------------------------
    # Run experiments
    # -------------------------
    for subset_name, subset_df in subsets.items():

        print(f"\nProcessing subset: {subset_name}")

        subset_df = build_dataset(subset_df)
        subset_df = filter_by_row_pickle(subset_df, WIEDER_ROWS_PATH)

        subset_dir = BASE_DIR / subset_name

        for target in targets:

            print(f"\n=== Target: {target} ===")

            for radius in bit_radii:
                for n_bits in bit_lengths:

                    print(f"radius={radius}, bits={n_bits}")

                    X, df_valid = build_features(
                        subset_df,
                        radius,
                        n_bits,
                        homa_df
                    )

                    # -------------------------
                    # target cleanup
                    # -------------------------
                    if target not in df_valid.columns:
                        raise ValueError(f"{target} not found in dataframe")

                    df_valid = df_valid.dropna(subset=[target])

                    y = df_valid[target].values

                    # -------------------------
                    # feature names
                    # -------------------------

                    n_total = X.shape[1]
                    n_arom = 1
                    n_fp = n_total - n_arom

                    feature_names = (
                        ["relative_homa_difference"] +
                        [f"fp_bit_{i}" for i in range(n_fp)]
                    )

                    # -------------------------
                    # variants
                    # -------------------------

                    X_base = X

                    variants = {
                        "no_arom_diff": (
                            X_base[:, n_arom:],
                            [f"fp_bit_{i}" for i in range(n_fp)]
                        ),

                        "with_arom_diff": (
                            X_base,
                            feature_names
                        )
                    }

                    # -------------------------
                    # removing constant features + scaling + saving
                    # -------------------------

                    for variant_name, (X_curr, feature_names_curr) in variants.items():


                        X_curr, feature_names_curr, n_dropped = remove_constant_features(
                            X_curr,
                            feature_names_curr
                        )

                        print(
                            f"{variant_name}: dropped {n_dropped} "
                            f"constant features; "
                            f"{X_curr.shape[1]} features remaining"
                        )

                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_curr)

                        config = {
                            "subset": subset_name,
                            "radius": radius,
                            "n_bits": n_bits,
                            "target": target,
                            "n_features": X_scaled.shape[1],
                            "variant": variant_name,
                            "scaling": "standard_all_features"
                        }

                        dataset_id = (
                            f"{target}_r{radius}_b{n_bits}_"
                            f"{variant_name}_{subset_name}_v1"
                        )

                        out_dir = subset_dir / dataset_id


                        save_dataset(
                            X_scaled,
                            y,
                            feature_names_curr,
                            config,
                            out_dir
                        )
                        
                        save_dataset_metadata(
                            df_valid,
                            target,
                            out_dir
                        )


                        registry_subset = df_valid[
                            [
                                "row",
                                "t1_smiles",
                                "t2_smiles",
                                target,
                            ]
                        ].copy()
                        
                        registry_subset["dataset_id"] = dataset_id
                        registry_subset["subset"] = subset_name
                        registry_subset["radius"] = radius
                        registry_subset["n_bits"] = n_bits
                        registry_subset["variant"] = variant_name
                        registry_subset["target"] = target
                        
                        dataset_registry_rows.append(registry_subset)

                        registry.append({
                            "dataset_id": dataset_id,
                            "subset": subset_name,
                            "path": str(out_dir),
                            "radius": radius,
                            "n_bits": n_bits,
                            "n_features": X_scaled.shape[1],
                            "variant": variant_name,
                            "target": target,
                            "scaled": True
                        })

                        print(f"saved {subset_name}/{dataset_id}")

                    # -------------------------
                    # update registry
                    # -------------------------
                    registry_path = BASE_DIR / "registry.json"

                    with open(registry_path, "w") as f:
                        json.dump(registry, f, indent=2)


                    # -------------------------
                    # Save global SMILES registry
                    # -------------------------
                    if dataset_registry_rows:
                
                        registry_df = pd.concat(
                            dataset_registry_rows,
                            ignore_index=True
                        )
                
                        registry_df.to_csv(
                            BASE_DIR / "dataset_registry_with_smiles.csv",
                            index=False
                        )
                
                        print(
                            f"Saved global registry: "
                            f"{BASE_DIR / 'dataset_registry_with_smiles.csv'}"
                        )
                
  
                
                



            print(f"\nSaved dataset registry at {registry_path}")


if __name__ == "__main__":
    main()
    