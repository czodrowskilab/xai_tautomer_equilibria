import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple

from xai_tautomer_equilibria.util.preprocess import (
    build_tautomer_pair,
    build_dataset,
    standardize_columns,
    filter_by_row_pickle,
)

from xai_tautomer_equilibria.util.ring_analysis import (
    extract_relevant_ring_indices,
    extract_ring_bond_types,
    extract_ring_bond_types_neighbor,
)

from xai_tautomer_equilibria.features.features import featurize_pair

from xai_tautomer_equilibria.config.paths import (
    TAUTOBASE_PATH,
    WIEDER_ROWS_PATH)

from xai_tautomer_equilibria.config.paths import PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



# -------------------------
# Build feature matrix
# -------------------------
def build_features(
    df: pd.DataFrame,
    radius: int,
    n_bits: int
) -> Tuple[np.ndarray, pd.DataFrame]:

    features = []
    valid_rows = []

    for row in df.itertuples(index=False):
        feat = featurize_pair(
          #  row._asdict()["t1-smiles"],
          #  row._asdict()["t2-smiles"],
            row.t1_smiles,
            row.t2_smiles,
            radius=radius,
            n_bits=n_bits
        )

        if feat is not None:
            features.append(feat)
            valid_rows.append(row)

    X = np.array(features)
    df_valid = pd.DataFrame(valid_rows)

    return X, df_valid

def save_dataset(X, y, feature_names, config, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # save arrays
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    # save metadata
    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

# -------------------------
# Main pipeline
# -------------------------
def main():

    # -------------------------
    # Load raw data
    # -------------------------
    df = pd.read_csv(TAUTOBASE_PATH)

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
            t1_mol, t2_mol = build_tautomer_pair(row.t1_smiles, row.t2_smiles)

            rel_t1, rel_t2 = extract_relevant_ring_indices(
                t1_mol, t2_mol,
                row.ring_idx_t1_list_rdkit,
                row.ring_idx_t2_list_rdkit,
                row.rxmol_ring_idx_t1,
                row.rxmol_ring_idx_t2,
                row.nicszz_vals_t1,
                row.nicszz_vals_t2
            )

            ring_change = extract_ring_bond_types(t1_mol, rel_t1) != \
                          extract_ring_bond_types(t2_mol, rel_t2)

            neighbor_change = extract_ring_bond_types_neighbor(t1_mol, rel_t1) != \
                              extract_ring_bond_types_neighbor(t2_mol, rel_t2)

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
    subsets = {
        "change_in_arom_r": df[df["ring_change"]],
        "change_near": df[df["neighbor_change"] & ~df["ring_change"]],
        "no_change_near": df[~df["neighbor_change"] & ~df["ring_change"]],
    }

    subsets["no_change_in_arom_r"] = pd.concat(
        [subsets["change_near"], subsets["no_change_near"]],
        ignore_index=True
    )
    # -------------------------
    # Feature configs
    # -------------------------
    bit_lengths = [1024, 2048, 4096, 8192]
    bit_radii = [2, 3, 4]

    BASE_DIR = PROCESSED_DIR  # keep consistent root

    registry = []

    # -------------------------
    # Run experiments
    # -------------------------
    for subset_name, subset_df in subsets.items():

        print(f"\nProcessing subset: {subset_name}")

        subset_df = build_dataset(subset_df)
        subset_df = filter_by_row_pickle(subset_df, WIEDER_ROWS_PATH)

        subset_dir = BASE_DIR / subset_name

        for radius in bit_radii:
            for n_bits in bit_lengths:

                print(f"radius={radius}, bits={n_bits}")

                X, df_valid = build_features(subset_df, radius, n_bits)

                # -------------------------
                # remove constant features
                # -------------------------
                X_df = pd.DataFrame(X)

                mask = X_df.nunique() != 1
                X_df = X_df.loc[:, mask]

                X = X_df.values
                y = df_valid["ddg"].values

                feature_names = X_df.columns.tolist()

                config = {
                    "subset": subset_name,
                    "radius": radius,
                    "n_bits": n_bits,
                    "target": "ddg",
                    "n_features": X.shape[1]
                }

                dataset_id = f"r{radius}_b{n_bits}_v1"
                out_dir = subset_dir / dataset_id

                save_dataset(X, y, feature_names, config, out_dir)

                # -------------------------
                # registry entry (CRITICAL for ML systems)
                # -------------------------
                registry.append({
                    "dataset_id": dataset_id,
                    "subset": subset_name,
                    "path": str(out_dir),
                    "radius": radius,
                    "n_bits": n_bits,
                    "n_features": X.shape[1]
                })

                print(f"saved {subset_name}/{dataset_id}")
                
                registry_path = BASE_DIR / "registry.json"

                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2)

                print(f"\nSaved dataset registry at {registry_path}")


if __name__ == "__main__":
    main()