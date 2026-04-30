import pandas as pd
import numpy as np
import pickle

from util.preprocess import build_tautomer_pair, build_dataset, standardize_columns, filter_by_row_pickle
from util.ring_analysis import extract_relevant_ring_indices, extract_ring_bond_types, extract_ring_bond_types_neighbor
from features.features import featurize_pair


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
            row._asdict()["t1-smiles"],
            row._asdict()["t2-smiles"],
            radius=radius,
            n_bits=n_bits
        )

        if feat is not None:
            features.append(feat)
            valid_rows.append(row)

    X = np.array(features)
    df_valid = pd.DataFrame(valid_rows)

    return X, df_valid



# -------------------------
# Main pipeline
# -------------------------
def main():

    # -------------------------
    # Load data
    # -------------------------
    PATH = "./data/tautobase_edited.csv"
    df = pd.read_csv(PATH)

    # -------------------------
    # Standardize schema
    # -------------------------
    df = standardize_columns(df)

    # -------------------------
    # Build ring features
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

    results_df = pd.DataFrame(results)
    df = df.merge(results_df, on="row")

    # -------------------------
    # Create subsets
    # -------------------------
    mols_w_change = df[df["ring_change"]]
    mols_no_change = df[~df["ring_change"]]

    mols_change_near = mols_no_change[mols_no_change["neighbor_change"]]
    mols_no_change_near = mols_no_change[~mols_no_change["neighbor_change"]]
    mols_w_arom_combined =  pd.concat([mols_change_near, mols_no_change_near], ignore_index=True)
    subsets = {
        "change": mols_w_change,
        "change_near": mols_change_near,
        "no_change_near": mols_no_change_near,
        "mols_w_arom_combined": mols_w_arom_combined
    }

    # -------------------------
    # Feature configs
    # -------------------------
    bit_lengths = [1024, 2048, 4096, 8192]
    bit_radii = [2, 3, 4]


    # -------------------------
    # Run experiments
    # -------------------------
    for name, subset_df in subsets.items():

        print(f"\nSubset: {name}")

        subset_df = build_dataset(subset_df)
        subset_df = filter_by_row_pickle(subset_df, "wieder_rows.pkl")

        for radius in bit_radii:
            for n_bits in bit_lengths:

                print(f"radius={radius}, bits={n_bits}")

                X, df_valid = build_features(subset_df, radius, n_bits)

                y = df_valid["ddG"].values

                np.save(f"./data/X_{name}_r{radius}_b{n_bits}.npy", X)
                np.save(f"./data/y_{name}_r{radius}_b{n_bits}.npy", y)

                print(f"saved {name}_r{radius}_b{n_bits}")



if __name__ == "__main__":
    main()