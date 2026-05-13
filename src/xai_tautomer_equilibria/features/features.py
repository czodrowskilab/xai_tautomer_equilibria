"""
Feature engineering utilities for molecular ML models.
"""

from typing import List
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs


# -------------------------
# Molecule processing
# -------------------------
def standardize_smiles(smiles: str, neutralize: bool = True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            block = BlockLogs()  # Block all RDkit logging

            # Normalizing functional groups
            clean_mol = rdMolStandardize.Cleanup(mol)

            # Get parent fragments
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

            # Neutralise
            if neutralize:
                uncharger = rdMolStandardize.Uncharger()
                uncharged_parent_clean_mol = uncharger.uncharge(
                    parent_clean_mol
                )
            else:
                uncharged_parent_clean_mol = parent_clean_mol

            del block
            return uncharged_parent_clean_mol

        return None

    except Exception:
        return None


# -------------------------
# Fingerprints
# -------------------------
def morgan_fingerprint(
    mol,
    radius: int = 2,
    n_bits: int = 2048
) -> np.ndarray:
    """Generate Morgan fingerprint."""

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    return np.array(fp)


# -------------------------
# Pair features
# -------------------------
def difference_fingerprint(
    fp1: np.ndarray,
    fp2: np.ndarray,
    binary: bool = False
) -> np.ndarray:
    """
    Compute difference between two fingerprints.

    binary=True → absolute difference (0/1)
    binary=False → signed difference (-1,0,1)
    """

    if binary:
        return np.abs(fp1 - fp2)

    return fp1 - fp2


# -------------------------
# Aromaticity features
# -------------------------
def load_aromaticity_features(
    row_id,
    homa_df,
    mci_df,
    nics_df
) -> np.ndarray:

    def get_value(df, colname):

        match = df.loc[df["row"] == row_id]

        if match.empty:
            return np.nan

        return match[colname].values[0]

    homa_val = get_value(homa_df, "relative_homa_difference")
    mci_val = get_value(mci_df, "relative_mci_difference")
    nics_val = get_value(nics_df, "relative_nicszz_difference")

    values = np.array([homa_val, mci_val, nics_val], dtype=float)

    if np.isnan(values).any():
        return None

    return values
# -------------------------
# High-level feature builder
# -------------------------
def featurize_pair(
    smiles1: str,
    smiles2: str,
    aromaticity_features: np.ndarray = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Convert SMILES pair into feature vector.
    """

    mol1 = standardize_smiles(smiles1)
    mol2 = standardize_smiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    fp1 = morgan_fingerprint(mol1, radius, n_bits)
    fp2 = morgan_fingerprint(mol2, radius, n_bits)

    diff = difference_fingerprint(fp1, fp2)

    # Append aromaticity descriptors  
    if aromaticity_features is not None:
        diff = np.concatenate([diff, aromaticity_features])


    return diff


# -------------------------
# Batch processing
# -------------------------
def featurize_dataframe(
    df,
    col1="t1-smiles",
    col2="t2-smiles",
    row_col="row",
    homa_csv="homa_values.csv",
    mci_csv="mci_values.csv",
    nics_csv="nicszz_values.csv",
):
    """
    Apply feature generation to entire dataframe.
    """

    # Load aromaticity tables
    homa_df = pd.read_csv(homa_csv)
    mci_df = pd.read_csv(mci_csv)
    nics_df = pd.read_csv(nics_csv)

    features = []
    valid_idx = []

    for i, (row_id, s1, s2) in enumerate(
        zip(df[row_col], df[col1], df[col2])
    ):

        aromaticity_feat = load_aromaticity_features(
            row_id,
            homa_df,
            mci_df,
            nics_df
        )

        # Skip rows with missing aromaticity values
        if aromaticity_feat is None:
            continue

        feat = featurize_pair(
            s1,
            s2,
            aromaticity_features=aromaticity_feat
        )

        if feat is not None:
            features.append(feat)
            valid_idx.append(i)

    return (
        np.array(features),
        df.iloc[valid_idx].reset_index(drop=True)
    )