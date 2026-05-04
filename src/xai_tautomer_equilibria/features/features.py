"""
Feature engineering utilities for molecular ML models.
"""

from typing import List
import numpy as np
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
            block = BlockLogs() # Block all RDkit logging
            
            # Normalizing functional groups
            # https://molvs.readthedocs.io/en/latest/guide/standardize.html
            clean_mol = rdMolStandardize.Cleanup(mol) 
            # Get parents fragments
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

            # Neutralise
            if neutralize:
                uncharger = rdMolStandardize.Uncharger()
                uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            else:
                uncharged_parent_clean_mol = parent_clean_mol
            del block # Release logging block to previous state
            return(uncharged_parent_clean_mol)
        else:
            return None
    except Exception:
        return None



# -------------------------
# Fingerprints
# -------------------------
def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate Morgan fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )
    return np.array(fp)


# -------------------------
# Pair features (important for your task)
# -------------------------
def difference_fingerprint(fp1: np.ndarray, fp2: np.ndarray, binary: bool = False) -> np.ndarray:
    """
    Compute difference between two fingerprints.
    
    binary=True → absolute difference (0/1)
    binary=False → signed difference (-1,0,1)
    """
    if binary:
        return np.abs(fp1 - fp2)
    return fp1 - fp2


# -------------------------
# High-level feature builder
# -------------------------
def featurize_pair(
    smiles1: str,
    smiles2: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert SMILES pair into feature vector."""
    
    mol1 = standardize_smiles(smiles1)
    mol2 = standardize_smiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    fp1 = morgan_fingerprint(mol1, radius, n_bits)
    fp2 = morgan_fingerprint(mol2, radius, n_bits)

    diff = difference_fingerprint(fp1, fp2)

    return diff


# -------------------------
# Batch processing
# -------------------------
def featurize_dataframe(df, col1="t1-smiles", col2="t2-smiles"):
    """Apply feature generation to entire dataframe."""
    
    features = []
    valid_idx = []

    for i, (s1, s2) in enumerate(zip(df[col1], df[col2])):
        feat = featurize_pair(s1, s2)
        if feat is not None:
            features.append(feat)
            valid_idx.append(i)

    return np.array(features), df.iloc[valid_idx].reset_index(drop=True)