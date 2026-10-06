"""
Aromaticity feature utilities.

This module provides:
- Aromaticity values (NICSzz, HOMA, MCI)
- RDKit-based aromaticity checks
"""

from typing import List
from rdkit import Chem


# -------------------------
# Threshold-based aromaticity
# -------------------------
def classify_nics_aromaticity(
    values_t1: List[float],
    values_t2: List[float],
    threshold: float = 5.5,
) -> int:
    """
    Classify aromaticity change using NICS values.

    Returns:
        1 → change in aromaticity
        2 → no aromatic rings
        3 → aromatic → aromatic
        0 → fallback
    """

    # Check change
    for v1, v2 in zip(values_t1, values_t2):
        cond1 = v1 > threshold
        cond2 = v2 > threshold
        if (cond1 or cond2) and not (cond1 and cond2):
            return 1

    # Aromatic → aromatic
    if any(v1 < threshold and v2 < threshold for v1, v2 in zip(values_t1, values_t2)):
        return 3

    # No aromatic rings
    if all(v1 > threshold for v1 in values_t1) and all(v2 > threshold for v2 in values_t2):
        return 2

    return 0


def classify_homa_aromaticity(
    values_t1: List[float],
    values_t2: List[float],
    threshold: float = 0.5,
) -> int:
    """Same logic for HOMA index."""

    for v1, v2 in zip(values_t1, values_t2):
        cond1 = v1 < threshold
        cond2 = v2 < threshold
        if (cond1 or cond2) and not (cond1 and cond2):
            return 1

    if any(v1 > threshold and v2 > threshold for v1, v2 in zip(values_t1, values_t2)):
        return 3

    if all(v1 < threshold for v1 in values_t1) and all(v2 < threshold for v2 in values_t2):
        return 2

    return 0


# -------------------------
# RDKit-based aromaticity
# -------------------------
def is_ring_aromatic(mol: Chem.Mol, atom_indices: List[int]) -> bool:
    """
    Check if a set of atoms forms an aromatic ring (MMFF model).
    """

    mol = Chem.Mol(mol)  # copy
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MMFF94)

    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        if not atom.GetIsAromatic():
            return False

    return True


# -------------------------
# Pair comparison 
# -------------------------
def compare_ring_aromaticity(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    ring_indices1: List[List[int]],
    ring_indices2: List[List[int]],
) -> int:
    """
    Compare aromaticity between two molecules.

    Returns:
        1 → change
        2 → non-aromatic
        3 → aromatic
        0 → unknown
    """

    sign_change = False
    aromatic_pair = False
    non_aromatic_pair = False

    for r1, r2 in zip(ring_indices1, ring_indices2):

        aro1 = is_ring_aromatic(mol1, r1)
        aro2 = is_ring_aromatic(mol2, r2)

        if aro1 != aro2:
            sign_change = True
            break

        if aro1 and aro2:
            aromatic_pair = True

        if not aro1 and not aro2:
            non_aromatic_pair = True

    if sign_change:
        return 1
    if aromatic_pair:
        return 3
    if non_aromatic_pair:
        return 2

    return 0