

from typing import List, Tuple
import ast
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rxnmapper import RXNMapper
from typing import Dict, Tuple, Iterable
from xai_tautomer_equilibria.util.preprocess import (
    standardize_mol,
    build_tautomer_pair,
)
from xai_tautomer_equilibria.util.aromaticity import is_ring_aromatic


def extract_relevant_ring_indices(
    t1_mol="t1-mol",
    t2_mol="t2_mol",
    rdkit_at_ind_1="ring_idx_t1_list_rdkit",
    rdkit_at_ind_2="ring_idx_t2_list_rdkit",
    reaction_idx_list_t1="rxmol_ring_idx_t1",
    reaction_idx_list_t2="rxmol_ring_idx_t2",
    aro_vals_1="nicszz_vals_t1",
    aro_vals_2="nicszz_vals_t2",
):
    """
    Compute relevant ring atom indices for both tautomers.

    Returns:
        (list_t1, list_t2)
    """

    relevant_ring_idx_t1 = []
    relevant_ring_idx_t2 = []
    aro_list1 = ast.literal_eval(aro_vals_1)
    aro_list2 = ast.literal_eval(aro_vals_2)
    rdkit_at_ind_1 = ast.literal_eval(rdkit_at_ind_1)
    rdkit_at_ind_2 = ast.literal_eval(rdkit_at_ind_2)

    reaction_idx_list_t1 = ast.literal_eval(reaction_idx_list_t1)
    reaction_idx_list_t2 = ast.literal_eval(reaction_idx_list_t2)



    zipped_list_a = [(x, y, z) for x, y, z in zip(aro_list1, reaction_idx_list_t1, rdkit_at_ind_1)]
    zipped_list_b = [(x, y, z) for x, y, z in zip(aro_list2, reaction_idx_list_t2, rdkit_at_ind_2)]
    for tuple_a in zipped_list_a:
        for tuple_b in zipped_list_b:
            if sorted(tuple_a[1]) == sorted(tuple_b[1]):  # if reaction at ids are the same (same rings in both mols)
                if is_ring_aromatic(t1_mol, tuple_a[2]) == True or  is_ring_aromatic(t2_mol, tuple_b[2]) == True:


                    relevant_ring_idx_t1.append(tuple_a[2])
                    relevant_ring_idx_t2.append(tuple_b[2])




    
    flattened_relevant_ring_idx_t1 = [int(item) for sublist in relevant_ring_idx_t1 for item in sublist]
    flattened_relevant_ring_idx_t2 = [int(item) for sublist in relevant_ring_idx_t2 for item in sublist]


    return (flattened_relevant_ring_idx_t1, flattened_relevant_ring_idx_t2)



            
def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol



def moveAtomMapsToNotes(m):
    for at in m.GetAtoms():
        if at.GetAtomMapNum():
            at.SetProp("atomNote",str(at.GetAtomMapNum()))

def extract_ring_bond_types(molecule: Chem.Mol, 
                            relevant_atoms_list: Iterable[int]
                            ) -> Dict[Tuple[int, int], Chem.rdchem.BondType]:
    """
    Get the bond types between the ring atoms and their neighbors.

    Parameters:
    - molecule (rdkit.Chem.rdchem.Mol): The molecule object.
    - relevant_atoms_list (list of int): List of atom indices.

    Returns:
    - dict: A dictionary where keys are tuples of atom indices and values are bond types.
    """

    moveAtomMapsToNotes(molecule) 
    Chem.Kekulize(molecule)
    bond_types = {}
    
    for at_idx in relevant_atoms_list:
        atom = molecule.GetAtomWithIdx(at_idx)
        for neighbor in atom.GetNeighbors():
                    # Check if the neighbor is not hydrogen
            if neighbor.GetAtomicNum() == 1:  # Skip hydrogen atoms
                continue
            neighbor_idx = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(at_idx, neighbor_idx)
            bond_type = bond.GetBondType()  # Gets bond type as a double
            bond_key = tuple(sorted([at_idx, neighbor_idx]))
            bond_types[bond_key] = bond_type

    return bond_types


def extract_ring_bond_types_neighbor(molecule: Chem.Mol, 
                                     relevant_atoms_list: Iterable[int]
                                     ) -> Dict[Tuple[int, int], Chem.rdchem.BondType]:
    """
    Get the bond types between the ring atoms and their neighbors.
    
    Parameters:
    - molecule (rdkit.Chem.rdchem.Mol): The molecule object.
    - relevant_atoms_list (list of int): List of ring atom indices.
    
    Returns:
    - dict: A dictionary where keys are tuples of atom indices and values are bond types.
    """
    moveAtomMapsToNotes(molecule)  
    Chem.Kekulize(molecule)
    bond_types = {}
    
    for at_idx in relevant_atoms_list:
        atom = molecule.GetAtomWithIdx(at_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in relevant_atoms_list and neighbor.GetAtomicNum() != 1:  # Avoid hydrogen atoms
                bond = molecule.GetBondBetweenAtoms(at_idx, neighbor_idx)
                if bond:
                    bond_type = bond.GetBondType()  # Gets bond type as a double
                    bond_key = tuple(sorted([at_idx, neighbor_idx]))
                    bond_types[bond_key] = bond_type

                for neighbor_2 in neighbor.GetNeighbors():
                    neighbor_2_idx = neighbor_2.GetIdx()
                    if neighbor_2_idx not in relevant_atoms_list and neighbor_2.GetAtomicNum() != 1:  # Avoid hydrogen atoms
                        bond = molecule.GetBondBetweenAtoms(neighbor_idx, neighbor_2_idx)
                        if bond:
                            bond_type = bond.GetBondType() 
                            bond_key = tuple(sorted([neighbor_idx, neighbor_2_idx]))
                            bond_types[bond_key] = bond_type

    return bond_types

