"""
Data loading and preprocessing utilities.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper
from rdkit.Chem import rdChemReactions


# Constants (move to config.py ideally)
DATA_DIR = Path("../data")
DDG_PATH = Path("../data/external/ddG_all_no_imag_SMD.pkl")


# -------------------------
# Loading
# -------------------------
def load_ddg_mapping(path: Path) -> dict:
    """Load ddG values and map row -> ddG."""
    with open(path, "rb") as f:
        ddg_data = pickle.load(f)

    return {row: ddg for ddg, row in ddg_data}


def load_tautobase(path: Path) -> pd.DataFrame:
    """Load tautobase dataset."""
    df = pd.read_csv(path)
    return df


# -------------------------
# Filtering
# -------------------------
def filter_water_only(df: pd.DataFrame, water_rows: set) -> pd.DataFrame:
    """Filter dataset for water solvent."""
    return df[df["row"].isin(water_rows)]


# -------------------------
# Feature merging
# -------------------------
def add_ddg(df: pd.DataFrame, ddg_map: dict) -> pd.DataFrame:
    """Add ddG column."""
    df = df.copy()
    df["ddG"] = df["row"].map(ddg_map)
    return df


# -------------------------
# Cleaning
# -------------------------
def drop_unused_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=columns, errors="ignore")


def remove_nan_targets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return df.dropna(subset=[target_col])



def standardize_mol(molecule):
    molecule = Chem.AddHs(molecule) 
    # Sanitize the molecule

    AllChem.SanitizeMol(molecule) #commented out bc it is done already below
    # AllChem.EmbedMolecule(molecule, randomSeed=1, maxAttempts = 5000) #https://github.com/rdkit/rdkit/issues/1433 #inserted maxattempts
    AllChem.EmbedMolecule(molecule, randomSeed=1)  #commented out bc it is done already below
    AllChem.UFFOptimizeMolecule(molecule) #commented out bc it is done already below
    molecule.GetConformer() #commented out bc it is done already below

    return molecule

_rxn_mapper = RXNMapper()  # reuse globally


def build_tautomer_pair(smi_t1, smi_t2):
    """
    Build RDKit molecules for a tautomer pair from a dataframe row.
    """

    rxns = [f"{smi_t1}>>{smi_t2}"]
    outputs = _rxn_mapper.get_attention_guided_atom_maps(rxns)

    out = outputs[0]
    rxn = rdChemReactions.ReactionFromSmarts(out["mapped_rxn"], useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)

    t1 = standardize_mol(trxn.GetReactants()[0])
    t2 = standardize_mol(trxn.GetProducts()[0])

    return (t1, t2)


RENAME_MAP = {
    "log_K_x": "K (T/E/M)",
    "smi_t1": "t1_smiles",
    "smi_t2": "t2_smiles",
    "murcko_smiles_x": "murcko_smiles",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataset column names into a consistent schema.
    """
    df = df.copy()

    missing = [c for c in RENAME_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df.rename(columns=RENAME_MAP)


def filter_by_row_pickle(df: pd.DataFrame, pickle_path: str, row_col: str = "row") -> pd.DataFrame:
    """
    Filter dataframe to only include rows listed in a pickle file.
    """
    with open(pickle_path, "rb") as f:
        valid_rows = pickle.load(f)

    df = df.copy()

    return df[df[row_col].astype(int).isin(valid_rows)].reset_index(drop=True)


# -------------------------
# Pipeline entrypoint
# -------------------------
def build_dataset(df):
    """Main dataset construction pipeline."""
    
    ddg_map = load_ddg_mapping(DDG_PATH)

    #df = load_tautobase(DATA_DIR / "tautobase.csv")

    df = add_ddg(df, ddg_map)
    df = remove_nan_targets(df, "ddG")
    df = remove_nan_targets(df, "t1-smiles")
    df = remove_nan_targets(df, "t2-smiles")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)