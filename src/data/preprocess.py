"""
Data loading and preprocessing utilities.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Constants (move to config.py ideally)
DATA_DIR = Path("./data")
DDG_PATH = Path("./dG_calculation/data/ddG_all_no_imag_SMD.pkl")


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


# -------------------------
# Pipeline entrypoint
# -------------------------
def build_dataset():
    """Main dataset construction pipeline."""
    
    ddg_map = load_ddg_mapping(DDG_PATH)

    df = load_tautobase(DATA_DIR / "tautobase.csv")

    df = add_ddg(df, ddg_map)
    df = remove_nan_targets(df, "ddG")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)