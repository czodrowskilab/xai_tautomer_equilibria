import os
import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import MolDraw2DSVG

import shap

# =================================================
# IMPORTANT: XFP PATH SETUP (your missing part)
# =================================================
sys.dont_write_bytecode = True
sys.path.append('/mnt/czolab-storage-01/limmiche/x-fp/src')

from xfp.xfp_tree_explainer import XFPTreeExplainer


# =================================================
# BASE PATHS
# =================================================
BASE_DIR = Path("../../processed")
registry = pd.read_csv(BASE_DIR / "dataset_registry_with_smiles.csv")


# =================================================
# SHAP / MODEL (YOU MUST PROVIDE THIS)
# =================================================
# rf_model_loaded MUST exist in your environment
# (load from pickle or MLflow)
#
# with open("model.pkl", "rb") as f:
#     rf_model_loaded = pickle.load(f)

shap_explainer = shap.TreeExplainer(rf_model_loaded)


# =================================================
# CORE XFP PIPELINE WRAPPER
# =================================================
def build_xfp_features(df, nBits, bit_radius):

    xfp = XFPTreeExplainer(shap_explainer)

    # IMPORTANT STEP (this is what you forgot)
    xfp.fetch_fp_from_df(
        df,
        mol_col="molecule",
        n_bits=nBits,
        radius=bit_radius,
        canonicalize=False
    )

    return xfp.input_df


# =================================================
# FRAGMENT FUNCTION (your version kept)
# =================================================
def generate_frags(smi, atom_index, radius):

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index, useHs=True)

    atoms = {atom_index}
    for bidx in env:
        atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
        atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())

    smarts = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atoms),
        bondsToUse=env,
        allBondsExplicit=True,
        rootedAtAtom=atom_index,
        canonical=True,
    )

    return smarts


# =================================================
# BIT ANALYSIS (UNCHANGED IDEA, CLEANED)
# =================================================
def get_values_and_smiles_for_keys(df, keys, smiles_col="smiles", folder_name="analysis"):

    os.makedirs(folder_name, exist_ok=True)

    for key in keys:
        print(f"Processing bit {key}")

        frag_counts = Counter()

        for _, row in df.iterrows():

            smi = row[smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            for atom in mol.GetAtoms():
                try:
                    frag = generate_frags(smi, atom.GetIdx(), radius=2)
                    if frag:
                        frag_counts[(frag, 2)] += 1
                except:
                    continue

        # save results
        for i, ((smarts, r), count) in enumerate(frag_counts.items()):

            mol_smarts = Chem.MolFromSmarts(smarts)
            if mol_smarts is None:
                continue

            d2d = MolDraw2DSVG(600, 600)
            d2d.DrawMolecule(mol_smarts, legend=str(count))
            d2d.FinishDrawing()

            svg = d2d.GetDrawingText()

            safe = re.sub(r'[^A-Za-z0-9]', '_', smarts)[:80]

            out_path = os.path.join(
                folder_name,
                f"bit_{key}_frag_{i}_{count}_{safe}.svg"
            )

            with open(out_path, "w") as f:
                f.write(svg)


# =================================================
# MAIN LOOP OVER REGISTRY
# =================================================
subsets = registry["subset"].unique()
targets = registry["target"].unique()
radii = registry["radius"].unique()
n_bits_list = registry["n_bits"].unique()
variants = registry["variant"].unique()

print("Starting full pipeline...\n")

for subset_name in subsets:
    for target in targets:
        for radius in radii:
            for n_bits in n_bits_list:
                for variant in variants:

                    sel = registry[
                        (registry["subset"] == subset_name) &
                        (registry["target"] == target) &
                        (registry["radius"] == radius) &
                        (registry["n_bits"] == n_bits) &
                        (registry["variant"] == variant)
                    ]

                    if sel.empty:
                        continue

                    row = sel.iloc[0]
                    dataset_path = Path(row["path"])
                    dataset_id = row["dataset_id"]

                    print("\n" + "=" * 70)
                    print(f"Dataset: {dataset_id}")

                    # =================================================
                    # LOAD DATASET
                    # =================================================
                    df = pd.read_csv(dataset_path / "dataset_rows.csv")

                    df_t1 = df[["t1_smiles", target]].rename(columns={"t1_smiles": "smiles"})
                    df_t2 = df[["t2_smiles", target]].rename(columns={"t2_smiles": "smiles"})

                    df_long = pd.concat([df_t1, df_t2], ignore_index=True)
                    df_long.columns = ["smiles", "log_K"]

                    # =================================================
                    # ADD MOLECULE COLUMN (IMPORTANT FOR XFP)
                    # =================================================
                    df_long["molecule"] = df_long["smiles"].apply(
                        lambda x: Chem.MolFromSmiles(x)
                    )

                    df_long = df_long.dropna(subset=["molecule"])

                    # =================================================
                    # XFP FEATURE BUILD (YOUR MISSING STEP)
                    # =================================================
                    try:
                        xfp_df = build_xfp_features(
                            df_long,
                            nBits=n_bits,
                            bit_radius=radius
                        )

                        print("XFP input_df ready:", xfp_df.shape)

                    except Exception as e:
                        print(f"XFP failed for {dataset_id}: {e}")
                        continue

                    # =================================================
                    # ANALYSIS STEP
                    # =================================================
                    try:
                        get_values_and_smiles_for_keys(
                            xfp_df,
                            keys=[650, 807, 1917, 623, 1602],
                            smiles_col="smiles",
                            folder_name=f"analysis_{dataset_id}"
                        )
                    except Exception as e:
                        print(f"Analysis failed {dataset_id}: {e}")

print("\nDONE")