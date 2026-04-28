import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.rdBase import BlockLogs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import AddHs


# In[17]:


#loading ddG data:
import pickle

bit_lengths = [1024, 2048, 4096, 8192]
bit_radii = [2, 3, 4]
data_subset_list = ["mols_w_arom_and_change", "mols_w_arom_and_change_near_rel_system", "mols_w_arom_and_no_change_near_rel_system", "mols_w_arom_combined"]

for subs in data_subset_list: 

    data_subset = subs

    for bit_radius in bit_radii:
        bit_radius = bit_radius

        for bit_lenght in bit_lengths:
            bit_lenght = bit_lenght
    
            #data_subset = "mols_w_arom_and_change_near_rel_system" #mols_w_arom_and_no_change_near_rel_system, mols_w_arom_and_change_near_rel_system, mols_w_arom_and_change, mols_w_arom_combined, mols_w_arom_and_change
            
           # bit_lenght = 512
        
          #  bit_radius = 2
            
            solvation_model = "smd" # gas smd
            
            if solvation_model == "smd":    
                filter_water = True # True # filter mols only solvated in water
            else:
                filter_water = False
            
            manual_filter_on = True
            
            arom_indice = "nicszz"#"HOMA"
            
            mols_with_change_near_ring = True # False # True
            
            inclue_row = True #for outlier analysis we need to include the sample name/row/smiles code for the outlier
            
            # which subset should the model train on:
            subset = 'arom_to_arom' #'arom_to_arom' #'non_arom_to_arom_rows' #"no_arom_mols" #"mols_w_arom_rows" # non_arom_to_arom_rows
            
            
            simple_bits = False #True # chose simple difference FP (0 = bit is the same; 1 = bit is differenct in both tautomers)
            
            
            # File path to load the list
            #if subset == "no_arom_mols":
            #    file_path = "./data/ddG_and_failed_files_list_using_correct_value_plus_non_ring_mols/ddG_all.pkl"
            #else:
            #    file_path = "./data/ddG_and_failed_files_list_using_correct_value/ddG_all.pkl"
            
            
            #file_path = "./data/ddG_and_failed_files_list_using_correct_value_plus_non_ring_mols_no_imag/ddG_all.pkl" # b3lyp model gas
            
            
            if solvation_model == "smd":
            
                file_path = "./dG_calculation/data/ddG_all_no_imag_SMD.pkl" # SMD model
            
            if solvation_model == "gas":
            
                file_path = "./dG_calculation/data/ddG_all_no_imag.pkl" # gas model
            
            # Load the list from the file
            with open(file_path, 'rb') as file:
                ddG_and_row = pickle.load(file)
            
            ddG_row_dic = {}
            
            for ddG, row in ddG_and_row:
                ddG_row_dic[row] = ddG
            
            ddG_row_dic
            


          
            
            def transform(smiles, neutralize=True):
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
            
            
            
            
            # #### MorganFP generator
            # generates morgan finger prints with a certain radius 'mfp_r' to set in the Parameter section
            
            # In[21]:
            
            
            def MFP_generator(mol):
                return(AllChem.GetMorganFingerprintAsBitVect(mol, radius= bit_radius, nBits=bit_lenght))
            

            ####################### read data set: #######################
            
            tautobase = pd.read_csv("./data/b3lyp_tautobase_subset.txt")
            
            #tautobase_experimental = pd.read_csv("./data/Tautobase_edited.txt", sep = '\t') # data base without aromaticity (HOMA etc.) values
            
            
            #data base with arom values:
            
            if arom_indice == "HOMA":
                tautobase_experimental = pd.read_csv("./data/with_arom_vals/tautobase_sum_homa_vals.csv") #HOMA values
            else:
                #tautobase_experimental = pd.read_csv("./data/with_arom_vals/tautobase_sum_nicszz_vals_no_imag.csv") #NICS values ,b3lyp gas
                tautobase_experimental = pd.read_csv(f"./nicszz/nicszz_values_generation/tautobase_sum_nicszz_vals_no_imag_{solvation_model}.csv") #NICS values SMD
            
                
            tautobase_experimental = tautobase_experimental.rename(columns={'log_K_x': 'K (T/E/M)'})
            tautobase_experimental = tautobase_experimental.rename(columns={'smi_t1': 't1-smiles'})
            tautobase_experimental = tautobase_experimental.rename(columns={'smi_t2': 't2-smiles'})
            tautobase_experimental = tautobase_experimental.rename(columns={'murcko_smiles_x': 'murcko_smiles'})
            
            

        
            
            ####################### get rows with water only #######################
            tautobase_with_rings_orig = pd.read_csv("./preopt_xtb_crest/df_data/tautobase_w_rings.csv")
            tautobase_with_rings_orig.columns
            
            
            # In[27]:
            
            
            tautobase_with_rings_water_only = tautobase_with_rings_orig[tautobase_with_rings_orig["Solvent"] == "Water"]
            len(tautobase_with_rings_water_only)
            
            
            # In[28]:
            
            
            water_rows = set(tautobase_with_rings_water_only["Row Nr"])
            
            
            # In[29]:
            
            
            if filter_water == True:
                tautobase_experimental = tautobase_experimental[tautobase_experimental["row"].isin(water_rows)]



            
            import rdkit
            import ast
            
            def standardization(molecule):
                mol = Chem.AddHs(molecule) #commented out bc it is done already below
                # Sanitize the molecule
            
                AllChem.SanitizeMol(mol) #commented out bc it is done already below
                # AllChem.EmbedMolecule(molecule, randomSeed=1, maxAttempts = 5000) #https://github.com/rdkit/rdkit/issues/1433 #inserted maxattempts
                AllChem.EmbedMolecule(mol, randomSeed=1)  #commented out bc it is done already below
                AllChem.UFFOptimizeMolecule(mol) #commented out bc it is done already below
                mol.GetConformer() #commented out bc it is done already below
            
                return mol
            
            def moveAtomMapsToNotes(m):
                for at in m.GetAtoms():
                    if at.GetAtomMapNum():
                        at.SetProp("atomNote",str(at.GetAtomMapNum()))
            
            
            def all_ring_idx_rxnmapper(mol):
                ring_info = Chem.GetSymmSSSR(mol)
                return [list(ring_idx_list) for ring_idx_list in ring_info]
            
            
          
            def check_aromaticity(atom_id_list, mol):
                aromatic_list = []
                molecule = standardization(mol)
                moveAtomMapsToNotes(molecule)
            
                Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_DEFAULT) #added this line
            
              #  Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_CUSTOM) 
            
            
             #   for atoms in ast.literal_eval(atom_id_list):
            
                for atoms in atom_id_list:
                    ring_list = []
                    is_aromatic_ring = True
            
                    for atom in atoms:
                        atom = int(atom)
                        at = molecule.GetAtomWithIdx(atom)
                        reaction_idx = at.GetProp("atomNote")
                        if not at.GetIsAromatic():
                            is_aromatic_ring = False
                            break
                        for neighbor in at.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if int(neighbor_idx) not in atoms:
                                bond = molecule.GetBondBetweenAtoms(atom, neighbor_idx)
                                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                                    is_aromatic_ring = False
                                    break
                        if not is_aromatic_ring:
                            break
                        ring_list.append(int(reaction_idx))
                    
                    if is_aromatic_ring:
                        aromatic_list.append(ring_list)
                    else:
                        aromatic_list.append([])  # Add an empty list if the ring is not fully aromatic
            
                return aromatic_list
            
            
            
            import ast


            
            def nicszz_arom_check(arom_list_t1, arom_list_t2):
                list1 = ast.literal_eval(arom_list_t1)
                list2 = ast.literal_eval(arom_list_t2)
                
                # First check: Sign change
                sign_change_found = False
                for i, ii in zip(list1, list2):
                    condition_i = i > 5.5
                    condition_ii = ii > 5.5
                    sign_change = (condition_i or condition_ii) and not (condition_i and condition_ii)
                    if sign_change:
                        sign_change_found = True
                        break
                
                if sign_change_found:
                    return 1
                
                # Second check: aro to aro
                aro_to_aro = False
                for i, ii in zip(list1, list2):
                    if i < 5.5 and ii < 5.5:
                        aro_to_aro = True
                        break
                
                if aro_to_aro:
                    return 3
                
                # Third check: no arom involved
                no_arom_involved = all(i > 5.5 for i in list1) and all(ii > 5.5 for ii in list2)
                
                if no_arom_involved:
                    return 2
                
                return 0
            

            
            def homa_arom_check(arom_list_t1, arom_list_t2):
                list1 = ast.literal_eval(arom_list_t1)
                list2 = ast.literal_eval(arom_list_t2)
                
                # First check: Sign change
                sign_change_found = False
                for i, ii in zip(list1, list2):
                    condition_i = i < 0.5
                    condition_ii = ii < 0.5
                    sign_change = (condition_i or condition_ii) and not (condition_i and condition_ii)
                    if sign_change:
                        sign_change_found = True
                        break
                
                if sign_change_found:
                    return 1
                
                # Second check: aro to aro
                aro_to_aro = False
                for i, ii in zip(list1, list2):
                    if i > 0.5 and ii > 0.5:
                        aro_to_aro = True
                        break
                
                if aro_to_aro:
                    return 3
                
                # Third check: no arom involved
                no_arom_involved = all(i < 0.5 for i in list1) and all(ii < 0.5 for ii in list2)
                
                if no_arom_involved:
                    return 2
                
                return 0
            
            
            
            def check_aromaticity_MMFF(atom_id_list, mol):
            
                molecule = standardization(mol)
               # moveAtomMapsToNotes(molecule)
            
                Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_MMFF94) #added this line
                #Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_DEFAULT) #added this line
            
              #  Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_CUSTOM) 
            
            
             #   for atoms in ast.literal_eval(atom_id_list):
            
            
                is_aromatic_ring = True
            
                for atom in atom_id_list:
                    atom = int(atom)
                    at = molecule.GetAtomWithIdx(atom)
                ##       reaction_idx = at.GetProp("atomNote")
                    if not at.GetIsAromatic():
                        is_aromatic_ring = False
                        break
                return is_aromatic_ring
            
            
            # In[39]:
            
            
            from rxnmapper import RXNMapper
            from rdkit.Chem import rdChemReactions
            import os
            
            arom_check = []
            
                    
          
            
            def compare_aromaticity(aro_list1, reaction_idx_list_t1, rdkit_at_ind_1,
                                     aro_list2, reaction_idx_list_t2, rdkit_at_ind_2,
                                     t1_mol, t2_mol):
                
                zipped_list_a = [(x, y, z) for x, y, z in zip(aro_list1, reaction_idx_list_t1, rdkit_at_ind_1)]
                zipped_list_b = [(x, y, z) for x, y, z in zip(aro_list2, reaction_idx_list_t2, rdkit_at_ind_2)]
                
            
                sign_change_found = False
                aromatic_pair_found = False
                non_aromatic_pair_found = False
                
                for tuple_a in zipped_list_a:
                    for tuple_b in zipped_list_b:
                        if sorted(tuple_a[1]) == sorted(tuple_b[1]):
                            if check_aromaticity_MMFF(tuple_a[2], t1_mol) != check_aromaticity_MMFF(tuple_b[2], t2_mol):
                                sign_change_found = True
                                break  # Exit inner loop early if one mismatch is found
                            if check_aromaticity_MMFF(tuple_a[2], t1_mol) and check_aromaticity_MMFF(tuple_b[2], t2_mol):
                                aromatic_pair_found = True
                            if not check_aromaticity_MMFF(tuple_a[2], t1_mol) and not check_aromaticity_MMFF(tuple_b[2], t2_mol):
                                non_aromatic_pair_found = True
                    if sign_change_found:
                        break  # Exit outer loop early if one mismatch is found
                
                if sign_change_found:
                    return 1
                elif aromatic_pair_found:
                    return 3
                elif non_aromatic_pair_found:
                    return 2
                else:
                    return 0 
            
                
            ####################### compare aromaticity:


          
            
            if arom_indice == "HOMA":
            
                for arom_vals_t1, arom_vals_t2 in zip(tautobase_experimental["homa_vals_t1"], tautobase_experimental["homa_vals_t2"]):
                    arom_change = nicszz_arom_check(arom_vals_t1, arom_vals_t2)
                    arom_check.append(arom_change)
                    
            #elif arom_indice == "nicszz":
            #    for arom_vals_t1, arom_vals_t2 in zip(tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"]):
            #        arom_change = nicszz_arom_check(arom_vals_t1, arom_vals_t2)
            #        #arom_change = homa_arom_check(arom_vals_t1, arom_vals_t2)
            #        arom_check.append(arom_change)
            else: 
                #for aro_vals_1, aro_vals_2, rdkit_at_ind_1, rdkit_at_ind_2 in zip(tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"], tautobase_experimental["ring_idx_t1_list_rdkit"], tautobase_experimental["ring_idx_t2_list_rdkit"]):
                for smi_t1, smi_t2, rdkit_at_ind_1, rdkit_at_ind_2, reaction_idx_list_t1, reaction_idx_list_t2, aro_vals_1, aro_vals_2 in zip( tautobase_experimental["t1-smiles"], tautobase_experimental["t2-smiles"], tautobase_experimental["ring_idx_t1_list_rdkit"], tautobase_experimental["ring_idx_t2_list_rdkit"], tautobase_experimental["rxmol_ring_idx_t1"], tautobase_experimental["rxmol_ring_idx_t2"], tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"]):
                    
                    rxns = [f'{smi_t1}>>{smi_t2}'] 
                    rxn_mapper = RXNMapper()
                    outputs = rxn_mapper.get_attention_guided_atom_maps(rxns)
                    for out in outputs:
                        smiles_reaction = out['mapped_rxn']
                        rxn = rdChemReactions.ReactionFromSmarts(smiles_reaction,useSmiles=True)
                        trxn = rdChemReactions.ChemicalReaction(rxn)
                        t2_mol = trxn.GetProducts()[0]
                        t2_mol = standardization(t2_mol)
                        t1_mol = trxn.GetReactants()[0]
                        t1_mol = standardization(t1_mol)
                    
                    
            
                    aro_list1 = ast.literal_eval(aro_vals_1)
                    aro_list2 = ast.literal_eval(aro_vals_2)
                    rdkit_at_ind_1 = ast.literal_eval(rdkit_at_ind_1)
                    rdkit_at_ind_2 = ast.literal_eval(rdkit_at_ind_2)
            
                    reaction_idx_list_t1 = ast.literal_eval(reaction_idx_list_t1)
                    reaction_idx_list_t2 = ast.literal_eval(reaction_idx_list_t2)
            
                    arom_change = compare_aromaticity(aro_list1, reaction_idx_list_t1, rdkit_at_ind_1,
                                            aro_list2, reaction_idx_list_t2, rdkit_at_ind_2,
                                            t1_mol, t2_mol)
                    arom_check.append(arom_change)
            print(arom_check)
            
            

            
            # In[40]:
            
            
            tautobase_experimental["arom_change"] = arom_check


          
            ####################### aromaticity in ring?: relevant atoms list: if either aro1 or aro2 negative: append rdkit at ind 1 and 2 #######################
            
            relevant_ring_idx_list_t1 = []
            
            relevant_ring_idx_list_t2 = []
            
            #for aro_vals_1, aro_vals_2, rdkit_at_ind_1, rdkit_at_ind_2 in zip(tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"], tautobase_experimental["ring_idx_t1_list_rdkit"], tautobase_experimental["ring_idx_t2_list_rdkit"]):
            for smi_t1, smi_t2, rdkit_at_ind_1, rdkit_at_ind_2, reaction_idx_list_t1, reaction_idx_list_t2, aro_vals_1, aro_vals_2 in zip( tautobase_experimental["t1-smiles"], tautobase_experimental["t2-smiles"],tautobase_experimental["ring_idx_t1_list_rdkit"], tautobase_experimental["ring_idx_t2_list_rdkit"], tautobase_experimental["rxmol_ring_idx_t1"], tautobase_experimental["rxmol_ring_idx_t2"], tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"]):
                
                
                
                rxns = [f'{smi_t1}>>{smi_t2}'] 
                rxn_mapper = RXNMapper()
                outputs = rxn_mapper.get_attention_guided_atom_maps(rxns)
                for out in outputs:
                    smiles_reaction = out['mapped_rxn']
                    rxn = rdChemReactions.ReactionFromSmarts(smiles_reaction,useSmiles=True)
                    trxn = rdChemReactions.ChemicalReaction(rxn)
                    t2_mol = trxn.GetProducts()[0]
                    t2_mol = standardization(t2_mol)
                    t1_mol = trxn.GetReactants()[0]
                    t1_mol = standardization(t1_mol)
                    
            
                
                    
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
                                if check_aromaticity_MMFF(tuple_a[2], t1_mol) == True or  check_aromaticity_MMFF(tuple_b[2], t2_mol) == True:
            
            
                                    relevant_ring_idx_t1.append(tuple_a[2])
                                    relevant_ring_idx_t2.append(tuple_b[2])
            
            
            
            
                    
                    flattened_relevant_ring_idx_t1 = [int(item) for sublist in relevant_ring_idx_t1 for item in sublist]
                    flattened_relevant_ring_idx_t2 = [int(item) for sublist in relevant_ring_idx_t2 for item in sublist]
            
                    relevant_ring_idx_list_t1.append(flattened_relevant_ring_idx_t1)
                    relevant_ring_idx_list_t2.append(flattened_relevant_ring_idx_t2)
            
                        
            tautobase_experimental["relevant_ring_idx_t1"] = relevant_ring_idx_list_t1
            tautobase_experimental["relevant_ring_idx_t2"] = relevant_ring_idx_list_t2
            
            
            
            # Filtering out rows with empty lists
            mols_atleast_one_aromatic = tautobase_experimental[tautobase_experimental['relevant_ring_idx_t2'].apply(lambda x: len(x) > 0)]
            




################ count hydrogens : do i need this:

         
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            
            
            ######
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit.Chem import rdDepictor
            from rdkit.Chem.Draw import rdMolDraw2D
            from PIL import Image
            from io import BytesIO
            import pandas as pd
            import subprocess
            import re
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit.Chem import rdChemReactions
            from rdkit.Chem.Draw import rdMolDraw2D
            from rdkit.Chem.Draw import IPythonConsole
            from IPython.display import SVG, display
            from rxnmapper import RXNMapper
            import os
            #visualize rdkit atom indices
            #https://iwatobipen.wordpress.com/2017/02/25/draw-molecule-with-atom-index-in-rdkit/
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import IPythonConsole
            IPythonConsole.ipython_useSVG = True
            
            
            import rdkit
            import ast
            
            def standardization(molecule):
                molecule = Chem.AddHs(molecule) #commented out bc it is done already below
                # Sanitize the molecule
            
                AllChem.SanitizeMol(molecule) #commented out bc it is done already below
                # AllChem.EmbedMolecule(molecule, randomSeed=1, maxAttempts = 5000) #https://github.com/rdkit/rdkit/issues/1433 #inserted maxattempts
                AllChem.EmbedMolecule(molecule, randomSeed=1)  #commented out bc it is done already below
                AllChem.UFFOptimizeMolecule(molecule) #commented out bc it is done already below
                molecule.GetConformer() #commented out bc it is done already below
            
                return molecule
            
             
            def mol_with_atom_index( mol ):
                atoms = mol.GetNumAtoms()
                for idx in range( atoms ):
                    mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
                return mol
            
            
            
            def moveAtomMapsToNotes(m):
                for at in m.GetAtoms():
                    if at.GetAtomMapNum():
                        at.SetProp("atomNote",str(at.GetAtomMapNum()))
            
            
            
            # Function to count the number of hydrogens attached to ring atoms
            def count_hydrogens_on_ring_atoms(molecule):
                ring_info = molecule.GetRingInfo()
                ring_atoms = set()
                
                # Collect all atoms that are part of rings
                for ring in ring_info.AtomRings():
                    ring_atoms.update(ring)
                
                # Count hydrogens attached to each ring atom
                hydrogen_counts = {}
                for atom_idx in ring_atoms:
                    atom = molecule.GetAtomWithIdx(atom_idx)
                    hydrogen_count = atom.GetTotalNumHs()
                    hydrogen_counts[atom_idx] = hydrogen_count
                
                return hydrogen_counts
            


           #################also for the molecule add the rxn molecule: need to generate this (look for this in this code)
            
            from rdkit import Chem
            from rdkit.Chem import rdmolops, Draw, rdDepictor, rdChemReactions
            from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
            from PIL import Image
            from io import BytesIO
            import pandas as pd
            import subprocess
            import re
            import ast
            from IPython.display import SVG, display
            from rxnmapper import RXNMapper
            import os
            
            IPythonConsole.ipython_useSVG = True
            
            import rdkit
            import ast
            
            def standardization(molecule):
                molecule = Chem.AddHs(molecule) #commented out bc it is done already below
                # Sanitize the molecule
            
                AllChem.SanitizeMol(molecule) #commented out bc it is done already below
                # AllChem.EmbedMolecule(molecule, randomSeed=1, maxAttempts = 5000) #https://github.com/rdkit/rdkit/issues/1433 #inserted maxattempts
                AllChem.EmbedMolecule(molecule, randomSeed=1)  #commented out bc it is done already below
                AllChem.UFFOptimizeMolecule(molecule) #commented out bc it is done already below
                molecule.GetConformer() #commented out bc it is done already below
            
                return molecule
            

            def get_hydrogen_counts_only_relevant_rings(molecule, relevant_atoms_list):
                """
                Get the bond types between the ring atoms and their neighbors.
            
                Parameters:
                - molecule (rdkit.Chem.rdchem.Mol): The molecule object.
                - relevant_atoms_list (list of int): List of atom indices.
            
                Returns:
                - dict: A dictionary where keys are tuples of atom indices and values are bond types.
                """
             #   molecule = clear_aromatic_flags(molecule)
                moveAtomMapsToNotes(molecule)  # Ensure this is defined and does what you need
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
                        bond_type = bond.GetBondType()  # Gets bond type as a double (e.g., 1.0 for single, 2.0 for double)
                        bond_key = tuple(sorted([at_idx, neighbor_idx]))
                        bond_types[bond_key] = bond_type
            
                return bond_types
            
            
            def get_hydrogen_counts_only_relevant_rings_neighbor(molecule, relevant_atoms_list):
                """
                Get the bond types between the ring atoms and their neighbors.
                
                Parameters:
                - molecule (rdkit.Chem.rdchem.Mol): The molecule object.
                - relevant_atoms_list (list of int): List of ring atom indices.
                
                Returns:
                - dict: A dictionary where keys are tuples of atom indices and values are bond types.
                """
               # molecule = clear_aromatic_flags(molecule)
                moveAtomMapsToNotes(molecule)  # Ensure this is defined and does what you need
                Chem.Kekulize(molecule)
                bond_types = {}
                
                for at_idx in relevant_atoms_list:
                    atom = molecule.GetAtomWithIdx(at_idx)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if neighbor_idx not in relevant_atoms_list and neighbor.GetAtomicNum() != 1:  # Avoid hydrogen atoms
                            bond = molecule.GetBondBetweenAtoms(at_idx, neighbor_idx)
                            if bond:
                                bond_type = bond.GetBondType()  # Gets bond type as a double (e.g., 1.0 for single, 2.0 for double)
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
            
            
            
            mols_with_ring_change = []
            
            mols_without_ring_change = []
            
            temp_dict_1 = []
            temp_dict_2 = []
            
            for smi_t1, smi_t2, row_nr, relevant_ring_idx_t1, relevant_ring_idx_t2 in zip(tautobase_experimental["t1-smiles"], tautobase_experimental["t2-smiles"], tautobase_experimental["row"], tautobase_experimental["relevant_ring_idx_t1"], tautobase_experimental["relevant_ring_idx_t2"]):
            
                new_at_idx_list_list_t1 = [] 
            
                rxns = [f'{smi_t1}>>{smi_t2}'] 
                rxn_mapper = RXNMapper()
                outputs = rxn_mapper.get_attention_guided_atom_maps(rxns)
                for out in outputs:
                    smiles_reaction = out['mapped_rxn']
                    rxn = rdChemReactions.ReactionFromSmarts(smiles_reaction,useSmiles=True)
                    trxn = rdChemReactions.ChemicalReaction(rxn)
                    t2_mol = trxn.GetProducts()[0]
                    t2_mol = standardization(t2_mol)
                    t1_mol = trxn.GetReactants()[0]
                    t1_mol = standardization(t1_mol)
            
                    idx_H_count_t1 = get_hydrogen_counts_only_relevant_rings(t1_mol, relevant_ring_idx_t1)
                    idx_H_count_t2 = get_hydrogen_counts_only_relevant_rings(t2_mol, relevant_ring_idx_t2)
                    temp_dict_1.append(idx_H_count_t1)
                    temp_dict_2.append(idx_H_count_t2)
                    if idx_H_count_t1 != idx_H_count_t2:
                        mols_with_ring_change.append(row_nr)
                    else:
                        mols_without_ring_change.append(row_nr)
            

        ################## create subsets 
            
            # for the mols with aromaticity and change:
            
            mols_w_arom_and_change = tautobase_experimental[tautobase_experimental['row'].isin(mols_with_ring_change)]
            mols_w_arom_and_change.to_csv("mols_w_arom_and_change.csv")
            
            
            # split furhter into: does it change aromaticity?
            
            
            tautobase_taut_involved_arom_change = mols_w_arom_and_change[mols_w_arom_and_change["arom_change"] == 1]
            
            tautobase_taut_involved_arom_change.to_csv("tautobase_taut_involved_arom_change.csv")
            
            
            tautobase_taut_involved_no_arom_change = mols_w_arom_and_change[(mols_w_arom_and_change["arom_change"] == 2) | (mols_w_arom_and_change["arom_change"] == 3)] #either from aro to aro or non aro to aro
            
            tautobase_taut_involved_no_arom_change.to_csv("tautobase_taut_involved_no_arom_change.csv")
            

          
            mols_without_ring_change
            
            
            # In[ ]:
            
            
            # for the mols with aromaticity and change:
            
            
            mols_w_arom_and_no_change = tautobase_experimental[tautobase_experimental['row'].isin(mols_without_ring_change)]
            mols_w_arom_and_no_change.to_csv("mols_w_arom_and_no_change.csv")
            
            
            #filter further: is tautomerism near a aromatic site (either t1 or t2):
            
            

        
            change_near_relevant_ring = []
            
            no_change_near_relevant_ring = []
            
            
            
            for smi_t1, smi_t2, row_nr, relevant_ring_idx_t1, relevant_ring_idx_t2 in zip(mols_w_arom_and_no_change["t1-smiles"], mols_w_arom_and_no_change["t2-smiles"], mols_w_arom_and_no_change["row"], mols_w_arom_and_no_change["relevant_ring_idx_t1"], mols_w_arom_and_no_change["relevant_ring_idx_t2"]):
            
                new_at_idx_list_list_t1 = [] 
            
                rxns = [f'{smi_t1}>>{smi_t2}'] 
                rxn_mapper = RXNMapper()
                outputs = rxn_mapper.get_attention_guided_atom_maps(rxns)
                for out in outputs:
                    smiles_reaction = out['mapped_rxn']
                    rxn = rdChemReactions.ReactionFromSmarts(smiles_reaction,useSmiles=True)
                    trxn = rdChemReactions.ChemicalReaction(rxn)
                    t2_mol = trxn.GetProducts()[0]
                    t2_mol = standardization(t2_mol)
                    t1_mol = trxn.GetReactants()[0]
                    t1_mol = standardization(t1_mol)
            
                    idx_H_count_t1 = get_hydrogen_counts_only_relevant_rings_neighbor(t1_mol, relevant_ring_idx_t1)
                    idx_H_count_t2 = get_hydrogen_counts_only_relevant_rings_neighbor(t2_mol, relevant_ring_idx_t2)
            
                    if idx_H_count_t1 != idx_H_count_t2:
                        change_near_relevant_ring.append(row_nr)
                    else:
                        no_change_near_relevant_ring.append(row_nr)
            
            #splitting:
            
            mols_w_arom_and_change_near_rel_system = mols_w_arom_and_no_change[mols_w_arom_and_no_change['row'].isin(change_near_relevant_ring)]
            mols_w_arom_and_change_near_rel_system.to_csv("mols_w_arom_and_change_near_rel_system.csv")
            
            
            mols_w_arom_and_no_change_near_rel_system = mols_w_arom_and_no_change[mols_w_arom_and_no_change['row'].isin(no_change_near_relevant_ring)]
            mols_w_arom_and_no_change_near_rel_system.to_csv("mols_w_arom_and_no_change_near_rel_system.csv")
            
            

############# just to see which Rows in 'tautobase_experimental' not included in the merged data frame


         
            import pandas as pd
            
            # Assuming the data frames are already loaded and named accordingly
            # If not, you would first load them, e.g., tautobase_experimental = pd.read_csv('tautobase_experimental.csv')
            
            # Merge the four data frames
            merged_df = pd.concat([tautobase_taut_involved_arom_change,
                                   tautobase_taut_involved_no_arom_change,
                                   mols_w_arom_and_change_near_rel_system,
                                   mols_w_arom_and_no_change_near_rel_system])
            
            # Extract the "row" column from the merged data frame
            merged_rows = merged_df['row'].unique()
            
            # Extract the "row" column from the tautobase_experimental data frame
            experimental_rows = tautobase_experimental['row'].unique()
            
            # Identify rows in tautobase_experimental that are not in the merged data frame
            rows_not_in_merged = [row for row in experimental_rows if row not in merged_rows]
            
            # Alternatively, using pandas for better performance on large data sets
            rows_not_in_merged = tautobase_experimental[~tautobase_experimental['row'].isin(merged_rows)]
            
            # Output the result
            print("Rows in 'tautobase_experimental' not included in the merged data frame:")
            print(rows_not_in_merged)
            
            

############# left out manual filter part

      
            mols_w_arom_combined =  pd.concat([mols_w_arom_and_change_near_rel_system, mols_w_arom_and_no_change_near_rel_system], ignore_index=True)
        

############# which subset to work on:
          
            if data_subset == "mols_w_arom_and_change":
                tautobase_experimental = mols_w_arom_and_change 
            
            if data_subset == "mols_w_arom_and_no_change_near_rel_system":
                tautobase_experimental = mols_w_arom_and_no_change_near_rel_system 
            
            if data_subset == "mols_w_arom_and_change_near_rel_system":
                tautobase_experimental = mols_w_arom_and_change_near_rel_system
            
            if data_subset == "mols_w_arom_combined":
                tautobase_experimental = mols_w_arom_combined


############# count the tautomer types 27.04.26:
            def count_taut_type(taut1, taut2):
                """
                Check for different tautomer types. Multiple types can be True.
                Returns a list of integers representing tautomer types.
                0 = keto
                1 = enamine
                2 = thiketo
                3 = none of the above
                """
                types = []
            
                keto_tup = pattern_check(taut1, taut2, '*=[O,o]')
                enamine_tup = pattern_check(taut1, taut2, '*=[N,n]')
                thiketo_tup = pattern_check(taut1, taut2, '*=[S,s]')
            
                if keto_tup[0] != keto_tup[1]:
                    types.append(0)
                if enamine_tup[0] != enamine_tup[1]:
                    types.append(1)
                if thiketo_tup[0] != thiketo_tup[1]:
                    types.append(2)
            
                # If none of the above conditions were true, append 3
                if not types:
                    types.append(3)
            
                return types
    
            tautobase_experimental_copy = tautobase_experimental.copy()

            import pickle
            
            with open("wieder_rows.pkl", "rb") as f:
                numbers = pickle.load(f)


            tautobase_experimental_copy = tautobase_experimental_copy[
                tautobase_experimental_copy["row"].astype(int).isin(numbers)
            ]
            print(len(numbers))
            print(f"len of tautobase_experimental_copy is {tautobase_experimental_copy.shape}")


            

            print(f"count taut type for subsets:")
            print(f"len of this subset is: {len(tautobase_experimental_copy)}")


            
            taut_type_list = []

          
            for smi_t1, smi_t2, row_nr in zip(tautobase_experimental_copy["t1-smiles"],
                                               tautobase_experimental_copy["t2-smiles"],
                                               tautobase_experimental_copy["row"]):
            
                rxns = [f'{smi_t1}>>{smi_t2}'] 
                rxn_mapper = RXNMapper()
                outputs = rxn_mapper.get_attention_guided_atom_maps(rxns)
            
                for out in outputs:
                    smiles_reaction = out['mapped_rxn']
                    rxn = rdChemReactions.ReactionFromSmarts(smiles_reaction, useSmiles=True)
                    trxn = rdChemReactions.ChemicalReaction(rxn)
            
                    t2_mol = standardization(trxn.GetProducts()[0])
                    t1_mol = standardization(trxn.GetReactants()[0])
            
                    taut_types = count_taut_type(t1_mol, t2_mol)
                    taut_type_list.append(taut_types)
            
            # Store as list of lists
            tautobase_experimental_copy["taut_type"] = taut_type_list


            import itertools
            flat_taut_types = list(itertools.chain.from_iterable(taut_type_list))
            value_counts = pd.Series(flat_taut_types).value_counts()
            print(f"tautomer type count for this subset is {value_counts}")
                        
                        
            ############# valid_tautobase, do i need this part? #############

            valid_tautobase = tautobase_experimental.copy()
            
            valid_tautobase['t1-molecule'] = valid_tautobase['t1-smiles'].apply(transform)
            valid_tautobase['t2-molecule'] = valid_tautobase['t2-smiles'].apply(transform)
            
            valid_tautobase = valid_tautobase[valid_tautobase['t1-molecule'] != None] 
            valid_tautobase = valid_tautobase[valid_tautobase['t2-molecule'] != None] 
            
            valid_tautobase = valid_tautobase.dropna(subset=['t1-molecule'])





            ############# add ddG values #############
            tautobase_experimental["ddG"] = tautobase_experimental["row"].apply(lambda x: ddG_row_dic.get(str(x), np.nan))
            
            
          
            ############# dropping unnecessary  columns #############
            import pandas as pd
            
            aromat_string = "nicszz" # homa
            
            # Assuming your DataFrame is named df
            columns_to_drop = ['Unnamed: 0', 'Unnamed: 0_x', 'name_t1', 'ring_idx_t1', 'filename_t1','murcko_smiles_not_reduced_x','rxmol_ring_idx_t1', 'ring_idx_t1_list_rdkit', 'reaction_idx_t2', 'ring_idx_t2_list_rdkit','rxmol_ring_idx_t2', f'{aromat_string}_vals_t2', 'reaction_idx_t1','murcko_smiles_not_reduced_y', 'murcko_smiles_y','Unnamed: 0_y', 'name_t2', 'ring_idx_t2', 'filename_t2', 'log_K_y',f'{aromat_string}_vals_t1']
            
            
            tautobase_experimental.drop(columns=columns_to_drop, inplace=True)
            
            

            
            ############# dropping nan ddG values #############
            nan_count = tautobase_experimental["ddG"].isna().sum()
            
            print("Number of np.nan values in tautobase_experimental['ddG']: ", nan_count)
            
            
            # In[ ]:
            
            
            # dropping nan vals
            tautobase_experimental = tautobase_experimental.dropna(subset=['ddG'])
            

            ############# filtering molecules with K values and create working subset #############
            
            valid_tautobase = tautobase_experimental.dropna(subset=["K (T/E/M)"])
            valid_tautobase = valid_tautobase.reset_index(drop=True)




            ############# dropping mols for featurization ? maybe put this in another script #############
            
            tautobase['t1-molecule'] = tautobase[' t1-smiles'].apply(transform)
            tautobase['t2-molecule'] = tautobase[' t2-smiles'].apply(transform)
            
            tautobase = tautobase.rename(columns={' t1-smiles': 't1-smiles', ' t2-smiles': 't2-smiles'})
            
            ###########
            tautobase_experimental['t1-molecule'] = tautobase_experimental['t1-smiles'].apply(transform)
            tautobase_experimental['t2-molecule'] = tautobase_experimental['t2-smiles'].apply(transform)
            
            tautobase_experimental = tautobase_experimental[tautobase_experimental['t1-molecule'] != None] 
            tautobase_experimental = tautobase_experimental[tautobase_experimental['t2-molecule'] != None] 
            
            #tautobase_experimental = tautobase_experimental.rename(columns={' t1-smiles': 't1-smiles', ' t2-smiles': 't2-smiles'})
            tautobase_experimental = tautobase_experimental.dropna(subset=['t1-molecule'])
            #############valid_tautobase
            
            valid_tautobase['t1-molecule'] = valid_tautobase['t1-smiles'].apply(transform)
            valid_tautobase['t2-molecule'] = valid_tautobase['t2-smiles'].apply(transform)
            
            valid_tautobase = valid_tautobase[valid_tautobase['t1-molecule'] != None] 
            valid_tautobase = valid_tautobase[valid_tautobase['t2-molecule'] != None] 
            
            valid_tautobase = valid_tautobase.dropna(subset=['t1-molecule'])



           
            ############# calculating difference fingerprints for features using MFP #############
            valid_tautobase["t1-FP"] = valid_tautobase["t1-molecule"].apply(MFP_generator)
            valid_tautobase["t2-FP"] = valid_tautobase["t2-molecule"].apply(MFP_generator)
        
        
            valid_tautobase["t1_collision"] = valid_tautobase["t1-molecule"].apply(get_bit_coll)
            valid_tautobase["t2_collision"] = valid_tautobase["t2-molecule"].apply(get_bit_coll)
        

        
            # calculating difference fingerprints using maccs_keys
            #valid_tautobase["t1-FP"] = valid_tautobase["t1-molecule"].apply(generate_maccs_keys)
            #valid_tautobase["t2-FP"] = valid_tautobase["t2-molecule"].apply(generate_maccs_keys)
            
            differnce_vectors = []
            for bit_vector1, bit_vector2 in zip(valid_tautobase["t1-FP"], valid_tautobase["t2-FP"]):
                if simple_bits == True:
                    difference_vect = [abs(int(bit1 - bit2)) for bit1, bit2 in zip(bit_vector1, bit_vector2)] 
                else:
                    difference_vect = [int(bit1 - bit2) for bit1, bit2 in zip(bit_vector1, bit_vector2)]        
            
                differnce_vectors.append(difference_vect)
            
            
            
            valid_tautobase['difference_vector'] = differnce_vectors
            
            


   
            ############# create subset of valid tautobase, that can be used in ML model (only features and lables) #############
            
            if inclue_row == True:
                valid_tautobase_ML = valid_tautobase[['K (T/E/M)', f'relative {aromat_string} difference', 'ddG', 'row' ]]
            else:
                valid_tautobase_ML = valid_tautobase[['K (T/E/M)', f'relative {aromat_string} difference', 'ddG' ]]
            

            # In[ ]:
            
            
            valid_tautobase_ML.head()
            
            
            # In[ ]:
            
            
            # generate a Numpy array of sample size and fps. Join a sequence of arrays along x axis.
            difference_arr = np.stack(valid_tautobase["difference_vector"])
            
            #convert to data frame:
            
            difference_df = pd.DataFrame(difference_arr)
            
            difference_df = difference_df.reset_index(drop=True)
            
            valid_tautobase_ML = valid_tautobase_ML.reset_index(drop=True)
            
            
            
            #### Renaming function
            #renames columns of a data set by adding a prefix 'new_name' in front of it
            
            def rename(df_name, new_name):
                df = df_name.copy(deep=True)
                for i in range(len(df_name.columns)):
                    name = new_name + str(df_name.columns[i])
                    df.rename(columns={df_name.columns[i] : name}, inplace=True)
                return df
            
            difference_df = rename(difference_df, 'difference_bit_')
            
            # Concatenate along the x-axis (horizontally)
            valid_tautobase_ML = pd.concat([valid_tautobase_ML, difference_df], axis=1)


            ############# create subset of valid tautobase, that can be used in ML model (only features and lables) #############
            #for data base with arom/homa vals:
        
            #valid_tautobase_ML["log_K"] = valid_tautobase_ML["K (T/E/M)"]
        
            arom_string = "nicszz"
        
            valid_tautobase_ML.rename(columns = {"K (T/E/M)":"log_K"}, inplace = True)
        
            ################## For ML models predicting theoretical ddG vals ##################
            valid_tautobase.to_csv(f"./data/valid_data_w_arom_vals_ordered_by_{arom_string}{file_suffix}.csv", index = None)
            valid_tautobase_ML_ddG = valid_tautobase_ML.drop("log_K", axis=1)
            valid_tautobase_ML_ddG.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ddG_ordered_by_{arom_string}{file_suffix}.csv", index = None)
        
        
        
            valid_tautobase_ML_wo_arom_ddG = valid_tautobase_ML_ddG.copy()
            valid_tautobase_ML_wo_arom_ddG = valid_tautobase_ML_wo_arom_ddG.drop(f"relative {arom_string} difference", axis=1)
            valid_tautobase_ML_wo_arom_ddG.to_csv(f"./data/rand_forest_regr_data_ddG_ordered_by_{arom_string}{file_suffix}.csv", index = None)
        
        
        
        
        
            ################## For ML models predicting experimental log K vals ##################
        
            valid_tautobase.to_csv(f"./data/valid_data_w_arom_vals_ordered_by_{arom_string}{file_suffix}.csv", index = None)
            valid_tautobase_ML = valid_tautobase_ML.drop("ddG", axis=1)
            valid_tautobase_ML.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ordered_by_{arom_string}{file_suffix}.csv", index = None)
        
        
        
            valid_tautobase_ML_wo_arom = valid_tautobase_ML.copy()
            valid_tautobase_ML_wo_arom = valid_tautobase_ML_wo_arom.drop(f"relative {arom_string} difference", axis=1)
            valid_tautobase_ML_wo_arom.to_csv(f"./data/rand_forest_regr_data_ordered_by_{arom_string}{file_suffix}.csv", index = None)
        
        
        
        
        
            ################## columns with only identical values are deleted if the parameter 'delun' is set to true ##################
            delun = True
        
            if delun:
                len_un = len(valid_tautobase_ML.columns)
                valid_tautobase_ML_unique = valid_tautobase_ML.loc[:,valid_tautobase_ML.nunique()!=1]
                num_drop_unique = len_un - len(valid_tautobase_ML_unique.columns)
                valid_tautobase_ML_unique.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ordered_by_{arom_string}_unique_feat{file_suffix}.csv", index = None)
                print(f'number of dropped collumns with identical values: {num_drop_unique}')
        
                len_un = len(valid_tautobase_ML_ddG.columns)
                valid_tautobase_ML_ddG_unique =  valid_tautobase_ML_ddG.loc[:,valid_tautobase_ML_ddG.nunique()!=1]
                num_drop_unique = len_un - len(valid_tautobase_ML_ddG_unique.columns)
                valid_tautobase_ML_ddG_unique.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ddG_ordered_by_{arom_string}_unique_feat{file_suffix}.csv", index = None)
                print(f'number of dropped collumns with identical values: {num_drop_unique}')
        
        
            ################## removing columns that dont contain bits, but keep HOMA change and murcko scaffold ##################
        
        
            cols_to_remove = ['NumAromaticCarbocycles','CalcNumAromaticHeterocycles', 'CalcNumAromaticRings', 'change_conj_bonds', 'degree_change_O', 'degree_change_N', 'degree_change_S']
        
        
            # Filter the list to include only those columns that exist in the DataFrame
            cols_to_remove = [col for col in cols_to_remove if col in valid_tautobase_ML_unique.columns]
        
        
        
            #exp data:
            valid_tautobase_ML_unique_feat_only_bits = valid_tautobase_ML_unique.drop(columns = cols_to_remove)
            # Using list comprehension
            #columns_to_keep = [col for col in valid_tautobase_ML_unique_feat_only_bits.columns if not col.startswith('murcko_smiles')]
            #valid_tautobase_ML_unique_feat_only_bits = valid_tautobase_ML_unique_feat_only_bits[columns_to_keep]
            valid_tautobase_ML_unique_feat_only_bits.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ordered_by_{arom_string}_unique_feat_only_bits_plus_murcko{file_suffix}.csv", index = None)
        
        
        
            # Filter the list to include only those columns that exist in the DataFrame
            cols_to_remove = [col for col in cols_to_remove if col in valid_tautobase_ML_ddG_unique.columns]
        
        
            #theoreth. data:
            valid_tautobase_ML_ddG_unique_feat_only_bits = valid_tautobase_ML_ddG_unique.drop(columns = cols_to_remove)
            # Using list comprehension
            #columns_to_keep = [col for col in valid_tautobase_ML_ddG_unique_feat_only_bits.columns if not col.startswith('murcko_smiles')]
            #valid_tautobase_ML_ddG_unique_feat_only_bits = valid_tautobase_ML_ddG_unique_feat_only_bits[columns_to_keep]
            valid_tautobase_ML_ddG_unique_feat_only_bits.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ddG_ordered_by_{arom_string}_unique_feat_only_bits_plus_murcko{file_suffix}.csv", index = None)
        
        
        
            ################## removing columns that dont contain bits, but keep HOMA change ##################
        
        
            if subset == "no_arom_mols":
                cols_to_remove = ['NumAromaticCarbocycles','CalcNumAromaticHeterocycles', 'CalcNumAromaticRings', 'change_conj_bonds', 'degree_change_O', 'degree_change_N', 'degree_change_S', 'relative nicszz difference']
        
            else:
                cols_to_remove = ['NumAromaticCarbocycles','CalcNumAromaticHeterocycles', 'CalcNumAromaticRings', 'change_conj_bonds', 'degree_change_O', 'degree_change_N', 'degree_change_S']
        
        
            # Filter the list to include only those columns that exist in the DataFrame
            cols_to_remove = [col for col in cols_to_remove if col in valid_tautobase_ML_unique.columns]
        
        
            #exp data:
            valid_tautobase_ML_unique_feat_only_bits = valid_tautobase_ML_unique.drop(columns = cols_to_remove)
            # Using list comprehension
            columns_to_keep = [col for col in valid_tautobase_ML_unique_feat_only_bits.columns if not col.startswith('murcko_smiles')]
            valid_tautobase_ML_unique_feat_only_bits = valid_tautobase_ML_unique_feat_only_bits[columns_to_keep]
            valid_tautobase_ML_unique_feat_only_bits.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ordered_by_{arom_string}_unique_feat_only_bits{file_suffix}.csv", index = None)
        
            #theoreth. data:
            valid_tautobase_ML_ddG_unique_feat_only_bits = valid_tautobase_ML_ddG_unique.drop(columns = cols_to_remove)
            # Using list comprehension
            columns_to_keep = [col for col in valid_tautobase_ML_ddG_unique_feat_only_bits.columns if not col.startswith('murcko_smiles')]
            valid_tautobase_ML_ddG_unique_feat_only_bits = valid_tautobase_ML_ddG_unique_feat_only_bits[columns_to_keep]
            valid_tautobase_ML_ddG_unique_feat_only_bits.to_csv(f"./data/rand_forest_regr_data_w_arom_vals_ddG_ordered_by_{arom_string}_unique_feat_only_bits{file_suffix}.csv", index = None)
        
        
        #/data/rand_forest_regr_data_w_arom_vals_ordered_by_{aromat_string}_{features_used}.csv
        
        
        
        

            print("scripts done")
            
   
