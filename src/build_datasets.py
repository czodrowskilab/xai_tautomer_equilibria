
from util.preprocess import build_tautomer_pair
from util.ring_analysis import extract_relevant_ring_indices, extract_ring_bond_types, extract_ring_bond_types_neighbor







mols_with_ring_change = []

mols_without_ring_change = []

temp_dict_1 = []
temp_dict_2 = []


        
change_near_relevant_ring = []

no_change_near_relevant_ring = []
            
            

for smi_t1, smi_t2, rdkit_at_ind_1, rdkit_at_ind_2, reaction_idx_list_t1, reaction_idx_list_t2, aro_vals_1, aro_vals_2, row_nr in zip(tautobase_experimental["t1-smiles"], tautobase_experimental["t2-smiles"],tautobase_experimental["ring_idx_t1_list_rdkit"], tautobase_experimental["ring_idx_t2_list_rdkit"], tautobase_experimental["rxmol_ring_idx_t1"], tautobase_experimental["rxmol_ring_idx_t2"], tautobase_experimental["nicszz_vals_t1"], tautobase_experimental["nicszz_vals_t2"], tautobase_experimental["row"]):
    t1_mol, t2_mol = build_tautomer_pair(smi_t1, smi_t2)


    relevant_ring_idx_t1, relevant_ring_idx_t2 = extract_relevant_ring_indices(t1_mol, t2_mol, rdkit_at_ind_1, rdkit_at_ind_2, reaction_idx_list_t1, reaction_idx_list_t2, aro_vals_1, aro_vals_2)



    idx_H_count_t1 = extract_ring_bond_types(t1_mol, relevant_ring_idx_t1)
    idx_H_count_t2 = extract_ring_bond_types(t2_mol, relevant_ring_idx_t2)
    temp_dict_1.append(idx_H_count_t1)
    temp_dict_2.append(idx_H_count_t2)
    if idx_H_count_t1 != idx_H_count_t2:
        mols_with_ring_change.append(row_nr)
    else:
        mols_without_ring_change.append(row_nr)


    idx_H_count_t1_ = extract_ring_bond_types_neighbor(t1_mol, relevant_ring_idx_t1)
    idx_H_count_t2_ = extract_ring_bond_types_neighbor(t2_mol, relevant_ring_idx_t2)

    if idx_H_count_t1_ != idx_H_count_t2_:
        change_near_relevant_ring.append(row_nr)
    else:
        no_change_near_relevant_ring.append(row_nr)

mols_w_arom_and_change = tautobase_experimental[tautobase_experimental['row'].isin(mols_with_ring_change)]
mols_w_arom_and_no_change = tautobase_experimental[tautobase_experimental['row'].isin(mols_without_ring_change)]


mols_w_arom_and_change_near_rel_system = mols_w_arom_and_no_change[mols_w_arom_and_no_change['row'].isin(change_near_relevant_ring)]
mols_w_arom_and_change_near_rel_system.to_csv("mols_w_arom_and_change_near_rel_system.csv")


mols_w_arom_and_no_change_near_rel_system = mols_w_arom_and_no_change[mols_w_arom_and_no_change['row'].isin(no_change_near_relevant_ring)]
mols_w_arom_and_no_change_near_rel_system.to_csv("mols_w_arom_and_no_change_near_rel_system.csv")








def main():
    df = load_data()

    df = compute_relevant_ring_indices(df)
    df_splits = split_by_ring_change(df)

    save_splits(df_splits)

if __name__ == "__main__":
    main()