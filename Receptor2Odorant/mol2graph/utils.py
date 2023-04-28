from rdkit import Chem 

def get_num_atoms_and_bonds(smiles, IncludeHs = False, validate = False):
    mol = Chem.MolFromSmiles(smiles.strip())
    assert mol is not None

    if IncludeHs:
        mol = Chem.rdmolops.AddHs(mol)

    if validate:
        can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TODO: Check if this row is necessary.
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")

    return len(mol.GetAtoms()), len(mol.GetBonds())


if __name__ == '__main__':
    smiles = 'C/C/1=C/CC/C(=C\[C@H]2[C@H](C2(C)C)CC1)/C'
    print(get_num_atoms_and_bonds(smiles))