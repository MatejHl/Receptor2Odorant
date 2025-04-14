import numpy as np
from rdkit import Chem, DataStructs
from rdkit.DataManip.Metric import GetTanimotoSimMat
from rdkit.Chem import AllChem

def get_tanimoto_matrix(_temp):
    _temp = _temp._SMILES.apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2))
    mat = GetTanimotoSimMat(_temp.values.tolist())
    return np.mean(mat)