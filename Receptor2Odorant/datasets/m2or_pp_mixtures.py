import os
import pandas
import json
from shutil import copy
import numpy

from matplotlib import pyplot as plt

from Receptor2Odorant.mol2graph.utils import get_num_atoms_and_bonds

from Receptor2Odorant.base_cross_validation import BaseCVPostProcess
from Receptor2Odorant.database_utils import get_map_inchikey_to_isomericSMILES, get_map_inchikey_to_canonicalSMILES, enumerate_isomers, merge_cols_with_priority, pubchem_isoSMILES_to_RDKit_isoSMILES

# ---------
# Mixtures:
# ---------
class CVPP_Mixture_ConcatGraph(BaseCVPostProcess):
    def __init__(self, data_dir, auxiliary_data_path):
        name = 'mix__concatGraph'
        super(CVPP_Mixture_ConcatGraph, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir,
                'auxiliary_data_path' : self.auxiliary_data_path}

    def load_mols(self):
        mols = pandas.read_csv(os.path.join(self.data_dir, 'mols.csv'), sep = self.sep, index_col = 0, header = 0)
        return mols

    def load_auxiliary(self):
        auxiliary = {}
        for key in self.auxiliary_data_path.keys():
            if self.auxiliary_data_path[key] is not None:
                with open(self.auxiliary_data_path[key], 'r') as jsonfile:
                    auxiliary[key] = json.load(jsonfile)
            else:
                auxiliary[key] = {}
        return auxiliary

    @staticmethod
    def _update_map_inchikey_to_isomericSMILES(inchikeys_series, map_inchikey_to_isomericSMILES):
        """
        """
        current_idx = pandas.Index(map_inchikey_to_isomericSMILES.keys(), name = 'InChI Key')
        candidate_idx = inchikeys_series.dropna()
        candidate_idx = candidate_idx.str.split(' ').explode() # TODO: pandas FutureWarning for this row.
        candidate_idx = pandas.Index(candidate_idx.unique())
        new_idx = candidate_idx.difference(current_idx)
        if len(new_idx) > 0:
            print('Updating map_inchikey_to_isomericSMILES...')
            NEW = get_map_inchikey_to_isomericSMILES(new_idx.tolist())
            map_inchikey_to_isomericSMILES.update(NEW)
        return map_inchikey_to_isomericSMILES

    @staticmethod
    def _update_map_inchikey_to_canonicalSMILES(inchikeys_series, map_inchikey_to_canonicalSMILES):
        """
        """
        current_idx = pandas.Index(map_inchikey_to_canonicalSMILES.keys(), name = 'InChI Key')
        candidate_idx = inchikeys_series.dropna()
        candidate_idx = candidate_idx.str.split(' ').explode() # TODO: pandas FutureWarning for this row.
        candidate_idx = pandas.Index(candidate_idx.unique())
        new_idx = candidate_idx.difference(current_idx)
        if len(new_idx) > 0:
            print('Updating map_inchikey_to_canonicalSMILES...')
            NEW = get_map_inchikey_to_canonicalSMILES(new_idx.tolist())
            map_inchikey_to_canonicalSMILES.update(NEW)
        return map_inchikey_to_canonicalSMILES

    def update_auxiliary(self, auxiliary, data):
        """
        """
        map_inchikey_to_isomericSMILES = self._update_map_inchikey_to_isomericSMILES(data['InChI Key'], auxiliary['map_inchikey_to_isomericSMILES'])
        with open(os.path.join(self.working_dir, 'map_inchikey_to_isomericSMILES.json'), 'w') as jsonfile:
            json.dump(map_inchikey_to_isomericSMILES, jsonfile)
        auxiliary['map_inchikey_to_isomericSMILES'] = map_inchikey_to_isomericSMILES

        map_inchikey_to_canonicalSMILES = self._update_map_inchikey_to_canonicalSMILES(data['InChI Key'], auxiliary['map_inchikey_to_canonicalSMILES'])
        with open(os.path.join(self.working_dir, 'map_inchikey_to_canonicalSMILES.json'), 'w') as jsonfile:
            json.dump(map_inchikey_to_canonicalSMILES, jsonfile)
        auxiliary['map_inchikey_to_canonicalSMILES'] = map_inchikey_to_canonicalSMILES

        return auxiliary

    @staticmethod
    def _get_SMILES_mono(row, _map_to_isomeric):
        """
        Notes:
        ------
        Use enumerate_isomers to go from canonicalSMILES to list of isomericSMILES.
        """
        if row['InChI Key'] == row['InChI Key']:
            isoSMILES = _map_to_isomeric[row['InChI Key']]
        else:
            isomers = enumerate_isomers(row['canonicalSMILES'])
            if len(isomers) > 1:
                raise ValueError('{} is not mono'.format(row['canonicalSMILES']))
            isoSMILES = isomers[0]
        return isoSMILES

    @staticmethod
    def _get_SMILES_sum_of_isomers(row, _map_to_canonical):
        """
        Unroll InChI keys to set of isomers separated by \'.\'.
        Using enumerate_isomers to go from canonicalSMILES to list of isomericSMILES.

        Notes:
        ------
        I. Ensamble approach
        """
        # TODO: How to process unrolled isomers?
        if row['InChI Key'] == row['InChI Key']:
            canonicalSMILES = _map_to_canonical[row['InChI Key']]
            isomers = enumerate_isomers(canonicalSMILES)
        else:
            isomers = enumerate_isomers(row['canonicalSMILES'])
        if len(isomers) < 1:
            raise ValueError('Only one isomer found for sum of isomers {}'.format(row['InChI Key']))
        isoSMILES = '.'.join(isomers)
        return isoSMILES

    @staticmethod
    def _get_SMILES_mixture(row, _map_to_canonical):
        """
        Unroll InChI keys to set of isomers separated by \'.\'.
        Using enumerate_isomers to go from canonicalSMILES to list of isomericSMILES.

        Notes:
        ------
        I. Ensamble approach

        WARNING: This is neglecting concentration differences if they mix racemic + others in the same concentration.
        """
        # TODO: How to process unrolled isomers?
        if row['InChI Key'] == row['InChI Key']:
            isomers = []
            for x in row['InChI Key'].split(' '):
                _canonicalSMILES = _map_to_canonical[x]
                _isomers = enumerate_isomers(_canonicalSMILES)
                isomers += _isomers
        else:
            isomers = []
            for x in row['canonicalSMILES'].split(' '):
                _isomers = enumerate_isomers(x)
                isomers += _isomers
        isoSMILES = '.'.join(isomers)
        return isoSMILES

    def load_and_process_and_save_mols(self):
        mols = self.load_mols()
        auxiliary = self.load_auxiliary()
        auxiliary = self.update_auxiliary(auxiliary, mols)
    
        mols_mono = mols[mols['Mixture'] == 'mono'].copy()
        if not mols_mono.empty:
            mols_mono['_SMILES'] = mols_mono.apply(lambda x: self._get_SMILES_mono(x, _map_to_isomeric = auxiliary['map_inchikey_to_isomericSMILES']), axis = 1)
    
        mols_sum_of_isomers = mols[mols['Mixture'] == 'sum of isomers'].copy()
        if not mols_sum_of_isomers.empty:
            mols_sum_of_isomers['_SMILES'] = mols_sum_of_isomers.apply(lambda x: self._get_SMILES_sum_of_isomers(x, _map_to_canonical = auxiliary['map_inchikey_to_canonicalSMILES']), axis = 1)
    
        mols_mixture = mols[mols['Mixture'] == 'mixture'].copy()
        if not mols_mixture.empty:
            mols_mixture['_SMILES'] = mols_mixture.apply(lambda x: self._get_SMILES_mixture(x, _map_to_canonical = auxiliary['map_inchikey_to_canonicalSMILES']), axis = 1)
            
        mols = pandas.concat([mols_mono, mols_sum_of_isomers, mols_mixture])
        mols.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep=';')
        
        return mols[['_MolID', '_SMILES']]  

    def _postprocess(self, data, mols):
        data = pandas.merge(data, mols, how = 'left', on = 'mol_id')
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()
        mols = self.load_and_process_and_save_mols()

        if not data_train.empty:
            data_train = self._postprocess(data_train, mols)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid, mols)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        if not data_test.empty:
            data_test = self._postprocess(data_test, mols)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        self.save_hparams(prefix = 'mixture_')
        return None




class CVPP_Mixture_Racemic(CVPP_Mixture_ConcatGraph):
    """
    ('Discard chirality')
    Discard mixture and treat sum of isomers as racemic (using canonical smiles).
    """
    def __init__(self, data_dir, auxiliary_data_path):
        name = 'mix__racemic'
        super(CVPP_Mixture_ConcatGraph, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    @staticmethod
    def _get_SMILES_sum_of_isomers(row, _map_to_canonical):
        """
        set SMILES of sum of isomers to canonical SMILES.
        """
        # TODO: How to process unrolled isomers? TODO: What is this comment!?
        if row['InChI Key'] == row['InChI Key']:
            canonicalSMILES = _map_to_canonical[row['InChI Key']]
        else:
            canonicalSMILES = row['canonicalSMILES']
        return canonicalSMILES

    def load_and_process_and_save_mols(self):
        mols = self.load_mols()
        auxiliary = self.load_auxiliary()
        auxiliary = self.update_auxiliary(auxiliary, mols)
    
        mols_mono = mols[mols['Mixture'] == 'mono'].copy()
        if not mols_mono.empty:
            mols_mono['_SMILES'] = mols_mono.apply(lambda x: self._get_SMILES_mono(x, _map_to_isomeric = auxiliary['map_inchikey_to_isomericSMILES']), axis = 1)
    
        mols_sum_of_isomers = mols[mols['Mixture'] == 'sum of isomers'].copy()
        if not mols_sum_of_isomers.empty:
            mols_sum_of_isomers['_SMILES'] = mols_sum_of_isomers.apply(lambda x: self._get_SMILES_sum_of_isomers(x, _map_to_canonical = auxiliary['map_inchikey_to_canonicalSMILES']), axis = 1)

        mols = pandas.concat([mols_mono, mols_sum_of_isomers])
        mols.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep=';')
        return mols[['_MolID', '_SMILES']]

    def _postprocess(self, data, mols):
        data = data[data['mol_id'].isin(mols.index)] # Delete mixture molecules
        data = pandas.merge(data, mols, how = 'left', on = 'mol_id')
        return data


class CVPP_Mixture_Discard(CVPP_Mixture_ConcatGraph):
    """
    Discard both mixture and sum of isomers.
    """
    def __init__(self, data_dir, auxiliary_data_path):
        name = 'mix__discard'
        super(CVPP_Mixture_ConcatGraph, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    def update_auxiliary(self, auxiliary, data):
        """
        """
        map_inchikey_to_isomericSMILES = self._update_map_inchikey_to_isomericSMILES(data['InChI Key'], auxiliary['map_inchikey_to_isomericSMILES'])
        with open(os.path.join(self.working_dir, 'map_inchikey_to_isomericSMILES.json'), 'w') as jsonfile:
            json.dump(map_inchikey_to_isomericSMILES, jsonfile)
        auxiliary['map_inchikey_to_isomericSMILES'] = map_inchikey_to_isomericSMILES

        return auxiliary

    def load_and_process_and_save_mols(self):
        mols = self.load_mols()
        auxiliary = self.load_auxiliary()
        auxiliary = self.update_auxiliary(auxiliary, mols)
    
        mols_mono = mols[mols['Mixture'] == 'mono'].copy()
        if not mols_mono.empty:
            mols_mono['_SMILES'] = mols_mono.apply(lambda x: self._get_SMILES_mono(x, _map_to_isomeric = auxiliary['map_inchikey_to_isomericSMILES']), axis = 1)
    
        mols_mono.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep=';')
        return mols_mono[['_MolID', '_SMILES']] 

    def _postprocess(self, data, mols):
        data = data[data['mol_id'].isin(mols.index)] # Delete non-mono molecules
        data = pandas.merge(data, mols, how = 'left', on = 'mol_id')
        return data



class CVPP_Mixture_Weighting(BaseCVPostProcess):
    # TODO: Future work.
    pass


class CVPP_Mixture_Chiral_Only(CVPP_Mixture_ConcatGraph):
    """
    Keep only chiral molecules. This is primarily for testing.
    """
    def __init__(self, data_dir, auxiliary_data_path):
        name = 'mix__chiral_only'
        super(CVPP_Mixture_ConcatGraph, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    def update_auxiliary(self, auxiliary, data):
        """
        """
        map_inchikey_to_isomericSMILES = self._update_map_inchikey_to_isomericSMILES(data['InChI Key'], auxiliary['map_inchikey_to_isomericSMILES'])
        with open(os.path.join(self.working_dir, 'map_inchikey_to_isomericSMILES.json'), 'w') as jsonfile:
            json.dump(map_inchikey_to_isomericSMILES, jsonfile)
        auxiliary['map_inchikey_to_isomericSMILES'] = map_inchikey_to_isomericSMILES

        return auxiliary

    def load_and_process_and_save_mols(self):
        mols = self.load_mols()
        auxiliary = self.load_auxiliary()
        auxiliary = self.update_auxiliary(auxiliary, mols)
    
        mols_mono = mols[mols['Mixture'] == 'mono'].copy()
        if not mols_mono.empty:
            mols_mono['_SMILES'] = mols_mono.apply(lambda x: self._get_SMILES_mono(x, _map_to_isomeric = auxiliary['map_inchikey_to_isomericSMILES']), axis = 1)

        mols_mono = mols_mono[~mols_mono['_MolID'].str.contains('UHFFFAOYSA')] # Delete non-chiral
    
        mols_mono.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep=';')
        return mols_mono[['_MolID', '_SMILES']] 

    def _postprocess(self, data, mols):
        data = data[data['mol_id'].isin(mols.index)] # Delete non-mono molecules
        data = pandas.merge(data, mols, how = 'left', on = 'mol_id')
        return data