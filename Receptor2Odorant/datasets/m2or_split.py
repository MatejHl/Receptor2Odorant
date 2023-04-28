import os
import pandas
import json
from sklearn.model_selection import train_test_split
from shutil import copy
import numpy

import itertools
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import hdbscan
from sklearn.cluster import AgglomerativeClustering

from Receptor2Odorant.base_cross_validation import BaseCVSplit
from Receptor2Odorant.database_utils import get_map_inchikey_to_canonicalSMILES


class InadequateTestSetSizeError(Exception):
    pass


class TestOnly(BaseCVSplit):
    """
    For convenience when createing test set only.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        super(TestOnly, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'test_only'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))

        return full_data

    def func_split_data(self, data, seed, **kwargs):
        """
        This just copy paste full_data to data_test

        Paramters:
        ----------
        data : pandas.DataFrame
            dataframe returned by self.func_data 
        """
        return pandas.DataFrame([], columns=data.columns), pandas.DataFrame([], columns=data.columns), data


class EC50_TestOnly(TestOnly):
    """
    For convenience when createing test set only.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        super(EC50_TestOnly, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'ec50_test_only'

    def func_split_data(self, data, seed, **kwargs):
        """
        This just copy paste full_data to data_test

        Paramters:
        ----------
        data : pandas.DataFrame
            dataframe returned by self.func_data 
        """
        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))

        return pandas.DataFrame([], columns=data.columns), pandas.DataFrame([], columns=data.columns), data_ec50


class EC50_Random(BaseCVSplit):
    """
    Take randomly subset of EC50 data.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        super(EC50_Random, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'EC50_random_data'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'full_data.csv'), dst = os.path.join(self.working_dir, 'full_data.csv'))
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))

        return full_data


    def func_split_data(self, data, seed, **kwargs):
        """
        ec50_random_split
        """
        relative_valid_ratio = kwargs.get('relative_valid_ratio', 0.1)
        ec50_test_ratio = kwargs.get('ec50_test_ratio', None)

        assert (ec50_test_ratio > 0.0) and (ec50_test_ratio < 1.0)
        assert (relative_valid_ratio > 0.0) # and (valid_ratio < 1.0)

        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))
        # test split
        _, data_test = train_test_split(data_ec50, 
                                            test_size = ec50_test_ratio, 
                                            random_state = seed)
        assert len(data_test[data_test['Responsive'] == 1]) > 0.01*len(data_ec50)
        assert len(data_test[data_test['Responsive'] == 0]) > 0.01*len(data_ec50)
        if len(data_test) > 0.5*len(data_ec50):
            raise ValueError('Test data is taking more than 50% of all EC50 data.')

        data_train = data.loc[data.index.difference(data_test.index)]
        
        # valid split
        data_train, data_valid = train_test_split(data_train, 
                                            test_size = (relative_valid_ratio*len(data_ec50))/len(data_train),
                                            random_state = seed)
        print(len(data_test[data_test['Responsive'] == 1]))
        print(len(data_test[data_test['Responsive'] == 0]))
        return data_train, data_valid, data_test




# -----------------------------------
# Deorphanization splits - Random OR:
# -----------------------------------
class EC50_LeaveOut_OR(BaseCVSplit):
    """
    Take all occurences of given ORs, take its EC50 occurences as test set and discard Screening occurencies.

    Here we wnat to investigate the effect of predicting pairs for new sequences but which can be similar to 
    the training ones. E.g. predicting pairs for a new mutant.

    Notes:
    ------
    We keep mutants of selected sequences.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        super(EC50_LeaveOut_OR, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'EC50_LeaveOut_OR'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'full_data.csv'), dst = os.path.join(self.working_dir, 'full_data.csv'))
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))
        # copy(src = os.path.join(self.data_dir, 'CV_data_hparams.json'), dst = os.path.join(self.working_dir, 'CV_data_hparams.json'))

        return full_data

    def get_ORs(self, data, min_n_ligands, portion, seed):
        """
        Parameters:
        -----------
        data : pandas.DataFrame

        min_n_ligands : int
            minimum number of ligands that a sequence must have in order to be considered for test set.

        portion : float
            perentage of sequences with enough ligands to include in test set.

        seed : int
            random seed for numpy.random.default_rng 

        Notes:
        ------
        If we wnat to put mutants and wild types together:
            # def _select_ORs_with_enough_ligands(df, min_n_ligands):
            #     if df['num_ligands'].sum() >= min_n_ligands:
            #         return df.index.values
            #     else:
            #         return float('nan')
            # _idx = _seqs.groupby('_Sequence').apply(lambda x: self._select_ORs_with_enough_ligands(x, min_n_ligands = min_n_ligands))
            # _idx = _idx.dropna()
            # n_idx = len(_idx)
            # print('Number of unique sequences with more than (or equal) {} ligands: {} (mutants added to wild type)'.format(min_n_ligands, n_idx))
        """
        num_ligands = data.groupby('seq_id').apply(lambda x: x['Responsive'].sum())
        num_ligands.name = 'num_ligands'
        _idx = num_ligands[num_ligands >= min_n_ligands].index

        n_idx = len(_idx)
        print('Number of sequences with more than (or equal) {} ligands: {}'.format(min_n_ligands, n_idx))

        np_rng = numpy.random.default_rng(seed)
        _idx = np_rng.choice(_idx, size = int(numpy.ceil(portion*n_idx)))
        return _idx

    def func_split_data(self, data, seed, **kwargs):
        """
        """
        relative_valid_ratio = kwargs.pop('relative_valid_ratio', 0.1)
        
        portion_seqs = kwargs.pop('portion_seqs', 0.5)
        min_n_ligands = kwargs.pop('min_n_ligands', 5)

        max_test_size = kwargs.pop('max_test_size', 0.4)
        min_test_size = kwargs.pop('min_test_size', 0.1)

        # assert (test_ratio > 0.0) and (test_ratio < 1.0)
        assert (relative_valid_ratio > 0.0) # and (valid_ratio < 1.0)

        _idx = self.get_ORs(data, min_n_ligands = min_n_ligands, portion = portion_seqs, seed = seed)

        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))

        _max_test_size = max_test_size*len(data_ec50)
        _min_test_size = min_test_size*len(data_ec50)

        data_test = data_ec50[data_ec50['seq_id'].isin(_idx)]

        if len(data_test) > _max_test_size:
            raise InadequateTestSetSizeError('data_test is bigger than max size: Size: {}; max_size: {}'.format(len(data_test), _max_test_size))
        if len(data_test) < _min_test_size:
            raise InadequateTestSetSizeError('data_test is smaller than min size: Size: {}; min_size: {}'.format(len(data_test), _min_test_size))

        assert len(data_test[data_test['Responsive'] == 1]) > 0.01*len(data_ec50)
        assert len(data_test[data_test['Responsive'] == 0]) > 0.01*len(data_ec50)
        if len(data_test) > 0.5*len(data_ec50):
            raise ValueError('Test data is taking more than 50% of all EC50 data.')

        data_train = data.loc[data.index.difference(data_test.index)]

        # valid split
        data_train, data_valid = train_test_split(data_train, 
                                            test_size = (relative_valid_ratio*len(data_ec50))/len(data_train),
                                            random_state = seed)
        print('Number of positive in data_test: {}'.format(len(data_test[data_test['Responsive'] == 1])))
        print('Number of negative in data_test: {}'.format(len(data_test[data_test['Responsive'] == 0])))

        print('\nWARNING: EC50_LeaveOut_OR needs to be followed by discarding screening in Post-processing!!\n')

        return data_train, data_valid, data_test




# ------------------------------------
# Deorphanization splits - Cluster OR:
# ------------------------------------
class EC50_LeaveClusterOut_OR(BaseCVSplit):
    """
    Leave cluster of ORs out based on some similarity measure.

    Here we wnat to investigate the effect of predicting pairs for new sequences but which can be similar to 
    the training ones. E.g. predicting pairs for a new mutant.

    Notes:
    ------
    We keep mutants of selected sequences.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        self.auxiliary_data_path = split_kwargs.pop('auxiliary_data_path')

        super(EC50_LeaveClusterOut_OR, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'EC50_LeaveClusterOut_OR'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        seqs = pandas.read_csv(os.path.join(self.data_dir, 'seqs.csv'), sep = self.sep, index_col = 0, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'full_data.csv'), dst = os.path.join(self.working_dir, 'full_data.csv'))
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))
        return full_data, seqs

    def load_auxiliary(self):
        auxiliary = {}
        for key in self.auxiliary_data_path.keys():
            if self.auxiliary_data_path[key] is not None:
                auxiliary[key] = pandas.read_csv(self.auxiliary_data_path[key], sep=';', index_col=0)
            else:
                auxiliary[key] = {}
        return auxiliary

    def get_ORs(self, seqs, data, min_n_ligands, n_cluster_sample, seed, hdbscan_min_samples, hdbscan_min_cluster_size):
        """
        Parameters:
        -----------
        mols : pandas.DataFrame

        data : pandas.Dataframe

        min_n_ligands : int
            minimal number of sequences that a molecule needs to activate to be considered for test set.

        n_cluster_sample : int
            number of clusters to put to a test set.

        seed : int
            random seed for numpy.random.default_rng

        hdbscan_min_samples : int
            the number of samples in a neighbourhood for a point to be considered a core point in HDBSCAN.
            See: https://hdbscan.readthedocs.io/en/latest/api.html

        hdbscan_min_cluster_size : int
            the minimum size of clusters; single linkage splits that contain fewer points than this will be 
            considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.
            See: https://hdbscan.readthedocs.io/en/latest/api.html
        """
        num_ligands = data.groupby('seq_id').apply(lambda x: x['Responsive'].sum())
        num_ligands.name = 'num_ligands'
        _idx_ligands = num_ligands[num_ligands >= min_n_ligands].index

        n_idx = len(_idx_ligands)
        print('Number of ORs with more than (or equal) {} ligands: {}'.format(min_n_ligands, n_idx))

        auxiliary = self.load_auxiliary()

        def _rename(x):
            return x.split('+')[0].replace('>', '')
        auxiliary['seq_dist'].rename(index = _rename, columns = _rename, inplace = True)

        # Change seq_similarity ids to sorrespond to current ids in seqs:
        seq_dist = auxiliary['seq_dist'].copy()
        seq_dist_ids = auxiliary['seq_dist_ids'].copy()
        seq_dist_ids.index.name = 'seq_dist_id'

        _seqs = seqs.reset_index(drop=False)
        _seqs = _seqs[['seq_id', 'mutated_Sequence']]
        _seqs = _seqs.set_index('mutated_Sequence')

        map_seq_ids = seq_dist_ids.join(_seqs, on = 'mutated_Sequence')
        map_seq_ids = map_seq_ids['seq_id'].to_dict()

        seq_dist.rename(index = map_seq_ids, columns = map_seq_ids, inplace = True)

        # Tanimoto Clustering - HDBSCAN:
        hdbscan_clustreing = hdbscan.HDBSCAN(min_samples = hdbscan_min_samples, 
                                            min_cluster_size = hdbscan_min_cluster_size, 
                                            metric = 'precomputed', 
                                            cluster_selection_epsilon = 0.0)
        hdbscan_clustreing.fit(seq_dist.values)
        _cluster_labels = pandas.Series(hdbscan_clustreing.labels_, index = seq_dist.index, name = '_cluster_hdbscan')

        _cluster_labels = _cluster_labels[_cluster_labels >= 0] # NOTE: Ignoring "-1" cluster. 
        print('Number of unique clusters (not including "-1" cluster): {}'.format(len(_cluster_labels.unique())))

        # Sample clusters:
        np_rng = numpy.random.default_rng(seed)
        _sampled_cluster_labels = np_rng.choice(_cluster_labels.unique(), size = n_cluster_sample)
        _idx_sample = _cluster_labels[_cluster_labels.isin(_sampled_cluster_labels)].index

        _idx = _idx_sample.intersection(_idx_ligands)
        return _idx

    def func_split_data(self, data, seed, **kwargs):
        """
        """
        relative_valid_ratio = kwargs.get('relative_valid_ratio', 0.1)

        min_n_ligands = kwargs.get('min_n_ligands', 5)
        n_cluster_sample = kwargs.get('n_cluster_sample', 5)

        max_test_size = kwargs.get('max_test_size', 0.4)
        min_test_size = kwargs.get('min_test_size', 0.2)

        hdbscan_min_samples = kwargs.get('hdbscan_min_samples', 1)
        hdbscan_min_cluster_size = kwargs.get('hdbscan_min_cluster_size', 15)

        # assert (test_ratio > 0.0) and (test_ratio < 1.0)
        assert (relative_valid_ratio > 0.0) # and (valid_ratio < 1.0)

        data, seqs = data

        _idx = self.get_ORs(seqs, data, 
                                min_n_ligands = min_n_ligands, 
                                n_cluster_sample = n_cluster_sample, 
                                seed = seed, 
                                hdbscan_min_samples = hdbscan_min_samples,
                                hdbscan_min_cluster_size = hdbscan_min_cluster_size)

        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))

        _max_test_size = max_test_size*len(data_ec50)
        _min_test_size = min_test_size*len(data_ec50)

        data_test = data_ec50[data_ec50['seq_id'].isin(_idx)]

        if len(data_test) > _max_test_size:
            raise InadequateTestSetSizeError('data_test is bigger than max size: Size: {}; max_size: {}'.format(len(data_test), _max_test_size))
        if len(data_test) < _min_test_size:
            raise InadequateTestSetSizeError('data_test is smaller than min size: Size: {}; min_size: {}'.format(len(data_test), _min_test_size))

        assert len(data_test[data_test['Responsive'] == 1]) > 0.01*len(data_ec50)
        assert len(data_test[data_test['Responsive'] == 0]) > 0.01*len(data_ec50)
        if len(data_test) > 0.5*len(data_ec50):
            raise ValueError('Test data is taking more than 50% of all EC50 data.')

        data_train = data.loc[data.index.difference(data_test.index)]

        # valid split
        data_train, data_valid = train_test_split(data_train, 
                                            test_size = (relative_valid_ratio*len(data_ec50))/len(data_train),
                                            random_state = seed)
        print('Number of positive in data_test: {}'.format(len(data_test[data_test['Responsive'] == 1])))
        print('Number of negative in data_test: {}'.format(len(data_test[data_test['Responsive'] == 0])))

        print('\nWARNING: EC50_LeaveClusterOut_OR needs to be followed by discarding screening in Post-processing!!\n')

        return data_train, data_valid, data_test



# ------------------------------------------
# Deorphanization splits - Random Molecules:
# ------------------------------------------
class EC50_LeaveOut_Mol(BaseCVSplit):
    """
    Take all occurences of given molecules, take its EC50 occurences as test. Screening occurencies are NOT discarded here.
    This should be done in post-processing.

    Here we wnat to investigate the effect of predicting pairs for new sequences but which can be similar to 
    the training ones. E.g. predicting pairs for a new mutant.

    Notes:
    ------
    We keep mutants of selected sequences.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        super(EC50_LeaveOut_Mol, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'EC50_LeaveOut_Mol'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'full_data.csv'), dst = os.path.join(self.working_dir, 'full_data.csv'))
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))

        return full_data

    def get_Mols(self, data, min_n_ligands, portion, seed):
        """
        data : pandas.DataFrame

        min_n_ligands : int
            minimal number of sequences that a molecule needs to activate to be considered for test set.

        portion : float
            perentage of molecules with enough activated sequences to include in test set.

        seed : int
            random seed for numpy.random.default_rng
        """
        num_ligands = data.groupby('mol_id').apply(lambda x: x['Responsive'].sum())
        num_ligands.name = 'num_ligands'
        _idx = num_ligands[num_ligands >= min_n_ligands].index

        n_idx = len(_idx)
        print('Number of molecules with more than (or equal) {} ligands: {}'.format(min_n_ligands, n_idx))

        np_rng = numpy.random.default_rng(seed)
        _idx = np_rng.choice(_idx, size = int(numpy.ceil(portion*n_idx)))
        return _idx

    def func_split_data(self, data, seed, **kwargs):
        """
        """
        relative_valid_ratio = kwargs.get('relative_valid_ratio', 0.1)
        
        portion_mols = kwargs.get('portion_mols', 0.5)
        min_n_ligands = kwargs.get('min_n_ligands', 5)

        max_test_size = kwargs.get('max_test_size', 0.4)
        min_test_size = kwargs.get('min_test_size', 0.2)

        # assert (test_ratio > 0.0) and (test_ratio < 1.0)
        assert (relative_valid_ratio > 0.0) # and (valid_ratio < 1.0)

        _idx = self.get_Mols(data, min_n_ligands = min_n_ligands, portion = portion_mols, seed = seed)
        print('Number of unique molecules to leave out: {}'.format(len(_idx)))

        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))

        _max_test_size = max_test_size*len(data_ec50)
        _min_test_size = min_test_size*len(data_ec50)

        data_test = data_ec50[data_ec50['mol_id'].isin(_idx)]

        if len(data_test) > _max_test_size:
            raise InadequateTestSetSizeError('data_test is bigger than max size: Size: {}; max_size: {}'.format(len(data_test), _max_test_size))
        if len(data_test) < _min_test_size:
            raise InadequateTestSetSizeError('data_test is smaller than min size: Size: {}; min_size: {}'.format(len(data_test), _min_test_size))

        assert len(data_test[data_test['Responsive'] == 1]) > 0.01*len(data_ec50)
        assert len(data_test[data_test['Responsive'] == 0]) > 0.01*len(data_ec50)
        if len(data_test) > 0.5*len(data_ec50):
            raise ValueError('Test data is taking more than 50% of all EC50 data.')

        data_train = data.loc[data.index.difference(data_test.index)]

        # valid split
        data_train, data_valid = train_test_split(data_train, 
                                            test_size = (relative_valid_ratio*len(data_ec50))/len(data_train),
                                            random_state = seed)
        print('Number of positive in data_test: {}'.format(len(data_test[data_test['Responsive'] == 1])))
        print('Number of negative in data_test: {}'.format(len(data_test[data_test['Responsive'] == 0])))

        print('\nWARNING: EC50_LeaveOut_Mol needs to be followed by discarding screening in Post-processing!!\n')

        return data_train, data_valid, data_test




# -------------------------------------------
# Deorphanization splits - Cluster Molecules:
# -------------------------------------------
def tanimoto_similarity(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = DataStructs.TanimotoSimilarity(fp1,fp2)
    return s

def dist_func(smiles):
    mol1, mol2 = smiles 
    id1, smi1 = mol1
    id2, smi2 = mol2
    return {'id_1': id1, 'id_2' : id2, 'Distance' : 1 - tanimoto_similarity(smi1, smi2)}

class EC50_LeaveClusterOut_Mol(BaseCVSplit):
    """
    Take all occurences of given molecules, take its EC50 occurences as test. Screening occurencies are NOT discarded here.
    This should be done in post-processing.

    Here we wnat to investigate the effect of predicting pairs for new sequences but which can be similar to 
    the training ones. E.g. predicting pairs for a new mutant.

    Notes:
    ------
    We keep mutants of selected sequences.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        self.auxiliary_data_path = split_kwargs.pop('auxiliary_data_path')

        super(EC50_LeaveClusterOut_Mol, self).__init__(data_dir = data_dir, seed = seed, split_kwargs = split_kwargs)

        self.func_split_data_name = 'EC50_LeaveClusterOut_Mol'

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        mols = pandas.read_csv(os.path.join(self.data_dir, 'mols.csv'), sep = self.sep, index_col = 0, header = 0)
        # Copy-paste data:
        copy(src = os.path.join(self.data_dir, 'full_data.csv'), dst = os.path.join(self.working_dir, 'full_data.csv'))
        copy(src = os.path.join(self.data_dir, 'mols.csv'), dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'), dst = os.path.join(self.working_dir, 'seqs.csv'))
        return full_data, mols

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
        map_inchikey_to_canonicalSMILES = self._update_map_inchikey_to_canonicalSMILES(data['InChI Key'], auxiliary['map_inchikey_to_canonicalSMILES'])
        with open(os.path.join(self.working_dir, 'map_inchikey_to_canonicalSMILES.json'), 'w') as jsonfile:
            json.dump(map_inchikey_to_canonicalSMILES, jsonfile)
        auxiliary['map_inchikey_to_canonicalSMILES'] = map_inchikey_to_canonicalSMILES

        return auxiliary

    @staticmethod
    def _get_canonicalSMILES(row, _map_to_canonical):
        """
        """
        if row['InChI Key'] == row['InChI Key']:
            canonicalSMILES = _map_to_canonical[row['InChI Key']]
        else:
            canonicalSMILES = row['canonicalSMILES']
        return canonicalSMILES

    def get_Mols(self, mols, data, min_n_ligands, n_cluster_sample, seed, hdbscan_min_samples, hdbscan_min_cluster_size):
        """
        Parameters:
        -----------
        mols : pandas.DataFrame

        data : pandas.Dataframe

        min_n_ligands : int
            minimal number of sequences that a molecule needs to activate to be considered for test set.

        n_cluster_sample : int
            number of clusters to put to a test set.

        seed : int
            random seed for numpy.random.default_rng

        hdbscan_min_samples : int
            the number of samples in a neighbourhood for a point to be considered a core point in HDBSCAN.
            See: https://hdbscan.readthedocs.io/en/latest/api.html

        hdbscan_min_cluster_size : int
            the minimum size of clusters; single linkage splits that contain fewer points than this will be 
            considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.
            See: https://hdbscan.readthedocs.io/en/latest/api.html
        """
        num_ligands = data.groupby('mol_id').apply(lambda x: x['Responsive'].sum())
        num_ligands.name = 'num_ligands'
        _idx_ligands = num_ligands[num_ligands >= min_n_ligands].index

        n_idx = len(_idx_ligands)
        print('Number of molecules with more than (or equal) {} ligands: {}'.format(min_n_ligands, n_idx))

        auxiliary = self.load_auxiliary()
        auxiliary = self.update_auxiliary(auxiliary, mols)

        _smiles = mols.apply(lambda x: self._get_canonicalSMILES(x, _map_to_canonical = auxiliary['map_inchikey_to_canonicalSMILES']), axis = 1)
        
        pool = multiprocessing.Pool(processes=20)
        _dist = pool.map(dist_func, itertools.combinations(_smiles.iteritems(), 2))
        pool.close()
        pool.join()

        _df_dist = pandas.DataFrame(_dist)
        _df_dist.set_index(['id_1', 'id_2'], inplace = True)

        _data = _df_dist.copy()
        _data_T = _data.swaplevel()
        _data_diag = pandas.DataFrame(numpy.zeros(len(_smiles)), columns = ['Distance'], index = pandas.MultiIndex.from_arrays([_smiles.index, _smiles.index]))
        _data = pandas.concat([_data, _data_T, _data_diag])
        _data.sort_index(inplace = True)
        _dist_data = _data['Distance'].unstack()

        # Tanimoto Clustering - HDBSCAN:
        hdbscan_clustreing = hdbscan.HDBSCAN(min_samples = hdbscan_min_samples, min_cluster_size = hdbscan_min_cluster_size, metric = 'precomputed', cluster_selection_epsilon = 0.0)
        hdbscan_clustreing.fit(_dist_data.values)
        _cluster_labels = pandas.Series(hdbscan_clustreing.labels_, index = _dist_data.index, name = 'tanimoto_cluster_hdbscan')

        _cluster_labels = _cluster_labels[_cluster_labels >= 0] # NOTE: Ignoring "-1" cluster. 
        print('Number of unique clusters (not including "-1" cluster): {}'.format(len(_cluster_labels.unique())))

        # Sample clusters:
        np_rng = numpy.random.default_rng(seed)
        _sampled_cluster_labels = np_rng.choice(_cluster_labels.unique(), size = n_cluster_sample)
        _idx_sample = _cluster_labels[_cluster_labels.isin(_sampled_cluster_labels)].index
        
        _idx = _idx_sample.intersection(_idx_ligands)
        return _idx


    def func_split_data(self, data, seed, **kwargs):
        """
        """
        relative_valid_ratio = kwargs.get('relative_valid_ratio', 0.1)
        
        min_n_ligands = kwargs.get('min_n_ligands', 5)
        n_cluster_sample = kwargs.get('n_cluster_sample', 5)

        max_test_size = kwargs.get('max_test_size', 0.4)
        min_test_size = kwargs.get('min_test_size', 0.2)

        hdbscan_min_samples = kwargs.get('hdbscan_min_samples', 1)
        hdbscan_min_cluster_size = kwargs.get('hdbscan_min_cluster_size', 10)

        # assert (test_ratio > 0.0) and (test_ratio < 1.0)
        assert (relative_valid_ratio > 0.0) # and (valid_ratio < 1.0)

        data, mols = data

        _idx = self.get_Mols(mols, data, 
                                min_n_ligands = min_n_ligands, 
                                n_cluster_sample = n_cluster_sample, 
                                seed = seed, 
                                hdbscan_min_samples = hdbscan_min_samples,
                                hdbscan_min_cluster_size = hdbscan_min_cluster_size)
        print('Number of unique molecules to leave out: {}'.format(len(_idx)))

        data_ec50 = data[data['_DataQuality'] == 'ec50']
        print('Number of EC50 measurements: Positive: {}, Negative: {}'.format(len(data_ec50[data_ec50['Responsive'] == 1]), len(data_ec50[data_ec50['Responsive'] == 0])))

        _max_test_size = max_test_size*len(data_ec50)
        _min_test_size = min_test_size*len(data_ec50)

        data_test = data_ec50[data_ec50['mol_id'].isin(_idx)]

        if len(data_test) > _max_test_size:
            raise InadequateTestSetSizeError('data_test is bigger than max size: Size: {}; max_size: {}'.format(len(data_test), _max_test_size))
        if len(data_test) < _min_test_size:
            raise InadequateTestSetSizeError('data_test is smaller than min size: Size: {}; min_size: {}'.format(len(data_test), _min_test_size))

        assert len(data_test[data_test['Responsive'] == 1]) > 0.01*len(data_ec50)
        assert len(data_test[data_test['Responsive'] == 0]) > 0.01*len(data_ec50)
        if len(data_test) > 0.5*len(data_ec50):
            raise ValueError('Test data is taking more than 50% of all EC50 data.')

        data_train = data.loc[data.index.difference(data_test.index)]

        # valid split
        data_train, data_valid = train_test_split(data_train, 
                                            test_size = (relative_valid_ratio*len(data_ec50))/len(data_train),
                                            random_state = seed)
        print('Number of positive in data_test: {}'.format(len(data_test[data_test['Responsive'] == 1])))
        print('Number of negative in data_test: {}'.format(len(data_test[data_test['Responsive'] == 0])))

        print('\nWARNING: EC50_LeaveClusterOut_Mol needs to be followed by discarding screening in Post-processing!!\n')

        return data_train, data_valid, data_test