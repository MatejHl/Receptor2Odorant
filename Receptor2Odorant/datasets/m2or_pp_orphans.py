# -------------------------------------------------------------------------------------
# NOTE: All postprocess functions are not changing data_test (except CVPP for mixtures)
# -------------------------------------------------------------------------------------
import json
import os
from shutil import copy

import numpy
import pandas
from matplotlib import pyplot as plt
from Receptor2Odorant.mol2graph.utils import get_num_atoms_and_bonds
from Receptor2Odorant.base_cross_validation import BaseCVPostProcess


class CVPP_OrphansKeep(BaseCVPostProcess):
    """
    Keep orphans
    """
    def __init__(self, data_dir):
        name = 'orphans__keep'
        super(CVPP_OrphansKeep, self).__init__(name, data_dir)
        self.prefix = 'orphans_keep_'

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir}

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        # Just copy paste:
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        copy(src = os.path.join(self.data_dir, 'mols.csv'),
            dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'),
            dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = self.prefix)
        return None


class CVPP_OrphansDiscardBoth(BaseCVPostProcess):
    """
    Discard orphans from validation and train datasets
    """
    def __init__(self, data_dir):
        name = 'orphans__discard_both'
        super(CVPP_OrphansDiscardBoth, self).__init__(name, data_dir)
        self.prefix = 'orphans_discard_both_'

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir}

    def get_orphans(self, data):
        """
        """
        count_seq_responsive = data.groupby(by = 'seq_id').sum()['Responsive']
        orphan_seq_id = count_seq_responsive[count_seq_responsive == 0].index

        count_mol_responsive = data.groupby(by = 'mol_id').sum()['Responsive']
        orphan_mol_id = count_mol_responsive[count_mol_responsive == 0].index

        return orphan_mol_id, orphan_seq_id

    def _postprocess(self, data, orphan_mol_id, orphan_seq_id):
        """
        Discard screening both positive and negative.
        """
        return data[~((data['seq_id'].isin(orphan_seq_id))|(data['mol_id'].isin(orphan_mol_id)))]

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        orphan_mol_id, orphan_seq_id = self.get_orphans(data_train)

        if not data_train.empty:
            data_train = self._postprocess(data_train, orphan_mol_id, orphan_seq_id)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid, orphan_mol_id, orphan_seq_id)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # NOTE: data_test is just copy-paste with no postprocessing.
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        copy(src = os.path.join(self.data_dir, 'mols.csv'),
            dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'),
            dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = self.prefix)
        return None


class CVPP_OrphansDiscardOR(CVPP_OrphansDiscardBoth):
    """
    Discard orphan molecules from validation and train datasets
    """
    def __init__(self, data_dir):
        name = 'orphans__discard_OR'
        super(CVPP_OrphansDiscardBoth, self).__init__(name, data_dir)
        self.prefix = 'orphans_discard_OR_'

    def _postprocess(self, data, orphan_mol_id, orphan_seq_id):
        """
        Discard screening both positive and negative.
        """
        return data[~(data['seq_id'].isin(orphan_seq_id))]


class CVPP_OrphansDiscardMols(CVPP_OrphansDiscardBoth):
    """
    Discard orphan molecules from validation and train datasets
    """
    def __init__(self, data_dir):
        name = 'orphans__discard_mols'
        super(CVPP_OrphansDiscardBoth, self).__init__(name, data_dir)
        self.prefix = 'orphans_discard_mols_'

    def _postprocess(self, data, orphan_mol_id, orphan_seq_id):
        """
        Discard screening both positive and negative.
        """
        return data[~(data['mol_id'].isin(orphan_mol_id))]