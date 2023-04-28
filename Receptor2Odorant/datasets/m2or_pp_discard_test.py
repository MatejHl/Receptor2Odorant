# -------------------------------------------------------------------------------------
# NOTE: All postprocess functions are not changing data_test (except CVPP for mixtures)
# -------------------------------------------------------------------------------------
import json
import os
from shutil import copy

import numpy
import pandas
from Receptor2Odorant.base_cross_validation import BaseCVPostProcess

# ----------------------------------------------
# Discarding ORs and molecules occuring in test:
# ----------------------------------------------
class CVPP_DiscardTest_Base(BaseCVPostProcess):
    """
    Discard pairs containing ORs from test set.
    """
    def __init__(self, data_dir):
        name = 'DiscardTest_OR_Base'
        super(CVPP_DiscardTest_Base, self).__init__(name, data_dir)
        self._col = None
        self.prefix = 'discard_test_base_'

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir}

    def _postprocess(self, data, _idx):
        """
        Discard screening both positive and negative.
        """
        raise NotImplementedError('Define how to discard here...')

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        _idx = data_test[self._col].unique()

        if not data_train.empty:
            data_train = self._postprocess(data_train, _idx)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid, _idx)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # if not data_test.empty:
        #     data_test = self._postprocess(data_test)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        copy(src = os.path.join(self.data_dir, 'mols.csv'),
            dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'),
            dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = self.prefix)
        return None


class CVPP_DiscardTest_OR_DiscardNeg(CVPP_DiscardTest_Base):
    """
    Discard pairs containing ORs from test set.
    """
    def __init__(self, data_dir):
        name = 'DiscardTest_OR_DiscardNeg'
        super(CVPP_DiscardTest_Base, self).__init__(name, data_dir)
        self._col = 'seq_id'
        self.prefix = 'discard_test_OR_DiscardNeg_'

    def _postprocess(self, data, _idx):
        """
        Discard screening both positive and negative.
        """
        _to_discard = (data[self._col].isin(_idx))
        print('Number of screening data discarded: {}'.format(_to_discard.sum()))
        data = data[~_to_discard]
        return data


class CVPP_DiscardTest_OR_KeepNeg(CVPP_DiscardTest_Base):
    """
    Discard pairs containing ORs from test set.
    """
    def __init__(self, data_dir):
        name = 'DiscardTest_OR_KeepNeg'
        super(CVPP_DiscardTest_Base, self).__init__(name, data_dir)
        self._col = 'seq_id'
        self.prefix = 'discard_test_OR_KeepNeg_'

    def _postprocess(self, data, _idx):
        """
        Discard screening positive only.
        """
        _to_discard = ((data[self._col].isin(_idx))&(data['Responsive'] == 1))
        print('Number of screening data discarded: {}'.format(_to_discard.sum()))
        data = data[~_to_discard]
        return data


class CVPP_DiscardTest_Mol_DiscardNeg(CVPP_DiscardTest_Base):
    """
    Discard pairs containing ORs from test set.
    """
    def __init__(self, data_dir):
        name = 'DiscardTest_Mol_DiscardNeg'
        super(CVPP_DiscardTest_Base, self).__init__(name, data_dir)
        self._col = 'mol_id'
        self.prefix = 'discard_test_Mol_DiscardNeg_'

    def _postprocess(self, data, _idx):
        """
        Discard screening both positive and negative.
        """
        _to_discard = (data[self._col].isin(_idx))
        print('Number of screening data discarded: {}'.format(_to_discard.sum()))
        data = data[~_to_discard]
        return data