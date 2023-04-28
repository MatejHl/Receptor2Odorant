import os
import pandas
import json
from shutil import copy
import numpy

from matplotlib import pyplot as plt

from Receptor2Odorant.mol2graph.utils import get_num_atoms_and_bonds

from Receptor2Odorant.base_cross_validation import BaseCVPostProcess

# -----------------
# Sample confidence
# -----------------
class CVPP_Discard_Primary(BaseCVPostProcess):
    """
    Discard primary screening data.
    """
    def __init__(self, data_dir):
        name = 'quality__discard_primary'
        super(CVPP_Discard_Primary, self).__init__(name, data_dir)

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir}

    def _postprocess(self, data):
        data = data[data['_DataQuality'] != 'primaryScreening']
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        if not data_train.empty:
            data_train = self._postprocess(data_train)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # NOTE: data_test is just copy-paste with no postprocessing.
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        copy(src = os.path.join(self.data_dir, 'mols.csv'),
            dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'),
            dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = 'discard_primary')
        return None


class CVPP_addWeights_DataQuality(BaseCVPostProcess):
    """
    Weighting samples based on screening quality (Primary/Secondary/EC50) and combining the weights with others.

    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir, auxiliary_data_path):
        name = 'quality__screening_weight'
        super(CVPP_addWeights_DataQuality, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir,
                'auxiliary_data_path' : self.auxiliary_data_path}

    def load_auxiliary(self):
        auxiliary = {}
        with open(self.auxiliary_data_path['screening_confidence_probs'], 'r') as jsonfile:
            auxiliary['screening_confidence_probs'] = json.load(jsonfile)

        with open(os.path.join(self.working_dir, 'addWeights_DataQuality_auxiliary.json'), 'w') as jsonfile:
            json.dump(auxiliary, jsonfile)

        return auxiliary

    @staticmethod
    def _weight_screening(row, probs):
        if row['Responsive'] == 1:
            if row['_DataQuality'] == 'primaryScreening':
                weight = probs['posEC50_if_posPrimary']
            elif row['_DataQuality'] == 'secondaryScreening':
                weight = probs['posEC50_if_posSecondary']
            elif row['_DataQuality'] == 'ec50':
                weight = 1.0
        elif row['Responsive'] == 0:
            if row['_DataQuality'] == 'primaryScreening':
                weight = probs['negEC50_if_negPrimary']
            elif row['_DataQuality'] == 'secondaryScreening':
                weight = probs['negEC50_if_negSecondary']
            elif row['_DataQuality'] == 'ec50':
                weight = 1.0
        return weight

    def _postprocess(self, data, auxilary):
        data['dataQuality_weight'] = data.apply(lambda x: self._weight_screening(x, auxilary['screening_confidence_probs']), axis = 1)
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()
        auxilary = self.load_auxiliary()

        if not data_train.empty:
            data_train = self._postprocess(data_train, auxilary)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid, auxilary)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # NOTE: data_test is just copy-paste with no postprocessing.
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)
        
        copy(src = os.path.join(self.data_dir, 'mols.csv'),
            dst = os.path.join(self.working_dir, 'mols.csv'))
        copy(src = os.path.join(self.data_dir, 'seqs.csv'),
            dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = 'addWeights_DataQuality_')
        return None


class CVPP_Naive_DataQuality(BaseCVPostProcess):
    """
    Weighting samples based on screening quality (Primary/Secondary/EC50) and combining the weights with others.

    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir):
        name = 'quality__naive'
        super(CVPP_Naive_DataQuality, self).__init__(name, data_dir)

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

        self.save_hparams(prefix = 'Naive_')
        return None