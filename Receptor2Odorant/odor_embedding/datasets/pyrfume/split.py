import os
import pandas
from shutil import copy

from Receptor2Odorant.base_cross_validation import BaseCVSplit

class Random_split(BaseCVSplit):
    """
    """
    def func_split_data(self, data, seed, **kwargs):
        """
        function that takes data as input and outputs test_data, validation_data
        and train_data dataframes.

        Needs to be overwriten by user. By default calls self.random_data.

        Paramters:
        ----------
        data : pandas.DataFrame
            dataframe returned by self.func_data 
        """
        return self.random_data(data, seed, **kwargs)