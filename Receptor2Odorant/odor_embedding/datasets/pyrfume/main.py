import os
import time

from Receptor2Odorant.odor_embedding.datasets.pyrfume.preprocess import *
from Receptor2Odorant.odor_embedding.datasets.pyrfume.split import *

if __name__ == '__main__':
    cv_working_dir = '/mnt/Receptor2Odorant/Data'
    cv_prepro = Pyrfume_PreProcess(base_working_dir = cv_working_dir, data_path=None)
    cv_prepro.CV_data()

    split_working_dir = cv_prepro.working_dir

    random_split = Random_split(data_dir = split_working_dir,
                                seed = int(time.time()), 
                                split_kwargs = {'valid_ratio' : 0.1,
                                                'test_ratio' : None})                     
    random_split.CV_split()