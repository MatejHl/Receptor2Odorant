import os
import pandas
import json
import datetime

from sklearn.model_selection import train_test_split


class BaseCVPostProcess():
    """
    Class for processing datasets after creating train, valid and test. It is mainly
    intended for small dataset-wise preprocessing such as createing class or sample weights
    or aggregating classes (like putting classes with only few datapoints into [UNK] class)
    """
    def __init__(self, name, data_dir):
        """
        Parameters:
        -----------
        name : str
            name of the post-processing method. A new directory with this name will be created
            in data_dir.

        data_dir : str
            path to data directory. New subdirectories with results will be created here.
        """
        self.name = name
        self.data_dir = data_dir
        if name is not None:
            self.working_dir = os.path.join(data_dir, name)
        else:
            self.working_dir = data_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.sep = ';'
        self.orient = 'columns'

    def load_data(self):
        data_train = pandas.read_csv(os.path.join(self.data_dir, 'data_train.csv'), sep = self.sep, index_col = None, header = 0)
        data_valid = pandas.read_csv(os.path.join(self.data_dir, 'data_valid.csv'), sep = self.sep, index_col = None, header = 0)
        data_test  = pandas.read_csv(os.path.join(self.data_dir, 'data_test.csv'), sep = self.sep, index_col = None, header = 0)
        return data_train, data_valid, data_test

    def load_json_data(self):
        data_train = pandas.read_json(os.path.join(self.data_dir, 'data_train.json'), orient = self.orient)
        data_valid = pandas.read_json(os.path.join(self.data_dir, 'data_valid.json'), orient = self.orient)
        data_test  = pandas.read_json(os.path.join(self.data_dir, 'data_test.json'), orient = self.orient)
        return data_train, data_valid, data_test

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        raise NotImplementedError('Needs to be implemented by user.')

    def save_hparams(self, prefix = None):
        filename = 'hparams.json'
        if self.name is None and prefix is None:
            raise ValueError('hparams cannot be saved if both self.Name and prefix is None. Consider giving prefix in save_hparams.')
        if prefix is not None:
            filename = prefix + filename
        
        hparams = self.serialize_hparams()
        hparams.update({'working_dir' : self.working_dir})
        with open(os.path.join(self.working_dir, filename), 'w') as outfile:
            json.dump(hparams, outfile)

    def postprocess(self):
        """
        apply postprocessing on data_train, data_valid, data_test and save 
        results to the corresponding subfolder. 

        Needs to be overwritten by user. All hyperparams should be saved using save_hparams.
        """
        data_train, data_valid, data_test = self.load_data()
        raise NotImplementedError('Needs to be implemented by the user.')
        


class BaseCVPreProcess:
    """
    base class for data preparation before spliting.
    """
    def __init__(self, base_working_dir, data_path):
        """
        Parameters:
        -----------
        base_working_dir : str
            working directory from which data directories are created.

        data_path : str
            path to raw data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        self.base_working_dir = base_working_dir
        self.data_path = data_path

        self.read_data_name = None
        self.func_data_name = None

    def _get_working_dir(self):
        """
        Create direcotry as a subdirectory in base_working_dir. The direcotry is created in 
        <self.base_working_dir>/<self.func_data_name>

        This function is called whenever CV_data is called.
        """
        _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.working_dir = os.path.join(self.base_working_dir, self.func_data_name + '_' + _datetime)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def read_data(self, data_path):
        """
        function to read raw data into a DataFrame.

        Paramters:
        ----------
        data_path : str or dict
            path to data. It could dictionary if needed or basically any type, that can be serialized into json.
        """
        raise NotImplementedError('Define this function in subclass.')

    def func_data(self, raw_data):
        """
        function that takes dataframe as input and returns data
        cleared from some datapoints (like only data with "Odor type" not NaN).

        Needs to be overwriten by user.

        Paramters:
        ----------
        raw_data : pandas.DataFrame
            dataframe returned by self.read_data
        """
        raise NotImplementedError('Define this function in subclass.')

    def _serialize_attributes(self):
        if not self.read_data_name or not self.func_data_name:
            raise ValueError('functions must have names. Check if all have read_data_name or func_data_name or func_split_data_name set.')
        hparams = { 'working_dir' : self.working_dir,
                    'data_path' : self.data_path,
                    'read_data' : self.read_data_name,
                    'func_data' : self.func_data_name}
        self.hparams = hparams
        return hparams

    def save_hparams(self):
        hparams = self._serialize_attributes()
        with open(os.path.join(self.working_dir, 'CV_data_hparams.json'), 'w') as outfile:
            json.dump(hparams, outfile)

    def CV_data(self):
        """
        Creating test dataset by randomly taking rows from dataset and
        using the rest for training and validation splitted by valid_ratio.

        Parameters:
        -----------
        working_dir : str
            working directory

        func_data : object
            function that takes df as input and returns data
            cleared from some datapoints (like only data with "Odor type" not NaN).

        func_split_data : object
            function that takes data as input and outputs test_data, validation_data
            and train_data dataframes.

        kwargs:
            kwargs are passed to func_split_data.
        """
        self._get_working_dir() # Create dir if not exists
        # Read data:
        raw_data = self.read_data(self.data_path)

        # Get dataset:
        data = self.func_data(raw_data)

        # Save full data:
        data.to_csv(os.path.join(self.working_dir, 'full_data.csv'), sep=';', index = True, header = True)

        self.save_hparams()

        return None


class BaseCVSplit:
    """
    base class for splitting logic.
    """
    def __init__(self, data_dir, seed = None, split_kwargs = {}):
        """
        Parameters:
        -----------
        data_dir : str
            directory containing full preprocessed data.

        split_kwargs : dict:
            kwargs are passed to func_split_data.
        """
        self.data_dir = data_dir
        self.seed = seed
        # kwargs for func_split_data
        self.split_kwargs = split_kwargs

        self.func_split_data_name = 'random_data'

        self.sep = ';'

    def _get_working_dir(self):
        """
        Create direcotry as a subdirectory in base_working_dir. The direcotry is created in 
        <self.data_dir>/<self.func_split_data_name>/<_datetime>
        where _datetime is current datetime in "%Y%m%d-%H%M%S" format.

        This function is called whenever CV_split is called.
        """
        _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.working_dir = os.path.join(self.data_dir, self.func_split_data_name, _datetime)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def load_data(self):
        """
        function to read preprocessed full data into a DataFrame.
        """
        full_data = pandas.read_csv(os.path.join(self.data_dir, 'full_data.csv'), sep = self.sep, index_col = None, header = 0)
        return full_data

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

    def _serialize_attributes(self):
        if not self.func_split_data_name:
            raise ValueError('functions must have names. Check if all have func_split_data_name set.')
        hparams = { 'working_dir' : self.working_dir,
                    'func_split_data' : self.func_split_data_name,
                    'split_kwargs' : self.split_kwargs,
                    'seed' : self.seed}
        self.hparams = hparams
        return hparams

    def save_hparams(self):
        hparams = self._serialize_attributes()
        with open(os.path.join(self.working_dir, 'CV_split_hparams.json'), 'w') as outfile:
            json.dump(hparams, outfile)

    def CV_split(self):
        """
        Creating test dataset by randomly taking rows from dataset and
        using the rest for training and validation splitted by valid_ratio.

        Parameters:
        -----------
        working_dir : str
            working directory

        func_data : object
            function that takes df as input and returns data
            cleared from some datapoints (like only data with "Odor type" not NaN).

        func_split_data : object
            function that takes data as input and outputs test_data, validation_data
            and train_data dataframes.

        kwargs:
            kwargs are passed to func_split_data.
        """
        self._get_working_dir() # Create dir if not exists

        data = self.load_data()

        # Train, Validation and Test data:
        data_train, data_valid, data_test = self.func_split_data(data, seed = self.seed, **self.split_kwargs)

        # Save train data:
        if data_train is not None:
            print("Shape of data_train:    {}".format(data_train.shape))
            data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep = ';', index = False, header = True)

        # Save validation data:
        if data_valid is not None:
            print("Shape of data_valid:    {}".format(data_valid.shape))
            data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)

        # Save test data:
        if data_test is not None:
            print("Shape of data_test:    {}".format(data_test.shape))
            data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        self.save_hparams()

        return None

    def random_data(self, data, seed, **kwargs):
        """
        split the data from pandas.

        Paramters:
        ----------
        data : pandas.DataFrame

        valid_ratio : float
            value in (0, 1). Percentage of data to put in validation set.

        test_ratio : float
            value in (0, 1). Percentage of data to put in test set.

        seed : int
            random seed
        """
        valid_ratio = kwargs.get('valid_ratio', 0.1)
        test_ratio = kwargs.get('test_ratio', None)

        assert (valid_ratio > 0.0) and (valid_ratio < 1.0)

        # valid split
        data_train, data_valid = train_test_split(data, 
                                            test_size = valid_ratio, 
                                            random_state = seed)
        # test split
        if test_ratio is not None:
            assert (test_ratio > 0.0) and (test_ratio < 1.0)
            data_train, data_test = train_test_split(data_train, 
                                                test_size = test_ratio/(1.0-valid_ratio),
                                                random_state = seed)
        else:
            data_test = pandas.DataFrame([])

        return data_train, data_valid, data_test

    def test_only(self, data, seed, **kwargs):
        """
        Transform pre-existing test dataset. Transformation is done in cross_validation_data function and 
        all transformed data are saved so all returned _data here should be None, including test_data.
        """
        return None, None, None