import os
import pandas
import ast
import json
import datetime
import pyrfume
import itertools
import pubchempy
import numpy
import re

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from Receptor2Odorant.base_cross_validation import BaseCVPreProcess

class Pyrfume_PreProcess(BaseCVPreProcess):
    def __init__(self, base_working_dir, data_path):
        super(Pyrfume_PreProcess, self).__init__(base_working_dir, data_path)
        self.read_data_name = 'pyrfume_download'
        self.func_data_name = 'pyrfume_base'

        self.pubchempy_batch_size = 200
        self.is_weighted = True
        self.min_occurence = 30

    
    def read_data(self, data_path):
        """
        """
        
        def _arctander(name_of_paper = "arctander_1960"):
            identifier = pyrfume.load_data(f'{name_of_paper}/identifiers.csv')
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior_1_sparse.csv')
            arctander = behavior.merge(identifier, left_index=True, right_index=True).reset_index().drop(columns=['ChemicalName',"CAS","Stimulus"]).rename(columns={"ChastretteDetails":"Arctander","new_CID":"CID"})
            return arctander

        def _ifra(name_of_paper = "ifra_2019"):
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior.csv')
            ifra = behavior.reset_index().rename(columns={"Descriptor 1":"ifra_1","Descriptor 2":"ifra_2","Descriptor 3":"ifra_3"})
            return ifra
        
        def _keller(name_of_paper = 'keller_2016'):
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior.csv')
            keller = behavior.reset_index().drop(columns=behavior.columns[1:]).rename(columns={"Descriptor":"Keller_2016"})
            return keller
        
        def _leffingwell(name_of_paper = 'leffingwell'):
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior_sparse.csv')
            leffingwell = behavior.reset_index().drop(columns=['IsomericSMILES','Raw Labels']).rename(columns={"Labels":"leffingwell"})
            return leffingwell
  
        def _sigma(name_of_paper = "sigma_2014"):
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior_sparse.csv')
            sigma = behavior.reset_index().rename(columns={"descriptors":"sigma_2014"})
            return sigma

        def _dravniek(name_of_paper = 'dravnieks_1985'):
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior_2.csv')
            identifier = pyrfume.load_data(f'{name_of_paper}/identifiers.csv')
            dravniek = identifier.merge(pandas.DataFrame(behavior.apply(lambda x: x.nlargest(n=5).index.to_list(), axis=1)).reset_index().rename(columns={0:"Dravniek"}), on='Stimulus').drop(columns=['Stimulus','CAS','Conc','Name'])
            return dravniek

        def _goodscent(name_of_paper = 'goodscents'):
            molecules = pyrfume.load_data(f'{name_of_paper}/molecules.csv').reset_index()
            identifer = pyrfume.load_data(f'{name_of_paper}/identifiers.csv').reset_index()
            behavior = pyrfume.load_data(f'{name_of_paper}/behavior_sparse.csv').reset_index()
            goodscent = identifer.merge(behavior, on='TGSC ID', how='right').drop_duplicates('TGSC ID').dropna().merge(molecules, on='CID').drop(columns=['TGSC ID','MolecularWeight','IUPACName','name',"IsomericSMILES"])
            goodscent['Tags'] = goodscent.Tags.apply(lambda x: x.strip('{}').replace('\'','').replace('"','').split(', '))
            goodscent.rename(columns={'Tags':'Goodscent'}, inplace=True)
            return goodscent

        def clean_categorical_values(x):
            descriptors = []
            for i in x:
                if i == i:
                    if isinstance(i, list) and len(i) > 0:
                        for item in i:
                            descriptors.append(item)
                    if type(i) == str and re.match('\[', i):
                        if isinstance(ast.literal_eval(i),list) and len(i) > 0:
                            corrected = ast.literal_eval(i)
                            for item in corrected:
                                descriptors.append(item)
                    elif type(i) == str:
                        string = i.split(',')
                        for item in string:
                            descriptors.append(item)
            return descriptors

        
        raw_data = _arctander().merge(_ifra(), on='CID',how='outer').merge(_leffingwell(), on='CID',how='outer').merge(_sigma(),on='CID',how='outer' ).merge(_dravniek(),on='CID',how='outer').merge(_goodscent(),on="CID",how="outer").merge(_keller(),on="CID",how="outer")
        raw_data = raw_data.drop(index=raw_data[(raw_data.CID <= 0) | (raw_data.CID.isna())].index)
        raw_data.drop_duplicates(subset = ['CID'],inplace=True)
        raw_data['CID'] = raw_data['CID'].astype(int)
        raw_data = raw_data.set_index('CID')
        raw_data['merged_Descriptors'] = raw_data.apply(lambda x: clean_categorical_values(x), axis=1)
        raw_data = raw_data.drop(columns=['Arctander','ifra_1','ifra_2','ifra_3','leffingwell','sigma_2014','Dravniek','Goodscent','Keller_2016'])
        
        return raw_data

    def func_data(self, raw_data):
        
        def _select_descriptors(x):
            x =  [string.lower().strip() for string in x]
            return [element for element in x if element in selected_descriptors]
        
        def calculating_class_weights(y_true):
            number_dim = numpy.shape(y_true)[1]
            weights = numpy.empty([number_dim, 2])
            for i in range(number_dim):
                weights[i] = compute_class_weight(class_weight = 'balanced', classes = [0.,1.], y = y_true[:, i])
            return weights

        def compute_smiles(cids_list):
            def batch(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield iterable[ndx:min(ndx + n, l)]
            
            i = 0
            _smiles = dict()
            for batch_cid in batch(cids_list, n = self.pubchempy_batch_size):
                print('pubchempy batch: {}'.format(i))
                batch_smiles = {i.cid : i.isomeric_smiles for i in pubchempy.get_compounds(batch_cid)}
                _smiles.update(batch_smiles)
                i += 1
            
            return _smiles

        # Remove empty ones
        raw_data = raw_data[raw_data.merged_Descriptors.str.len() != 0]

        # Flatten and clean the list
        flat_descriptors = list(itertools.chain(*raw_data['merged_Descriptors'].values.tolist()))
        flat_descriptors = [string.lower().strip() for string in flat_descriptors]

        # remove ones occuring less than 30 times
        selected_descriptors = []
        for i in set(flat_descriptors):
            if flat_descriptors.count(i) >= self.min_occurence:
                selected_descriptors.append(i)
        raw_data['merged_Descriptors'] = raw_data.merged_Descriptors.apply(_select_descriptors)

        # Get SMILES
        smiles = compute_smiles(raw_data.index.astype(int).values.tolist())
        smiles = pandas.Series(smiles, name = 'SMILES')
        smiles.index.name = 'CID'
        smiles.to_csv(os.path.join(self.working_dir, 'CID_to_SMILES.csv'), sep=';', index = True)

        raw_data['SMILES'] = smiles
        df_smiles = raw_data.reset_index()[['CID','SMILES', 'merged_Descriptors']]

        # Get Encoding
        toencode = raw_data.copy()
        mlb = MultiLabelBinarizer()
        _df = pandas.DataFrame(mlb.fit_transform(toencode['merged_Descriptors']),columns=mlb.classes_)
        encoded = _df.set_index(toencode.index)
        encoded['Values'] = encoded.values.tolist()
        encoded_2 = pandas.Series(encoded.Values.values.tolist(), index=encoded.index, name="Values")

        # Save mapping to classes.
        classes_mapping = {name : i for i, name in enumerate(mlb.classes_)}
        with open(os.path.join(self.working_dir, "classes_mapping.json"), 'w') as outfile:
            json.dump(classes_mapping, outfile)

        # Merge Smiles and Encodings
        preprocess_data = df_smiles.merge(encoded_2, on='CID',how='inner')

        # Remove Mixtures and non-Organic molecules
        preprocess_data = preprocess_data.drop(index=preprocess_data[(preprocess_data.SMILES.str.contains(r'^[A-Z][a-z]$')) | (preprocess_data.SMILES.str.contains(r'\.')) | (preprocess_data.SMILES.str.contains(r'\[*\+|\-\]'))].index)

        if self.is_weighted:
            # Compute Weights
            weight_matrix = calculating_class_weights(numpy.stack(preprocess_data.Values))
            negative_weight_matrix = [item[0] for item in weight_matrix]
            positive_weight_matrix = [item[1] for item in weight_matrix]
            weight_for_eachclasses = numpy.stack(preprocess_data.Values) * positive_weight_matrix + negative_weight_matrix * (1-numpy.stack(preprocess_data.Values))
            preprocess_data['Weight'] = weight_for_eachclasses.tolist()

        return preprocess_data
