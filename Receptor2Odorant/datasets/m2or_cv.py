import os
import pandas
import json
import numpy

from Receptor2Odorant.base_cross_validation import BaseCVPreProcess
from Receptor2Odorant.database_utils import perform_mutation, merge_cols_with_priority


class CV_ORpairs(BaseCVPreProcess):
    def __init__(self, base_working_dir, data_path):
        super(CV_ORpairs, self).__init__(base_working_dir, data_path)

        self.read_data_name = 'pairs_all'
        self.func_data_name = 'all'

    def read_data(self, data_path):
        if isinstance(data_path['pairs'], pandas.DataFrame):
            data = data_path['pairs']
            data_path['pairs'] = data.__class__.__name__
            print(data)
        else:
            data = pandas.read_csv(data_path['pairs'], sep=';', index_col = 0)
        df_uniprot = pandas.read_csv(data_path['uniprot_sequences'], sep = ';', index_col = 0)

        # Load auxillary:
        if 'map_inchikey_to_isomericSMILES' in data_path.keys() and data_path['map_inchikey_to_isomericSMILES'] is not None:
            with open(data_path['map_inchikey_to_isomericSMILES'], 'r') as jsonfile:
                map_inchikey_to_isomericSMILES = json.load(jsonfile)
        else:
            map_inchikey_to_isomericSMILES = {}

        if 'map_inchikey_to_canonicalSMILES' in data_path.keys() and data_path['map_inchikey_to_canonicalSMILES'] is not None:
            with open(data_path['map_inchikey_to_canonicalSMILES'], 'r') as jsonfile:
                map_inchikey_to_canonicalSMILES = json.load(jsonfile)
        else:
            map_inchikey_to_canonicalSMILES = {}
        
        raw_data = {'data' : data[['Mutation', 'Uniprot ID', 'Sequence',    # _Sequence
                                    'InChI Key', 'canonicalSMILES', # _mol
                                    'Parameter', # Test set construction
                                    'Value', 'Unit', 'Value_Screen', 'Unit_Screen', # Multiple concentrations
                                    'nbr_measurements', 'Type', 'Cell_line', 'Co_transfection', 'Assay System', 'Tag',    # Confidence score
                                    'Responsive',  
                                    'Mixture', 'DOI', 'Reference']],
                    'df_uniprot' : df_uniprot['Uniprot_Sequence'],
                    'map_inchikey_to_isomericSMILES' : map_inchikey_to_isomericSMILES,
                    'map_inchikey_to_canonicalSMILES' : map_inchikey_to_canonicalSMILES}

        return raw_data


    @staticmethod
    def _create_seq_id(data):
        seqs = data[['Mutation', 'Uniprot ID', '_Sequence', 'mutated_Sequence']].copy()
        seqs = seqs[~seqs['mutated_Sequence'].duplicated()]
        seqs.reset_index(drop = True, inplace = True)
        seqs.index.name = 'seq_id'
        seqs.index = 's_' + seqs.index.astype(str)
        seqs.reset_index(drop = False, inplace = True)

        data = pandas.merge(data, seqs[['seq_id', 'mutated_Sequence']], on = 'mutated_Sequence', how = 'left')
        return data, seqs

    @staticmethod
    def _create_mol_id(data):
        mols = data[['_MolID', 'InChI Key', 'canonicalSMILES', 'Mixture']].copy()
        mols = mols[~mols['_MolID'].duplicated()]
        mols.reset_index(drop = True, inplace = True)
        mols.index.name = 'mol_id'
        mols.index = 'm_' + mols.index.astype(str)
        mols.reset_index(drop = False, inplace = True)

        data = pandas.merge(data, mols[['mol_id', '_MolID']], on = '_MolID', how = 'left')
        return data, mols
    
    @staticmethod
    def _change_units(x, _from, _to):
        try:
            x = float(x)
        except:
            return x
        if _from == 'uM' and _to == 'Log(M)':
            return numpy.log10(x) - 6.0 # uM = 10^-6 M
        elif _from == 'Log(M)' and _to == 'uM':
            return 10.0**(x + 6.0)
        elif _from == 'mM' and _to == 'uM':
            return x*(10.0**3)

    @staticmethod
    def _func_data_ec50(data_ec50):
        pairs_ec50 = data_ec50.groupby(['mol_id', 'seq_id']).apply(lambda x: x['Responsive'].mean()) # TODO: apply more complex function returning {'Responsive', 'sample_weight'}
        if ((pairs_ec50 > 0.0)&(pairs_ec50 < 1.0)).any():
            _ec50_inconsitent_idx = pairs_ec50[((pairs_ec50 > 0.0)&(pairs_ec50 < 1.0))].index
            print('INFO: To discard EC50 inconsistent: {}'.format(len(_ec50_inconsitent_idx)))
            pairs_ec50 = pairs_ec50.loc[pairs_ec50.index.difference(_ec50_inconsitent_idx)]
        pairs_ec50 = pairs_ec50.astype(int)
        pairs_ec50.name = 'Responsive'
        pairs_ec50 = pairs_ec50.to_frame()
        pairs_ec50['_DataQuality'] = 'ec50'
        return pairs_ec50

    @staticmethod
    def _func_data_screening(data_screening):
        """
        """
        def num_unique_value_screen(_df):
            return len(_df['Value_Screen'].unique())

        def _is_sorted(s):
            return all(s.iloc[i] <= s.iloc[i+1] for i in range(len(s) - 1))
            
        _screening_consistency_ordering = data_screening.groupby(['mol_id', 'seq_id']).apply(lambda x: _is_sorted(x.sort_values(['Value_Screen', 'Responsive'])['Responsive']))
        _screening_consistency_ordering.name = 'Check'
        _screening_inconsistent_ordering_idx = _screening_consistency_ordering[~_screening_consistency_ordering].index
        _screening_consistent_ordering_idx = _screening_consistency_ordering[_screening_consistency_ordering].index
        print('INFO: To discard because of screening ordering (Inconsistent through Value_Screen): {}'.format(len(_screening_inconsistent_ordering_idx)))

        _screening_consistency_per_value = data_screening.groupby(['mol_id', 'seq_id', 'Value_Screen']).apply(lambda x: ((x['Responsive']==1).all() or (x['Responsive']==0).all()))
        _screening_consistency_per_value.name = 'Check'
        _screening_consistency_per_value = _screening_consistency_per_value.reset_index('Value_Screen')
        _screening_inconsitent_per_value_idx = _screening_consistency_per_value[~_screening_consistency_per_value['Check']].index
        _screening_inconsitent_per_value_idx = _screening_inconsitent_per_value_idx.drop_duplicates()
        _screening_consitent_per_value_idx = _screening_consistency_per_value[_screening_consistency_per_value['Check']].index
        _screening_consitent_per_value_idx = _screening_consitent_per_value_idx.drop_duplicates()
        print('INFO: To discard screening inconsistent per Value_Screen: {}'.format(len(_screening_inconsitent_per_value_idx)))

        _screening_inconsitent_idx = _screening_inconsitent_per_value_idx.union(_screening_inconsistent_ordering_idx)
        print('INFO: To discard screening inconsistent: {}'.format(len(_screening_inconsitent_idx)))

        pairs_screening = data_screening.groupby(['mol_id', 'seq_id']).apply(lambda x: (x['Responsive']==1).any()).astype(int) # TODO: apply more complex function returning {'Responsive', 'sample_weight'}
        pairs_screening.name = 'Responsive'
        pairs_screening = pairs_screening.to_frame()
        pairs_screening = pairs_screening.loc[pairs_screening.index.difference(_screening_inconsitent_idx)] # TODO: Discarding inconsistent here. Do we want to do it?

        _count_unique_concentrations = data_screening.groupby(['mol_id', 'seq_id']).apply(num_unique_value_screen)
        _count_unique_concentrations.name = 'num_unique_value_screen'

        pairs_screening = pairs_screening.join(_count_unique_concentrations, how='left')        

        pairs_primary = pairs_screening[pairs_screening['num_unique_value_screen'] == 1].copy()
        pairs_primary['_DataQuality'] = 'primaryScreening'
        pairs_secondary = pairs_screening[pairs_screening['num_unique_value_screen'] > 1].copy()
        pairs_secondary['_DataQuality'] = 'secondaryScreening'

        return pairs_primary, pairs_secondary


    def func_data(self, raw_data):
        data = raw_data['data'].copy()

        # Process sequenes:
        if data['Uniprot ID'].isna().all():
            data['Uniprot_Sequence'] = float('nan')
        else:
            data = data.join(raw_data['df_uniprot'], on = 'Uniprot ID', how = 'left')
        data['_Sequence'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'Uniprot_Sequence', secondary_col = 'Sequence'), axis = 1)
        data['mutated_Sequence'] = data.apply(lambda x: perform_mutation(x, mutation_col = 'Mutation', seq_col = '_Sequence'), axis = 1)
        
        # Create molecules ID:
        data['_MolID'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'InChI Key', secondary_col = 'canonicalSMILES'), axis = 1)

        # Change units:
        data.loc[data['Unit'] == 'uM', 'Value'] = data[data['Unit'] == 'uM']['Value'].apply(lambda x: self._change_units(x, _from = 'uM', _to = 'Log(M)'))
        data.loc[data['Unit'] == 'uM', 'Unit'] = 'Log(M)'
        data.loc[data['Unit_Screen'] == 'mM', 'Value_Screen'] = data[data['Unit_Screen'] == 'mM']['Value_Screen'].apply(lambda x: self._change_units(x, _from = 'mM', _to = 'uM'))
        data.loc[data['Unit_Screen'] == 'mM', 'Unit_Screen'] = 'uM'

        # Create unique ids for sequence and molecules
        data, seqs = self._create_seq_id(data)
        data, mols = self._create_mol_id(data)

        # Save sequences and molecules:
        seqs.to_csv(os.path.join(self.working_dir, 'seqs.csv'), sep = ';', index = False)
        mols.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep = ';', index = False)

        # process EC50:
        data_ec50 = data[data['Parameter'] == 'ec50']
        pairs_ec50 = self._func_data_ec50(data_ec50)

        # process Screening:
        data_screening = data[data['Parameter'] != 'ec50']

        if not data_screening.empty:
            pairs_primary, pairs_secondary = self._func_data_screening(data_screening)
            # NOTE: Delete EC50. Data quality weights are estimated separately using overlaps, so we don't need it here.
            pairs_primary = pairs_primary.loc[pairs_primary.index.difference(pairs_ec50.index)] 
            pairs_secondary = pairs_secondary.loc[pairs_secondary.index.difference(pairs_ec50.index)]

            pairs = pandas.concat([pairs_ec50, pairs_primary, pairs_secondary])
        else:
            pairs = pairs_ec50
        print('Label distribution: Positive: {}  Negative: {}'.format(len(pairs[pairs['Responsive'] == 1]), len(pairs[pairs['Responsive'] == 0])))
        
        pairs['Responsive'] = pairs['Responsive'].astype(int)

        return pairs


class CV_ORpairs_DiscardMix(CV_ORpairs):
    def __init__(self, base_working_dir, data_path): # , seed = None, split_kwargs = {}):
        super(CV_ORpairs, self).__init__(base_working_dir, data_path) # , seed, split_kwargs)

        self.read_data_name = 'pairs_all'
        self.func_data_name = 'mixDiscard'

    def func_data(self, raw_data):
        data = raw_data['data'].copy()

        # Process sequenes:
        if data['Uniprot ID'].isna().all():
            data['Uniprot_Sequence'] = float('nan')
        else:
            data = data.join(raw_data['df_uniprot'], on = 'Uniprot ID', how = 'left')
        data['_Sequence'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'Uniprot_Sequence', secondary_col = 'Sequence'), axis = 1)
        data['mutated_Sequence'] = data.apply(lambda x: perform_mutation(x, mutation_col = 'Mutation', seq_col = '_Sequence'), axis = 1)
        
        # Create molecules ID:
        data['_MolID'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'InChI Key', secondary_col = 'canonicalSMILES'), axis = 1)

        # Change units:
        data.loc[data['Unit'] == 'uM', 'Value'] = data[data['Unit'] == 'uM']['Value'].apply(lambda x: self._change_units(x, _from = 'uM', _to = 'Log(M)'))
        data.loc[data['Unit'] == 'uM', 'Unit'] = 'Log(M)'
        data.loc[data['Unit_Screen'] == 'mM', 'Value_Screen'] = data[data['Unit_Screen'] == 'mM']['Value_Screen'].apply(lambda x: self._change_units(x, _from = 'mM', _to = 'uM'))
        data.loc[data['Unit_Screen'] == 'mM', 'Unit_Screen'] = 'uM'

        # NOTE: Discard mixtures:
        _n_w_mix = len(data)
        data = data[data['Mixture'] != 'mixture']
        print('INFO: To discard mixtures (non-unique): {}'.format(_n_w_mix - len(data)))

        # Create unique ids for sequence and molecules
        data, seqs = self._create_seq_id(data)
        data, mols = self._create_mol_id(data)

        # Save sequences and molecules:
        seqs.to_csv(os.path.join(self.working_dir, 'seqs.csv'), sep = ';', index = False)
        mols.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep = ';', index = False)

        # process EC50:
        data_ec50 = data[data['Parameter'] == 'ec50']
        pairs_ec50 = self._func_data_ec50(data_ec50)

        # process Screening:
        data_screening = data[data['Parameter'] != 'ec50']

        if not data_screening.empty:
            pairs_primary, pairs_secondary = self._func_data_screening(data_screening)
            # NOTE: Delete EC50. Data quality weights are estimated separately using overlaps, so we don't need it here.
            pairs_primary = pairs_primary.loc[pairs_primary.index.difference(pairs_ec50.index)] 
            pairs_secondary = pairs_secondary.loc[pairs_secondary.index.difference(pairs_ec50.index)]

            pairs = pandas.concat([pairs_ec50, pairs_primary, pairs_secondary])
        else:
            pairs = pairs_ec50
        print('Label distribution: Positive: {}  Negative: {}'.format(len(pairs[pairs['Responsive'] == 1]), len(pairs[pairs['Responsive'] == 0])))
        
        pairs['Responsive'] = pairs['Responsive'].astype(int)

        return pairs




class CV_ORpairs_OnlyMix(CV_ORpairs):
    def __init__(self, base_working_dir, data_path):
        super(CV_ORpairs, self).__init__(base_working_dir, data_path)

        self.read_data_name = 'pairs_all'
        self.func_data_name = 'mixOnly'

    @staticmethod
    def _func_data_screening(data_screening):
        """
        """
        def num_unique_value_screen(_df):
            return len(_df['Value_Screen'].unique())

        def _is_sorted(s):
            return all(s.iloc[i] <= s.iloc[i+1] for i in range(len(s) - 1))
            
        _screening_consistency_ordering = data_screening.groupby(['mol_id', 'seq_id']).apply(lambda x: _is_sorted(x.sort_values(['Value_Screen', 'Responsive'])['Responsive']))
        _screening_consistency_ordering.name = 'Check'
        _screening_inconsistent_ordering_idx = _screening_consistency_ordering[~_screening_consistency_ordering].index
        _screening_consistent_ordering_idx = _screening_consistency_ordering[_screening_consistency_ordering].index
        print('INFO: To discard because of screening ordering (Inconsistent through Value_Screen): {}'.format(len(_screening_inconsistent_ordering_idx)))

        _screening_consistency_per_value = data_screening.groupby(['mol_id', 'seq_id', 'Value_Screen']).apply(lambda x: ((x['Responsive']==1).all() or (x['Responsive']==0).all()))
        _screening_consistency_per_value.name = 'Check'
        _screening_consistency_per_value = _screening_consistency_per_value.reset_index('Value_Screen')
        _screening_inconsitent_per_value_idx = _screening_consistency_per_value[~_screening_consistency_per_value['Check']].index
        _screening_inconsitent_per_value_idx = _screening_inconsitent_per_value_idx.drop_duplicates()
        _screening_consitent_per_value_idx = _screening_consistency_per_value[_screening_consistency_per_value['Check']].index
        _screening_consitent_per_value_idx = _screening_consitent_per_value_idx.drop_duplicates()
        print('INFO: To discard screening inconsistent per Value_Screen: {}'.format(len(_screening_inconsitent_per_value_idx)))

        _screening_inconsitent_idx = _screening_inconsitent_per_value_idx.union(_screening_inconsistent_ordering_idx)
        print('INFO: To discard screening inconsistent: {}'.format(len(_screening_inconsitent_idx)))

        pairs_screening = data_screening.groupby(['mol_id', 'seq_id']).apply(lambda x: (x['Responsive']==1).any()).astype(int) # TODO: apply more complex function returning {'Responsive', 'sample_weight'}
        pairs_screening.name = 'Responsive'
        pairs_screening = pairs_screening.to_frame()
        # pairs_screening = pairs_screening.loc[pairs_screening.index.difference(_screening_inconsitent_idx)] # TODO: Discarding inconsistent here. Do we want to do it?

        _count_unique_concentrations = data_screening.groupby(['mol_id', 'seq_id']).apply(num_unique_value_screen)
        _count_unique_concentrations.name = 'num_unique_value_screen'

        pairs_screening = pairs_screening.join(_count_unique_concentrations, how='left')        

        pairs_primary = pairs_screening[pairs_screening['num_unique_value_screen'] == 1].copy()
        pairs_primary['_DataQuality'] = 'primaryScreening'
        pairs_secondary = pairs_screening[pairs_screening['num_unique_value_screen'] > 1].copy()
        pairs_secondary['_DataQuality'] = 'secondaryScreening'

        return pairs_primary, pairs_secondary

    def func_data(self, raw_data):
        data = raw_data['data'].copy()

        # Process sequenes:
        if data['Uniprot ID'].isna().all():
            data['Uniprot_Sequence'] = float('nan')
        else:
            data = data.join(raw_data['df_uniprot'], on = 'Uniprot ID', how = 'left')
        data['_Sequence'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'Uniprot_Sequence', secondary_col = 'Sequence'), axis = 1)
        data['mutated_Sequence'] = data.apply(lambda x: perform_mutation(x, mutation_col = 'Mutation', seq_col = '_Sequence'), axis = 1)
        
        # Create molecules ID:
        data['_MolID'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'InChI Key', secondary_col = 'canonicalSMILES'), axis = 1)

        # Change units:
        data.loc[data['Unit'] == 'uM', 'Value'] = data[data['Unit'] == 'uM']['Value'].apply(lambda x: self._change_units(x, _from = 'uM', _to = 'Log(M)'))
        data.loc[data['Unit'] == 'uM', 'Unit'] = 'Log(M)'
        data.loc[data['Unit_Screen'] == 'mM', 'Value_Screen'] = data[data['Unit_Screen'] == 'mM']['Value_Screen'].apply(lambda x: self._change_units(x, _from = 'mM', _to = 'uM'))
        data.loc[data['Unit_Screen'] == 'mM', 'Unit_Screen'] = 'uM'

        # Discard mono and sum of isomers:
        data['Mixture'] = data['Mixture'].str.lower()
        data = data[data['Mixture'] == 'mixture']
        print('INFO: Kept mixtures (non-unique): {}'.format(len(data)))

        # Create unique ids for sequence and molecules
        data, seqs = self._create_seq_id(data)
        data, mols = self._create_mol_id(data)

        # Save sequences and molecules:
        seqs.to_csv(os.path.join(self.working_dir, 'seqs.csv'), sep = ';', index = False)
        mols.to_csv(os.path.join(self.working_dir, 'mols.csv'), sep = ';', index = False)

        # process EC50:
        data_ec50 = data[data['Parameter'].str.lower() == 'ec50']
        if not data_ec50.empty:
            pairs_ec50 = self._func_data_ec50(data_ec50)

        # process Screening:
        data_screening = data[data['Parameter'] != 'ec50']

        if not data_screening.empty:
            pairs_primary, pairs_secondary = self._func_data_screening(data_screening)
            
            if not data_ec50.empty: # NOTE: This is a quick fix only.
                # NOTE: Delete EC50. Data quality weights are estimated separately using overlaps, so we don't need it here.
                pairs_primary = pairs_primary.loc[pairs_primary.index.difference(pairs_ec50.index)] 
                pairs_secondary = pairs_secondary.loc[pairs_secondary.index.difference(pairs_ec50.index)]
                pairs = pandas.concat([pairs_ec50, pairs_primary, pairs_secondary])
            else:
                pairs = pandas.concat([pairs_primary, pairs_secondary])
        else:
            pairs = pairs_ec50
        print('Label distribution: Positive: {}  Negative: {}'.format(len(pairs[pairs['Responsive'] == 1]), len(pairs[pairs['Responsive'] == 0])))
        
        pairs['Responsive'] = pairs['Responsive'].astype(int)

        return pairs