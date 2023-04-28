import os
import pandas
import json
import numpy

from Receptor2Odorant.database_utils import perform_mutation, merge_cols_with_priority


class ScreeningConfidence:
    def __init__(self, base_working_dir, data_path):
        """
        Parameters:
        -----------
        base_working_dir : str
            working directory from which data directories are created.

        data_path : str
            path to raw data.
        """
        self.base_working_dir = base_working_dir
        self.data_path = data_path

    def read_data(self, data_path):
        if isinstance(data_path['pairs'], pandas.DataFrame):
            data = data_path['pairs']
        else:
            data = pandas.read_csv(data_path['pairs'], sep=';', index_col = 0)

        print('''WARNING: Taking only Mainland 2015 data for confidence estimation 
            because there we know how to distinguish between primary and secondary sreeening''')
        data = data[data['DOI'] == 'https://doi.org/10.1038/sdata.2015.2']

        df_uniprot = pandas.read_csv(data_path['uniprot_sequences'], sep = ';', index_col = 0)
        
        raw_data = {'data' : data[['Mutation', 'Uniprot ID', 'Sequence',    # _Sequence
                                    'InChI Key', 'canonicalSMILES', # _mol
                                    'Parameter', # Test set construction
                                    'Value', 'Unit', 'Value_Screen', 'Unit_Screen', # Multiple concentrations
                                    # 'nbr_measurements', 'Type', 'Cell_line', 'Co_transfection', 'Assay System', 'Tag',    # Confidence score
                                    'Responsive',  
                                    'Mixture', 'DOI', 'Reference']],
                    'df_uniprot' : df_uniprot['Uniprot_Sequence']}

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
        mols = data[['_MolID']].copy()
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
    def num_unique_value_screen(_df):
        return len(_df['Value_Screen'].unique())


    def main(self):
        raw_data = self.read_data(self.data_path)

        data = raw_data['data'].copy()

        # Process sequenes:
        if data['Uniprot ID'].isna().all():
            data['Uniprot_Sequence'] = float('nan')
        else:
            data = data.join(raw_data['df_uniprot'], on = 'Uniprot ID', how = 'left')
        
        data['_Sequence'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'Uniprot_Sequence', secondary_col = 'Sequence'), axis = 1)
        data['mutated_Sequence'] = data.apply(lambda x: perform_mutation(x, mutation_col = 'Mutation', seq_col = '_Sequence'), axis = 1)
        
        # Process molecules:
        data['_MolID'] = data.apply(lambda x: merge_cols_with_priority(x, primary_col = 'InChI Key', secondary_col = 'canonicalSMILES'), axis = 1)

        # Change units:
        data.loc[data['Unit'] == 'uM', 'Value'] = data[data['Unit'] == 'uM']['Value'].apply(lambda x: self._change_units(x, _from = 'uM', _to = 'Log(M)'))
        data.loc[data['Unit'] == 'uM', 'Unit'] = 'Log(M)'
        data.loc[data['Unit_Screen'] == 'mM', 'Value_Screen'] = data[data['Unit_Screen'] == 'mM']['Value_Screen'].apply(lambda x: self._change_units(x, _from = 'mM', _to = 'uM'))
        data.loc[data['Unit_Screen'] == 'mM', 'Unit_Screen'] = 'uM'

        # Create unique ids for sequence and molecules
        data, seqs = self._create_seq_id(data)
        data, mols = self._create_mol_id(data)

        # process EC50:
        data_ec50 = data[data['Parameter'] == 'ec50']
        pairs_ec50 = data_ec50.groupby(['mol_id', 'seq_id']).apply(lambda x: x['Responsive'].mean()) # TODO: apply more complex function returning {'Responsive', 'sample_weight'}
        # Check for EC50 both responsive and non-responsive:
        if ((pairs_ec50 > 0.0)&(pairs_ec50 < 1.0)).any():
            _ec50_inconsitent_idx = pairs_ec50[((pairs_ec50 > 0.0)&(pairs_ec50 < 1.0))].index
            print('INFO: To discard EC50 inconsistent: {}'.format(len(_ec50_inconsitent_idx)))
            pairs_ec50 = pairs_ec50.loc[pairs_ec50.index.difference(_ec50_inconsitent_idx)]

        pairs_ec50 = pairs_ec50.astype(int)
        pairs_ec50.name = 'Responsive'
        pairs_ec50 = pairs_ec50.to_frame()
        pairs_ec50['_Parameter'] = 'ec50'

        # process Screening:
        data_screening = data[data['Parameter'] != 'ec50']

        _count_unique_concentrations = data_screening.groupby(['mol_id', 'seq_id']).apply(self.num_unique_value_screen)
        pairs_primary_idx = _count_unique_concentrations[_count_unique_concentrations == 1].index
        pairs_secondary_idx = _count_unique_concentrations[_count_unique_concentrations > 1].index

        if data_screening.empty:
            raise ValueError('No screening data.')
        
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

        pairs_primary_idx = pairs_primary_idx.difference(_screening_inconsitent_idx) # TODO: Discarding inconsistent here. Do we want to do it?
        pairs_secondary_idx = pairs_secondary_idx.difference(_screening_inconsitent_idx) # TODO: Discarding inconsistent here. Do we want to do it?
        
        pairs_primary = pairs_screening.loc[pairs_primary_idx]
        pairs_primary = pairs_primary.to_frame()
        pairs_primary['_Parameter'] = 'primaryScreening'

        pairs_secondary = pairs_screening.loc[pairs_secondary_idx]
        pairs_secondary = pairs_secondary.to_frame()
        pairs_secondary['_Parameter'] = 'secondaryScreening'

        probs = {}

        primary_vs_ec50 = pairs_ec50.join(pairs_primary, how = 'inner', lsuffix = '_ec50', rsuffix = '_primary')
        posPrimary_vs_ec50 = primary_vs_ec50[primary_vs_ec50['Responsive_primary'] == 1]
        probs['posEC50_if_posPrimary'] = sum(posPrimary_vs_ec50['Responsive_ec50'] == 1) / len(posPrimary_vs_ec50)
        probs['negEC50_if_posPrimary'] = sum(posPrimary_vs_ec50['Responsive_ec50'] == 0) / len(posPrimary_vs_ec50)
        negPrimary_vs_ec50 = primary_vs_ec50[primary_vs_ec50['Responsive_primary'] == 0]
        probs['posEC50_if_negPrimary'] = sum(negPrimary_vs_ec50['Responsive_ec50'] == 1) / len(negPrimary_vs_ec50)
        probs['negEC50_if_negPrimary'] = sum(negPrimary_vs_ec50['Responsive_ec50'] == 0) / len(negPrimary_vs_ec50)

        secondary_vs_ec50 = pairs_ec50.join(pairs_secondary, how = 'inner', lsuffix = '_ec50', rsuffix = '_secondary')
        posSecondary_vs_ec50 = secondary_vs_ec50[secondary_vs_ec50['Responsive_secondary'] == 1]
        probs['posEC50_if_posSecondary'] = sum(posSecondary_vs_ec50['Responsive_ec50'] == 1) / len(posSecondary_vs_ec50)
        probs['negEC50_if_posSecondary'] = sum(posSecondary_vs_ec50['Responsive_ec50'] == 0) / len(posSecondary_vs_ec50)
        negSecondary_vs_ec50 = secondary_vs_ec50[secondary_vs_ec50['Responsive_secondary'] == 0]
        probs['posEC50_if_negSecondary'] = sum(negSecondary_vs_ec50['Responsive_ec50'] == 1) / len(negSecondary_vs_ec50)
        probs['negEC50_if_negSecondary'] = sum(negSecondary_vs_ec50['Responsive_ec50'] == 0) / len(negSecondary_vs_ec50)

        with open(os.path.join(self.base_working_dir, 'Screening_confidence_probabilities.json'), 'w') as outfile:
            json.dump(probs, outfile)

        return probs

    
