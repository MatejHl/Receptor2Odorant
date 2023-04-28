# Utility functions for uniprot database https://www.uniprot.org

from typing import List
from io import StringIO
from urllib import parse
from urllib.request import Request, urlopen
import pubchempy
import pandas
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


class MutationError(Exception):
    pass

class UniprotNotFoundError(Exception):
    pass

class UniprotMultipleOutputError(Exception):
    pass

# -----------
# Downloades:
# -----------
def get_uniprot_sequences(uniprot_ids: List, check_consistency: bool = True) -> pandas.DataFrame:
        """
        Retrieve uniprot sequences based on a list of uniprot sequence identifier.

        For large lists it is recommended to perform batch retrieval.

        Parameters:
        ----------
        uniprot_ids: list 
            list of uniprot identifier

        Returns:
        --------
        df : pandas.DataFrame
            pandas dataframe with uniprot id column, sequence column and query column.

        References:
        -----------
        Original script:
        https://www.biostars.org/p/94422/

        Documentation which columns are available:
        https://www.uniprot.org/help/uniprotkb%5Fcolumn%5Fnames
        """
        url = 'https://www.uniprot.org/uploadlists/'  # This is the webserver to retrieve the Uniprot data
        params = {
            'from': "ACC+ID",
            'to': 'ACC',
            'format': 'tab',
            'query': " ".join(uniprot_ids),
            'columns': 'id,sequence'}

        data = parse.urlencode(params)
        data = data.encode('ascii')
        req = Request(url, data)
        with urlopen(req) as response:
            res = response.read()
        res = res.decode("utf-8")
        if res == '':
            raise UniprotNotFoundError('Result from uniprot is empty. Input: {}'.format(uniprot_ids))
        df_fasta = pandas.read_csv(StringIO(res), sep="\t")
        df_fasta.columns = ["Entry", "Uniprot_Sequence", "Query"]
        # it might happen that 2 different ids for a single query id are returned, split these rows
        df_fasta = df_fasta.assign(Query=df_fasta['Query'].str.split(',')).explode('Query')
        if check_consistency:
            set_uniprot_ids = set(uniprot_ids)
            set_output_ids = set(df_fasta['Entry'])
            intersect = set_output_ids.intersection(set_uniprot_ids)
            if len(intersect) < len(set_uniprot_ids):
                raise UniprotNotFoundError('Some uniprot IDs were not found: {}'.format(set_uniprot_ids.difference(set_output_ids)))
            elif len(intersect) > len(set_uniprot_ids):
                raise UniprotMultipleOutputError('More uniprot IDs found than inputs. Difference: {}'.format(set_output_ids.difference(set_uniprot_ids)))
        return df_fasta


def get_map_inchikey_to_isomericSMILES(inchikey):
    mols = pubchempy.get_compounds(inchikey, 'inchikey')
    return {mol.inchikey: mol.isomeric_smiles for mol in mols if mol is not None}


def get_map_inchikey_to_canonicalSMILES(inchikey):
    mols = pubchempy.get_compounds(inchikey, 'inchikey')
    return {mol.inchikey: mol.canonical_smiles for mol in mols if mol is not None}
    

# ----------
# Molecules:
# ----------
def enumerate_isomers(canonicalSMILES):
    """
    Get ismoers for a given canonical SMILES.

    Parameters:
    -----------
    canonicalSMILES : str
        canonical SMILES.

    Returns:
    --------
    isomers : list
        list of isomers found by rdkit EnumerateStereoisomers.

    References:
    -----------
    https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html
    """
    mol = Chem.MolFromSmiles(canonicalSMILES)
    isomers = tuple(EnumerateStereoisomers(mol))
    isomers = [Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers]
    return isomers


def pubchem_isoSMILES_to_RDKit_isoSMILES(isomericSMILES):
    """
    This is a precaution to ensure we have the same engine to create SMILES.

    Parameters:
    -----------
    isomericSMILES : str
        isomeric SMILES.

    Returns:
    --------
    isoSMILES : str
        RDKit generated isomeric SMILES
    """
    mol = Chem.MolFromSmiles(isomericSMILES)
    isoSMILES = Chem.MolToSmiles(x, isomericSmiles=True)
    return isoSMILES
    

# ----------
# Sequences:
# ----------
def _perform_mutation(mutations, seq):
    """
    Main logic for performing mutations.
    """    
    for mutation in mutations:
        # print('->' + mutation)
        _from = mutation[0]
        _to = mutation[-1]
        _position = int(mutation[1:-1]) - 1
        if seq[_position] == _from:
            seq = seq[:_position] + _to + seq[_position+1:]
        else:
            left_pos = _position - 5
            right_pos = _position + 4
            if _position < 5: left_pos = 0
            if _position > len(seq) - 4: right_pos = len(seq)
            raise MutationError('Expected letter {} on position {} in sequence arround: {}. Found {}. Mutation: {}'.format(_from, _position, seq[left_pos:right_pos], seq[_position], mutation))
    return seq


def perform_mutation(row, mutation_col = 'Mutation', seq_col = 'Sequence'):
    """
    Perform mutation on a row in pandas.DataFrame. This function is designed to be used in pandas apply.

    Paramters:
    ----------
    row : pandas.Series
        row of pandas dataframe

    mutation_col : str
        name of the column with mutation information.

    seq_col : str
        name of the column with sequences.

    Returns:
    --------
    mutated_seq : str
        mutated sequence
    """
    try:
        if row[mutation_col] == row[mutation_col] and row[seq_col] == row[seq_col]:
            seq = row[seq_col]
            mutations = row[mutation_col].strip().split('_')
            return _perform_mutation(mutations, seq)
        else:
            return row[seq_col]
    except MutationError as e:
        # print(e)
        raise e
        # return float('nan')
    except Exception as e:
        print(row)
        print(row[seq_col])
        raise  e


def merge_cols_with_priority(row, primary_col = 'Uniprot_Sequence', secondary_col = 'Sequence'): # merge_seqs
    """
    merge two columns. primary column is kept and only if entry is missing use secondary_col.
    This should be used in pandas apply.

    Paramters:
    ----------
    row : pandas.Series
        row of pandas dataframe.
    
    primary_col : str
        name of the main column.

    secondary_col : str
        name of the secondary column used only when info in the first is missing.

    Returns:
    --------
    entry : Any
        value to put to the merged column.
    """
    if row[primary_col] == row[primary_col]:
        return row[primary_col]
    else:
        return row[secondary_col]


if __name__ == '__main__':
    print(enumerate_isomers('CCCCC[C@H]1CCCC(=O)O1'))