import re
import numpy
import pandas
import json
import jraph
import jax
from jax import numpy as jnp
import functools
import copy

from Receptor2Odorant.mol2graph.utils import get_num_atoms_and_bonds

from Receptor2Odorant.mol2graph.jraph.convert import smiles_to_jraph
from Receptor2Odorant.mol2graph.exceptions import NoBondsError

from Receptor2Odorant.utils import deserialize_to_jraph, create_line_graph_and_pad, create_line_graph, pad_graph_and_line_graph, pad_graph
from Receptor2Odorant.base_loader import BaseDataset, BaseDataLoader


# raise Exception('Check if line graph is necessary...')


class ProtBERTDataset(BaseDataset):
    """
    """
    def __init__(self, data_csv, seq_col, mol_col, label_col, weight_col = None,
                atom_features = ['AtomicNum'], bond_features = ['BondType'], 
                line_graph_max_size = None,
                line_graph = True,
                **kwargs):
        """
        Parameters:
        -----------
        data_csv : str
            path to csv

        seq_col : str
            name of the column with protein sequence

        mol_col : str
            name of the column with smiles
        
        label_col : str
            name of the column with labels (in case of multilabel problem, 
            labels should be in one column)

        atom_features : list
            list of atom features.

        bond_features : list
            list of bond features.

        **kwargs
            IncludeHs
            sep
            seq_sep

        Notes:
        ------
        sequence representations are retrived in Collate.
        """
        self.data_csv = data_csv
        self.sep = kwargs.get('sep', ';')

        self.mol_col = mol_col
        self.seq_col = seq_col
        self.label_col = label_col
        self.weight_col = weight_col

        self.IncludeHs = kwargs.get('IncludeHs', False)
        self.self_loops = kwargs.get('self_loops', False)

        self.atom_features = atom_features
        self.bond_features = bond_features

        self.line_graph = line_graph
        self.line_graph_max_size = line_graph_max_size

        self.data = self.read()

    def _read_graph(self, smiles):
        """
        """
        try:
            G = smiles_to_jraph(smiles, u = None, validate = False, IncludeHs = self.IncludeHs,
                            atom_features = self.atom_features, bond_features = self.bond_features,
                            self_loops = self.self_loops)
        except NoBondsError:
            return float('nan')
        if self.line_graph:
            return (G, create_line_graph(G, max_size = self.line_graph_max_size))
        else:
            return (G, )

    def read(self):
        if isinstance(self.data_csv, pandas.DataFrame):
            if self.weight_col is not None:
                df = self.data_csv[[self.mol_col, self.seq_col, self.label_col, self.weight_col]]
            else:
                df = self.data_csv[[self.mol_col, self.seq_col, self.label_col]]
        else:
            if self.weight_col is not None:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_col, self.label_col, self.weight_col])
            else:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_col, self.label_col])

        smiles = df[self.mol_col].drop_duplicates()
        smiles.index = smiles
        smiles = smiles.apply(self._read_graph)
        smiles.dropna(inplace = True)
        smiles.rename('_graphs', inplace = True)

        df = df.join(smiles, on = self.mol_col, how = 'left')
        
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # TODO: previously jax DeviceArray and this raised assertionError 
        # in pandas. See if this behaviour changes in new padnas
        index = numpy.asarray(index)
        sample = self.data.iloc[index]
        
        seq = sample[self.seq_col]
        seq = ' '.join(list(seq))
        seq = re.sub(r"[UZOB]", "X", seq)

        mol = sample['_graphs']
        label = sample[self.label_col]
        if self.weight_col is not None:
            sample_weight = sample[self.weight_col]
            return seq, mol, (label, sample_weight)
        else:
            return seq, mol, label



def transpose_batch(batch):
    """
    Move first dimension of pytree into batch. I.e. pytree with (n_parts, n_elements, (batch_size, features, ...)) will be 
    changed to (n_elements, (n_parts, batch_size, features, ...)).
    
    Example:
    --------
    List of tuples [(X, Y, Z)] with dim(X) = (batch_size, x_size), dim(Y) = (batch_size, y_size), dim(Z) = (batch_size, z_size)
    is chaged to tuple (X', Y', Z') where dim(X') = (1, batch_size, x_size), dim(Y) = (1, batch_size, y_size), dim(Z) = (1, batch_size, z_size).
    """
    return jax.tree_multimap(lambda *x: jnp.stack(x, axis = 0), *batch)



class ProtBERTCollate:
    """
    Bookkeeping and clean manipulation with collate function.

    The aim of this class is to ease small modifications of some parts of collate function without
    the need for code redundancy. In Loader use output of make_collate as a collate function.
    """
    def __init__(self, tokenizer, padding_n_node, padding_n_edge, line_graph = True, n_partitions = 0, seq_max_length = 2048):
        self.tokenizer = tokenizer
        self.padding_n_node = padding_n_node
        self.padding_n_edge = padding_n_edge
        self.n_partitions = n_partitions
        self.seq_max_length = seq_max_length

        if line_graph:
            self._graph_collate = functools.partial(self._graph_collate_with_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)
        else:
            self._graph_collate = functools.partial(self._graph_collate_without_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)

    def _seq_collate(self, batch):
        """
        """
        tokenizer = self.tokenizer
        n_partitions = self.n_partitions
        seqs = dict(tokenizer(batch, return_tensors='np', padding = 'max_length', max_length = self.seq_max_length, truncation = True)) # 2048
        if 'position_ids' not in seqs.keys():
                seqs['position_ids'] = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(seqs['input_ids']).shape[-1]), seqs['input_ids'].shape)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions
            _seqs = []
            for i in range(n_partitions):
                _seq = {}
                for key in seqs.keys():
                    _seq[key] = seqs[key][i*partition_size:(i+1)*partition_size]
                _seqs.append(_seq)
            return _seqs
        else:
            return seqs

    @staticmethod
    def _graph_collate_with_line_graph(batch, padding_n_node, padding_n_edge, n_partitions):
        """
        For n_edges in a padding graph for line graph see https://stackoverflow.com/questions/6548283/max-number-of-paths-of-length-2-in-a-graph-with-n-nodes
        We expect on average degree of a node to be 3 (C has max degree 4, but it has often implicit hydrogen/benzene ring/double bond)
        Error will be raised if the assumption is not enough.

        Notes:
        ------
        Most of the molecules have small number of edges, so for them padding can be small. Thus padding is branched into two branches, one for small graph
        and the other for big graphs. This will triger retracing twice in jitted processing, but for most molecules only small version will be used.
        """
        mols = []
        line_mols = []
        for mol, line_mol in batch:
            if len(mol.senders) == 0 or len(mol.receivers) == 0:
                print(mol)
                raise ValueError('Molecule with no bonds encountered.')
            if len(line_mol.senders) == 0 or len(line_mol.receivers) == 0:
                print(line_mol)
                raise ValueError('Molecule with no edges encountered (line molecule with no bonds).')
            padded_mol, padded_line_mol = pad_graph_and_line_graph(mol, 
                                                                line_mol, 
                                                                padding_n_node = padding_n_node, 
                                                                padding_n_edge = padding_n_edge)
            mols.append(padded_mol)
            line_mols.append(padded_line_mol)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            return [(jraph.batch(mols[i*partition_size:(i+1)*partition_size]), 
                     jraph.batch(line_mols[i*partition_size:(i+1)*partition_size])) for i in range(n_partitions)]
        else:
            return (jraph.batch(mols), jraph.batch(line_mols))

    @staticmethod
    def _graph_collate_without_line_graph(batch, padding_n_node, padding_n_edge, n_partitions):
        """
        For n_edges in a padding graph for line graph see https://stackoverflow.com/questions/6548283/max-number-of-paths-of-length-2-in-a-graph-with-n-nodes
        We expect on average degree of a node to be 3 (C has max degree 4, but it has often implicit hydrogen/benzene ring/double bond)
        Error will be raised if the assumption is not enough.

        Notes:
        ------
        Most of the molecules have small number of edges, so for them padding can be small. Thus padding is branched into two branches, one for small graph
        and the other for big graphs. This will triger retracing twice in jitted processing, but for most molecules only small version will be used.
        """
        mols = []
        for mol in batch:
            mol = mol[0] # Output of dataset __getitem__ is a tuple even if line_graph = False.
            if len(mol.senders) == 0 or len(mol.receivers) == 0:
                print(mol)
                raise ValueError('Molecule with no bonds encountered.')
            padded_mol = pad_graph(mol, 
                                padding_n_node = padding_n_node, 
                                padding_n_edge = padding_n_edge)
            mols.append(padded_mol)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            return [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jraph.batch(mols)

    def _label_collate(self, batch):
        """
        """
        n_partitions = self.n_partitions
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions
            return [jnp.stack(batch[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)] # [numpy.stack(batch[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jnp.stack(batch)

    def make_collate(self):
        """
        Create collate function that is the input to Loader.
        """
        n_partitions = self.n_partitions
        
        def _collate(batch):
            if isinstance(batch[0], str):
                return self._seq_collate(batch)
            elif isinstance(batch[0], numpy.integer) or isinstance(batch[0], numpy.floating):
                return self._label_collate(batch)
            elif isinstance(batch[0], (tuple,list)):
                if isinstance(batch[0][0], jraph.GraphsTuple):
                    return self._graph_collate(batch)
                else:
                    transposed = zip(*batch)
                    _batch = tuple([_collate(samples) for samples in transposed])
                    if n_partitions > 0:
                        return tuple(zip(*_batch))
                    else:
                        return _batch
            else:
                raise ValueError('Unexpected type passed from dataset to loader: {}'.format(type(batch[0])))

        def collate(batch):
            batch = _collate(batch)
            if n_partitions > 0:
                batch = transpose_batch(batch)
            return batch
        
        return collate


class ProtBERTLoader(BaseDataLoader):
    """
    Paramters:
    ----------
    padding_n_node : int
        maximum number of nodes in one graph. Final padding size of a batch will be padding_n_nodes * batch_size

    padding_n_edge : int
        maximum number of edges in one graph. Final padding size of a batch will be padding_n_nodes * batch_size
    """
    def __init__(self, dataset, collate_fn,
                    batch_size=1,
                    n_partitions = 0,
                    shuffle=False, 
                    rng=None, 
                    drop_last=False):

        self.n_partitions = n_partitions
        if n_partitions > 0:
            assert batch_size % self.n_partitions == 0

        super(self.__class__, self).__init__(dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        rng = rng,
        drop_last = drop_last,
        collate_fn = collate_fn,
        )


# --------------------
# Precomputed ProtBERT
# --------------------
class ProtBERTDatasetPrecomputeBERT(BaseDataset):
    """
    consider introducing mol_buffer to save already preprocessed graphs.
    """
    def __init__(self, data_csv, seq_id_col, mol_col, label_col, weight_col = None,
                atom_features = ['AtomicNum'], bond_features = ['BondType'], 
                line_graph_max_size = None,
                line_graph = True,
                **kwargs):
        """
        Parameters:
        -----------
        data_csv : str
            path to csv

        seq_id_col : str
            name of the column with protein IDs

        mol_col : str
            name of the column with smiles
        
        label_col : str
            name of the column with labels (in case of multilabel problem, 
            labels should be in one column)

        atom_features : list
            list of atom features.

        bond_features : list
            list of bond features.

        **kwargs
            IncludeHs
            sep
            seq_sep

        Notes:
        ------
        sequence representations are retrived in Collate.
        """
        self.data_csv = data_csv
        self.sep = kwargs.get('sep', ';')

        self.mol_col = mol_col
        self.seq_id_col = seq_id_col
        self.label_col = label_col
        self.weight_col = weight_col

        self.IncludeHs = kwargs.get('IncludeHs', False)
        self.self_loops = kwargs.get('self_loops', False)

        self.atom_features = atom_features
        self.bond_features = bond_features

        self.line_graph = line_graph
        self.line_graph_max_size = line_graph_max_size

        self.data = self.read()

    def _read_graph(self, smiles):
        """
        """
        try:
            G = smiles_to_jraph(smiles, u = None, validate = False, IncludeHs = self.IncludeHs,
                            atom_features = self.atom_features, bond_features = self.bond_features,
                            self_loops = self.self_loops)
        except NoBondsError:
            return float('nan')
        # return G
        if self.line_graph:
            return (G, create_line_graph(G, max_size = self.line_graph_max_size))
        else:
            return (G, )

    def read(self):
        if isinstance(self.data_csv, pandas.DataFrame):
            if self.weight_col is not None:
                df = self.data_csv[[self.mol_col, self.seq_id_col, self.label_col, self.weight_col]]
            else:
                df = self.data_csv[[self.mol_col, self.seq_id_col, self.label_col]]
        else:
            if self.weight_col is not None:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_id_col, self.label_col, self.weight_col])
            else:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_id_col, self.label_col])

        smiles = df[self.mol_col].drop_duplicates()
        smiles.index = smiles
        smiles = smiles.apply(self._read_graph)
        smiles.dropna(inplace = True)
        smiles.rename('_graphs', inplace = True)

        df = df.join(smiles, on = self.mol_col, how = 'left')
        
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # TODO: previously jax DeviceArray and this raised assertionError 
        # in pandas. See if this behaviour changes in new padnas
        index = numpy.asarray(index)
        sample = self.data.iloc[index]
        
        seq_id = sample[self.seq_id_col]
        seq_id = str(seq_id)
        
        mol = sample['_graphs']

        if mol is None or not mol == mol:
            raise Exception('Molecule {} in sample {} is None or NaN'.format(mol, sample))

        label = sample[self.label_col]
        if self.weight_col is not None:
            sample_weight = sample[self.weight_col]
            return seq_id, mol, (label, sample_weight)
        else:
            return seq_id, mol, label



# ------------------------
# Precomputed ProtBERT CLS
# ------------------------
class ProtBERTCollatePrecomputeBERT_CLS(ProtBERTCollate):
    def __init__(self, bert_table, padding_n_node, padding_n_edge, n_partitions, line_graph = True, from_disk = False):
        self.bert_table = bert_table
        self.padding_n_node = padding_n_node
        self.padding_n_edge = padding_n_edge
        self.n_partitions = n_partitions

        if line_graph:
            self._graph_collate = functools.partial(self._graph_collate_with_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)
        else:
            self._graph_collate = functools.partial(self._graph_collate_without_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)

        if not from_disk:
            bert_dict = {}
            for row in self.bert_table.iterrows():
                bert_dict[row['id'].decode('utf-8')] = row['hidden_states']
            print('Table loaded...')
            self._seq_collate = functools.partial(self._seq_collate_from_ram, bert_dict = bert_dict, n_partitions = n_partitions)
        else:
            self._seq_collate = functools.partial(self._seq_collate_from_disk, bert_table = bert_table, n_partitions = n_partitions)
        
    @staticmethod
    def _seq_collate_from_disk(batch, bert_table, n_partitions):
        seqs_hidden_states = []
        for _id in batch:
            x = list(bert_table.where('(id == b"{}")'.format(_id)))
            if len(x) == 0:
                raise ValueError('No record found in bert_table for id: {}'.format(_id))
            seqs_hidden_states.append(x[0]['hidden_states'])

        if n_partitions > 0:
            seqs = []
            partition_size = len(batch) // n_partitions
            for i in range(n_partitions):
                seqs.append(jnp.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
        else:
            seqs = jnp.stack(seqs_hidden_states)
        return seqs

    @staticmethod
    def _seq_collate_from_ram(batch, bert_dict, n_partitions):
        seqs_hidden_states = []
        for _id in batch:
            x = bert_dict[_id]
            seqs_hidden_states.append(x)

        if n_partitions > 0:
            seqs = []
            partition_size = len(batch) // n_partitions
            for i in range(n_partitions):
                seqs.append(jnp.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
        else:
            seqs = jnp.stack(seqs_hidden_states)

        return seqs




if __name__ == '__main__':
    # Tests are run here...

    import os
    from transformers import BertTokenizer
    from transformers import FlaxBertModel, BertConfig
    import time
    import flax

    from Receptor_odorant.JAX.BERT_GNN.CLS_GTransformer.make_init import get_tf_specs
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if True:
        hparams = {'BATCH_SIZE' : 800, # 30,
                    'N_EPOCH' : 10000,
                    'N_PARTITIONS' : 0,
                    'LEARNING_RATE' : 0.001,
                    'SAVE_FREQUENCY' : 50,
                    'LOG_IMAGES_FREQUENCY' : 50,
                    'LOSS_OPTION' : 'cross_entropy',
                    'LINE_GRAPH_MAX_SIZE_MULTIPLIER' : 5,
                    # model hparmas:
                    'ATOM_FEATURES' : ('AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                    'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic'),
                    'BOND_FEATURES' : ('BondType', 'Stereo', 'IsAromatic'),
                    }
        
        _datacase = os.path.join('chemosimdb', 'all_20220513-203629', 'EC50_random_data',  '20220513-203728', 'quality__screening_weight', 'mix__concatGraph')
        _h5file = os.path.join('BERT_GNN', 'Data', 'chemosimdb', 'all_20220513-203629', 'PrecomputeProtBERT_CLS', 'ProtBERT_CLS.h5')
        datadir = os.path.join('BERT_GNN',  'Data', _datacase)

        dataparams = {'DATACASE' : _datacase,
                    'SIZE_CUT' : 'size_cut_atom32_bond64',
                    'TRAIN' : 'data_train_small.csv',
                    'VALID' : 'data_valid_small.csv',
                    'BERT_H5FILE' : _h5file,
                    'seq_id_col' : 'seq_id',
                    'mol_col' : '_SMILES',
                    'label_col' : 'Responsive',
                    'weight_col' : None,
                    'valid_weight_col' : None,
                    'loader_output_type' : 'tf',
                    }

        import tables
        h5file = tables.open_file(dataparams['BERT_H5FILE'], mode = 'r', title="TapeBERT")
        bert_table = h5file.root.bert.BERTtable
        from_disk = False

        collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                    padding_n_node = 32, padding_n_edge = 64,
                                                    n_partitions = hparams['N_PARTITIONS'],
                                                    from_disk = from_disk)

        dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, dataparams['SIZE_CUT'], dataparams['TRAIN']),
                            mol_col = dataparams['mol_col'],
                            seq_id_col = dataparams['seq_id_col'], # Gene is only sequence id.
                            label_col = dataparams['label_col'],
                            weight_col = dataparams['weight_col'],
                            atom_features = hparams['ATOM_FEATURES'],
                            bond_features = hparams['BOND_FEATURES'],
                            line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                            )

        _loader = ProtBERTLoader(dataset, 
                            batch_size = hparams['BATCH_SIZE'],
                            collate_fn = collate.make_collate(),
                            shuffle = False,     # NOTE: shuffle is redundant for tf.data.Dataset here.
                            rng = jax.random.PRNGKey(int(time.time())),
                            drop_last = False,
                            n_partitions = hparams['N_PARTITIONS'])

        valid_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, dataparams['SIZE_CUT'], dataparams['VALID']),
                            mol_col = dataparams['mol_col'],
                            seq_id_col = dataparams['seq_id_col'], # Gene is only sequence id.
                            label_col = dataparams['label_col'],
                            weight_col = dataparams['valid_weight_col'],
                            atom_features = hparams['ATOM_FEATURES'],
                            bond_features = hparams['BOND_FEATURES'],
                            line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                            )

        _valid_loader = ProtBERTLoader(valid_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

        # print(dataset.data[['seq_id', '_SMILES', 'Responsive']])
        # print(valid_dataset.data[['seq_id', '_SMILES', 'Responsive']])
        big_dataset = dataset + valid_dataset
        _big_loader = ProtBERTLoader(big_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

        for i, ele in enumerate(_big_loader):
            print(i*hparams['BATCH_SIZE'])