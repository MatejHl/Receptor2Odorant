import re
import numpy
import pandas
import json
import jraph
import jax
from jax import numpy as jnp
import functools

from Receptor2Odorant.mol2graph.jraph.convert import smiles_to_jraph
from Receptor2Odorant.mol2graph.jraph.convert import NoBondsError

from Receptor2Odorant.utils import deserialize_to_jraph, create_line_graph_and_pad, pad_graph
from Receptor2Odorant.base_loader import BaseDataset, BaseDataLoader


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


class Collate:
    """
    Bookkeeping and clean manipulation with collate function.

    The aim of this class is to ease small modifications of some parts of collate function without
    the need for code redundancy. In Loader use output of make_collate as a collate function.
    """
    def __init__(self, padding_n_node, padding_n_edge, n_partitions = 0):
        self.padding_n_node = padding_n_node
        self.padding_n_edge = padding_n_edge
        self.n_partitions = n_partitions

        self._graph_collate = functools.partial(self._graph_collate_without_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)

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
            if len(mol.senders) == 0 or len(mol.receivers) == 0:
                print(mol)
                raise ValueError('Molecule with no bonds encountered.')
            padded_mol = pad_graph(mol, 
                                padding_n_node = padding_n_node, 
                                padding_n_edge = padding_n_edge,
                                )
            mols.append(padded_mol)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            return [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jraph.batch(mols)

    def _numeric_collate(self, batch):
        """
        """
        n_partitions = self.n_partitions
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions
            return [jnp.stack(batch[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jnp.stack(batch)

    def make_collate(self):
        """
        Create collate function that is the input to Loader.
        """
        n_partitions = self.n_partitions
        
        def _collate(batch):
            if isinstance(batch[0], numpy.ndarray):
                return self._numeric_collate(batch)
            elif isinstance(batch[0], jraph.GraphsTuple):
                return self._graph_collate(batch)
            elif isinstance(batch[0], numpy.integer) or isinstance(batch[0], numpy.floating):
                return self._numeric_collate(batch)
            elif isinstance(batch[0], (tuple,list)): 
                transposed = zip(*batch)
                _batch = tuple([_collate(samples) for samples in transposed])
                if n_partitions > 0:
                    return tuple(zip(*_batch))
                else:
                    return _batch
            else:
                print(batch[0])
                raise ValueError('Unexpected type passed from dataset to loader: {}'.format(type(batch[0])))

        def collate(batch):
            batch = _collate(batch)
            if n_partitions > 0:
                batch = transpose_batch(batch)
            return batch
        
        return collate


class Loader(BaseDataLoader):
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

class DatasetBuilder(BaseDataset):
    """
    consider introducing mol_buffer to save already preprocessed graphs.

    Note:
    -----
    Molecules with no bonds are discarded.
    """
    def __init__(self, data_csv, mol_col, label_col, weight_col = None,
                atom_features = ['AtomicNum'], bond_features = ['BondType'],
                **kwargs):

        self.data_csv = data_csv
        self.sep = kwargs.get('sep', ';')

        self.mol_col = mol_col 
        self.label_col = label_col 
        self.weight_col = weight_col

        self.IncludeHs = kwargs.get('IncludeHs', False)

        self.atom_features = atom_features
        self.bond_features = bond_features

        self.data = self.read()

    @staticmethod
    def _cast(x):
        if isinstance(x, str): # Cast list writen in CSV.
            _x = x.strip().lower()
            if _x[0] == '[' and _x[-1] == ']': # expecting list..
                return numpy.array([float(ele) for ele in _x[1:-1].split(', ')])
            else:
                return x
        elif numpy.isnan(x).any():
            return x
        elif isinstance(x, int):
            return numpy.int32(x)
        elif isinstance(x, float):
            return numpy.float32(x)
        else:
            return x

    def _read_graph(self, smiles):
        """
        """
        try:
            G = smiles_to_jraph(smiles, u = None, validate = False, IncludeHs = self.IncludeHs,
                            atom_features = self.atom_features, bond_features = self.bond_features)
        except NoBondsError:
            return float('nan')
        except AssertionError:
            return float('nan')
        return G

    def read(self):
        column_list=[self.mol_col]
        if self.label_col is not None:
            column_list.append(self.label_col)
        if self.weight_col is not None:
            column_list.append(self.weight_col)
        if isinstance(self.data_csv, pandas.DataFrame):
            df = self.data_csv[column_list]
        else:
            df = pandas.read_csv(self.data_csv, sep = self.sep, usecols=column_list)

        smiles = df[self.mol_col].drop_duplicates()
        smiles.index = smiles
        smiles = smiles.apply(self._read_graph)
        smiles.dropna(inplace = True)
        smiles.rename('_graphs', inplace = True)

        df = df.join(smiles, on = self.mol_col, how = 'inner')

        if df.isna().any().any():
            print(df.isna().any())
            raise Exception('Data contains NaNs')

        if self.weight_col is not None:
            df['_weight'] = df[self.weight_col].apply(self._cast)
        if self.label_col is not None:
            df['_label'] = df[self.label_col].apply(self._cast)
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index = numpy.asarray(index)
        sample = self.data.iloc[index]

        mol = sample['_graphs']
        if self.label_col is None:
            if self.weight_col is not None:
                sample_weight = numpy.asarray(sample['_weight'], numpy.float32)
                return mol, (None, sample_weight)
            else:
                return (mol,)
            
        else:
            label = numpy.asarray(sample['_label'], numpy.float32)
            if self.weight_col is not None:
                sample_weight = numpy.asarray(sample['_weight'], numpy.float32)
                return mol, (label, sample_weight)
            else:
                return mol, label