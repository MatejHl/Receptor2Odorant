import jax
from jax import numpy as jnp
import jraph
import time

from Receptor2Odorant.utils import pad_graph

def make_init_model(model, batch_size, atom_features, bond_features, padding_n_node, padding_n_edge, n_partitions = 0):
    """
    Parameters:
    -----------
    seq_embedding_size : int
        size of embedding. 1024 for ProtBERT, 768 for TapeBERT
    """
    num_node_features = len(atom_features)
    num_edge_features = len(bond_features)
    key = jax.random.PRNGKey(int(time.time()))
    def init_model(rngs):
        key_nodes, key_edges = jax.random.split(key)
        num_nodes = jax.random.randint(key_nodes, minval=10, maxval=padding_n_node - 1, shape = ())
        num_edges = jax.random.randint(key_edges, minval=30, maxval=padding_n_edge - 1, shape = ())

        _mol = jraph.GraphsTuple(nodes = jnp.ones(shape = (num_nodes, num_node_features), dtype = jnp.float32),
                                edges = jnp.ones(shape = (num_edges, num_edge_features), dtype = jnp.float32),
                                receivers = jnp.concatenate([jnp.arange(num_edges -1) + 1, jnp.array([0], dtype=jnp.int32)]),  
                                senders = jnp.arange(num_edges),                                                              
                                n_node = jnp.array([num_nodes]),
                                n_edge = jnp.array([num_edges]),
                                globals = None,
                                )
        batch = [_mol for _ in range(batch_size)]
        mols = []
        for mol in batch: 
            padded_mol = pad_graph(mol, 
                                padding_n_node = padding_n_node, 
                                padding_n_edge = padding_n_edge,
                                )           
            mols.append(padded_mol)

        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            mols = [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            mols = jraph.batch(mols)
        params = model.init(rngs,  mols, deterministic = False, return_embeddings=False)
        return params
    return init_model