import jax
from jax import numpy as jnp
import jraph
import time

from Receptor2Odorant.utils import create_line_graph_and_pad, pad_graph

def make_init_model(model, batch_size, n_partitions = 0, seq_embedding_size = 1024, padding_n_node = 32, padding_n_edge = 64, num_node_features = 8, num_edge_features = 2, self_loops = False, line_graph = True):
    """
    Parameters:
    -----------
    seq_embedding_size : int
        size of embedding. 1024 for ProtBERT, 768 for TapeBERT
    """
    key = jax.random.PRNGKey(int(time.time()))
    def init_model(rngs):
        key_nodes, key_edges, key_S = jax.random.split(key, 3)
        num_nodes = jax.random.randint(key_nodes, minval=10, maxval=padding_n_node - 1, shape = ())
        num_edges = jax.random.randint(key_edges, minval=30, maxval=padding_n_edge - 1, shape = ())
        
        S = jax.random.normal(key = key_S, shape = (batch_size, 5 * seq_embedding_size))

        _mol = jraph.GraphsTuple(nodes = jnp.ones(shape = (num_nodes, num_node_features), dtype = jnp.float32),
                                edges = jnp.ones(shape = (num_edges, num_edge_features), dtype = jnp.float32),
                                receivers = jnp.concatenate([jnp.arange(num_edges -1) + 1, jnp.array([0], dtype=jnp.int32)]),  # circle
                                senders = jnp.arange(num_edges),                                                               # circle
                                n_node = jnp.array([num_nodes]),
                                n_edge = jnp.array([num_edges]),
                                globals = None,
                                )

        if self_loops:
            senders = jnp.concatenate([jnp.arange(num_nodes), _mol.senders])
            receivers = jnp.concatenate([jnp.arange(num_nodes), _mol.receivers])
            _mol = _mol._replace(edges = None, receivers = receivers, senders = senders)

        batch = [_mol for _ in range(batch_size)]

        if not line_graph:
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
            G = mols
        else:
            mols = []
            line_mols = []
            for mol in batch:
                padded_mol, padded_line_mol = create_line_graph_and_pad(mol, 
                                                                        padding_n_node = padding_n_node, 
                                                                        padding_n_edge = padding_n_edge,
                                                                        )
                mols.append(padded_mol)
                line_mols.append(padded_line_mol)
            if n_partitions > 0:
                partition_size = len(batch) // n_partitions        
                mols, line_mols = [(jraph.batch(mols[i*partition_size:(i+1)*partition_size]), 
                                    jraph.batch(line_mols[i*partition_size:(i+1)*partition_size])) for i in range(n_partitions)]
            else:
                mols, line_mols = (jraph.batch(mols), jraph.batch(line_mols))
            G = (mols, line_mols)

        params = model.init(rngs, (S, G), deterministic = False)
        return params
    return init_model
