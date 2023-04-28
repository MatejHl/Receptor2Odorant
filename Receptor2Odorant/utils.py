import os
import json
import functools
import jraph
import numpy
import flax
import jax
from jax import numpy as jnp
from flax.training import train_state
from typing import Any, Callable, Sequence, Optional, Iterable

from tensorflow.experimental import dlpack as tfdlpack
from jax import dlpack as jdlpack

from Receptor2Odorant.mol2graph.jraph.convert import smiles_to_jraph
from Receptor2Odorant.mol2graph.exceptions import NoBondsError


class TrainState_with_epoch(train_state.TrainState):
    """
    Epoch starts at 1.
    """
    epoch : int = 1

class TrainState_with_epoch_and_rngs(train_state.TrainState):
    """
    Epoch starts at 1.
    """
    epoch : int = 1
    rngs : Any = None


# -----------------
# Tensorflow utils:
# -----------------
def jax_to_tf(arr):
  return tfdlpack.from_dlpack(jdlpack.to_dlpack(arr))

def tf_to_jax(arr):
  return jdlpack.from_dlpack(tfdlpack.to_dlpack(arr))


# -----------
# Flax utils:
# -----------
import collections
import itertools
from jax.interpreters import xla
from flax.jax_utils import _pmap_device_order
def prefetch_to_device(iterator, size, devices=None):
    """
    References:
    -----------
    https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    """
    queue = collections.deque()
    devices = devices or _pmap_device_order()
    if len(devices) > 1:
        raise ValueError('In case of multiple devices use flax.jax_utils.prefetch_to_device')

    def _prefetch(xs):
      if hasattr(jax, "device_put"):  # jax>=0.2.0
        # return jax.device_put_sharded(list(xs), devices)
        return jax.device_put(xs, devices[0])
      else:
        raise NotImplementedError('This brench is not modified comparad to flax.jax_utils.prefetch_to_device')
        aval = jax.xla.abstractify(xs)
        assert xs.shape[0] == len(devices), (
            "The first dimension of the iterator's ndarrays is not "
            "equal to the number of devices.")
        buffers = [xla.device_put(x, devices[i])
                   for i, x in enumerate(xs)]
        return jax.pxla.ShardedDeviceArray(aval, buffers)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
      for data in itertools.islice(iterator, n):
        queue.append(jax.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
      yield queue.popleft()
      enqueue(1)


def find_params_by_node_name(params, node_name):
    """
    References:
    -----------
    https://github.com/google/flax/discussions/1654
    """
    def _is_leaf_fun(x):
        if isinstance(x, Iterable) and jax.tree_util.all_leaves(x.values()):
            return True
        return False

    def _get_key_finder(key):
        def _finder(x):
            value = x.get(key)
            return None if value is None else {key: value}
        return _finder

    filtered_params = jax.tree_map(_get_key_finder(node_name), params, is_leaf=_is_leaf_fun)

    return filtered_params

# ------------
# Graph utils:
# ------------
def _get_line_graph(nodes, senders, receivers, size):
    # NOTE: equivalent to jnp.sum(n_node), but jittable:
    sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    n = jax.ops.segment_sum(1, senders, num_segments = sum_n_node) # in_degree
    m = jax.ops.segment_sum(1, receivers, num_segments = sum_n_node) # out_degree
    n_paths = m*n
    n_edges = jnp.sum(n_paths).astype(int)
    a, b = jnp.where(senders == jnp.expand_dims(receivers, axis = -1), size = size)
    return a, b, n_edges, n_paths

_get_line_graph = jax.jit(_get_line_graph, static_argnames = ['size']) # JIT


def get_line_graph(G, _get_line_graph_func):
    """
    if batch_output:
        n_paths = m*n
        n_node = G.n_edge
        node_offsets = jnp.cumsum(G.n_node[:-1])
        n_edge = jax.tree_map(jnp.sum, jnp.split(n_paths, node_offsets))
    """
    a, b, n_edges, _ =  _get_line_graph_func(G.nodes, G.senders, G.receivers)
    line_senders = a[:n_edges]
    line_receivers = b[:n_edges]
    # n_node = jnp.array([G.edges.shape[0]])
    n_node = jnp.array([jnp.sum(G.n_edge).astype(int)])
    n_edge = jnp.array([n_edges])
    line_G = jraph.GraphsTuple(nodes = G.edges,
                                edges = G.nodes[G.receivers[line_senders]],
                                receivers = line_receivers,
                                senders = line_senders,
                                globals = G.globals,
                                n_node = n_node,
                                n_edge = n_edge)
    return line_G


def create_line_graph_and_pad(mol, padding_n_node, padding_n_edge, line_edge_multiplier = 9):
    _get_line_graph_func = functools.partial(_get_line_graph, size = (line_edge_multiplier + 1) * padding_n_node)
    line_mol = get_line_graph(mol, _get_line_graph_func = _get_line_graph_func)
    padded_mol = jraph.pad_with_graphs(mol, 
                                n_node = padding_n_node, 
                                n_edge = padding_n_edge, 
                                n_graph=2)
    padded_line_mol = jraph.pad_with_graphs(line_mol, 
                                n_node = padding_n_edge, 
                                n_edge = line_edge_multiplier * padding_n_node, 
                                n_graph=2)
    # padded_mol = padded_mol._replace(globals = jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0))
    padded_line_mol = padded_line_mol._replace(globals = jnp.expand_dims(jraph.get_node_padding_mask(padded_line_mol), axis=0))
    padded_mol = padded_mol._replace(globals = {'node_padding_mask': jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0),
                                                'edge_padding_mask' : jnp.expand_dims(jraph.get_edge_padding_mask(padded_mol), axis=0)})
    return padded_mol, padded_line_mol


def create_line_graph(mol, max_size):
    _get_line_graph_func = functools.partial(_get_line_graph, size = max_size) # 10 * padding_n_node)
    line_mol = get_line_graph(mol, _get_line_graph_func = _get_line_graph_func)
    return line_mol

def pad_graph(mol, padding_n_node, padding_n_edge):
    padded_mol = jraph.pad_with_graphs(mol, 
                                n_node = padding_n_node, 
                                n_edge = padding_n_edge, 
                                n_graph=2)
    # padded_mol = padded_mol._replace(globals = jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0))
    padded_mol = padded_mol._replace(globals = {'node_padding_mask': jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0),
                                                'edge_padding_mask' : jnp.expand_dims(jraph.get_edge_padding_mask(padded_mol), axis=0)})
    return padded_mol

def pad_graph_and_line_graph(mol, line_mol, padding_n_node, padding_n_edge):
    padded_mol = jraph.pad_with_graphs(mol, 
                                n_node = padding_n_node, 
                                n_edge = padding_n_edge, 
                                n_graph=2)
    padded_line_mol = jraph.pad_with_graphs(line_mol, 
                                n_node = padding_n_edge, 
                                n_edge = 9 * padding_n_node, 
                                n_graph=2)
    # padded_mol = padded_mol._replace(globals = jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0))
    # padded_line_mol = padded_line_mol._replace(globals = jnp.expand_dims(jraph.get_node_padding_mask(padded_line_mol), axis=0))
    padded_mol = padded_mol._replace(globals = {'node_padding_mask': jnp.expand_dims(jraph.get_node_padding_mask(padded_mol), axis=0),
                                                'edge_padding_mask' : jnp.expand_dims(jraph.get_edge_padding_mask(padded_mol), axis=0)})
    return padded_mol, padded_line_mol    

# --------------
# Serialization:
# --------------
def smiles_to_jraph_and_serialize(smiles, u = None, validate=False, IncludeHs = False, atom_features = ['AtomicNum'], bond_features = ['BondType']):
    """
    function to create jraph.GraphsTuple from smiles and transform it to list of numpy arrays.

    Order is: [nodes, edges, receivers, senders, globals, n_node, n_edge]
    """
    try:
        G = smiles_to_jraph(smiles, u = u, validate = validate, IncludeHs = IncludeHs,
                        atom_features = atom_features, bond_features = bond_features)
    except NoBondsError:
        return float('nan')
    return [numpy.array(ele) if ele is not None else None for ele in G]


def deserialize_to_jraph(values):
    """
    Create jraph.GraphsTuple from array created by smiles_to_jraph_and_serialize
    """
    values = [jnp.array(val) if val is not None else None for val in values]
    return jraph.GraphsTuple(nodes = values[0],
                            edges = values[1],
                            receivers = values[2],
                            senders = values[3],
                            globals = values[4],
                            n_node = values[5],
                            n_edge = values[6])


def serialize_BERT_hidden_states(hidden_states):
    _hidden_states = list(numpy.stack(hidden_states, axis = 1)) # Transpose
    return _hidden_states


def _serialize_hparam(val):
    """
    cast hparams value to correct format.

    dictionary is transformed to string K(key)_V(value)__K(key)_V(value)
    list is transformed to string value__value

    Notes:
    ------
    If there are any _ at the begining or end of values or keys, ValueError is raised
    """
    if isinstance(val, dict):
        _vals = []
        for key in val.keys():
            if '-' in str(key) or '-' in str(val[key]):
                raise ValueError('- found in key or value. Key: {}, val[key]: {}'.format(key, val[key]))
            _vals.append('k' + str(key) + '-V' + str(val[key]))
        return ';'.join(_vals)    
    elif isinstance(val, list):
        _vals = []
        for ele in val:
            if '-' in str(ele):
                raise ValueError('- found in key or value. Element: {}'.format(ele))
            _vals.append(str(ele))
        return ';'.join(_vals)
    elif val is None:
        return 'None'
    else:
        return val


# -------
# Vocabs:
# -------
def save_vocabs(layers, vocabs_path):
    if not os.path.isdir(vocabs_path): 
        os.makedirs(vocabs_path)
    for l in layers:
        # First two elements in vocabulary are ignored because of padding and oov values.
        with open(os.path.join(vocabs_path, l.name + '.json'), 'w') as vocab_json:
            json.dump([int(v) for v in l.get_vocabulary()][2:], vocab_json) 
    return None

def load_vocabs(layers, vocabs_path):
    for l in layers:
        with open(os.path.join(vocabs_path, l.name + '.json'), 'r') as vocab_json:
            l.set_vocabulary(json.load(vocab_json))
    return None


# -------
# Others:
# -------
def get_activation_function_by_name(name):
    """
    """
    if name == 'celu':
        return flax.linen.celu
    elif name == 'tanh':
        return jnp.tanh