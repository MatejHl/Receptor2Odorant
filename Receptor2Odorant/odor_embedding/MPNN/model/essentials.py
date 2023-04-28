from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp
import jraph
import flax

class MPNN_update_edge(flax.linen.Module):
    edge_embedding_size : int
    widening_factor : int

    @flax.linen.compact
    def __call__(self, inputs):
        edges, sent_attributes, received_attributes = inputs
        _sent = flax.linen.Dense(self.edge_embedding_size)(sent_attributes)
        _received = flax.linen.Dense(self.edge_embedding_size)(received_attributes)
        gate = jnp.concatenate([_sent, _received], axis = -1)
        gate = flax.linen.Dense(self.edge_embedding_size)(gate)
        gate = flax.linen.sigmoid(gate)

        update = flax.linen.relu(flax.linen.Dense(self.edge_embedding_size)(edges))
        x = gate * edges + (1 - gate) * update
        return x


class MPNN_update_node(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, inputs):
        x, sent_messages = inputs
        x, _ = flax.linen.GRUCell()(carry = x, inputs = sent_messages)
        # x = flax.linen.LayerNorm()(x) # <---------- TODO: is this necessary ? ?
        return x


class TNDMPNNStep(flax.linen.Module):
    """
    TruncatedNormalDynamicMessagePassing
    """
    edge_embedding_size : int
    
    def make_network(self, update_edge_func, update_node_func):
        """
        Notes:
        ------
        jraph.GraphNetwork is a Callable function not a class.
        """
        def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
            """
            """
            m = update_edge_func((edges, sent_attributes, received_attributes))
            return m

        def aggregate_edges_for_nodes_fn(messages, indices, num_segments):
            return jax.ops.segment_sum(messages, indices, num_segments)

        def update_node_fn(nodes, sent_messages, received_messages, global_attributes):
            new_nodes = update_node_func((nodes, sent_messages))
            return new_nodes

        gn = jraph.GraphNetwork(update_edge_fn = update_edge_fn, 
                        update_node_fn = update_node_fn, 
                        update_global_fn = None, 
                        aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn, 
                        # aggregate_nodes_for_globals_fn = <function segment_sum>, 
                        # aggregate_edges_for_globals_fn = <function segment_sum>, 
                        attention_logit_fn = None, 
                        # attention_normalize_fn = <function segment_softmax>, 
                        attention_reduce_fn = None)
        return gn


    def setup(self):
        self.update_edge_func = MPNN_update_edge(self.edge_embedding_size, widening_factor = 2)
        self.update_node_func = MPNN_update_node()
        self.gn = self.make_network(update_edge_func = self.update_edge_func,
                                    update_node_func = self.update_node_func)


    def __call__(self, graph, dummy):
        """
        Notes:
        ------
        dummy and duplicate G in output are used to be compatible with flax.linen.scan
        """
        G = graph
        G = self.gn(G)
        return G, G


class TruncatedNormalDynamicMessagePassing(flax.linen.Module):
    edge_embedding_size : int
    a : int
    b : int
    mean : float
    stddev : float

    def _get_propagation_steps(self, deterministic, rng = None):
        """
        The values are drawn from a normal distribution with specified mean and standard deviation, 
        discarding and re-drawing any samples that are more than two standard deviations from the mean.
        
        References:
        -----------
        https://jax.readthedocs.io/en/latest/_autosummary/jax.random.truncated_normal.html#jax.random.truncated_normal
        """
        if not deterministic:
            if rng is None:
                rng = self.make_rng('num_steps')
            a = (self.a - self.mean)/self.stddev
            b = (self.b - self.mean)/self.stddev
            propagation_steps = jax.random.truncated_normal(rng, lower = a, upper = b, shape=())
            propagation_steps = jnp.round(self.stddev * propagation_steps + self.mean)
        else:
            propagation_steps = jnp.round(self.mean)
        return propagation_steps.astype(jnp.int32)

    @flax.linen.compact
    def __call__(self, inputs, deterministic, rng = None):
        """
        In order to be able to use jax.jit with random number of hops, MPNN is calculated for maximal possible number of hops and
        then the state corresponding to the desired number of hops is chosen.

        References:
        -----------
        https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.scan.html
        """
        G = inputs
        num_steps = self._get_propagation_steps(deterministic = deterministic, rng = rng)
        mpnn = flax.linen.transforms.scan(TNDMPNNStep, 
                                        variable_broadcast='params', 
                                        length = self.b, 
                                        split_rngs={'params': False})(self.edge_embedding_size, name = 'TNDMPNNStep')
        xs = jnp.arange(0, self.b)
        _, states = mpnn(G, xs)
        G = jax.tree_map(lambda x: x[num_steps-1, ...], states)
        return G


class MPNN(flax.linen.Module):
    num_propagation_steps : int
    edge_embedding_size : int

    @flax.linen.compact
    def __call__(self, inputs):
        """
        In order to be able to use jax.jit with random number of hops, MPNN is calculated for maximal possible number of hops and
        then the state corresponding to the desired number of hops is chosen.

        References:
        -----------
        https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.scan.html
        """
        G = inputs
        mpnn = TNDMPNNStep(self.edge_embedding_size, name = 'TNDMPNNStep')
        for _ in range(self.num_propagation_steps):
            G, _ = mpnn(G, G) # Second input and output is dummy because of truncated normal.
        return G


# -----------
# Embeddings:
# -----------

class AtomicNumEmbedding(flax.linen.Module):
    """
    Embedding for atomic number.
    """
    num_features : int

    def setup(self):
        # num_embeddings = max atomic number in the first 4 rows of periodic table = 36
        self.node_embed = flax.linen.Embed(num_embeddings = 36, features = self.num_features)

    def __call__(self, inputs):
        X = inputs
        X = X - 1 # shift to indices (atomic number starts from 1).
        return self.node_embed(X.astype(jnp.int32))


class ChiralTagEmbedding(flax.linen.Module):
    """
    Embedding for chiral tag.

    References:
    -----------
    [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.ChiralType.values
    """
    num_features : int

    def setup(self):
        # See [1] for num_embeddings explanation
        self.node_embed = flax.linen.Embed(num_embeddings = 4, features = self.num_features)

    def __call__(self, inputs):
        X = inputs
        return self.node_embed(X.astype(jnp.int32))


class HybridizationEmbedding(flax.linen.Module):
    """
    Embedding for hybridization.

    References:
    -----------
    [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.HybridizationType.values
    """
    num_features : int

    def setup(self):
        # See [1] for num_embeddings explanation
        self.node_embed = flax.linen.Embed(num_embeddings = 8, features = self.num_features)

    def __call__(self, inputs):
        X = inputs
        return self.node_embed(X.astype(jnp.int32))


class BondTypeEmbedding(flax.linen.Module):
    """
    Embedding for bond type.

    References:
    -----------
    [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType.values
    """
    num_features : int

    def setup(self):
        # See [1] for num_embeddings explanation 
        self.edge_embed = flax.linen.Embed(num_embeddings = 22, features = self.num_features)

    def __call__(self, inputs):
        E = inputs
        return self.edge_embed(E.astype(jnp.int32))


class StereoEmbedding(flax.linen.Module):
    """
    Embedding for bond stereo.

    References:
    -----------
    [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values
    """
    num_features : int

    def setup(self):
        # See [1] for num_embeddings explanation 
        self.edge_embed = flax.linen.Embed(num_embeddings = 6, features = self.num_features)

    def __call__(self, inputs):
        E = inputs
        return self.edge_embed(E.astype(jnp.int32))