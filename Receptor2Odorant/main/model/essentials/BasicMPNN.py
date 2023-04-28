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
        x = jnp.concatenate([sent_attributes, received_attributes, edges], axis = -1)
        x = flax.linen.Dense(self.edge_embedding_size)(x)
        x = flax.linen.relu(x)
        return x


class MPNN_update_node(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, inputs):
        x, received_messages = inputs
        x, _ = flax.linen.GRUCell()(carry = x, inputs = received_messages)
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
            # new_nodes = update_node_func((nodes, sent_messages))
            new_nodes = update_node_func((nodes, received_messages))
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


class BasicTruncatedNormalDynamicMessagePassing(flax.linen.Module):
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
        # edges_old = G.edges
        _, states = mpnn(G, xs)
        G = jax.tree_map(lambda x: x[num_steps-1, ...], states)
        # G = G._replace(edges = edges_old)
        return G