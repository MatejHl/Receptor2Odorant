from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp

import jraph


def ECCLayer(mlp, root = None):
    """
    References:
    -----------
    [1] https://arxiv.org/abs/1704.02901
    [2] https://graphneural.network/layers/convolution/#eccconv

    Notes:
    ------
    In order to use bias without root, you need to make root function such that is outputs only bias.

    Edges are updated during processing. If you want to keep edges fixed consider replacing them back after computation.
    """
    def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
        return jnp.einsum('ij,ijk->ik',sent_attributes, mlp(edges))

    def aggregate_edges_for_nodes_fn(messages, indices, num_segments):
        return jax.ops.segment_sum(messages, indices, num_segments)

    def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
        X = sent_attributes
        if root is not None:
            X += root(nodes)
        return X

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


def GlobalAttnSumPoolLayer(mlp):
    """
    Attentoin pooling.

    NOTE: This function assumes empty globals in the graph!

    Parameters:
    -----------
    mlp : Callable
        map to logits. This function is there to govern trainable parameters.
        It is a map to 1-D space corresponding to getting logits for attention weights.
        For FLAX this would be something like flax.linen.Dense(features = 1, use_bias = False)

    Returns:
    --------
    mols: jraph.GraphsTuple
        original graph with 

    Examples:
    ---------
    class Weight(flax.linen.Module):
            @flax.linen.compact
            def __call__(self, x):
                return flax.linen.Dense(features = 1, use_bias = False)(x)

        weight = Weight()
        params_weight = weight.init(rng2, mols.nodes)
        def weight_call(x):
            return weight.apply(params_weight, x)

        pool = GlobalAttnSumPoolLayer(weight_call)

        mols_new = pool(mols)
        print(mols_new.globals)
    """
    def aggregate_nodes_for_globals_fn(nodes, node_gr_idx, n_graph):
        a = mlp(nodes) # map to logits
        a = jraph.segment_softmax(a, node_gr_idx, num_segments = n_graph)
        return jax.ops.segment_sum(a * nodes, node_gr_idx, num_segments = n_graph)

    def update_global_fn(node_attributes, edge_attribtutes, globals_):
        return node_attributes

    gn = jraph.GraphNetwork(update_edge_fn = None, 
                        update_node_fn = None, 
                        update_global_fn = update_global_fn, 
                        aggregate_edges_for_nodes_fn = None, 
                        aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn, 
                        # aggregate_edges_for_globals_fn = <function segment_sum>, 
                        attention_logit_fn = None, 
                        # attention_normalize_fn = <function segment_softmax>, 
                        attention_reduce_fn = None)

    return gn