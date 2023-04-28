"""
Message passing Attention layer
"""
from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp
import flax

from Receptor2Odorant.main.layers.attention import MultiHeadDotProductAttention_with_AttnWeights_and_QKV


class MessagePassingSingleHeadAttention(flax.linen.Module):
    """
    Returns:
    --------
    H : jax.ndarray
        Updated nodes. Jax array that has the same shape and type as mols.nodes. It can be used to replace old 
        node representation by `mols._replace(nodes = H)`
    
    Notes:
    ------
    batch_size needs to be known in front, because in general it can not be infered from the graph.
    """
    _mpnn_Q : Callable
    _mpnn_K : Callable
    _mpnn_V : Callable
    num_heads : int
    # depth : int
    d_model : int
    batch_size : int
    dropout_rate : float

    def setup(self):
        self.mha = MultiHeadDotProductAttention_with_AttnWeights_and_QKV(num_heads = self.num_heads, 
                                                                        dropout_rate = self.dropout_rate,
                                                                        )

    def __call__(self, inputs, deterministic = True):
        mols, mols_mask = inputs

        mols_Q = self._mpnn_Q(mols, deterministic = deterministic)
        mols_K = self._mpnn_K(mols, deterministic = deterministic)
        mols_V = self._mpnn_V(mols, deterministic = deterministic)

        H_Q = mols_Q.nodes
        H_K = mols_K.nodes
        H_V = mols_V.nodes

        H_Q = jnp.reshape(H_Q, (self.batch_size, -1, self.d_model)) 
        H_K = jnp.reshape(H_K, (self.batch_size, -1, self.d_model))
        H_V = jnp.reshape(H_V, (self.batch_size, -1, self.d_model))
        
        attn_mask = jnp.einsum('bi,bj->bij', mols_mask, mols_mask)
        attn_mask = jnp.expand_dims(attn_mask, axis = 1)

        H = self.mha(inputs_q = H_Q, inputs_k = H_K, inputs_v = H_V, mask = attn_mask, deterministic = deterministic)
        H = H * jnp.expand_dims(mols_mask, axis = -1)

        H = jnp.reshape(H, (-1, self.d_model))

        return H # updated nodes


class PositionwiseFeedForwardNetwork(flax.linen.Module):
    widening_factor : int

    @flax.linen.compact
    def __call__(self, inputs):
        x = inputs
        hidden_size = x.shape[-1]
        x = flax.linen.Dense(self.widening_factor * hidden_size, use_bias = True)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(hidden_size, use_bias = True)(x)
        return x

class GraphProcessingEncoderLayer(flax.linen.Module):
    """

    """
    _mpnn_Q : Callable
    _mpnn_K : Callable
    _mpnn_V : Callable
    num_heads : int
    d_model : int
    dropout_rate : float
    widening_factor : int

    @flax.linen.compact
    def __call__(self, inputs, deterministic):
        mols, mols_mask = inputs
        batch_size = mols_mask.shape[0]

        H = MessagePassingSingleHeadAttention( _mpnn_Q = self._mpnn_Q,
                            _mpnn_K = self._mpnn_K,
                            _mpnn_V = self._mpnn_V,
                            num_heads = self.num_heads,
                            d_model = self.d_model,
                            batch_size = batch_size,
                            dropout_rate = self.dropout_rate,
                            name = 'MPmha')((mols, mols_mask), deterministic)
        H = H + mols.nodes # Residual connection
        H = flax.linen.LayerNorm(name = 'layernorm_MPmha')(H)

        _H = PositionwiseFeedForwardNetwork(self.widening_factor, name = 'FFN')(H)
        H = _H + H
        H = flax.linen.LayerNorm(name = 'layernorm_FFN')(H)
        
        return H