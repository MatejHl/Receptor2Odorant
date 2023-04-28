from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp
import flax
import jraph

from Receptor2Odorant.main.layers.gnn import GlobalAttnSumPoolLayer

from Receptor2Odorant.main.model.essentials.embeddings import *


class _multiHead_GAT(flax.linen.Module):
    """
    Notes:
    ------
    This is not very efficient implementation! ! !
    """
    num_heads : int = 6
    node_d_model : int = 72
    #
    heads = []

    @staticmethod
    def make_attention_logit_fn(linear_fun):
        def attention_logit_fn(sent_attributes: jnp.ndarray, 
                            received_attributes: jnp.ndarray,
                            edges: jnp.ndarray) -> jnp.ndarray:
            del edges
            x = jnp.concatenate((sent_attributes, received_attributes), axis=1)
            return linear_fun(x)
        return attention_logit_fn

    @staticmethod
    def make_attention_query_fn(fun):
        def attention_query_fn(nodes):
            return fun(nodes)
        return attention_query_fn

    @flax.linen.compact        
    def __call__(self, G):
        updated_nodes = []
        for i in range(self.num_heads):
            _G = jraph.GAT(attention_query_fn = self.make_attention_query_fn(flax.linen.Dense(features = self.node_d_model // self.num_heads, use_bias = True)),
                        attention_logit_fn = self.make_attention_logit_fn(flax.linen.Dense(features = self.node_d_model // self.num_heads)),
                        node_update_fn = None)(G)
            updated_nodes.append(_G.nodes)
        nodes = jnp.concatenate(updated_nodes, axis = -1)
        G = G._replace(nodes = G.nodes + nodes) # residual
        return G
    

class Simple_GAT_model(flax.linen.Module):
    node_d_model : int = 72
    edge_d_model : int = 36
    atom_features : Sequence = ('AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic')
    bond_features : Sequence = () # Bond features must be empty
    # Internal attributes:
    _eps = 10e-7
    atom_embed_funcs = {}
    atom_embed_features_pos = {}
    atom_other_features_pos = []
    edge_embed_funcs = {}
    edge_embed_features_pos = {}
    edge_other_features_pos = []

    def setup(self):
        # Atom embedding:
        # NOTE: cleanning needs to be done becuase after init atom_other_features_pos is updated
        #       Without cleaning this would lead to size mismatch.
        self.atom_embed_funcs.clear()
        self.atom_embed_features_pos.clear()
        self.atom_other_features_pos.clear()
        for i, name in enumerate(self.atom_features):
            if name == 'AtomicNum':
                self.atomic_num_embed = AtomicNumEmbedding(self.node_d_model)
                self.atom_embed_funcs[name] = self.atomic_num_embed
                self.atom_embed_features_pos[name] = i
            elif name == 'ChiralTag':
                self.chiral_tag_embed = ChiralTagEmbedding(self.node_d_model)
                self.atom_embed_funcs[name] = self.chiral_tag_embed
                self.atom_embed_features_pos[name] = i
            elif name == 'Hybridization':
                self.hybridization_embed = HybridizationEmbedding(self.node_d_model)
                self.atom_embed_funcs[name] = self.hybridization_embed
                self.atom_embed_features_pos[name] = i
            else:
                self.atom_other_features_pos.append(i)

        # Edge embedding:
        # NOTE: cleaning, see above.
        self.edge_embed_funcs.clear()
        self.edge_embed_features_pos.clear()
        self.edge_other_features_pos.clear()
        for i, name in enumerate(self.bond_features):
            if name == 'BondType':
                self.bond_type_embed = BondTypeEmbedding(self.edge_d_model)
                self.edge_embed_funcs[name] = self.bond_type_embed
                self.edge_embed_features_pos[name] = i
            elif name == 'Stereo':
                self.stereo_embed = StereoEmbedding(self.edge_d_model)
                self.edge_embed_funcs[name] = self.stereo_embed
                self.edge_embed_features_pos[name] = i
            else:
                self.edge_other_features_pos.append(i)

        # OR processing:
        self.OR_dense_1 = flax.linen.Dense(256)
        self.OR_dense_2 = flax.linen.Dense(self.node_d_model)
        self.OR_LayerNorm = flax.linen.LayerNorm()

        self.X_proj_non_embeded = flax.linen.Dense(self.node_d_model)
        self.E_proj_non_embeded = flax.linen.Dense(self.edge_d_model)

        # ----------- 0:
        self.mhGAT_0 = _multiHead_GAT(node_d_model = self.node_d_model)
        # ----------- 1:
        self.mhGAT_1 = _multiHead_GAT(node_d_model = self.node_d_model)
        # ----------- 2:
        self.mhGAT_2 = _multiHead_GAT(node_d_model = self.node_d_model)
        # ----------- 3:
        self.mhGAT_3 = _multiHead_GAT(node_d_model = self.node_d_model)
        # ----------- 4:
        self.mhGAT_4 = _multiHead_GAT(node_d_model = self.node_d_model)
        # ----------- OUTPUT:
        
        self.GlobalPool_logits = flax.linen.Dense(features = 1, use_bias = False)
        self.GlobalPool = GlobalAttnSumPoolLayer(self.GlobalPool_logits)

        self.dropout = flax.linen.Dropout(rate = 0.5)
        self.out = flax.linen.Dense(features = 1, use_bias = True)


    def __call__(self, inputs, deterministic):
        S, mols = inputs
        batch_size = S.shape[0]

        assert 2*batch_size == len(mols.n_node)

        S = self.OR_dense_1(S)
        S = flax.linen.relu(S)
        S = self.OR_dense_2(S)
        S = self.OR_LayerNorm(S)

        mols_padding_mask = mols.globals['node_padding_mask']
        mols = mols._replace(globals = None)

        # Embedding for atoms:
        X = mols.nodes
        _X_embed_tree = jax.tree_multimap(lambda idx, embed_fun: embed_fun(X[:, idx]), self.atom_embed_features_pos, self.atom_embed_funcs)
        _S_mols = jnp.repeat(S, repeats = X.shape[0]//batch_size, axis=0, total_repeat_length = X.shape[0])
        _X_other = jnp.concatenate([X[:, self.atom_other_features_pos], _S_mols], axis = -1)
        _X_other = self.X_proj_non_embeded(_X_other)

        # Combining embeddings:
        _X = sum(jax.tree_leaves(_X_embed_tree)) + _X_other
        _X = _X * jnp.reshape(mols_padding_mask, newshape=(-1, 1)) # Set padding features back to 0.
        mols = mols._replace(nodes = _X)

        # Main:
        # ----------- 0:
        mols = self.mhGAT_0(mols)
        # ----------- 1:
        mols = self.mhGAT_1(mols)
        # ----------- 2:
        mols = self.mhGAT_2(mols)
        # ----------- 3:
        mols = self.mhGAT_3(mols)
        # ----------- 4:
        mols = self.mhGAT_4(mols)
        # ----------- OUTPUT:
        mols = self.GlobalPool(mols)

        x = mols.globals
        # assert jnp.abs(jnp.sum(x[1::2])) < self._eps # Because of LayerNorm sum(x[1::2]) won't be 0
        x = x[::2] # Because of how padding is done, every second graph is a padding graph.

        x = self.dropout(x, deterministic = deterministic)
        x = self.out(x)
        return x