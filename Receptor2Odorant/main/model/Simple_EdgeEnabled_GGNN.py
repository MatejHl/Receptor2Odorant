from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp
import flax
import jraph

from Receptor2Odorant.main.layers.gnn import GlobalAttnSumPoolLayer

from Receptor2Odorant.main.model.essentials.embeddings import *
from Receptor2Odorant.main.model.essentials.EdgeGatedMPNN import *


class _EdgeEnabled_GGNN_model(flax.linen.Module):
    edge_embedding_size : int
    num_steps : int

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
        mpnn = flax.linen.transforms.scan(TNDMPNNStep, 
                                        variable_broadcast='params', 
                                        length = self.num_steps, 
                                        split_rngs={'params': False})(self.edge_embedding_size, name = 'TNDMPNNStep')
        xs = jnp.arange(0, self.num_steps)
        # Since the number of steps is fixed, we can take the last state of the carry. In the case
        # of random hops this is not possible that's why we need we take states and take the num_steps-th
        # state which is the same as the carry after num_steps.
        G, _ = mpnn(G, xs)
        return G


class Simple_EdgeEnabled_GGNN_model(flax.linen.Module):
    node_d_model : int = 72
    edge_d_model : int = 36
    atom_features : Sequence = ('AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic')
    bond_features : Sequence = ('BondType', 'IsAromatic') # ('BondType', 'Stereo', 'IsAromatic')
    num_steps : int = 6
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

        self.mpnn = _EdgeEnabled_GGNN_model(edge_embedding_size = self.edge_d_model, 
                                            num_steps = self.num_steps)

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
        line_mols_padding_mask = mols.globals['edge_padding_mask']
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

        # Embedding for edges:
        E = mols.edges # line_mols.nodes
        _E_embed_tree = jax.tree_multimap(lambda idx, embed_fun: embed_fun(E[:, idx]), self.edge_embed_features_pos, self.edge_embed_funcs)
        _E_other = E[:, self.edge_other_features_pos]
        _E_other = self.E_proj_non_embeded(_E_other)

        # Combining embeddings:
        _E = sum(jax.tree_leaves(_E_embed_tree)) + _E_other

        # NOTE: Line below is here for seting padding edges to 0. 
        #       This is redundant and is here just as a precaution.
        _E = _E * jnp.reshape(line_mols_padding_mask, newshape=(-1, 1)) # Set padding features back to 0. 

        # Main:
        mols = mols._replace(edges = _E)
        # MPNN:
        mols = self.mpnn(mols)
        mols = self.GlobalPool(mols)

        x = mols.globals
        # assert jnp.abs(jnp.sum(x[1::2])) < self._eps # Because of LayerNorm sum(x[1::2]) won't be 0
        x = x[::2] # Because of how padding is done, every second graph is a padding graph.

        x = self.dropout(x, deterministic = deterministic)
        x = self.out(x)
        return x

