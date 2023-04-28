from typing import Any, Callable, Sequence, Optional
import jax
from jax import numpy as jnp
import flax

from Receptor2Odorant.main.layers.attention import MultiHeadDotProductAttention_with_AttnWeights
from Receptor2Odorant.main.layers.gnn import ECCLayer, GlobalAttnSumPoolLayer
from Receptor2Odorant.main.layers.MPattns import GraphProcessingEncoderLayer

from Receptor2Odorant.main.model.essentials.BasicMPNN import BasicTruncatedNormalDynamicMessagePassing
from Receptor2Odorant.main.model.essentials.embeddings import *


class endCLS_normal_QK_model(flax.linen.Module):
    node_d_model : int = 72
    edge_d_model : int = 36
    edge_embedding_size : int = 36
    E_edge_embedding_size : int = 72
    atom_features : Sequence = ('AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic')
    bond_features : Sequence = ('BondType', 'IsAromatic') # ('BondType', 'Stereo', 'IsAromatic')
    # Internal attributes:
    _eps = 10e-7
    atom_embed_funcs = {}
    atom_embed_features_pos = {}
    atom_other_features_pos = []
    edge_embed_funcs = {}
    edge_embed_features_pos = {}
    edge_other_features_pos = []

    def setup(self):

        assert self.edge_d_model == self.edge_embedding_size

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
        self.dense_1 = flax.linen.Dense(256)
        self.dense_2 = flax.linen.Dense(self.node_d_model)
        self.LayerNorm = flax.linen.LayerNorm()

        self.X_proj_non_embeded = flax.linen.Dense(self.node_d_model) # kernel_regularizer is missing 
        self.E_proj_non_embeded = flax.linen.Dense(self.edge_d_model) # kernel_regularizer is missing

        # ----------- 0:
        self.mha_mpnn_Q = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_mpnn_K = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        # self.mha_mpnn_V = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha = GraphProcessingEncoderLayer(_mpnn_Q = self.mha_mpnn_Q, 
                                            _mpnn_K = self.mha_mpnn_K, 
                                            _mpnn_V = self.mha_mpnn_K, 
                                            num_heads = 6, d_model = self.node_d_model, dropout_rate = 0.1, widening_factor = 8)
        
        # ----------- 1:
        self.mha_mpnn_Q_1 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_mpnn_K_1 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        # self.mha_mpnn_V_1 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_1 = GraphProcessingEncoderLayer(_mpnn_Q = self.mha_mpnn_Q_1, 
                                              _mpnn_K = self.mha_mpnn_K_1, 
                                              _mpnn_V = self.mha_mpnn_K_1, 
                                              num_heads = 6, d_model = self.node_d_model, dropout_rate = 0.1, widening_factor = 8)
        
        # ----------- 2:
        self.mha_mpnn_Q_2 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_mpnn_K_2 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        # self.mha_mpnn_V_2 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_2 = GraphProcessingEncoderLayer(_mpnn_Q = self.mha_mpnn_Q_2, 
                                              _mpnn_K = self.mha_mpnn_K_2, 
                                              _mpnn_V = self.mha_mpnn_K_2, 
                                              num_heads = 6, d_model = self.node_d_model, dropout_rate = 0.1, widening_factor = 8)

        # ----------- 3:
        self.mha_mpnn_Q_3 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_mpnn_K_3 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        # self.mha_mpnn_V_3 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_3 = GraphProcessingEncoderLayer(_mpnn_Q = self.mha_mpnn_Q_3, 
                                              _mpnn_K = self.mha_mpnn_K_3, 
                                              _mpnn_V = self.mha_mpnn_K_3, 
                                              num_heads = 6, d_model = self.node_d_model, dropout_rate = 0.1, widening_factor = 8)

        # ----------- 4:
        self.mha_mpnn_Q_4 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_mpnn_K_4 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        # self.mha_mpnn_V_4 = BasicTruncatedNormalDynamicMessagePassing(edge_embedding_size=self.edge_embedding_size, a=3, b=9, mean=6.0, stddev=1.0)
        self.mha_4 = GraphProcessingEncoderLayer(_mpnn_Q = self.mha_mpnn_Q_4, 
                                              _mpnn_K = self.mha_mpnn_K_4, 
                                              _mpnn_V = self.mha_mpnn_K_4, 
                                              num_heads = 6, d_model = self.node_d_model, dropout_rate = 0.1, widening_factor = 8)

        # ----------- OUTPUT:
        self.ECC_MLP = flax.linen.DenseGeneral(features = (self.node_d_model, self.node_d_model), use_bias = False)
        self.ECC_root = flax.linen.Dense(features = self.node_d_model, use_bias = True)
        self.ECC = ECCLayer(self.ECC_MLP, self.ECC_root)

        self.GlobalPool_logits = flax.linen.Dense(features = 1, use_bias = False)
        self.GlobalPool = GlobalAttnSumPoolLayer(self.GlobalPool_logits)

        self.dropout = flax.linen.Dropout(rate = 0.5)
        self.out = flax.linen.Dense(features = 1, use_bias = True)


    def __call__(self, inputs, deterministic):
        S, mols = inputs
        batch_size = S.shape[0]

        assert 2*batch_size == len(mols.n_node)

        mols_padding_mask = mols.globals['node_padding_mask']
        line_mols_padding_mask = mols.globals['edge_padding_mask']
        mols = mols._replace(globals = None)

        # Embedding for atoms:
        X = mols.nodes
        _X_embed_tree = jax.tree_multimap(lambda idx, embed_fun: embed_fun(X[:, idx]), self.atom_embed_features_pos, self.atom_embed_funcs)
        _X_other = X[:, self.atom_other_features_pos]
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
        _E = _E * jnp.reshape(line_mols_padding_mask, newshape=(-1, 1)) # Set padding features back to 0.

        # Main:
        mols = mols._replace(edges = _E)
        # ----------- 0:
        X = self.mha((mols, mols_padding_mask), deterministic = deterministic) # Nodes are changed.
        mols = mols._replace(nodes = X)
        # ----------- 1:
        X = self.mha_1((mols, mols_padding_mask), deterministic = deterministic) # Nodes are changed.
        mols = mols._replace(nodes = X)
        # ----------- 2:
        X = self.mha_2((mols, mols_padding_mask), deterministic = deterministic) # Nodes are changed.
        mols = mols._replace(nodes = X)
        # ----------- 3:
        X = self.mha_3((mols, mols_padding_mask), deterministic = deterministic) # Nodes are changed.
        mols = mols._replace(nodes = X)
        # ----------- 4:
        X = self.mha_4((mols, mols_padding_mask), deterministic = deterministic) # Nodes are changed.
        mols = mols._replace(nodes = X)
        # ----------- OUTPUT:
        mols = self.ECC(mols) # Edges are updated here, but not used anymore
        mols = self.GlobalPool(mols)

        x = mols.globals
        # assert jnp.abs(jnp.sum(x[1::2])) < self._eps # Because of LayerNorm sum(x[1::2]) won't be 0
        x = x[::2] # Because of how padding is done, every second graph is a padding graph.

        x = jnp.concatenate([x, S], axis = -1)

        x = self.dense_1(x)
        x = flax.linen.relu(x)
        x = self.dense_2(x)
        x = self.LayerNorm(x)

        x = self.dropout(x, deterministic = deterministic)
        x = self.out(x)
        return x
