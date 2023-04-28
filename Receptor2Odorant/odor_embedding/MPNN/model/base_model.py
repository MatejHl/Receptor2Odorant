from typing import Any, Callable, Sequence, Optional
import flax

from Receptor2Odorant.odor_embedding.MPNN.model.essentials import *


def GlobalAttnSumPoolLayer(mlp):
    """
    This function assumes empty globals in the graph!

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




class VanillaMPNN(flax.linen.Module):
    num_classes : int
    bond_features : Sequence
    atom_features : Sequence 
    node_d_model : int = 72
    edge_d_model : int = 36
    edge_embedding_size : int = 36
    E_edge_embedding_size : int = 72

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


        self.codes_dense_1 = flax.linen.Dense(256)
        self.codes_dense_2 = flax.linen.Dense(self.node_d_model)
        self.codes_LayerNorm = flax.linen.LayerNorm()
        
        self.dense_1 = flax.linen.Dense(392)
        self.dense_2 = flax.linen.Dense(392)
        self.dense_3 = flax.linen.Dense(392)
        self.dense_4 = flax.linen.Dense(392)

        self.LayerNorm = flax.linen.LayerNorm()
        self.GroupNorm = flax.linen.GroupNorm(num_groups=64)

        self.dense_map1 = flax.linen.Dense(409)
        self.dense_map2 = flax.linen.Dense(409)
        self.dense_map3 = flax.linen.Dense(409)

        self.X_proj_non_embeded = flax.linen.Dense(self.node_d_model) 
        self.E_proj_non_embeded = flax.linen.Dense(self.edge_d_model)

        self.mpnn = MPNN(num_propagation_steps = 5, edge_embedding_size=self.edge_embedding_size)

        self.GlobalPool_logits = flax.linen.Dense(features = 1, use_bias = False)
        self.GlobalPool = GlobalAttnSumPoolLayer(self.GlobalPool_logits)

        

        self.dropout = flax.linen.Dropout(rate = 0.12)
        self.out = flax.linen.Dense(features = self.num_classes, use_bias = True)

    

    def __call__(self, inputs, deterministic, return_embeddings):

        mols = inputs
        mols = mols._replace(globals = None)

        # Embedding for atoms:
        X = mols.nodes
        _X_embed_tree = jax.tree_multimap(lambda idx, embed_fun: embed_fun(X[:, idx]), self.atom_embed_features_pos, self.atom_embed_funcs)
        _X_other = X[:, self.atom_other_features_pos]
        _X_other = self.X_proj_non_embeded(_X_other)

        # Combining embeddings:
        _X = sum(jax.tree_leaves(_X_embed_tree)) + _X_other
        mols = mols._replace(nodes = _X)

        #Embedding for edges:
        E = mols.edges
        _E_embed_tree = jax.tree_multimap(lambda idx, embed_fun: embed_fun(E[:, idx]), self.edge_embed_features_pos, self.edge_embed_funcs)
        _E_other = E[:, self.edge_other_features_pos]
        _E_other = self.E_proj_non_embeded(_E_other)

        # Combining embeddings:
        _E = sum(jax.tree_leaves(_E_embed_tree)) + _E_other
        mols = mols._replace(edges = _E)

        # Main:
        # -----------
        mols = self.mpnn(mols)
        mols = self.GlobalPool(mols)

        x = mols.globals
        x = x[::2] 

        x = self.dense_1(x)
        x = flax.linen.relu(x)
        x = self.dense_2(x)
        x = flax.linen.relu(x) 
        x = self.dense_3(x)
        x = flax.linen.relu(x) 
        x = self.dense_4(x)
        x_embed = self.LayerNorm(x)
        
        x = self.dropout(x_embed, deterministic = deterministic)
        x = self.out(x)
        if return_embeddings:
            return x, x_embed
        else:
            return x