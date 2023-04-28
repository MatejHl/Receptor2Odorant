from typing import Any, Callable, Sequence, Optional
from jax import numpy as jnp
import flax

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