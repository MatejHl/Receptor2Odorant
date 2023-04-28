import copy
import pandas
import jax
from jax import numpy as jnp
import tensorflow as tf

from Receptor2Odorant.utils import jax_to_tf


def default_collate_fn(samples):
    X = jnp.array([sample[0] for sample in samples])
    Y = jnp.array([sample[1] for sample in samples])
    return X, Y


class BaseDataset(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __add__(self, other):
        """
        We assume existance of self.data
        """
        if len(self.data.columns) != len(other.data.columns):
            raise ValueError('Number of columns differ.')
        if not (self.data.columns == other.data.columns).all():
            raise ValueError('Column names are different in self and other.')
        data = pandas.concat([self.data, other.data], ignore_index = True).copy()
        new_class = copy.deepcopy(self)
        new_class.data = data
        return new_class


class BaseDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, rng=None, drop_last=False, collate_fn=default_collate_fn):
        """
        Adapted from Pytorch Dataloader implementation.
        Parameters:
        -----------
        dataset : object
            a class with the __len__ and __getitem__ implemented.
        
        batch_size : int
            size of each batch.
        
        shuffle : bool
            whether to shuffle the dataset upon epoch ending.
        
        collate_fn : object
            Function. How the samples are collated.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.shuffle = shuffle
        self.rng = rng
        if rng is None and shuffle:
            raise ValueError('Provide RNG if shuffle is True.')
        self.drop_last = drop_last
        self.reset() # Get indices
        self.collate_fn = collate_fn
        

    def __getitem__(self, idx):
        if len(self) <= idx:
            raise IndexError("Index out of range")
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        samples = []
        for i in indices:
            data = self.dataset[i]
            samples.append(data)
        return self.collate_fn(samples)

    def __len__(self):
        if self.drop_last:
            if self.length < self.batch_size:
                raise ValueError('Length of data is smaller than batch size and drop_last is True')
            return int(jnp.floor(self.length / self.batch_size))
        else:
            return int(jnp.ceil(self.length / self.batch_size))

    def reset(self):
        seq = jnp.arange(self.length)
        if self.shuffle:
            _, rng = jax.random.split(self.rng, 2)
            shuffled_idx = jax.random.permutation(rng, seq)
            self.indices = shuffled_idx
            self.rng = rng
            # print(self.indices)
        else:
            self.indices = seq

    def _jax_generator(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def _tf_generator(self):
        for idx in range(len(self)):
            output =  jax.tree_map(lambda x: jax_to_tf(x), self.__getitem__(idx))
            yield output

    def _get_output_signature(self, n_partitions):
        example = self.__getitem__(0)
        if n_partitions == 0:
            func = lambda x: tf.TensorSpec(shape = (None, ) + x.shape[1:], dtype = x.dtype)
        elif n_partitions > 0:
            func = lambda x: tf.TensorSpec(shape = (n_partitions, None, ) + x.shape[2:], dtype = x.dtype)
        output_signature = jax.tree_map(func, example)
        self.reset() # shuffle loader
        return output_signature

    def tf_Dataset(self, output_signature):
        return tf.data.Dataset.from_generator(self._jax_generator, output_signature = output_signature)

    def tf_Dataset_by_example(self, n_partitions):
        output_signature = self._get_output_signature(n_partitions)
        return tf.data.Dataset.from_generator(self._jax_generator, output_signature = output_signature)