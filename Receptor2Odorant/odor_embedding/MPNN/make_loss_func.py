import optax
import jax
from jax import numpy as jnp
from sklearn.metrics import hamming_loss

def make_loss_func(is_weighted, num_classes, option = 'cross_entropy'):
    if option == 'cross_entropy':
        if is_weighted:
            def loss_func(logits, labels):
                labels, sample_weights = labels
                labels = jnp.asarray(labels, dtype = jnp.float32)
                # one_hot_labels = jax.nn.one_hot(labels, num_classes = num_classes)
                loss_val = optax.sigmoid_binary_cross_entropy(logits, labels)
                weighted_loss_val = sample_weights * loss_val
                return jnp.mean(weighted_loss_val)
        else:
            def loss_func(logits, labels):
                labels = jnp.asarray(labels, dtype = jnp.float32)
                # one_hot_labels = jax.nn.one_hot(labels, num_classes = num_classes)
                loss_val = optax.sigmoid_binary_cross_entropy(logits, labels)
                return jnp.mean(loss_val)
    
    if option == 'mse':
        if is_weighted:
            def loss_func(logits, labels):
                labels, sample_weights = labels
                loss_val = optax.l2_loss(logits,targets = labels)
                weighted_loss_val = sample_weights * loss_val
                return jnp.mean(weighted_loss_val)
        else:
            def loss_func(logits, labels):
                loss_val = optax.huber_loss(logits,targets = labels)
                return jnp.mean(loss_val)

    if option == 'hamming':
        if is_weighted:
            def loss_func(logits, labels):
                labels, sample_weights = labels
                loss_val = hamming_loss(labels, logits)
                weighted_loss_val = sample_weights * loss_val
                return jnp.mean(weighted_loss_val)
        else:
            def loss_func(logits, labels): 
                labels = jnp.asarray(labels, dtype = jnp.float32)
                loss_val = optax.sigmoid_binary_cross_entropy(logits,labels)
                return jnp.mean(loss_val)
        

    return loss_func