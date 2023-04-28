import optax
from jax import numpy as jnp

from Receptor2Odorant.losses import focal_loss

def make_loss_func(is_weighted, option = 'cross_entropy'):
    if option == 'cross_entropy':
        if is_weighted:
            def loss_func(logits, labels):
                labels, sample_weights = labels
                logits = jnp.squeeze(logits)
                labels = jnp.asarray(labels, dtype = jnp.float32)
                loss_val = optax.sigmoid_binary_cross_entropy(logits, labels)
                weighted_loss_val = sample_weights * loss_val
                return jnp.mean(weighted_loss_val)
        else:
            def loss_func(logits, labels):
                logits = jnp.squeeze(logits)
                labels = jnp.asarray(labels, dtype = jnp.float32)
                loss_val = optax.sigmoid_binary_cross_entropy(logits, labels)
                return jnp.mean(loss_val)

    elif option == 'focal':
        if is_weighted:
            def loss_func(logits, labels):
                labels, sample_weights = labels
                logits = jnp.squeeze(logits)
                labels = jnp.asarray(labels, dtype = jnp.float32)
                loss_val = focal_loss(logits, labels)
                weighted_loss_val = sample_weights * loss_val
                return jnp.mean(weighted_loss_val)
        else:
            def loss_func(logits, labels):
                logits = jnp.squeeze(logits)
                labels = jnp.asarray(labels, dtype = jnp.float32)
                loss_val = focal_loss(logits, labels)
                return jnp.mean(loss_val)
    return loss_func