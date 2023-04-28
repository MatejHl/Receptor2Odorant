import chex
import jax
from jax import numpy as jnp

def focal_loss(logits, labels, gamma = 2.0):
    """
    Focal loss. See [1] and [2].
    
    Based on sigmoid_binary_cross_entropy from optax [3].
    
    Paramters:
    ----------
    logits : jax.numpy.DeviceArray
        Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels : jax.numpy.DeviceArray 
        The target probabilities for each class, must have a shape
        broadcastable to that of `logits`;

    References:
    -----------
    [1] https://arxiv.org/pdf/1708.02002.pdf
    [2] https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy
    [3] https://github.com/deepmind/optax/blob/master/optax/_src/loss.py#L116#L139
    """
    chex.assert_type([logits], float)
    log_p = jax.nn.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    log_not_p = jax.nn.log_sigmoid(-logits)

    sigmoidal = jax.nn.sigmoid(logits)
    p_t = (labels * sigmoidal) + ((1 - labels) * (1 - sigmoidal))
    focal_factor = jnp.power(1.0 - p_t, gamma)
    bce = -labels * log_p - (1. - labels) * log_not_p
    
    return focal_factor * bce