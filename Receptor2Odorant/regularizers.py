from typing import Optional

import chex
import jax
import jax.numpy as jnp
import optax

def safe_mean_squares(x: chex.Array, min_ms: chex.Numeric) -> chex.Array:
    """Returns `maximum(mean(square(x)), min_norm)` with correct grads.
    The gradients of `maximum(mean(square(x)), min_norm)` at 0.0
    is `NaN`, because jax will evaluate both branches of the `jnp.maximum`. This
    function will instead return the correct gradient of 0.0 also in such setting.
    See reference for details.
    Paramters:
    ----------
    x : jnp.array

    min_ms : float
        lower bound for the returned norm.

    Returns:
    --------
    The safe mean squared of the input vector, accounting for correct gradient.

    Notes:
    ------
    if complex numbers are expected see optax._src.abs_sq instead of jnp.square

    References:
    -----------
    https://github.com/deepmind/optax/blob/master/optax/_src/numerics.py#L84#L100
    """
    ms = jnp.mean(jnp.square(x))
    x = jnp.where(ms <= min_ms, jnp.ones_like(x), x)
    return jnp.where(ms <= min_ms, min_ms, jnp.mean(jnp.square(x)))


def l2_norm_regularizer(
    param: chex.Array,
    alpha : float,
    ) -> chex.Array:
    """Calculates the L2 loss for a vector of parameters.
        Paramters:
        ----------
        param : jnp.array
            tensor of parameters

        Returns:
        --------
        alpha * (sum of squared elements of param)
    """
    return alpha * safe_mean_squares(param, min_ms = 0.0)


def l1_norm_regularizer(
    param: chex.Array,
    alpha : float,
    ) -> chex.Array:
    """Calculates the L1 loss for a vector of parameters.
        Paramters:
        ----------
        param : jnp.array
            tensor of parameters

        Returns:
        --------
        alpha * (sum of abs elements of param)
    """
    return alpha * jnp.mean(jnp.abs(param))

