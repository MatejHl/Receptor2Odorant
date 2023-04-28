
import jax
from flax.traverse_util import flatten_dict

from typing import Optional
from Receptor2Odorant.utils import find_params_by_node_name

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

def make_regularization_loss(params_path, alpha, option = 'l2'):
    """
    TODO: Would 'scan'-like make loops faster? 
    """
    if params_path == 'ALL':
        raise NotImplementedError('Put case of regularizing all')
    elif params_path == 'KERNEL' or params_path == 'BIAS':
        node_name = params_path.lower()
        if option == 'l2':
            def reg_loss_func(params):
                flat_params = find_params_by_node_name(params, node_name)
                loss = sum(l2_norm_regularizer(x, alpha) for x in jax.tree_leaves(flat_params))
                return loss
        elif option == 'l1':
            def reg_loss_func(params):
                flat_params = find_params_by_node_name(params, node_name)
                loss = sum(l1_norm_regularizer(x, alpha) for x in jax.tree_leaves(flat_params))
                return loss
    else:
        if option == 'l2':
            def reg_loss_func(params):
                flat_params = flatten_dict(params, keep_empty_nodes=False, is_leaf=None, sep='/')
                loss = 0.0
                for key, x in flat_params.items():
                    if key in params_path:
                        loss += l2_norm_regularizer(x, alpha)
                    else:
                        loss += 0.0
                return loss
        elif option == 'l1':
            def reg_loss_func(params):
                flat_params = flatten_dict(params, keep_empty_nodes=False, is_leaf=None, sep='/')
                loss = 0.0
                for key, x in flat_params.items():
                    if key in params_path:
                        loss += l1_norm_regularizer(x, alpha)
                    else:
                        loss += 0.0
                return loss
        return reg_loss_func    