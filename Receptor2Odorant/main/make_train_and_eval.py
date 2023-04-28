import jax
from jax import numpy as jnp

from Receptor2Odorant.utils import tf_to_jax

from Receptor2Odorant.main.make_loss_func import make_loss_func
from Receptor2Odorant.main.make_compute_metrics import make_compute_metrics

def make_train_step(loss_func, init_rngs, reg_loss_func = None):
    if reg_loss_func is not None:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1]) + reg_loss_func(params)
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    else:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1])
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    return jax.jit(train_step)


def make_eval_step():
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        return logits
    return jax.jit(eval_step)


def make_train_epoch(is_weighted, loss_option, init_rngs, logger, reg_loss_func = None, loader_output_type = 'jax'):
    """
    Helper function to create train_epoch function.
    """
    loss_func = make_loss_func(is_weighted = is_weighted, option = loss_option)
    compute_metrics = make_compute_metrics(is_weighted = is_weighted, loss_option = loss_option, use_jit = True)
    train_step = make_train_step(loss_func = loss_func, init_rngs = init_rngs, reg_loss_func = reg_loss_func)
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in enumerate(loader):
                state, logits, _ = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                batch_metrics.append(metrics)
            loader.reset()
            return state, batch_metrics
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                state, logits, _ = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                batch_metrics.append(metrics)
            return state, batch_metrics
    return train_epoch


def make_valid_epoch(loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    compute_metrics = make_compute_metrics(is_weighted = False, loss_option = loss_option, use_jit = True)
    eval_step = make_eval_step()
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                labels = batch[2]
                if isinstance(labels, (list, tuple)): # Ignore weights.
                    labels = labels[0]
                batch = (batch[0], batch[1], labels)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            valid_loader.reset()
            return batch_metrics
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch) 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                batch = (batch[0], batch[1], labels)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch




# --------
# jax.pmap
# --------
# NOTE: pmap is not tested.
def make_train_step_pmap(loss_func, init_rngs, reg_loss_func = None):
    if reg_loss_func is not None:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1]) + reg_loss_func(params)
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    else:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1])
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    return jax.pmap(train_step, axis_name='batch')


def make_eval_step_pmap():
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        return logits
    return jax.pmap(eval_step, axis_name='batch')


def make_train_epoch_pmap(is_weighted, loss_option, init_rngs, logger, reg_loss_func = None, loader_output_type = 'jax'):
    """
    Helper function to create train_epoch function.
    """
    loss_func = make_loss_func(is_weighted = is_weighted, option = loss_option)
    compute_metrics = make_compute_metrics(is_weighted = is_weighted, loss_option = loss_option, use_jit = True)
    train_step = make_train_step_pmap(loss_func = loss_func, init_rngs = init_rngs, reg_loss_func = reg_loss_func)
    if loader_output_type == 'jax':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in enumerate(loader):
                state, logits, _ = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return state, batch_metrics
    elif loader_output_type == 'tf':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                state, logits, grads = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return state, batch_metrics
    return train_epoch


def make_valid_epoch_pmap(loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    compute_metrics = make_compute_metrics(is_weighted = False, loss_option = loss_option, use_jit = True)
    eval_step = make_eval_step_pmap()

    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                batch = (batch[0], batch[1], labels)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                batch = (batch[0], batch[1], labels)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch