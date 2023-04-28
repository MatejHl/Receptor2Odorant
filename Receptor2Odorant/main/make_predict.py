import jax
from jax import numpy as jnp

from Receptor2Odorant.utils import tf_to_jax
from Receptor2Odorant.utils import serialize_BERT_hidden_states

def make_predict_step(return_intermediates = False):
    if return_intermediates:
        def predict_step(state, batch):
            logits, intermediates = state.apply_fn(state.params, batch, deterministic = True, mutable=['intermediates'])
            pred_probs = jax.nn.sigmoid(logits)
            return pred_probs, intermediates
    else:
        def predict_step(state, batch):
            logits = state.apply_fn(state.params, batch, deterministic = True)
            pred_probs = jax.nn.sigmoid(logits)
            return pred_probs
    return jax.jit(predict_step)


def make_predict_epoch(logger, loader_output_type = 'jax'):
    """
    Helper function to create predict_epoch function.
    """
    predict_step = make_predict_step()
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in enumerate(predict_loader):
                batch = (batch[0], batch[1])
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            predict_loader.reset()
            return batch_predictions
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in predict_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                batch = (batch[0], batch[1])
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            return batch_predictions
    return predict_epoch


# --------
# jax.pmap
# --------
# NOTE: pmap is not tested.
def make_predict_step_pmap():
    def predict_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        pred_probs = jax.nn.sigmoid(logits)
        return pred_probs
    return jax.pmap(predict_step, axis_name='batch')


def make_predict_epoch_pmap(logger, loader_output_type = 'jax'):
    """
    Helper function to create predict_epoch function.
    """
    predict_step = make_predict_step_pmap()

    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in enumerate(predict_loader):
                batch = (batch[0], batch[1])
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            return batch_predictions
    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in predict_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                batch = (batch[0], batch[1])
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            return batch_predictions
    return predict_epoch


# ---------------
# predict single:
# ---------------
def make_apply_bert(bert_model):
    """
    Take the last 5 CLS layers.
    """
    # @jax.jit
    def apply_bert(seq):
        bert_output = bert_model.module.apply({'params': bert_model.params}, **seq, deterministic = True,
                             output_attentions = False,
                             output_hidden_states = True, 
                             return_dict = True)
        S = bert_output.hidden_states
        S = jnp.stack(S[-5:], axis = 1)
        S = jnp.reshape(S[:, :, 0, :], newshape = (S.shape[0], -1))
        return S
    return apply_bert

def make_predict_single_epoch(logger, bert_model, loader_output_type = 'jax', return_intermediates = False):
    """
    Helper function to create predict_epoch function.
    """
    predict_step = make_predict_step(return_intermediates = return_intermediates)
    apply_bert = make_apply_bert(bert_model)
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in enumerate(predict_loader):
                S = apply_bert(batch[0]) # Apply BERT
                batch = (S, batch[1])
                output = predict_step(state, batch)
                batch_predictions.append(output)
            predict_loader.reset()
            return batch_predictions
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in predict_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                S = apply_bert(batch[0]) # Apply BERT
                batch = (S, batch[1])
                # batch = flax.jax_utils.replicate(batch)
                output = predict_step(state, batch)
                batch_predictions.append(output)
            return batch_predictions
    return predict_epoch
