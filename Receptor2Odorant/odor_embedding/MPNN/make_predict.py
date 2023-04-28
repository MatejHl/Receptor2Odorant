import jax
from jax import numpy as jnp

from Receptor2Odorant.utils import tf_to_jax

def make_predict_step(return_embeddings = False):
    if return_embeddings:
        def predict_step(state, batch):
            logits, embedding = state.apply_fn(state.params, batch[0], deterministic = True, return_embeddings=True)
            return logits, embedding
    else:
        def predict_step(state, batch):
            logits = state.apply_fn(state.params, batch[0], deterministic = True, return_embeddings=False)
            return logits, None
    return jax.jit(predict_step)


def make_predict_epoch(num_classes, loss_option, logger, return_embeddings, loader_output_type = 'jax', ):
    """
    Helper function to create predict function.
    """
    predict_step = make_predict_step(return_embeddings = return_embeddings)

    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_logits = []
            batch_embedding = []
            for batch in predict_loader:
                batch = (batch[0],)
                logits, embeddings = predict_step(state, batch)
                if return_embeddings:
                    batch_logits.append(logits)
                    batch_embedding.append(embeddings)
                else:
                    batch_logits.append(logits)
                    batch_embedding = None
            predict_loader.reset()            
            logits_concat = jnp.concatenate(batch_logits)
            embedding_concat = jnp.concatenate(batch_embedding)
            return logits_concat, embedding_concat

    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_logits = []
            batch_embedding = []
            for batch in predict_loader:
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                batch = (batch[0],)
                logits, embeddings = predict_step(state, batch)
                if return_embeddings:
                    batch_logits.append(logits)
                    batch_embedding.append(embeddings)
                else:
                    batch_logits.append(logits)
                    batch_embedding = None
            logits_concat = jnp.concatenate(batch_logits)
            embedding_concat = jnp.concatenate(batch_embedding)
            return logits_concat, embedding_concat
    return predict_epoch