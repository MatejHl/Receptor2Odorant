import os
import sys
import functools
import pickle
import json
import time
import numpy as np
import jax
from jax import numpy as jnp
import flax
from flax import serialization
from objax.jaxboard import SummaryWriter, Summary, Image, DelayedScalar

from matplotlib import pyplot as plt

from Receptor2Odorant.main.loader import ProtBERTDatasetPrecomputeBERT, ProtBERTLoader, ProtBERTCollatePrecomputeBERT_CLS
from Receptor2Odorant.main.make_init import make_init_model
from Receptor2Odorant.main.make_create_optimizer import make_create_optimizer
from Receptor2Odorant.main.make_predict import make_predict_epoch, make_predict_epoch_pmap
from Receptor2Odorant.main.select_model import get_model_by_name
from Receptor2Odorant.utils import _serialize_hparam

import logging


def main_predict(hparams):
    """
    """
    logdir = hparams['LOGGING_PARENT_DIR']
    data_csv = hparams['PREDICT_CSV_PATH']

    model_class = get_model_by_name(hparams['MODEL_NAME'])
    model = model_class(atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    if hparams['SELF_LOOPS']:
        hparams['PADDING_N_EDGE'] = hparams['PADDING_N_EDGE'] + hparams['PADDING_N_NODE'] # NOTE: Because of self_loops
        if len(hparams['BOND_FEATURES']) > 0:
            raise ValueError('Can not have both bond features and self_loops.')

    logger = logging.getLogger('main_predict')
    logger.setLevel(logging.INFO) # logging.DEBUG

    import tables
    h5file = tables.open_file(hparams['BERT_H5FILE'], mode = 'r', title="TapeBERT")
    bert_table = h5file.root.bert.BERTtable

    collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                padding_n_node = hparams['PADDING_N_NODE'],
                                                padding_n_edge = hparams['PADDING_N_EDGE'],
                                                n_partitions = hparams['N_PARTITIONS'],
                                                from_disk = hparams['PYTABLE_FROM_DISK'],
                                                line_graph = hparams['LINE_GRAPH'])

    if not hparams['PYTABLE_FROM_DISK']:
        h5file.close()
        print('Table closed...')

    predict_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = data_csv,
                        mol_col = hparams['MOL_COL'],
                        seq_id_col = hparams['SEQ_ID_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = None,
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                        self_loops = hparams['SELF_LOOPS'],
                        line_graph = hparams['LINE_GRAPH'],
                        )

    _predict_loader = ProtBERTLoader(predict_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = None,
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        predict_loader = _predict_loader
    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        predict_loader = _predict_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        predict_loader = predict_loader.cache()
        predict_loader = predict_loader.prefetch(buffer_size = 4)
        logger.info('loader_output_type = {}'.format(hparams['LOADER_OUTPUT_TYPE']))

    key1, key2 = jax.random.split(jax.random.PRNGKey(int(time.time())), 2)
    key_params, _key_num_steps, key_num_steps, key_dropout = jax.random.split(key1, 4)

    # Initializations:
    start = time.time()
    logger.info('jax_version = {}'.format(jax.__version__))
    logger.info('flax_version = {}'.format(flax.__version__))
    logger.info('from_disk = {}'.format(hparams['PYTABLE_FROM_DISK']))
    logger.info('Initializing...')
    init_model = make_init_model(model, 
                                batch_size = hparams['BATCH_SIZE'], 
                                seq_embedding_size = 1024,
                                padding_n_node = hparams['PADDING_N_NODE'],
                                padding_n_edge = hparams['PADDING_N_EDGE'], 
                                num_node_features = len(hparams['ATOM_FEATURES']), 
                                num_edge_features = len(hparams['BOND_FEATURES']), 
                                self_loops = hparams['SELF_LOOPS'],
                                line_graph = hparams['LINE_GRAPH'])
    params = init_model(rngs = {'params' : key_params, 'dropout' : key_dropout, 'num_steps' : _key_num_steps}) # jax.random.split(key1, jax.device_count()))
    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    # This is needed to create state:
    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], warmup_steps = hparams['OPTIMIZATION']['WARMUP_STEPS'], transition_steps = hparams['OPTIMIZATION']['TRANSITION_EPOCHS']*(len(predict_dataset)/hparams['BATCH_SIZE']))
    init_state, scheduler = create_optimizer(params, rngs = {'dropout' : key_dropout, 'num_steps' : key_num_steps}, learning_rate = hparams['LEARNING_RATE'])

    # Restore params:
    if hparams['RESTORE_FILE'] is not None:
        logger.info('Restoring parameters from {}'.format(hparams['RESTORE_FILE']))
        with open(hparams['RESTORE_FILE'], 'rb') as pklfile:
            bytes_output = pickle.load(pklfile)
        state = serialization.from_bytes(init_state, bytes_output)
        logger.info('Parameters restored...')
    else:
        state = init_state

    if hparams['N_PARTITIONS'] > 0:
        state = flax.jax_utils.replicate(state)
        predict_epoch = make_predict_epoch_pmap(logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
    else:
        predict_epoch = make_predict_epoch(logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # Log hyperparams and create logdir:
    os.makedirs(logdir)
    _hparams = {}
    for key in hparams.keys():
        _hparams[key] = _serialize_hparam(hparams[key])
    hparams_logs = _hparams
    with open(os.path.join(logdir, 'hparams_predict.json'), 'w') as jsonfile:
        json.dump(hparams_logs, jsonfile)

    # --------
    # PREDICT:
    # --------
    start = time.time()
    predictions = predict_epoch(state, predict_loader)
    end = time.time()
    logger.info('TIME: predict_epoch: {}'.format(end - start))
    predictions_np = jax.device_get(predictions)

    predictions_np = np.concatenate(predictions_np)
    print(predictions_np.shape)
    df = predict_dataset.data.copy()[[hparams['SEQ_ID_COL'], hparams['MOL_COL']]]
    df['pred'] = np.squeeze(predictions_np)

    # Save predictions:
    df.to_csv(os.path.join(logdir, 'predict.csv'), sep=';')

    logger.info('Finished...')
    return None
    