import os
import sys
import functools
import pandas 
import pickle
import time
import datetime
import json
import numpy as np
import jax
from jax import numpy as jnp
import flax
from flax import serialization
from flax.training import train_state
from objax.jaxboard import SummaryWriter, Summary

from Receptor2Odorant.odor_embedding.metrics import log_confusion_matrix

from Receptor2Odorant.odor_embedding.MPNN.loader import Loader, Collate, DatasetBuilder
from Receptor2Odorant.odor_embedding.MPNN.model.base_model import VanillaMPNN
from Receptor2Odorant.utils import _serialize_hparam

from Receptor2Odorant.odor_embedding.MPNN.make_init import make_init_model
from Receptor2Odorant.odor_embedding.MPNN.make_create_optimizer import make_create_optimizer
from Receptor2Odorant.odor_embedding.MPNN.make_train_and_eval import make_train_epoch, make_valid_epoch
from Receptor2Odorant.odor_embedding.MPNN.make_regularization_loss import make_regularization_loss

import logging

def main_train(hparams):
    datadir = os.path.join(hparams['DATA_PARENT_DIR'], hparams['DATACASE'])
    logdir = os.path.join(hparams['LOGGING_PARENT_DIR'], hparams['DATACASE'])

    restore_file = hparams['RESTORE_FILE']

    model = VanillaMPNN(num_classes = hparams['NUM_CLASSES'], atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    logger = logging.getLogger('main_train')
    logger.setLevel(logging.INFO)
    _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(logdir, model.__class__.__name__, _datetime)
    os.makedirs(logdir)
    os.mkdir(os.path.join(logdir, 'ckpts'))
    logger_file_handler = logging.FileHandler(os.path.join(logdir, 'run.log'))
    logger_stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(logger_file_handler)
    logger.addHandler(logger_stdout_handler)

    collate = Collate(padding_n_node = hparams['PADDING_N_NODE'], 
                    padding_n_edge = hparams['PADDING_N_EDGE'],
                    n_partitions = hparams['N_PARTITIONS'])

    dataset = DatasetBuilder(data_csv = os.path.join(datadir, hparams['TRAIN_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = hparams['WEIGHT_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        )

    _loader = Loader(dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = True,
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = True,
                        n_partitions = hparams['N_PARTITIONS'])

    valid_dataset = DatasetBuilder(data_csv = os.path.join(datadir, hparams['VALID_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        label_col = hparams['LABEL_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        )

    _valid_loader = Loader(valid_dataset, 
                        batch_size = len(valid_dataset),
                        collate_fn = collate.make_collate(),
                        shuffle = True,
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])
        
    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        loader = _loader
        valid_loader = _valid_loader
    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        loader = _loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        loader = loader.cache()
        loader = loader.shuffle(buffer_size = len(_loader))
        loader = loader.prefetch(buffer_size = 4)
        
        valid_loader = _valid_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        valid_loader = valid_loader.cache()
        valid_loader = valid_loader.shuffle(buffer_size = len(_valid_loader))
        valid_loader = valid_loader.prefetch(buffer_size = 4)
        logger.info('loader_output_type = {}'.format(hparams['LOADER_OUTPUT_TYPE']))

    key1, key2 = jax.random.split(jax.random.PRNGKey(int(time.time())), 2)
    key_params, _key_num_steps, key_num_steps, key_dropout = jax.random.split(key1, 4)

    # Initializations:
    start = time.time()
    logger.info('jax_version = {}'.format(jax.__version__))
    logger.info('flax_version = {}'.format(flax.__version__))
    logger.info('Initializing...')
    init_model = make_init_model(model, 
                                batch_size = hparams['BATCH_SIZE'], 
                                atom_features = model.atom_features, 
                                bond_features = model.bond_features,
                                padding_n_node = hparams['PADDING_N_NODE'],
                                padding_n_edge = hparams['PADDING_N_EDGE'])
    
    params = init_model(rngs = {'params' : key_params, 'dropout' : key_dropout}) 
    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], transition_steps = 800*(len(dataset)/hparams['BATCH_SIZE']))
    init_state, scheduler = create_optimizer(params, rngs = {'dropout' : key_dropout}, learning_rate = hparams['LEARNING_RATE'])

    # Restore params:
    if restore_file is not None:
        logger.info('Restoring parameters from {}'.format(restore_file))
        with open(restore_file, 'rb') as pklfile:
            bytes_output = pickle.load(pklfile)
        state = serialization.from_bytes(init_state, bytes_output)
        logger.info('Parameters restored...')
    else:
        state = init_state    

    reg_loss_func_embed = make_regularization_loss(params_path = ['params/atomic_num_embed/node_embed/embedding',
                                                            'params/chiral_tag_embed/node_embed/embedding',
                                                            'params/hybridization_embed/node_embed/embedding',
                                                            'params/X_proj_non_embeded/kernel',
                                                            'params/X_proj_non_embeded/bias',
                                                            'params/bond_type_embed/edge_embed/embedding',
                                                            'params/E_proj_non_embeded/kernel',
                                                            'params/E_proj_non_embeded/bias',
                                                            'params/stereo_embed/edge_embed/embedding',
                                                            ], alpha = 0.01, option = 'l1')
    reg_loss_func_kernel = make_regularization_loss(params_path = ['kernel',
                                                            ], alpha = 0.01, option = 'l2')
    reg_loss_func_kernel_2 = make_regularization_loss(params_path = ['kernel',  
                                                            ], alpha = 0.01, option = 'l1') 
    def reg_loss_func(params):
        return reg_loss_func_embed(params) + reg_loss_func_kernel(params) + reg_loss_func_kernel_2(params) 

    train_epoch = make_train_epoch(is_weighted = True, num_classes = model.num_classes, loss_option = hparams['LOSS_OPTION'], init_rngs = state.rngs, logger = logger, reg_loss_func = reg_loss_func, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
    valid_epoch = make_valid_epoch(num_classes = model.num_classes, loss_option = hparams['LOSS_OPTION'], logger = logger,  loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # Log hyperparams:
    _hparams = {}
    for key in hparams.keys():
        _hparams[key] = _serialize_hparam(hparams[key])
    hparams_logs = _hparams
    hparams_logs.update({'DATACASE' : hparams['DATACASE']})
    with open(os.path.join(logdir, 'hparams_logs.json'), 'w') as jsonfile:
        json.dump(hparams_logs, jsonfile)

    _dataparams = {}
    for key in hparams.keys():
        _dataparams[key] = _serialize_hparam(hparams[key])
    with open(os.path.join(logdir, 'dataparams_logs.json'), 'w') as jsonfile:
        json.dump(_dataparams, jsonfile)

    # Training:
    _, key = jax.random.split(key2, 2) 
    logger.info('Training...')
    # ---- NEW
    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    valid_writer = SummaryWriter(os.path.join(logdir, 'validation'))
    # ----
    for _ in range(hparams['N_EPOCH']):
        logger.info('Epoch:  {}'.format(state.epoch))

        start = time.time()
        state, batch_metrics = train_epoch(state, loader)
        end = time.time()
        logger.info('TIME: train_epoch: {}'.format(end - start))
        batch_metrics_np = jax.device_get(batch_metrics)
        auc = [metrics.pop('auc') for metrics in batch_metrics_np]
        hamming = sum([metrics.pop('hamming') for metrics in batch_metrics_np])
        report = [metrics.pop('report') for metrics in batch_metrics_np]
        report = pandas.DataFrame.from_dict(report[0], orient='index')
        f1 = report.loc['micro avg', 'f1-score']
        precision = report.loc['micro avg', 'precision']
        recall = report.loc['micro avg', 'recall']
        loss = np.mean([metrics.pop('loss') for metrics in batch_metrics_np])
        lr = scheduler(state.step)
        logger.info('train loss: {}'.format(loss))
        logger.info(f'hamming: {hamming}')
        
        # Write train metrics:
        summary = Summary()
        summary.scalar('hamming', hamming)
        summary.scalar('auc', auc)
        summary.scalar('epoch_loss', loss)
        summary.scalar('f1', f1)
        summary.scalar('precision', precision)
        summary.scalar('recall', recall)
        summary.scalar('learning_rate', lr)
        train_writer.write(summary, step = state.epoch)
        train_writer.writer.flush()

        # VALID:
        start = time.time()
        valid_metrics = valid_epoch(state, valid_loader)
        end = time.time()
        logger.info('TIME: valid_epoch: {}'.format(end - start))
        valid_metrics_np = jax.device_get(valid_metrics)
        hamming = sum([metrics.pop('hamming') for metrics in valid_metrics_np])
        report = [metrics.pop('report') for metrics in valid_metrics_np]
        auc = [metrics.pop('auc') for metrics in valid_metrics_np]
        report = pandas.DataFrame.from_dict(report[0], orient='index')
        f1 = report.loc['micro avg', 'f1-score']
        precision = report.loc['micro avg', 'precision']
        recall = report.loc['micro avg', 'recall']
        loss = np.mean([metrics.pop('loss') for metrics in valid_metrics_np])
        logger.info('valid loss: {}'.format(loss))
        logger.info(f'valid hamming: {hamming}')

        # Write valid metrics:
        summary = Summary()
        summary.scalar('hamming', hamming)
        summary.scalar('auc', auc)
        summary.scalar('epoch_loss', loss)
        summary.scalar('f1', f1)
        summary.scalar('precision', precision)
        summary.scalar('recall', recall)
        valid_writer.write(summary, step = state.epoch)
        valid_writer.writer.flush()
        
        # Update epoch number:
        state = state.replace(epoch = state.epoch + 1)
    
        # Save current state:
        if state.epoch%hparams['SAVE_FREQUENCY'] == 0:
            bytes_output = serialization.to_bytes(state)
            with open(os.path.join(logdir, 'ckpts', 'state_e' + str(state.epoch) + '.pkl'), 'wb') as pklfile:
                pickle.dump(bytes_output, pklfile)
            logger.info('State {} saved...'.format('state_e' + str(state.epoch)))

    # ---- NEW
    train_writer.close()
    valid_writer.close()
    # ----

    
