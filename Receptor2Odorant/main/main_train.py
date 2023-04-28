import os
import sys
import functools
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

from Receptor2Odorant.metrics import log_confusion_matrix, log_roc_curve, precision_recall_fscore, MCC, roc_auc, roc_curve, precision_recall_curve, average_precision_score, log_pr_curve # log_roc_curve_OLD, roc_auc_OLD

from Receptor2Odorant.main.loader import ProtBERTDatasetPrecomputeBERT, ProtBERTLoader, ProtBERTCollatePrecomputeBERT_CLS

from Receptor2Odorant.utils import _serialize_hparam

from Receptor2Odorant.main.make_init import make_init_model
from Receptor2Odorant.main.make_create_optimizer import make_create_optimizer
from Receptor2Odorant.main.make_train_and_eval import make_train_epoch, make_valid_epoch, make_train_epoch_pmap, make_valid_epoch_pmap
from Receptor2Odorant.main.make_regularization_loss import make_regularization_loss

from Receptor2Odorant.main.select_model import get_model_by_name

import logging


def main_train(hparams):
    """
    """
    if hparams['SIZE_CUT_DIRNAME'] is None:
        hparams.update({'BIG_SWITCH_EPOCH' : hparams['N_EPOCH'] + 1})

    datadir = os.path.join(hparams['DATA_PARENT_DIR'], hparams['DATACASE'])
    logdir = os.path.join(hparams['LOGGING_PARENT_DIR'], hparams['DATACASE'])
    restore_file = hparams['RESTORE_FILE']

    model_class = get_model_by_name(hparams['MODEL_NAME'])
    model = model_class(atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    if hparams['SELF_LOOPS']:
        hparams['PADDING_N_EDGE'] = hparams['PADDING_N_EDGE'] + hparams['PADDING_N_NODE'] # NOTE: Because of self_loops
        if hparams['SIZE_CUT_DIRNAME'] is not None:
            hparams['BIG_PADDING_N_EDGE'] = hparams['BIG_PADDING_N_EDGE'] + hparams['BIG_PADDING_N_NODE'] # NOTE: Because of self_loops
        if len(hparams['BOND_FEATURES']) > 0:
            raise ValueError('Can not have both bond features and self_loops.')

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
    

    # ---------
    # Datasets:
    # ---------
    import tables
    h5file = tables.open_file(hparams['BERT_H5FILE'], mode = 'r', title="ProtBERT")
    bert_table = h5file.root.bert.BERTtable

    collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                padding_n_node = hparams['PADDING_N_NODE'], 
                                                padding_n_edge = hparams['PADDING_N_EDGE'],
                                                n_partitions = hparams['N_PARTITIONS'],
                                                from_disk = hparams['PYTABLE_FROM_DISK'],
                                                line_graph = hparams['LINE_GRAPH'])

    if hparams['SIZE_CUT_DIRNAME'] is not None:
        big_collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                padding_n_node = hparams['BIG_PADDING_N_NODE'], 
                                                padding_n_edge = hparams['BIG_PADDING_N_EDGE'],
                                                n_partitions = hparams['N_PARTITIONS'],
                                                from_disk = hparams['PYTABLE_FROM_DISK'],
                                                line_graph = hparams['LINE_GRAPH'])

    if not hparams['PYTABLE_FROM_DISK']:
        h5file.close()
        print('Table closed...')

    dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, hparams['SIZE_CUT_DIRNAME'], hparams['TRAIN_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        seq_id_col = hparams['SEQ_ID_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = hparams['WEIGHT_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        class_alpha = hparams['CLASS_ALPHA'],
                        line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                        self_loops = hparams['SELF_LOOPS'],
                        line_graph = hparams['LINE_GRAPH'],
                        )

    _loader = ProtBERTLoader(dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = True,     # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = True,
                        n_partitions = hparams['N_PARTITIONS'])

    valid_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, hparams['SIZE_CUT_DIRNAME'], hparams['VALID_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        seq_id_col = hparams['SEQ_ID_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = hparams['VALID_WEIGHT_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                        self_loops = hparams['SELF_LOOPS'],
                        line_graph = hparams['LINE_GRAPH'],
                        )

    _valid_loader = ProtBERTLoader(valid_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = True,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = True,
                        n_partitions = hparams['N_PARTITIONS'])

    if hparams['SIZE_CUT_DIRNAME'] is not None:
        _big_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, hparams['SIZE_CUT_DIRNAME'], hparams['BIG_TRAIN_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        seq_id_col = hparams['SEQ_ID_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = hparams['WEIGHT_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        class_alpha = hparams['CLASS_ALPHA'],
                        line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * big_collate.padding_n_node,
                        self_loops = hparams['SELF_LOOPS'],
                        line_graph = hparams['LINE_GRAPH'],
                        )
        big_dataset = dataset + _big_dataset

        _big_loader = ProtBERTLoader(big_dataset, 
                            batch_size = hparams['BIG_BATCH_SIZE'],
                            collate_fn = big_collate.make_collate(),
                            shuffle = True,     # NOTE: shuffle is redundant for tf.data.Dataset here.
                            rng = jax.random.PRNGKey(int(time.time())),
                            drop_last = True,
                            n_partitions = hparams['N_PARTITIONS'])

        _big_valid_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, hparams['SIZE_CUT_DIRNAME'], hparams['BIG_VALID_CSV_NAME']),
                        mol_col = hparams['MOL_COL'],
                        seq_id_col = hparams['SEQ_ID_COL'],
                        label_col = hparams['LABEL_COL'],
                        weight_col = hparams['VALID_WEIGHT_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * big_collate.padding_n_node,
                        self_loops = hparams['SELF_LOOPS'],
                        line_graph = hparams['LINE_GRAPH'],
                        )
        big_valid_dataset = valid_dataset + _big_valid_dataset

        _big_valid_loader = ProtBERTLoader(big_valid_dataset, 
                                batch_size = hparams['BIG_BATCH_SIZE'],
                                collate_fn = big_collate.make_collate(),
                                shuffle = True,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                                rng = jax.random.PRNGKey(int(time.time())),
                                drop_last = True,
                                n_partitions = hparams['N_PARTITIONS'])


    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        loader = _loader
        valid_loader = _valid_loader
        if hparams['SIZE_CUT_DIRNAME'] is not None:
            big_loader = _big_loader
            big_valid_loader = _big_valid_loader

    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        loader = _loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        loader = loader.cache() # NOTE: This loads full dataset.
        loader = loader.shuffle(buffer_size = len(_loader))
        loader = loader.prefetch(buffer_size = 4)

        valid_loader = _valid_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        valid_loader = valid_loader.cache() # NOTE: This loads full dataset.
        valid_loader = valid_loader.shuffle(buffer_size = len(_valid_loader))
        valid_loader = valid_loader.prefetch(buffer_size = 4)

        if hparams['SIZE_CUT_DIRNAME'] is not None:
            big_loader = _big_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
            big_loader = big_loader.cache()
            big_loader = big_loader.shuffle(buffer_size = len(_big_loader))
            big_loader = big_loader.prefetch(buffer_size = 4)

            big_valid_loader = _big_valid_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
            big_valid_loader = big_valid_loader.cache()
            big_valid_loader = big_valid_loader.shuffle(buffer_size = len(_big_valid_loader))
            big_valid_loader = big_valid_loader.prefetch(buffer_size = 4)

    # ----------------
    # Initializations:
    # ----------------
    key1, key2 = jax.random.split(jax.random.PRNGKey(int(time.time())), 2)
    key_params, _key_num_steps, key_num_steps, key_dropout = jax.random.split(key1, 4)

    logger.info('jax_version = {}'.format(jax.__version__))
    logger.info('flax_version = {}'.format(flax.__version__))
    logger.info('from_disk = {}'.format(hparams['PYTABLE_FROM_DISK']))
    logger.info('model_name = {}'.format(hparams['MODEL_NAME']))
    logger.info('loader_output_type = {}'.format(hparams['LOADER_OUTPUT_TYPE']))

    # Initializations:
    start = time.time()
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
    params = init_model(rngs = {'params' : key_params, 'dropout' : key_dropout, 'num_steps' : _key_num_steps})
    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    transition_steps = hparams['OPTIMIZATION']['TRANSITION_EPOCHS']*(len(dataset)/hparams['BATCH_SIZE'])
    if hparams['SIZE_CUT_DIRNAME'] is not None:
        transition_steps += hparams['OPTIMIZATION']['TRANSITION_EPOCHS']*(len(_big_dataset)/hparams['BIG_BATCH_SIZE'])

    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], warmup_steps = hparams['OPTIMIZATION']['WARMUP_STEPS'], transition_steps = transition_steps)
    init_state, scheduler = create_optimizer(params, rngs = {'dropout' : key_dropout, 'num_steps' : key_num_steps}, learning_rate = hparams['LEARNING_RATE'])

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
    def reg_loss_func(params):
        return reg_loss_func_embed(params) + reg_loss_func_kernel(params)

    if hparams['N_PARTITIONS'] > 0:
        state = flax.jax_utils.replicate(state)
        train_epoch = make_train_epoch_pmap(is_weighted = True, loss_option = hparams['LOSS_OPTION'], init_rngs = state.rngs, logger = logger, reg_loss_func = reg_loss_func, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
        valid_epoch = make_valid_epoch_pmap(loss_option = hparams['LOSS_OPTION'], logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
    else:
        train_epoch = make_train_epoch(is_weighted = True, loss_option = hparams['LOSS_OPTION'], init_rngs = state.rngs, logger = logger, reg_loss_func = reg_loss_func, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
        valid_epoch = make_valid_epoch(loss_option = hparams['LOSS_OPTION'], logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # Log hyperparams:
    _hparams = {}
    for key in hparams.keys():
        _hparams[key] = _serialize_hparam(hparams[key])
    hparams_logs = _hparams
    # hparams_logs.update({'DATACASE' : _datacase})
    with open(os.path.join(logdir, 'hparams_logs.json'), 'w') as jsonfile:
        json.dump(hparams_logs, jsonfile)

    # Training:
    _, key = jax.random.split(key2, 2) 
    logger.info('Training...')
    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    valid_writer = SummaryWriter(os.path.join(logdir, 'validation'))

    if hparams['N_PARTITIONS'] > 0:
        epoch = state.epoch[0]
    else:
        epoch = state.epoch
    while epoch <= hparams['N_EPOCH']:
        logger.info('Epoch:  {}'.format(epoch))
        if epoch == hparams['BIG_SWITCH_EPOCH']:
            logger.info('Switching to Big loader...')

        start = time.time()
        if epoch >= hparams['BIG_SWITCH_EPOCH']: # TODO: This doesn't make sense if there is no BIG!
            state, batch_metrics = train_epoch(state, big_loader)
        else:
            state, batch_metrics = train_epoch(state, loader)
        end = time.time()
        logger.info('TIME: train_epoch: {}'.format(end - start))
        batch_metrics_np = jax.device_get(batch_metrics)
        CM = sum([metrics.pop('confusion_matrix') for metrics in batch_metrics_np])
        loss = np.mean([metrics.pop('loss') for metrics in batch_metrics_np])
        CM_per_threshold = jax.tree_multimap(lambda *x: sum(x), *[metrics.pop('confusion_matrix_per_threshold') for metrics in batch_metrics_np])
        lr = scheduler(state.step)
        logger.info('train loss: {}'.format(loss))
        logger.info('train accuracy: 0: {}  1: {}'.format(*list(np.diag(CM)/(np.sum(CM, axis = 1) + 10e-9))))
        # with SummaryWriter(os.path.join(logdir, 'train')) as train_writer:
        if True:
            summary = Summary()
            roc_curve_values = roc_curve(CM_per_threshold, drop_intermediate=True)
            pr_curve_values = precision_recall_curve(CM_per_threshold)
            AveP = average_precision_score(pr_curve_values)
            if epoch%hparams['LOG_IMAGES_FREQUENCY'] == 0:
                summary['epoch_confusion_matrix'] = log_confusion_matrix(CM, class_names=['0', '1'])
                summary['epoch_ROC_curve'] = log_roc_curve(roc_curve_values)
                summary['epoch_PR_curve'] = log_pr_curve(pr_curve_values, average_precision = AveP)
            summary.scalar('AUC_ROC', roc_auc(roc_curve_values))
            summary.scalar('AveP', AveP)
            precision, recall, f_score = precision_recall_fscore(CM)
            summary.scalar('precision', precision)
            summary.scalar('recall', recall)
            summary.scalar('f_score', f_score)
            summary.scalar('MCC', MCC(CM))
            summary.scalar('epoch_loss', loss)
            summary.scalar('learning_rate', lr)
            train_writer.write(summary, step = int(epoch))
            train_writer.writer.flush()

        # VALID:
        start = time.time()
        if epoch >= hparams['BIG_SWITCH_EPOCH']:
            valid_metrics = valid_epoch(state, big_valid_loader)
        else:
            valid_metrics = valid_epoch(state, valid_loader)
        end = time.time()
        logger.info('TIME: valid_epoch: {}'.format(end - start))
        valid_metrics_np = jax.device_get(valid_metrics)
        CM = sum([metrics.pop('confusion_matrix') for metrics in valid_metrics_np])
        loss = np.mean([metrics.pop('loss') for metrics in valid_metrics_np])
        CM_per_threshold = jax.tree_multimap(lambda *x: sum(x), *[metrics.pop('confusion_matrix_per_threshold') for metrics in valid_metrics_np])
        logger.info('valid loss: {}'.format(loss))
        logger.info('valid accuracy: 0: {}  1: {}'.format(*list(np.diag(CM)/(np.sum(CM, axis = 1) + 10e-9))))
        
        # Write metrics:
        summary = Summary()
        roc_curve_values = roc_curve(CM_per_threshold, drop_intermediate=True)
        pr_curve_values = precision_recall_curve(CM_per_threshold)
        AveP = average_precision_score(pr_curve_values)
        if epoch%hparams['LOG_IMAGES_FREQUENCY'] == 0:
            summary['epoch_confusion_matrix'] = log_confusion_matrix(CM, class_names=['0', '1'])
            summary['epoch_ROC_curve'] = log_roc_curve(roc_curve_values)
            summary['epoch_PR_curve'] = log_pr_curve(pr_curve_values, average_precision = AveP)
        summary.scalar('AUC_ROC', roc_auc(roc_curve_values))
        summary.scalar('AveP', AveP)
        precision, recall, f_score = precision_recall_fscore(CM)
        summary.scalar('precision', precision)
        summary.scalar('recall', recall)
        summary.scalar('f_score', f_score)
        summary.scalar('MCC', MCC(CM))
        summary.scalar('epoch_loss', loss)
        valid_writer.write(summary, step = int(epoch))
        valid_writer.writer.flush()

        # Save current state:
        if epoch%hparams['SAVE_FREQUENCY'] == 0:
            bytes_output = serialization.to_bytes(state)
            with open(os.path.join(logdir, 'ckpts', 'state_e' + str(epoch) + '.pkl'), 'wb') as pklfile:
                pickle.dump(bytes_output, pklfile)
            logger.info('State {} saved...'.format('state_e' + str(epoch)))

        # Update epoch number:
        state = state.replace(epoch = jax.tree_map(lambda x: x + 1, state.epoch))
        
        # Get epoch for while condition:
        if hparams['N_PARTITIONS'] > 0:
            epoch = state.epoch[0]
        else:
            epoch = state.epoch


    # ---- NEW
    train_writer.close()
    valid_writer.close()
    # ----
    logger.info('Finished...')
    return None