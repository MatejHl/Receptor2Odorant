import os
import pickle
import time
import numpy as np
import jax
from jax import numpy as jnp
import flax
from flax import serialization
from objax.jaxboard import SummaryWriter, Summary, Image, DelayedScalar

import tables

from matplotlib import pyplot as plt

from Receptor2Odorant.metrics import log_confusion_matrix, log_roc_curve, precision_recall_fscore, MCC, roc_auc, roc_curve, precision_recall_curve, average_precision_score, true_negative_rate, log_pr_curve

from Receptor2Odorant.main.loader import ProtBERTDatasetPrecomputeBERT, ProtBERTLoader, ProtBERTCollatePrecomputeBERT_CLS
from Receptor2Odorant.main.make_init import make_init_model
from Receptor2Odorant.main.make_create_optimizer import make_create_optimizer
from Receptor2Odorant.main.make_train_and_eval import make_valid_epoch, make_valid_epoch_pmap
from Receptor2Odorant.main.select_model import get_model_by_name

import logging


def main_eval(hparams):
    """
    """
    datadir = os.path.join(hparams['DATA_PARENT_DIR'], hparams['DATACASE'])
    data_csv = os.path.join(datadir, hparams['VALID_CSV_NAME'])

    model_class = get_model_by_name(hparams['MODEL_NAME'])
    model = model_class(atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    if hparams['SELF_LOOPS']:
        hparams['PADDING_N_EDGE'] = hparams['PADDING_N_EDGE'] + hparams['PADDING_N_NODE'] # NOTE: Because of self_loops
        if len(hparams['BOND_FEATURES']) > 0:
            raise ValueError('Can not have both bond features and self_loops.')

    logger = logging.getLogger('main_eval')
    logger.setLevel(logging.INFO) # logging.DEBUG
    
    h5file = tables.open_file(hparams['BERT_H5FILE'], mode = 'r', title="BERT")
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

    valid_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = data_csv,
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
                        shuffle = False,
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        valid_loader = _valid_loader
    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        valid_loader = _valid_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        valid_loader = valid_loader.cache() # NOTE: This loads the full dataset.
        valid_loader = valid_loader.prefetch(buffer_size = 4)
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
                                line_graph = hparams['LINE_GRAPH']) # 768)
    params = init_model(rngs = {'params' : key_params, 'dropout' : key_dropout, 'num_steps' : _key_num_steps})
    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    # This is needed to create state:
    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], warmup_steps = hparams['OPTIMIZATION']['WARMUP_STEPS'], transition_steps = hparams['OPTIMIZATION']['TRANSITION_EPOCHS']*(len(valid_dataset)/hparams['BATCH_SIZE']))
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
        valid_epoch = make_valid_epoch_pmap(loss_option = hparams['LOSS_OPTION'], logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
    else:
        valid_epoch = make_valid_epoch(loss_option = hparams['LOSS_OPTION'], logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # ------
    # VALID:
    # ------
    start = time.time()
    valid_metrics = valid_epoch(state, valid_loader)
    end = time.time()
    logger.info('TIME: valid_epoch: {}'.format(end - start))
    valid_metrics_np = jax.device_get(valid_metrics)
    CM = sum([metrics.pop('confusion_matrix') for metrics in valid_metrics_np])
    loss = np.mean([metrics.pop('loss') for metrics in valid_metrics_np])
    CM_per_threshold = jax.tree_multimap(lambda *x: sum(x), *[metrics.pop('confusion_matrix_per_threshold') for metrics in valid_metrics_np])
    logger.info('valid loss: {}'.format(loss))
    logger.info('valid accuracy: 0: {}  1: {}'.format(*list(np.diag(CM)/(np.sum(CM, axis = 1) + 10e-9))))
    
    # Gather metrics:
    summary = Summary()
    roc_curve_values = roc_curve(CM_per_threshold, drop_intermediate=True)
    pr_curve_values = precision_recall_curve(CM_per_threshold)
    AveP = average_precision_score(pr_curve_values)
    summary['epoch_confusion_matrix'] = log_confusion_matrix(CM, class_names=['0', '1'])
    summary['epoch_ROC_curve'] = log_roc_curve(roc_curve_values)
    summary['epoch_PR_curve'] = log_pr_curve(pr_curve_values, average_precision = AveP)
    summary.scalar('AUC_ROC', roc_auc(roc_curve_values))
    summary.scalar('AveP', AveP)
    precision, recall, f_score = precision_recall_fscore(CM)
    summary.scalar('true_negative_rate', true_negative_rate(CM))
    summary.scalar('precision', precision)
    summary.scalar('recall', recall)
    summary.scalar('f_score', f_score)
    summary.scalar('MCC', MCC(CM))
    summary.scalar('epoch_loss', loss)

    result = {'DATA_CSV' : data_csv, 
                'RESTORE_FILE' : hparams['RESTORE_FILE'],
                'BERT_H5FILE' : hparams['BERT_H5FILE'],
                'NUM_TEST_DATA' : len(valid_dataset)
                }
    for key in summary.keys():
        print(key)
        if isinstance(summary[key], Image):
            pass
        elif isinstance(summary[key], DelayedScalar):
            print(summary[key].values)
            result[key] = jax.tree_map(lambda x: float(x), summary[key].values)
        print('--------')

    print(result)
    print(CM)
    logger.info('Finished...')
    return result
        
