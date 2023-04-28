import os
import sys
import pickle
import time
import datetime
import numpy
import jax
from jax import numpy as jnp
import flax
from flax import serialization

from Receptor2Odorant.odor_embedding.MPNN.loader import Loader, Collate, DatasetBuilder
from Receptor2Odorant.odor_embedding.MPNN.model.base_model import VanillaMPNN

from Receptor2Odorant.odor_embedding.MPNN.make_init import make_init_model
from Receptor2Odorant.odor_embedding.MPNN.make_create_optimizer import make_create_optimizer
from Receptor2Odorant.odor_embedding.MPNN.make_predict import make_predict_epoch
import logging

def main_predict(hparams):
    logdir = hparams['LOGGING_PARENT_DIR']
    restore_file = hparams['RESTORE_FILE']

    model = VanillaMPNN(num_classes = hparams['NUM_CLASSES'], atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    logger = logging.getLogger('main_predict')
    logger.setLevel(logging.INFO)
    _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(logdir, model.__class__.__name__, _datetime)
    os.makedirs(logdir)
    logger_file_handler = logging.FileHandler(os.path.join(logdir, 'run.log'))
    logger_stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(logger_file_handler)
    logger.addHandler(logger_stdout_handler)

    collate = Collate(padding_n_node = hparams['PADDING_N_NODE'], 
                    padding_n_edge = hparams['PADDING_N_EDGE'],
                    n_partitions = hparams['N_PARTITIONS'])

    predict_dataset = DatasetBuilder(data_csv = hparams['PREDICT_CSV_PATH'],
                        mol_col = hparams['MOL_COL'],
                        label_col = hparams['LABEL_COL'],
                        atom_features = model.atom_features,
                        bond_features = model.bond_features,
                        )

    _predict_loader = Loader(predict_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])
        
    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        predict_loader = _predict_loader
    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        predict_loader = _predict_loader.tf_Dataset_by_example(n_partitions = hparams['N_PARTITIONS'])
        predict_loader = predict_loader.cache()
        predict_loader = predict_loader.shuffle(buffer_size = len(_predict_loader))
        predict_loader = predict_loader.prefetch(buffer_size = 4)
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

    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], transition_steps = 800*(len(predict_dataset)/hparams['BATCH_SIZE']))
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

    loss_option = hparams['LOSS_OPTION']
    predict_epoch = make_predict_epoch(num_classes = model.num_classes, loss_option = loss_option, logger = logger, return_embeddings=True, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # Combine predictions and data
    logits, embedding = predict_epoch(state, predict_loader)
    prediction = numpy.array(jax.nn.sigmoid(logits))
    embedding = numpy.array(embedding)

    df = predict_dataset.data.copy()
    df['prediction'] = list(prediction)
    df['embed'] = list(embedding)

    from shutil import copy
    export_dir = logdir
    copy(src = hparams['PREDICT_CSV_PATH'], dst = os.path.join(export_dir, 'pyrfume_embedding_' + _datetime + '_source.csv'))
    df[[hparams['MOL_COL'], 'prediction', 'embed']].to_json(os.path.join(export_dir, 'pyrfume_embedding_' + _datetime + '_embed.json'), orient = 'index')

    