import os
import functools
import re
import jax
from jax import numpy as jnp
import numpy
import pandas
import json
import tables

from Receptor2Odorant.mol2graph.read import read_fasta

from Receptor2Odorant.utils import serialize_BERT_hidden_states
from Receptor2Odorant.base_loader import BaseDataset, BaseDataLoader

from transformers.models.bert.tokenization_bert import BertTokenizer


class PrecomputeBertDataset(BaseDataset):
    """
    """
    def __init__(self, data, seq_col, id_col,
                 orient='columns'):
        self.seq_col = seq_col
        self.id_col = id_col
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index = numpy.asarray(index)
        seq = self.data.iloc[index][self.seq_col]
        if not seq == seq:
            print(seq)
        seq = ' '.join(list(seq))
        seq = re.sub(r"[UZOB]", "X", seq)

        ids = self.data.iloc[index][self.id_col]
        return ids, seq


def collate_fn_seq_with_id(batch, tokenizer, n_partitions):
    """
    """
    ids, batch = zip(*batch)
    
    seqs = dict(tokenizer(batch, return_tensors='np', padding = 'max_length', max_length = 512, truncation = True)) # 2048
    if 'position_ids' not in seqs.keys():
            seqs['position_ids'] = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(seqs['input_ids']).shape[-1]), seqs['input_ids'].shape)
    if n_partitions > 0:
        partition_size = len(batch) // n_partitions
        _seqs = []
        for i in range(n_partitions):
            _seq = {}
            for key in seqs.keys():
                _seq[key] = seqs[key][i*partition_size:(i+1)*partition_size]
            _seqs.append(_seq)
        return ids, _seqs
    else:
        return ids, seqs


class PrecomputeBertLoader(BaseDataLoader):
    """
    """
    def __init__(self, dataset, tokenizer,
                    batch_size=1,
                    n_partitions = 0,
                    shuffle=False, 
                    rng=None, 
                    drop_last=False):

        self.n_partitions = n_partitions
        if n_partitions > 0:
            assert batch_size % self.n_partitions == 0

        super(self.__class__, self).__init__(dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        rng = rng,
        drop_last = drop_last,
        collate_fn = functools.partial(collate_fn_seq_with_id, tokenizer = tokenizer,
                                                            n_partitions = n_partitions),
        )


class PrecomputeProtBERT:
    def __init__(self, data_file, save_dir, save_folder_name = None, mode = 'a', id_col = 'UniProt ID', seq_col = 'seq', dbname = '', batch_size = 8, bert_model = None, tokenizer = None):
        """
        """
        if save_folder_name is None:
            save_folder_name = __class__.__name__

        self.data_file = data_file
        self.save_dir = save_dir
        self.save_dir = os.path.join(save_dir, save_folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.id_col = id_col 
        self.seq_col = seq_col
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.apply_bert = self._make_apply_bert(bert_model)

        self.db_id_len = 64
        self.dbname = dbname
        self.mode = mode

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.save_dir will be added
        to the dict in self.save_hparams.
        """
        return {'batch_size' : str(self.batch_size),
                'bert_model' : self.bert_model.__class__.__name__,
                'tokenizer' : self.tokenizer.__class__.__name__}

    def save_hparams(self):
        hparams = self.serialize_hparams()
        hparams.update({'data_file' : self.data_file,
                        'save_dir' : self.save_dir})
        with open(os.path.join(self.save_dir, 'hparams.json'), 'w') as outfile:
            json.dump(hparams, outfile)
        
    def create_h5file(self, expectedrows):
        # PyTable handling:
        class PrecomputeBERTtable(tables.IsDescription):
            id    = tables.StringCol(self.db_id_len)
            hidden_states = tables.Float32Col(shape = (31, 2048, 1024))
            attention_mask = tables.Int32Col(shape = (2048,))

        h5file = tables.open_file(os.path.join(self.save_dir, self.dbname), mode = self.mode, title="ProtBERT")
        group = h5file.create_group("/", name = 'bert', title = 'ProtBERTgroup')
        self.filters = tables.Filters(complevel = 1, complib = 'blosc')
        self.table = h5file.create_table(group, name = 'BERTtable', description = PrecomputeBERTtable, title = "ProtBERTtable",
                                        filters = self.filters, expectedrows = expectedrows)
        self.h5file = h5file
        print(h5file)
        return None

    def _make_apply_bert(self, bert_model):
        @jax.jit
        def apply_bert(seq):
            bert_output = bert_model.module.apply({'params': bert_model.params}, **seq, deterministic = True,
                                 output_attentions = False,
                                 output_hidden_states = True, 
                                 return_dict = True)
            return bert_output
        return apply_bert

    def _precompute_and_save(self, data):
        dataset = PrecomputeBertDataset(data, seq_col = self.seq_col, id_col = self.id_col)
        loader = PrecomputeBertLoader(dataset, tokenizer = self.tokenizer, batch_size = self.batch_size,
                    n_partitions = 0, shuffle=False, rng=None, drop_last=False)

        row = self.table.row

        for i, batch in enumerate(loader):
            ids, batch = batch
            attn_mask = batch['attention_mask']
            _batch = self.apply_bert(batch)
            hidden_states = serialize_BERT_hidden_states(_batch.hidden_states)

            for j in range(len(ids)):
                if len(ids[j]) > self.db_id_len:
                    raise ValueError('ID "{}" is too long for db_id_len: {}'.format(ids[j], self.db_id_len))
                row['id'] = ids[j]
                row['hidden_states'] = hidden_states[j]
                row['attention_mask'] = attn_mask[j]
                row.append()

            if i >= 10:
                self.table.flush()
        
        print('creating index...')
        self.table.cols.id.create_index(optlevel=9, kind='full', filters = self.filters) # Create index for finished table to speed up search
        self.table.flush()
        return None

    def load_data(self):
        _, ext = os.path.splitext(self.data_file)
        if ext == ".fasta" or ext == '.fa':
            df = read_fasta(self.data_file)
            df.name = self.seq_col
            df = df.to_frame()
            df.index.name = self.id_col
            df.reset_index(inplace = True)
        elif ext == '.csv':
            df = pandas.read_csv(self.data_file, sep = ';', index_col = None, header = 0, usecols = [self.id_col, self.seq_col])
        return df

    def precompute_and_save(self):
        data = self.load_data()

        data = data[[self.id_col, self.seq_col]]
        data = data[~data[self.id_col].duplicated()]

        print('Number of records to process:  {}'.format(len(data)))
        
        self.create_h5file(expectedrows = len(data))
        self._precompute_and_save(data)
        self.h5file.close()

        self.save_hparams()
        return None



class PrecomputeProtBERT_CLS(PrecomputeProtBERT):
    def __init__(self, data_file, save_dir, mode = 'a', id_col = 'UniProt ID', seq_col = 'seq', dbname = '', batch_size = 8, bert_model = None, tokenizer = None):
        super(PrecomputeProtBERT_CLS, self).__init__(data_file = data_file, 
                                                save_dir = save_dir,
                                                save_folder_name = __class__.__name__,
                                                mode = mode, 
                                                id_col = id_col, 
                                                seq_col = seq_col, 
                                                dbname = dbname, 
                                                batch_size = batch_size, 
                                                bert_model = bert_model, 
                                                tokenizer = tokenizer)
        self.id_col = id_col 
        self.seq_col = seq_col
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.apply_bert = self._make_apply_bert(bert_model)

        self.db_id_len = 64
        self.dbname = dbname
        self.mode = mode

    def create_h5file(self, expectedrows):
        # PyTable handling:
        class PrecomputeBERTtable(tables.IsDescription):
            id    = tables.StringCol(self.db_id_len)
            hidden_states = tables.Float32Col(shape = (5*1024,))

        h5file = tables.open_file(os.path.join(self.save_dir, self.dbname), mode = self.mode, title="ProtBERT")
        group = h5file.create_group("/", name = 'bert', title = 'ProtBERTgroup')
        self.filters = tables.Filters(complevel = 1, complib = 'blosc')
        self.table = h5file.create_table(group, name = 'BERTtable', description = PrecomputeBERTtable, title = "ProtBERTtable",
                                        filters = self.filters, expectedrows = expectedrows)
        self.h5file = h5file
        print(h5file)
        return None

    def _precompute_and_save(self, data):
        dataset = PrecomputeBertDataset(data, seq_col = self.seq_col, id_col = self.id_col)
        loader = PrecomputeBertLoader(dataset, tokenizer = self.tokenizer, batch_size = self.batch_size,
                    n_partitions = 0, shuffle=False, rng=None, drop_last=False)

        row = self.table.row

        for i, batch in enumerate(loader):
            ids, batch = batch
            _batch = self.apply_bert(batch)
            hidden_states = serialize_BERT_hidden_states(_batch.hidden_states)

            for j in range(len(ids)):
                if len(ids[j]) > self.db_id_len:
                    raise ValueError('ID "{}" is too long for db_id_len: {}'.format(ids[j], self.db_id_len))
                row['id'] = ids[j]
                row['hidden_states'] = jnp.concatenate(hidden_states[j][-5:, 0, :], axis = 0) # CLS
                row.append()

            if i >= 10:
                self.table.flush()
        
        print('creating index...')
        self.table.cols.id.create_index(optlevel=9, kind='full', filters = self.filters) # Create index for finished table to speed up search
        self.table.flush()
        return None