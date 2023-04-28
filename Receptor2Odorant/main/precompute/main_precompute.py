import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" # NOTE: This is due to RuntimeError: INTERNAL: Failed to allocate 16777216 bytes for new constant

def main_precompute(hparams):
    if hparams['MODEL_NAME'] == 'ProtBERT':
        from Receptor2Odorant.main.precompute.prot_bert import  PrecomputeProtBERT_CLS
        from transformers import BertTokenizer, BertConfig, FlaxBertModel
        
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, output_attentions=False)
        bert_model = FlaxBertModel.from_pretrained("Rostlab/prot_bert", from_pt = True, config = config)

        precomuteBERT_CLS = PrecomputeProtBERT_CLS(data_file = hparams['DATA_FILE'],
                                            save_dir = hparams['SAVE_DIR'],
                                            mode = 'w',
                                            dbname = hparams['DBNAME'],
                                            id_col = hparams['ID_COL'],
                                            seq_col = hparams['SEQ_COL'],
                                            batch_size = hparams['BATCH_SIZE'],
                                            bert_model = bert_model,
                                            tokenizer = tokenizer,
                                            )

        precomuteBERT_CLS.precompute_and_save()
        precomuteBERT_CLS.h5file.close()

    else:
        raise ValueError('Unknown MODEL_NAME: \t {}'.format(hparams['MODEL_NAME']))

    return None