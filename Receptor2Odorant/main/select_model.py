# Main model and ablations:
from Receptor2Odorant.main.model.dummy import DummyModel
from Receptor2Odorant.main.model.endCLS_mha_normal_QK import endCLS_mha_normal_QK_model
from Receptor2Odorant.main.model.endCLS_normal_QK import endCLS_normal_QK_model
from Receptor2Odorant.main.model.noGNN_QK import noGNN_QK_model
from Receptor2Odorant.main.model.normal_QK import normal_QK_model
from Receptor2Odorant.main.model.Transformer_GAT import Transformer_GAT_model

# GNN baselines:
from Receptor2Odorant.main.model.Simple_ECC import Simple_ECC_model
from Receptor2Odorant.main.model.Simple_EdgeEnabled_GGNN import Simple_EdgeEnabled_GGNN_model
from Receptor2Odorant.main.model.Simple_GAT import Simple_GAT_model


def get_model_by_name(name):
    """
    Utils to retrieve model given its name
    """
    if name == 'DummyModel':
        model_class = DummyModel
    elif name == 'endCLS_mha_normal_QK_model':
        model_class = endCLS_mha_normal_QK_model
    elif name == 'endCLS_normal_QK_model':
        model_class = endCLS_normal_QK_model
    elif name == 'noGNN_QK_model':
        model_class = noGNN_QK_model
    elif name == 'normal_QK_model':
        model_class = normal_QK_model
    elif name == 'Transformer_GAT_model':
        model_class = Transformer_GAT_model
    
    # Baselines:
    elif name == 'Simple_ECC_model':
        model_class = Simple_ECC_model
    elif name == 'Simple_EdgeEnabled_GGNN_model':
        model_class = Simple_EdgeEnabled_GGNN_model
    elif name == 'Simple_GAT_model':
        model_class = Simple_GAT_model

    
    # Other:
    else:
        raise ValueError('Unknown model name: {}'.format(name))

    return model_class    


