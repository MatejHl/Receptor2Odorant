
import jax
from flax.traverse_util import flatten_dict
from Receptor2Odorant.regularizers import l2_norm_regularizer, l1_norm_regularizer
from Receptor2Odorant.utils import find_params_by_node_name


def make_regularization_loss(params_path, alpha, option = 'l2'):
    """
    TODO: Would 'scan'-like make loops faster? 
    """
    if params_path == 'ALL':
        raise NotImplementedError('Put case of regularizing all')
    elif params_path == 'KERNEL' or params_path == 'BIAS':
        node_name = params_path.lower()
        if option == 'l2':
            def reg_loss_func(params):
                flat_params = find_params_by_node_name(params, node_name)
                loss = sum(l2_norm_regularizer(x, alpha) for x in jax.tree_leaves(flat_params))
                return loss
        elif option == 'l1':
            def reg_loss_func(params):
                flat_params = find_params_by_node_name(params, node_name)
                loss = sum(l1_norm_regularizer(x, alpha) for x in jax.tree_leaves(flat_params))
                return loss
    else:
        if option == 'l2':
            def reg_loss_func(params):
                flat_params = flatten_dict(params, keep_empty_nodes=False, is_leaf=None, sep='/')
                loss = 0.0
                for key, x in flat_params.items():
                    if key in params_path:
                        loss += l2_norm_regularizer(x, alpha)
                    else:
                        loss += 0.0
                return loss
        elif option == 'l1':
            def reg_loss_func(params):
                flat_params = flatten_dict(params, keep_empty_nodes=False, is_leaf=None, sep='/')
                loss = 0.0
                for key, x in flat_params.items():
                    if key in params_path:
                        loss += l1_norm_regularizer(x, alpha)
                    else:
                        loss += 0.0
                return loss
        return reg_loss_func    