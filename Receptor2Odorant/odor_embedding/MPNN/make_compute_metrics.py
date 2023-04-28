import jax
from jax import numpy as jnp

from sklearn.metrics import confusion_matrix, hamming_loss,classification_report, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, mean_absolute_error, mean_squared_error, multilabel_confusion_matrix

from Receptor2Odorant.odor_embedding.MPNN.make_loss_func import make_loss_func

def make_compute_metrics(is_weighted, num_classes, loss_option, num_thresholds = 200):
    loss_func = make_loss_func(is_weighted, num_classes, option = loss_option)
    labels_names = list(range(num_classes))
    def _compute_metrics(logits, labels):
        pred_labels = jax.nn.sigmoid(logits)
        auc = roc_auc_score(labels, pred_labels, average='micro')
        pred_labels = (pred_labels > 0.5) 
        conf_matrix = multilabel_confusion_matrix(y_true = labels, y_pred = pred_labels, labels = labels_names)
        hamming = hamming_loss(labels, pred_labels)
        report = classification_report(labels, pred_labels, output_dict=True, zero_division=True)
        
        return conf_matrix , hamming, report, auc

    if is_weighted:
        def compute_metrics(logits, labels):
            loss = loss_func(logits, labels)
            labels, _ = labels
            conf_matrix, hamming, report, auc = _compute_metrics(logits, labels)
            metrics = {'loss' : loss,
                    'confusion_matrix' : conf_matrix,
                    'hamming' : hamming,
                    'report' : report,
                    'auc': auc
                    }
            return metrics
    else:
        def compute_metrics(logits, labels):
            loss = loss_func(logits, labels)
            conf_matrix, hamming, report, auc = _compute_metrics(logits, labels)
            metrics = {'loss' : loss,
                    'confusion_matrix' : conf_matrix,
                    'hamming' : hamming,
                    'report' : report,
                    'auc': auc
                    }
            return metrics

    return compute_metrics
        