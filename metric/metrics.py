import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import torch


def f1_loss(y, predictions):
    y = y.data.cpu().numpy()
    predictions = predictions.data.cpu().numpy()
    number_of_labels = y.shape[1]
    pred_sorted = np.argsort(predictions, axis=1)

    # Use the true label count k_u per node u; pick the top-k label indices.
    num_labels = np.sum(y, axis=1)
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro


def BCE_loss(outputs: torch.Tensor, labels: torch.Tensor):
    loss = torch.nn.BCELoss()
    bce = loss(outputs, labels)
    return bce


def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC and AP score averaged across tasks
    '''

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []

    # Per-label AUC, macro-averaged. Labels with only positives or only
    # negatives are skipped since AUC is undefined there.
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    #return {'rocauc': sum(rocauc_list)/len(rocauc_list)}
    return sum(rocauc_list) / len(rocauc_list)


def ap_score(y_true, y_pred):

    ap_score = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    return ap_score


def ap_score_full(y_true, y_pred):
    """Return (macro_ap, micro_ap, per_label_ap_list) for multi-label predictions.

    Per-label AP is a list of length C; entries for labels with no positive
    instances in y_true are returned as float('nan') and excluded from the
    macro mean (sklearn's default behaviour).
    """
    yt = y_true.cpu().detach().numpy()
    yp = y_pred.cpu().detach().numpy()
    macro = float(average_precision_score(yt, yp, average="macro"))
    micro = float(average_precision_score(yt, yp, average="micro"))
    per_label_arr = average_precision_score(yt, yp, average=None)
    per_label = [float(v) for v in np.asarray(per_label_arr).ravel()]
    return macro, micro, per_label