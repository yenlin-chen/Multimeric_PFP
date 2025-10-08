'''This script provides definition and function of the most common
metrics for evaluating the performance of a regression model.'''

import torch
import numpy as np

########################################################################
# METRICS FOR CLASSIFICATION
########################################################################

def comp_tp_fp_tn_fn(output, true, thres=0.5):

    '''
    Computes TP, FP, TN, FN based on each class. The returned
    Ndarray is a 4D array with size equal to the number of predicted
    classes for each dimension.
    '''

    # output & true shape: (n_data, n_GO_terms)
    pred = torch.where(torch.sigmoid(output)>=thres, 1, 0)

    # shape: (n_GO_terms,)
    tp = torch.logical_and(pred==1, true==1).detach().sum(dim=0)
    fp = torch.logical_and(pred==1, true==0).detach().sum(dim=0)
    tn = torch.logical_and(pred==0, true==0).detach().sum(dim=0)
    fn = torch.logical_and(pred==0, true==1).detach().sum(dim=0)

    # shape: (n_GO_terms, 4)
    return torch.column_stack((tp, fp, tn, fn))

def comp_metrics_avg(tp_fp_tn_fn, metric_avg_type='macro'):

    # shape of tp_fp_tn_fn: (n_GO_terms, 4)

    if metric_avg_type.lower() == 'micro':
        raise NotImplementedError('Micro averaging is not implemented.')
        tp, fp, tn, fn = tp_fp_tn_fn.sum(dim=0)
    elif metric_avg_type.lower() == 'macro':
        tp, fp, tn, fn = (
            tp_fp_tn_fn[:,0], tp_fp_tn_fn[:,1],
            tp_fp_tn_fn[:,2], tp_fp_tn_fn[:,3]
        )
    else:
        raise ValueError('Accepts "macro" or "micro" for metric_avg_type.')

    # shape: (n_GO_terms,)
    prec = tp / (tp + fp)   # PPV
    # recall = torch.where(tp+fn != 0, tp / (tp + fn), 0) # TPR # FLAG
    recall = tp / (tp + fn) # TPR
    spec = tn / (tn + fp)   # TNR

    f1 = 2*recall*prec / (recall+prec)
    acc = (tp+tn) / (tp+fp+tn+fn)

    if metric_avg_type == 'macro':
        prec   = torch.nanmean(prec)
        recall = torch.nanmean(recall) # = balanced accuracy when macro
        spec   = torch.nanmean(spec)
        f1     = torch.nanmean(f1)
        acc    = torch.nanmean(acc)

    return prec.item(), recall.item(), spec.item(), f1.item(), acc.item()

########################################################################
# METRICS FOR REGRESSION
########################################################################

def pcc(pred, true):
    '''Pearson correlation coefficient between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    pred = pred - torch.mean(pred)
    true = true - torch.mean(true)

    numerator = torch.sum(pred * true)
    denominator = torch.sqrt(
        torch.sum(pred ** 2) * torch.sum(true ** 2)
    )

    return ( numerator / denominator ).item()

def rmse(pred, true):
    '''Root mean squared error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return ( torch.sqrt(torch.mean((pred - true) ** 2)) ).item()

def mae(pred, true):
    '''Mean absolute error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return ( torch.mean(torch.abs(pred - true)) ).item()

def mse(pred, true):
    '''Mean squared error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return ( torch.mean((pred - true) ** 2) ).item()

def r2(pred, true):
    '''R^2 score between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    numerator = torch.sum((pred - true) ** 2)
    denominator = torch.sum((true - torch.mean(true)) ** 2)

    return ( 1 - numerator / denominator ).item()

if __name__ == '__main__':

    from time import time
    import sklearn.metrics as m

    pred = torch.tensor([1,2,3,4,5.5], dtype=torch.float)
    true = torch.tensor([9,8,7,6,57], dtype=torch.float)

    print('r2')
    start = time()
    val = m.r2_score(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = r2(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('MSE')
    start = time()
    val = m.mean_squared_error(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = mse(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('RMSE')
    start = time()
    val = m.mean_squared_error(true, pred, squared=False)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = rmse(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('MAE')
    start = time()
    val = m.mean_absolute_error(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = mae(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
