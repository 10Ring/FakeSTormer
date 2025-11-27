#-*- coding: utf-8 -*-
import os

import numpy as np
from sklearn import metrics as cal_metrics
import torch
from sklearn.metrics import (
    average_precision_score, recall_score,
    balanced_accuracy_score, precision_score,
    f1_score 
)
from scipy.stats import gaussian_kde, beta

from losses.losses import _avg_sigmoid, _sigmoid


def bin_calculate_acc(preds, labels, targets=None, threshold=0.5):
    if torch.is_tensor(preds):
        if preds.shape[-1] > 1:
            preds = preds.softmax(dim=-1)
            if preds.shape[-1] == 2:
                preds = preds[:, -1]
            else:
                preds = preds.max(dim=-1, keepdim=True).values
                labels = labels.max(dim=-1, keepdim=True).values
        else:
            preds = preds.sigmoid()
        
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    
    preds_ = (preds >= threshold).astype(int)
    acc = np.mean((preds_ == labels).astype(int), axis=0)
    
    if acc.ndim >= 1:
        return acc[0]
    else:
        return acc


def hm_calculate_acc(preds, targets=None, labels=None, threshold=0.5):
    cls_ = _avg_sigmoid(preds)
    acc = bin_calculate_acc(cls_, labels, threshold=threshold)
    return acc


def hm_bin_calculate_acc(hm_preds, cls_preds, targets=None, labels=None, cls_lamda=0.05):
    # Select top hm_preds
    hm_preds_ = _sigmoid(hm_preds.clone())
    hm_preds_ = torch.reshape(hm_preds_, (hm_preds_.shape[0], hm_preds_.shape[1], -1))
    top_k = torch.topk(hm_preds_, 10, -1).values
    mean_hm_preds = torch.mean(top_k, -1)
    
    cls_preds_ = cls_lamda*cls_preds + (1-cls_lamda)*mean_hm_preds
    acc = bin_calculate_acc(cls_preds_, labels)
    return acc


def bin_calculate_auc_ap_ar(cls_preds, labels, metrics_base='binary', hm_preds=None, cls_lamda=0.1, threshold=0.5, apr=True):
    assert metrics_base in ['binary', 'heatmap', 'combine'], 'Metric base is only one of these values [binary, heatmap, combine]'
    
    if torch.is_tensor(cls_preds):
        if cls_preds.shape[-1] > 1:
            cls_preds = cls_preds.softmax(dim=-1)
            if cls_preds.shape[-1] == 2:
                cls_preds = cls_preds[:, -1]
            else:
                cls_preds = cls_preds.max(dim=-1, keepdim=True).values
                labels = labels.max(dim=-1, keepdim=True).values
        else:
            cls_preds = cls_preds.sigmoid()

        if metrics_base == 'combine':
            assert hm_preds is not None, 'Heatmap predict can not be None if metrics-base is combine'
            hm_preds = _sigmoid(hm_preds)
            hm_preds = torch.reshape(hm_preds, (hm_preds.shape[0], 1, -1))
            top_k = torch.topk(hm_preds, 10, -1).values
            mean_hm_preds = torch.mean(top_k, -1)
            cls_preds = cls_lamda*cls_preds + (1-cls_lamda)*mean_hm_preds

        labels = labels.cpu().numpy()
        cls_preds = cls_preds.cpu().numpy()
    fpr, tpr, thresholds = cal_metrics.roc_curve(labels, cls_preds, pos_label=1)

    metrics = {}
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    top_k_min_indices = np.argsort(distances)[:5]
    top_k_thresholds = thresholds[top_k_min_indices]
    metrics['best_thr'] = optimal_threshold
    metrics['thr_var'] = np.var(top_k_thresholds, ddof=1)

    #AUC
    metrics['auc'] = cal_metrics.auc(fpr, tpr)
    
    if apr:
        # AP metric
        ap = average_precision_score(labels, cls_preds)
        metrics['ap'] = ap
        
        # AR metric
        ar = recall_score(labels, (cls_preds >= threshold).astype(int), average='macro')
        metrics['ar'] = ar
        
        # mF1 metric
        metrics['mf1'] = (ap*ar*2)/(ap+ar)
        
        return metrics
    else:
        # False Negative Rate
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fpr - fnr))]
        eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]
        metrics['eer'] = eer

        metrics['bacc'] = balanced_accuracy_score(labels, (cls_preds >= threshold).astype(int))

        metrics['p'] = precision_score(labels, (cls_preds >= threshold).astype(int))

        metrics['r'] = recall_score(labels, (cls_preds >= threshold).astype(int))

        metrics['s'] = recall_score(labels, (cls_preds >= threshold).astype(int), pos_label=0)

        metrics['f1'] = f1_score(labels, (cls_preds >= threshold).astype(int))
        
        return metrics


def get_acc_mesure_func(task='binary'):
    if task == 'binary':
        return bin_calculate_acc
    elif task == 'heatmap':
        return hm_calculate_acc
    else:
        return hm_bin_calculate_acc


# Compute Empirical CDF
def empirical_cdf(data):
    sorted_data = np.sort(data)
    return sorted_data, np.arange(1, len(sorted_data) + 1) / len(sorted_data)


# Transformations
def apply_cdf_transform(neg_data, pos_data, cdf_type="empirical"):
    if cdf_type == "empirical":
        return empirical_cdf(neg_data)[1], empirical_cdf(pos_data)[1]  # Get the ECDF values
    elif cdf_type == "kde":
        kde_data1 = gaussian_kde(neg_data)
        kde_data2 = gaussian_kde(pos_data)

        x_values = np.linspace(0, 1, 11) # 0.0; 0.1; 0.2; ...

        cdf1 = np.cumsum(kde_data1(x_values))
        cdf1 /= cdf1[-1]

        cdf2 = np.cumsum(kde_data2(x_values))
        cdf2 /= cdf2[-1]
        return cdf1, cdf2
    elif cdf_type == "para":
        alp1, bta1, _, _ = beta.fit(neg_data, floc=0, fscale=1.0001)
        alp2, bta2, _, _ = beta.fit(pos_data, floc=0, fscale=1.0001)

        x_values = np.linspace(0, 1, 11) # 0.0; 0.1; 0.2; ...

        cdf1 = beta.cdf(x_values, alp1, bta1)
        cdf2 = beta.cdf(x_values, alp2, bta2)
        return cdf1, cdf2
    elif cdf_type == "quantile":
        q = np.linspace(0, 1, min(len(neg_data), len(pos_data)))  # Matching quantiles
        F_inv_P0 = np.quantile(neg_data, q)
        F_inv_P1 = np.quantile(pos_data, q)

        return F_inv_P0, F_inv_P1
    else:
        raise ValueError("Unknown CDF type")
