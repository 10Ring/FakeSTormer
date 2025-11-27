#-*- coding: utf-8 -*-
import os
import sys
if not os.getcwd() in sys.path:
    sys.path.insert(0, os.getcwd())
import argparse

import torch
import numpy as np
from scipy.stats import wasserstein_distance

from utils import load_file
from lib.metrics import (
    get_acc_mesure_func, bin_calculate_auc_ap_ar,
    apply_cdf_transform
)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reporting metrics based on saved predictions!')
    parser.add_argument('--saved_preds', '-f', help='Path to saved prediction file')
    parser.add_argument('--model_base', '-m', help='The input model is image-level or video-level', default='image')
    parser.add_argument('--metric_level', '-ml', help='Report metrics at image-level or video level', default='image')
    parser.add_argument('--apr', help='Report average metrics instead of simple ones', action='store_true')
    args = parser.parse_args()

    file = args.saved_preds
    model_base = args.model_base
    metric_level = args.metric_level
    apr = args.apr
    #Load data
    data = load_file(file_path=file)

    if "data" in data.keys():
        data = data['data']

    total_preds = []
    total_labels = []
    neg_preds = []
    pos_preds = []
    vid_preds = {}
    vid_labels = {}

    if model_base == 'image':
        for ip in data.keys():
            vid_id = os.path.dirname(ip)
            pred = data[ip][0]
            label = data[ip][1]

            # Just append to total preds
            total_preds.append(pred)
            total_labels.append(label)
            
            if vid_id in vid_preds.keys():
                vid_preds[vid_id].append(pred)
            else:
                vid_preds[vid_id] = [pred]
                vid_labels[vid_id] = [label]

        if metric_level == 'video':
            total_preds = [np.mean(vid_preds[k], keepdims=True) for k in vid_preds.keys()]
            total_labels = [vid_labels[k] for k in vid_labels.keys()]
    else:
        for ip in data.keys():
            vid_id = ip.split('/')[-1]
            pred = [v for idx, v in enumerate(data[ip]) if idx%2==0]
            pred = [np.array(pred).mean()]
            label = [data[ip][1]]

            # Just append to total preds
            total_preds.append(pred)
            total_labels.append(label)

    total_preds = sigmoid(np.array(total_preds))
    total_labels = np.array(total_labels)

    # Assigning predictions to neg/pos groups
    neg_preds = total_preds[total_labels < 1].squeeze()
    pos_preds = total_preds[total_labels == 1].squeeze()

    # Computing metric section
    acc_measure = get_acc_mesure_func('binary')
    acc_ = acc_measure(total_preds, targets=None, labels=total_labels)
    metrics = bin_calculate_auc_ap_ar(total_preds, total_labels, apr=apr)
    best_thr = metrics['best_thr']
    thr_var = metrics['thr_var']

    if apr:
        auc_, ap_, ar_, mf1_ = metrics['auc'], metrics['ap'], metrics['ar'], metrics['mf1']
        print(f'Current ACC, AUC, AP, AR, mF1, THR --- {acc_*100} -- {auc_*100} -- {ap_*100} -- {ar_*100} -- {mf1_*100} -- {best_thr}')
    else:
        bacc_, auc_, p_, r_, s_, f1_, eer_ = metrics['bacc'], metrics['auc'], metrics['p'], metrics['r'], metrics['s'], metrics['f1'], metrics['eer']
        print(f'Current ACC, BACC, AUC, P, R, S, F1, EER, THR, THR_VAR -- {acc_*100} -- {bacc_*100} -- {auc_*100} -- {p_*100} -- {r_*100} -- {s_*100} -- {f1_*100} -- {eer_*100} -- {best_thr:.3f} -- {thr_var:.6f}')

    # Computing Wasserstein distance
    for cdf_type in ['empirical', 'kde', 'para', 'quantile']:
        cdf1, cdf2 = apply_cdf_transform(neg_data=neg_preds, pos_data=pos_preds, cdf_type=cdf_type)

        # Customize the calculation of quantile
        if cdf_type != 'quantile':
            wd = wasserstein_distance(cdf1, cdf2)
        else:
            q = np.linspace(0, 1, min(len(neg_preds), len(pos_preds)))
            wd = np.trapz(np.abs(cdf1 - cdf2), q)
        
        print(f'Wasserstein Distance --- {cdf_type} --- {wd:.6f}')
