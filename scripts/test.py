#-*- coding: utf-8 -*-
import os
import sys
import time
import argparse
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import math
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from glob import glob
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image

from package_utils.transform import (
    final_transform,
    get_center_scale, 
    get_affine_transform,
)
from configs.get_config import load_config
from models import *
from package_utils.utils import vis_heatmap, save_file
from package_utils.tensors import masked_inputs
from package_utils.image_utils import load_image, crop_by_margin
from losses.losses import _sigmoid
from lib.metrics import get_acc_mesure_func, bin_calculate_auc_ap_ar
from datasets import DATASETS, build_dataset
from lib.core_function import AverageMeter
from logs.logger import Logger, LOG_DIR


def parse_args(args=None):
    arg_parser = argparse.ArgumentParser('Processing testing...')
    arg_parser.add_argument('--cfg', '-c', help='Config file', required=True)
    arg_parser.add_argument('--image', '-i', type=str, help='Image for the single testing mode!')
    arg_parser.add_argument('--video', '-v', type=str, help='Video for the single testing mode!')
    args = arg_parser.parse_args(args)
    
    return args


if __name__=='__main__':
    if sys.argv[1:] is not None:
        args = sys.argv[1:]
    else:
        args = sys.argv[:-1]
    args = parse_args(args)
    
    # Loading config file
    cfg = load_config(args.cfg)
    logger = Logger(task='testing')

    #Seed
    seed = cfg.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    task = cfg.TEST.subtask
    flip_test = cfg.TEST.flip_test
    logger.info('Flip Test is used --- {}'.format(flip_test))

    save_preds = cfg.TEST.save_preds
    pred_file = cfg.TEST.pred_file
    
    if task == 'test_img':
        assert args.image is not None, "Image can not be None with single image test mode!"
        logger.info('Turning on single image test mode...')
    if task == 'test_vid':
        assert args.video is not None, "Video can not be None with single video test mode!"
        assert os.path.exists(args.video), 'Video path must be valid, please check the path again!'
        logger.info('Turning on single video test mode...')
    else:
        logger.info('Turning on evaluation mode...')
    if task == 'eval' and cfg.DATASET.DATA.TEST.FROM_FILE:
        assert cfg.DATASET.DATA.TEST.ANNO_FILE is not None, "Annotation file can not be None with evaluation test mode!"
        assert len(cfg.DATASET.DATA.TEST.ANNO_FILE), "Annotation file can not be empty with evaluation test mode!"
        # assert os.access(cfg.DATASET.DATA.TEST.ANNO_FILE, os.R_OK), "Annotation file must be valid with evaluation test mode!"
    device_count = torch.cuda.device_count()
    
    # build and load/initiate pretrained model
    model = build_model(cfg.MODEL, MODELS).to(torch.float)
    logger.info('Loading weight ... {}'.format(cfg.TEST.pretrained))
    model = load_pretrained(model, cfg.TEST.pretrained)
    
    if device_count >= 1:
        model = nn.DataParallel(model, device_ids=cfg.TEST.gpus).cuda()
    else:
        model = model.cuda()
    
    # Define essential variables
    image = args.image
    vid = args.video
    test_file = cfg.TEST.test_file
    video_level = cfg.TEST.video_level
    aspect_ratio = cfg.DATASET.IMAGE_SIZE[1]*1.0 / cfg.DATASET.IMAGE_SIZE[0]
    pixel_std = 200
    rot = 0
    transforms = final_transform(cfg.DATASET)
    metrics_base = cfg.METRICS_BASE
    acc_measure = get_acc_mesure_func(metrics_base)
    no_shot_preds = cfg.TEST.no_shot_preds or 1
    
    model.eval()
    if image is not None and task == 'test_img':
        img = load_image(image)
        img = cv2.resize(img, (317, 317))
        img = img[18:(317-18), 18:(317-18), :]
        c, s = get_center_scale(img.shape[:2], aspect_ratio, pixel_std)
        trans = get_affine_transform(c, s, rot, cfg.DATASET.IMAGE_SIZE)
        input = cv2.warpAffine(img,
                               trans,
                               (int(cfg.DATASET.IMAGE_SIZE[0]), int(cfg.DATASET.IMAGE_SIZE[1])),
                               flags=cv2.INTER_LINEAR,
                              )
        with torch.no_grad():
            st = time.time()
            img_trans = transforms(input/255).to(torch.float)
            img_trans = torch.unsqueeze(img_trans, 0)
            if device_count > 0:
                img_trans = img_trans.cuda(non_blocking=True)
            
            outputs = model(img_trans)
            hm_outputs = outputs[0]['hm']
            cls_outputs = outputs[0]['cls'].sigmoid()
            hm_preds = _sigmoid(hm_outputs).cpu().numpy()
            if cfg.TEST.vis_hm:
                print(f'Heatmap max value --- {hm_preds.max()}')
                vis_heatmap(img, hm_preds[0], 'output_pred.jpg')
            label_pred = cls_outputs.cpu().numpy()
            label = 'Fake' if label_pred[0][-1] > cfg.TEST.threshold else 'Real'
            logger.info('Inferencing time --- {}'.format(time.time() - st))
            logger.info('{} --- {}'.format(label, label_pred[0][-1]))
            logger.info('-----------------***--------------------')
    if vid is not None and task == 'test_vid':
        print(vid)
        img_list = []
        n_frames = cfg.DATASET.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES
        assert n_frames is not None, 'Number of video frames can not be None!'
        # Load first n_frames inside the video
        img_paths = glob(f'{args.video}/*.png')
        img_paths = natsorted(img_paths) # correct the order of image paths
        img_paths = img_paths[:n_frames]
        for img_path in img_paths:
            img = Image.open(img_path)
            H, W = img.size
            img = img.crop((15, 15, W-15, H-15))
            img_list.append(img)
        
        # Transform images
        transformed_imgs = torch.tensor([]).cuda()
        for _i in img_list:
            img_resize = _i.resize((int(cfg.DATASET.IMAGE_SIZE[0]), int(cfg.DATASET.IMAGE_SIZE[1])))
            img_resize = np.array(img_resize)/255
            img_tensor = transforms(img_resize).to(torch.float)
            if device_count > 0:
                img_tensor = img_tensor.cuda(non_blocking=True)
            transformed_imgs = torch.cat((transformed_imgs, img_tensor.unsqueeze(0)), 0)
        
        with torch.no_grad():
            st = time.time()
            transformed_imgs = transformed_imgs.transpose(0,1).unsqueeze(0)
            outputs = model(transformed_imgs)
        
        hm_outputs = outputs[0]['hm']
        cls_outputs = outputs[0]['cls'].sigmoid()
        temp_loc_outputs = outputs[0]['temp_loc'].sigmoid()
        temp_loc_preds = temp_loc_outputs.cpu().numpy()
        hm_preds = _sigmoid(hm_outputs).cpu().numpy()
        
        if cfg.TEST.vis_hm:
            print(f'Heatmap max value --- {hm_preds.max()}')
            print(f'Heatmap min value --- {hm_preds.min()}')
            vis_heatmap(img_list, hm_preds[0], 'output_pred.jpg', temp_loc_preds=temp_loc_preds[0])
        
        label = 'Fake' if temp_loc_preds[0][-1] > cfg.TEST.threshold else 'Real'
        logger.info('Inferencing time --- {}'.format(time.time() - st))
        logger.info('{} --- {}'.format(label, temp_loc_preds[0]))
        logger.info('-----------------***--------------------')
    if task == 'eval':
        logger.info(f'Using metric-base {metrics_base} for evaluation!')
        logger.info(f'Video level evaluation mode: {video_level}')
        st = time.time()
        test_dataset = build_dataset(cfg.DATASET, 
                                     DATASETS,
                                     default_args=dict(split='test', config=cfg.DATASET))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                     shuffle=True,
                                     num_workers=cfg.DATASET.NUM_WORKERS)
        logger.info('Dataset loading time --- {}'.format(time.time() - st))

        apr = cfg.TEST.apr
        test_dataloader = tqdm(test_dataloader, dynamic_ncols=True)
        with torch.no_grad():
            # Make sure all tensors in same device
            total_preds = torch.tensor([]).cuda().to(dtype=torch.float)
            total_labels = torch.tensor([]).cuda().to(dtype=torch.float)
            vid_preds = {}
            vid_labels = {}

            # Achieving frame-level predictions to save into file
            pred_meta = {}

            for b, (inputs, labels, meta) in enumerate(test_dataloader):
                i_st = time.time()
                prev_pos_mask = None
                prev_hm_preds = None
                b_vid_ids = [vid for vid in meta['vid_id']]

                if 'img_path' in meta.keys():
                    b_data_paths = [ip for ip in meta['img_path']]
                elif 'vid_path' in meta.keys():
                    b_data_paths = [ip for ip in meta['vid_path']]
                else:
                    if save_preds: raise ValueError('There is no img or vid data for saving!')

                if device_count > 0:
                    inputs = inputs.to(dtype=torch.float).cuda()
                    labels = labels.to(dtype=torch.float).cuda()

                for i_shot in range(no_shot_preds): # multi-shot predictions
                    logger.info(f'Running the {i_shot} shot of predictions')
                    if i_shot > 0:
                        new_inputs, pos_mask = masked_inputs(inputs=inputs,
                                                             hm_preds=prev_hm_preds,
                                                             prev_pos_mask=prev_pos_mask, 
                                                             cfg=cfg.DATASET,
                                                             patch_size=16,
                                                             shot=i_shot,
                                                             debug=False,
                                                             vid_ids=b_vid_ids)
                        outputs = model(new_inputs)
                        prev_pos_mask = pos_mask
                    else:
                        outputs = model(inputs)
                    
                    # Applying Flip test
                    if flip_test:
                        if inputs.dim() == 4:
                            outputs_1 = model(inputs.flip(dims=(3,)))
                        else:
                            outputs_1 = model(inputs.flip(dims=(4,)))
                        
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                        if flip_test:
                            outputs_1 = outputs_1[0]
                    
                    #In case outputs contain a dict key
                    if isinstance(outputs, dict):
                        if flip_test:
                            hm_outputs = (outputs['hm'] + outputs_1['hm'])/2 if 'hm' in outputs.keys() else None
                            cls_outputs = (outputs['cls'] + outputs_1['cls'])/2
                            outputs_temp_loc = (outputs['temp_loc'] + outputs_1['temp_loc'])/2 if 'temp_loc' in outputs.keys() else None
                        else:
                            hm_outputs = outputs['hm'] if 'hm' in outputs.keys() else None
                            cls_outputs = outputs['cls']
                            outputs_temp_loc = outputs['temp_loc'] if 'temp_loc' in outputs.keys() else None
                        prev_hm_preds = hm_outputs
                    logger.info('Inferencing time --- {}'.format(time.time() - st))

                    #Grisping data item
                    for b_i in range(len(b_data_paths)):
                        if b_data_paths[b_i] not in pred_meta.keys():
                            pred_meta[b_data_paths[b_i]] = list((cls_outputs[b_i].clone().detach().item(), 
                                                                 labels[b_i].clone().detach().item()))
                        else:
                            pred_meta[b_data_paths[b_i]].extend(
                                list((cls_outputs[b_i].clone().detach().item(), 
                                      labels[b_i].clone().detach().item()))
                            )

                    if i_shot == (no_shot_preds-1):
                        if not video_level:
                            total_preds = torch.cat((total_preds, cls_outputs), 0)
                            total_labels = torch.cat((total_labels, labels), 0)
                        else:
                            for idx, vid_id in enumerate(b_vid_ids):
                                if vid_id in vid_preds.keys(): 
                                    vid_preds[vid_id] = torch.cat((vid_preds[vid_id], torch.unsqueeze(cls_outputs[idx], 0)), 0)
                                else:
                                    vid_preds[vid_id] = torch.unsqueeze(cls_outputs[idx].clone().detach(), 0).cuda().to(dtype=torch.float)
                                    vid_labels[vid_id] = torch.unsqueeze(labels[idx].clone().detach(), 0).cuda().to(dtype=torch.float)

            if video_level:
                for k in vid_preds.keys():
                    total_preds = torch.cat((total_preds, torch.mean(vid_preds[k], 0, keepdim=True)), 0)
                    total_labels = torch.cat((total_labels, vid_labels[k]), 0)

            acc_ = acc_measure(total_preds, targets=None, labels=total_labels, threshold=cfg.TEST.threshold)
            metrics = bin_calculate_auc_ap_ar(total_preds, total_labels, metrics_base=metrics_base, threshold=cfg.TEST.threshold, apr=apr)
            best_thr = metrics['best_thr']

            if apr:
                auc_, ap_, ar_, mf1_ = metrics['auc'], metrics['ap'], metrics['ar'], metrics['mf1']
                    
                logger.info(f'Current ACC, AUC, AP, AR, mF1, THR for {cfg.DATASET.DATA.TEST.FAKETYPE} --- {cfg.DATASET.DATA.TEST.LABEL_FOLDER} -- \
                    {acc_*100} -- {auc_*100} -- {ap_*100} -- {ar_*100} -- {mf1_*100} -- {best_thr}')
            else:
                bacc_, auc_, p_, r_, s_, f1_, eer_ = metrics['bacc'], metrics['auc'], metrics['p'], metrics['r'], metrics['s'], metrics['f1'], metrics['eer']

                logger.info(f'Current ACC, BACC, AUC, P, R, S, F1, EER, THR for {cfg.DATASET.DATA.TEST.FAKETYPE} --- {cfg.DATASET.DATA.TEST.LABEL_FOLDER} -- \
                    {acc_*100} -- {bacc_*100} -- {auc_*100} -- {p_*100} -- {r_*100} -- {s_*100} -- {f1_*100} -- {eer_*100} -- {best_thr}')

            if save_preds:
                logger.info(f'Preditions will be saved into -- {pred_file}')
                save_file(data=pred_meta, file_path=pred_file)
