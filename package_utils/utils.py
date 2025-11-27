#-*- coding: utf-8 -*-
import os
import simplejson as json
from copy import deepcopy
import logging

import cv2
import numpy as np
import plotly.graph_objects as go
import torch

from losses.losses import _sigmoid


def file_extention(file_path):
    f_name, f_extension = os.path.splitext(file_path)
    return f_name, f_extension


def make_dir(dir_path):
    if not os.path.exists(dir_path):
       os.mkdir(dir_path)


def vis_heatmap(images, heatmaps, file_name, **kwargs):
    temp_locs = kwargs.get('temp_loc_preds')
    
    # hm_h, hm_w = heatmaps.shape[1:]
    hm_h, hm_w = np.array(images[0]).shape[:2]
    
    masked_image = np.zeros((hm_h, hm_w*heatmaps.shape[0], 3), dtype=np.uint8)

    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap*255, 0, 255).astype(np.uint8)
        heatmap = np.squeeze(heatmap)
        
        # heatmap_h = heatmap.shape[0]
        # heatmap_w = heatmap.shape[1]
        
        if isinstance(images, list):
            # resized_image = cv2.resize(np.array(images[i]), (int(heatmap_h), int(heatmap_w)))
            heatmap = cv2.resize(heatmap, np.array(images[i]).shape[:2], interpolation=cv2.INTER_LINEAR)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image[:, hm_w*i:hm_w*(i+1), :] = colored_heatmap*0.7 + np.array(images[i])*0.3
        else:
            # resized_image = cv2.resize(images, (int(heatmap_h), int(heatmap_w)))
            heatmap = cv2.resize(heatmap, images.shape[:2], interpolation=cv2.INTER_LINEAR)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image[:, hm_w*i:hm_w*(i+1), :] = colored_heatmap*0.7 + images*0.3
        
        if temp_locs is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White color in BGR
            thickness = 2
            position = (20, 30)
            
            temp_loc = temp_locs[i]
            text = f'{temp_loc:.5f}'
            cv2.putText(masked_image[:, hm_w*i:hm_w*(i+1), :], text, position, font, font_scale, color, thickness)
    cv2.imwrite(file_name, masked_image)


def vis_3d_heatmap(heatmap, file_name):
    # Define the dimensions of the cuboid
    z_dim, y_dim, x_dim = heatmap.shape

    Z, Y, X = np.mgrid[:z_dim, :y_dim, :x_dim]

    fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=heatmap.flatten(),
            isomin=0.0,
            isomax=0.999,
            opacity=0.1,
            surface_count=25,
        ))
    fig.update_layout(scene_xaxis_showticklabels=False,
                      scene_yaxis_showticklabels=False,
                      scene_zaxis_showticklabels=False)
    fig.write_image(file_name)


def save_batch_heatmaps(batch_image, 
                        batch_heatmaps, 
                        file_name,
                        normalize=True,
                        batch_cls=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    batch_cls: ['batch_size, num_joints, 1]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)
    
    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        if batch_image.dim() == 4:
            image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()    
        else:
            image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 3, 0)\
                              .cpu().numpy()\

        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            if image.ndim == 4:
                image = image[j, :, :, :]

            resized_image = cv2.resize(image,
                                       (int(heatmap_width), int(heatmap_height)))

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if batch_cls is not None:
                cls = batch_cls[i][j].detach().cpu().numpy()
                colored_heatmap = cv2.putText(colored_heatmap, 
                                              f'Cls Pred: {cls}', 
                                              (heatmap_width*(j+1)-15, 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              1, 1, 
                                              cv2.LINE_AA)
            masked_image = colored_heatmap*0.7 + resized_image*0.3

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    cv2.imwrite(file_name, grid_image)


def debugging_panel(debug_cfg, 
                    batch_image, 
                    batch_heatmaps_gt, 
                    batch_heatmaps_pred, 
                    idx, 
                    normalize=True,
                    batch_cls_gt=None,
                    batch_cls_pred=None,
                    split='train'):
    if debug_cfg.save_hm_gt:
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_gt, 
                            f'samples/{split}_debugs/hm_gt_{idx}.jpg', 
                            normalize=normalize)
    
    if debug_cfg.save_hm_pred:
        batch_heatmaps_pred_ = _sigmoid(batch_heatmaps_pred.clone())
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_pred_, 
                            f'samples/{split}_debugs/hm_pred_{idx}.jpg',
                            normalize=normalize)


def save_file(data, file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f'Data has been saved to --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')


def load_file(file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'Data has been loaded from --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')

    return data


def draw_landmarks(image, landmarks):
    """This function is to draw facial landmarks into transformed images
    """
    assert landmarks is not None, "Landmarks can not be None!"
    
    img_cp = deepcopy(image)
    landmarks = landmarks.astype(int)
    
    for i, p in enumerate(landmarks):
        img_cp = cv2.circle(img_cp, (p[0], p[1]), 2, (0, 255, 0), 1)
    
    return img_cp


def draw_most_vul_points(blended_mask):
    """ Detecting and Drawing the most vulnerable points for visualization purpose
    """
    b_mask_cp = deepcopy(blended_mask)
    target_H, target_W, target_C = b_mask_cp.shape

    max_val = b_mask_cp[...,0].max()
    max_val = max_val if max_val > 0 else 1

    m_v_indices = np.where(b_mask_cp == max_val)

    for j,i in zip(m_v_indices[0], m_v_indices[1]):
        b_mask_cp[j,i] = (255, 0, 0)
    
    return b_mask_cp
