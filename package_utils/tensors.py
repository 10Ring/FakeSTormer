#-*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def masked_inputs(inputs:torch.tensor, 
                  hm_preds:torch.tensor, 
                  cfg:dict,
                  patch_size:int=16,
                  prev_pos_mask:torch.tensor=None,
                  debug=False,
                  shot=0,
                  vid_ids=None):
    '''
    Masked out inputs given positions extracted from Heatmap_preds for multi-shot predictions.
    args:
        inputs of shape (B, 3, T, H, W)
        hm of shape (B, 1, T, Hp, Wp) where Hp, Wp = H//patch_size, W//patch_size
    '''
    # Prepare positions to mask
    batch_size = inputs.shape[0]
    Hp, Wp = hm_preds.shape[-2:]
    H, W = inputs.shape[-2:]

    hm_preds = hm_preds.reshape(batch_size, hm_preds.shape[1], hm_preds.shape[2], -1)
    max_vals = hm_preds.max(axis=-1, keepdim=True)[0]
    # pos_ = hm_preds.eq(max_vals).float()[:,:,0,:].unsqueeze(2).repeat(1,1,hm_preds.shape[2],1).unsqueeze(-1) #Take the first pos matrix
    # pos_ = pos_.repeat(1,1,1,1,patch_size*patch_size).reshape(batch_size, pos_.shape[1], pos_.shape[2], H, W)
    pos_ = hm_preds.eq(max_vals).float().reshape(batch_size, 
                                                 hm_preds.shape[1], 
                                                 hm_preds.shape[2], 
                                                 Hp, Wp)
    pos_ = pos_[:,:,1,:,:].unsqueeze(2).repeat(1,1,hm_preds.shape[2],1,1)
    pos_ = F.interpolate(pos_, size=(hm_preds.shape[2], H, W), mode='nearest')

    # Prepare values to fill up
    hard_vals = torch.zeros((1,1,1,1,3), dtype=torch.float)
    mean = torch.as_tensor(cfg.TRANSFORM.normalize.mean)
    std = torch.as_tensor(cfg.TRANSFORM.normalize.std)
    normalized_vals = hard_vals.sub_(mean).div_(std).to(dtype=torch.float).cuda()

    inputs_ = inputs.permute(0, 2, 3, 4, 1)
    pos = pos_.permute(0, 2, 3, 4, 1)

    if prev_pos_mask is not None:
        pos = torch.logical_or(pos, prev_pos_mask).int()

    outs = torch.where(pos.repeat(1,1,1,1,3).bool(), normalized_vals, inputs_)
    inputs = outs.permute(0, 4, 1, 2, 3)

    if debug:
        if vid_ids is not None:
            vis_input_tensor(inputs[0], file_name=f'test_{vid_ids[0]}_{shot+1}.png')
        else:
            vis_input_tensor(inputs[0], file_name=f'test_{shot+1}.png')

    return inputs, pos


def vis_input_tensor(tensor, file_name, normalize=True):
    '''
    Visualize input tensor for debugging
    args:
        tensor of shape (3, H, W) or (3, T, H, W)
    '''
    assert tensor.ndim in [3, 4]
    H, W = tensor.shape[-2:]

    if normalize:
        tensor = tensor.clone()
        min = float(tensor.min())
        max = float(tensor.max())
        tensor.add_(-min).div_(max - min + 1e-5)
    
    if tensor.ndim == 4:
        depth = tensor.shape[1]
        inputs = tensor.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .permute(1, 2, 3, 0)\
                       .cpu().numpy()
    else:
        inputs = tensor.mul(255)\
                    .clamp(0, 255)\
                    .byte()\
                    .permute(1, 2, 0)\
                    .cpu().numpy()
        depth = 1

    grid_image = np.zeros((H, depth*W, 3), dtype=np.uint8)
    for i in range(depth):
        image = inputs[i]
        grid_image[0:H, i*W:(i+1)*W, :] = image
    
    Image.fromarray(grid_image).save(file_name)
