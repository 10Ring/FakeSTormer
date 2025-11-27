#-*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy

from .builder import LOSSES


def _sigmoid(hm):
    x = hm
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def _avg_sigmoid(hm):
    if hm.dim() == 4:
        x = torch.mean(hm, [2, 3])
    else:
        x = hm
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def f_cstency(cstency_hm_preds, cstency_hm_gt, feature='2D'):
    # Heatmap here that is original is returned from model without any modification
    cstency_matrix = torch.zeros_like(cstency_hm_gt).cuda()
    b_size = cstency_hm_preds.size(0)

    indices_ = cstency_hm_gt.view(b_size, -1).argmax(dim=-1)

    cst_hm_dim = cstency_hm_preds.size(1)

    if feature == '2D':
        # Handling 2D output features
        cst_hm_h = cstency_hm_preds.size(2)
        cst_hm_w = cstency_hm_preds.size(3)

        cstency_matrix_ = torch.matmul(
            cstency_hm_preds.view(b_size, cst_hm_dim, -1)[np.arange(b_size), :, indices_].view(b_size, 1, cst_hm_dim),
            cstency_hm_preds.view(b_size, cst_hm_dim, -1)
        )
        cstency_matrix_ = cstency_matrix_.view(b_size, cstency_hm_gt.size(1), cst_hm_h, cst_hm_w) / math.sqrt(cst_hm_dim)
    elif feature == '3D':
        # Handling 3D output features
        cst_hm_d = cstency_hm_preds.size(2)
        cst_hm_h = cstency_hm_preds.size(3)
        cst_hm_w = cstency_hm_preds.size(4)

        cstency_matrix_ = torch.matmul(
            cstency_hm_preds.view(b_size, cst_hm_dim, -1)[np.arange(b_size), :, indices_].view(b_size, 1, cst_hm_dim),
            cstency_hm_preds.view(b_size, cst_hm_dim, -1)
        )
        cstency_matrix_ = cstency_matrix_.view(b_size, cstency_hm_gt.size(1), cst_hm_d, cst_hm_h, cst_hm_w) / math.sqrt(cst_hm_dim)
    else:
        raise ValueError(f'{feature} output shape has not been supported!')
    
    cstency_matrix = cstency_matrix_.sigmoid_()
    
    return cstency_matrix


def _neg_pos_loss(hm_pred, hm_gt):
    pos_idxes = hm_gt > 0
    neg_idxes = ~pos_idxes
    batch_size = hm_gt.size(0)
    neg_pos_gt, neg_pos_pred = torch.zeros(batch_size, 1, dtype=torch.float).cuda(), \
                                    torch.zeros(batch_size, 1, dtype=torch.float).cuda()
    hm_pred_ = torch.squeeze(torch.clone(hm_pred))
    
    for i in range(batch_size):
        neg_pos_gt[i] = torch.sum(hm_gt[i][pos_idxes[i,:,:]]) - torch.sum(hm_gt[i][neg_idxes[i,:,:]])
        neg_pos_pred[i] = torch.sum(hm_pred_[i][pos_idxes[i,:,:]]) - torch.sum(hm_pred_[i][neg_idxes[i,:,:]])
        
    return torch.abs(neg_pos_pred), torch.abs(neg_pos_gt)


def _neg_loss(pred, gt, epsilon=0.1, noise_distribution=0.2, alpha=0.25, **kwargs):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    hm_mask = kwargs.get('hm_mask') # Removing non-computed self-attention positions in total loss

    loss = 0
    pos_inds = gt.eq(1.0).float()
    neg_inds = gt.lt(1.0).float()
    b_size = gt.shape[0] 

    if hm_mask is not None:
        pos_inds = hm_mask * pos_inds
        neg_inds = hm_mask * neg_inds

    neg_weights = torch.pow(1 - gt, 4)

    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * alpha
    pos_loss = (1 - epsilon) * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    pos_loss_noise = epsilon * torch.log(pred) * torch.pow(1 - pred, 2) * noise_distribution * pos_inds
    pos_loss = pos_loss + pos_loss_noise
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    loss *= alpha
    return loss


def _distance_hm_cls_loss(cos_sim_ins, hm_preds, hm_gts, label_preds, label_gts, alpha=0.25):
    b_size = hm_preds.size(0)
    hm_preds = hm_preds.view(b_size, -1)
    hm_gts = hm_gts.view(b_size, -1)
    pos_hm_loss = 0.0
    neg_hm_loss = 0.0

    for i in range(0, b_size//2):
        for j in range(0, b_size//2):
            pos_hm_loss += (1/2) * (1 - cos_sim_ins(hm_preds[i], hm_preds[j]))
            neg_hm_loss += (1/2) * (1 - cos_sim_ins(hm_preds[i], hm_preds[j + b_size//2]))

    cos_loss = pos_hm_loss/((b_size//2)**2) - neg_hm_loss/((b_size//2)**2)
    cos_loss = cos_loss * alpha
    return cos_loss


@LOSSES.register_module()
class BaseLoss(nn.Module):
    def __init__(self, 
                 cfg, 
                 **kwargs):
        self.cfg = cfg
        super().__init__()
        
        for k,v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)
        # Critetion ins
        self.mse_critetion = nn.MSELoss(reduction=self.cfg.mse_reduction)
        if hasattr(self, 'use_ce') and getattr(self, 'use_ce'):
            self.bce_critetion = nn.CrossEntropyLoss(reduction=self.ce_reduction)
        else:
            self.bce_critetion = nn.BCEWithLogitsLoss(reduction=self.cfg.ce_reduction) # For Binary Cross Entropy Loss
        self.ce_critetion = CrossEntropyLoss(reduction=self.cfg.ce_reduction) # For Cross Entropy Loss in general

        # Lambda coefs
        self.offset_lmda = self.cfg.offset_lmda
        self.cls_lmda = self.cfg.cls_lmda
        self.dst_hm_cls_lmda = self.cfg.dst_hm_cls_lmda
        self.hm_lmda = self.cfg.hm_lmda
        self.cstency_lmda = self.cfg.cstency_lmda

        # Others
        self.cos_sim_ins = nn.CosineSimilarity(dim=0, eps=1e-6)

    def _offset_loss(self, preds, gts, apply_filter=False):
        loss = 0
        coefs = gts.gt(0).float() if apply_filter else 1
        n_coefs = coefs.float().sum()
        
        loss = 0.5 * self.mse_critetion(preds * coefs, gts * coefs)
        loss /= (n_coefs + 1e-6)
        loss *= self.offset_lmda
        return loss
    
    def _cls_loss(self, preds, gts):
        loss = 0
        loss = self.bce_critetion(preds, gts)
        loss *= self.cls_lmda
        return loss
    
    def _consistency_loss(self, preds, gts, feature='2D'):
        loss = torch.zeros(1).cuda()
        encode_preds = f_cstency(preds, gts, feature=feature)
        # loss = self.bce_critetion(encode_preds.view(-1, 1), gts.view(-1, 1))
        loss = self.mse_critetion(encode_preds, gts)
        loss *= self.cstency_lmda
        return loss.sum()
    
    def _temp_loc_loss(self, preds, gts, alpha=0.25, gamma=2):
        """
        Calculating loss for temporal location
        agrs:
            preds: output prediction of temporal location
            gts: gt of temporal location
        """
        loss = self.bce_critetion(preds.view(-1).unsqueeze(-1), gts.view(-1).unsqueeze(-1))
        loss *= self.cfg.tmp_loc_lmda
        return loss


@LOSSES.register_module()
class BinaryCrossEntropy(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(BinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=self.reduction)
        
    def __call__(self, pred, y):
        return self.bce(pred, y)


@LOSSES.register_module()
class CombinedFocalLoss(BaseLoss):
    '''nn.Module warpper for focal loss'''
    def __init__(self, 
                 cfg,
                 use_target_weight, 
                 **kwargs):
        super(CombinedFocalLoss, self).__init__(cfg, **kwargs)
        self.hm_loss = _neg_loss
        self.feature = kwargs.get('feature') or '2D'
        
    def forward(self, 
                hm_outputs, hm_targets, 
                cls_preds, cls_gts,
                hm_mask=None,
                offset_preds=None,
                offset_gts=None,
                cstency_preds=None,
                cstency_gts=None,
                target_weight=None,
                temp_loc_preds=None,
                temp_loc_gts=None):
        loss_return = {}
        hm_outputs_ = torch.clone(hm_outputs)
        hm_outputs_ = _sigmoid(hm_outputs_)
        if hm_targets.dim() == 3:
            hm_targets = torch.unsqueeze(hm_targets, 1)
        
        loss_hm = self.hm_loss(hm_outputs_, hm_targets, alpha=self.hm_lmda, hm_mask=hm_mask)
        loss_return['hm'] = loss_hm
        loss_return['cls'] = self._cls_loss(cls_preds, cls_gts)
        
        if self.dst_hm_cls_lmda > 0:
            loss_return['dst_hm_cls'] = _distance_hm_cls_loss(self.cos_sim_ins, 
                                                              hm_outputs, 
                                                              hm_targets, 
                                                              cls_preds, 
                                                              cls_gts, 
                                                              alpha=self.dst_hm_cls_lmda)
        
        if self.offset_lmda > 0 and offset_preds is not None:
            loss_return['offset'] = self._offset_loss(offset_preds, 
                                                      offset_gts, 
                                                      apply_filter=True)
        
        if self.cstency_lmda > 0 and cstency_preds is not None:
            loss_return['cstency'] = self._consistency_loss(cstency_preds, 
                                                            cstency_gts,
                                                            feature=self.feature)
        
        if temp_loc_preds is not None and self.cfg.tmp_loc_lmda is not None:
            loss_temp_loc = self._temp_loc_loss(temp_loc_preds, temp_loc_gts)
            loss_return['temp_loc'] = loss_temp_loc
        
        return loss_return


@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, reduction='mean', lmda=1):
        super(JointsMSELoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.lmda = lmda

    def forward(self, output, target, target_weight=None, **kwargs):
        hm_mask = kwargs.get('hm_mask')
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        if hm_mask is not None:
            hm_mask_ = hm_mask.reshape((batch_size, num_joints, -1)).split(1,1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight and target_weight is not None:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                if hm_mask is not None:
                    heatmap_pred = heatmap_pred*hm_mask_[idx].squeeze()
                    heatmap_gt = heatmap_gt*hm_mask_[idx].squeeze()
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
                
        if self.reduction != 'mean':
            loss = loss * self.lmda / num_joints
        else:
            loss = loss * self.lmda

        return loss


@LOSSES.register_module()
class CombinedMSELoss(BaseLoss):
    def __init__(self,
                 cfg,
                 use_target_weight=False, 
                 **kwargs):
        super(CombinedMSELoss, self).__init__(cfg=cfg, **kwargs)
        self.criterion_hm = JointsMSELoss(use_target_weight=use_target_weight, 
                                          reduction=self.cfg.mse_reduction,
                                          lmda=self.cfg.hm_lmda)
        self.use_target_weight = use_target_weight
        self.feature = kwargs.get('feature') or '2D'

    def forward(self, 
                hm_outputs, 
                hm_targets, 
                cls_preds, 
                cls_gts, 
                hm_mask=None,
                target_weight=None,
                cstency_preds=None,
                cstency_gts=None,
                temp_loc_preds=None,
                temp_loc_gts=None,
                **kwargs):
        loss_return = {}
        loss_hm = self.criterion_hm(hm_outputs, 
                                    hm_targets, 
                                    target_weight=target_weight,
                                    hm_mask=hm_mask)
        loss_return['hm'] = loss_hm

        loss_cls = self._cls_loss(cls_preds/self.temperature, 
                                  cls_gts)
        loss_return['cls'] = loss_cls

        if self.cstency_lmda > 0 and cstency_preds is not None:
            loss_return['cstency'] = self._consistency_loss(cstency_preds, 
                                                            cstency_gts,
                                                            feature=self.feature)

        if temp_loc_preds is not None and self.cfg.tmp_loc_lmda is not None:
            loss_temp_loc = self._temp_loc_loss(temp_loc_preds, 
                                                temp_loc_gts)
            loss_return['temp_loc'] = loss_temp_loc

        return loss_return


@LOSSES.register_module()
class CombinedHeatmapBinaryLoss(nn.Module):
    def __init__(self, use_target_weight, cls_lmda=0.2, reduction='mean', cls_cal=True, **kwargs):
        super(CombinedHeatmapBinaryLoss, self).__init__()
        self.criterion_cls = BinaryCrossEntropy(reduction=reduction)
        self.criterion_hm = BinaryCrossEntropy(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.cls_lmda = cls_lmda if cls_cal else 0
        self.cls_cal = cls_cal

    def forward(self, hm_outputs, hm_targets, cls_preds, cls_gts, target_weight=None):
        batch_size = hm_outputs.size(0)
        hm_targets = hm_targets[:,:,:,0]
        hm_h = hm_outputs.size(2)
        hm_w = hm_outputs.size(3)
        total_pixels = hm_h * hm_w
        loss_hm = torch.zeros(1).cuda()
        hm_outputs_ = torch.clone(hm_outputs)
        hm_outputs_ = _sigmoid(hm_outputs_)
        
        for i in range(hm_h):
            for j in range(hm_w):
                loss_hm_ = self.criterion_hm(hm_outputs_[:,:,i,j], torch.unsqueeze(hm_targets[:,i,j], 1))
                loss_hm += loss_hm_
        
        loss_hm = loss_hm / total_pixels
        loss_return = {}
        loss_return['hm'] = loss_hm
        
        loss_cls = self.criterion_cls(cls_preds, cls_gts)
        loss_return['cls'] = loss_cls
        return loss_return


@LOSSES.register_module()
class CombinedPolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    """
    def __init__(self, use_target_weight, epsilon=2.0, cls_lmda=0.05, reduction='mean', cls_cal=True, **kwargs):
        super(CombinedPolyLoss, self).__init__()
        self.cls_critetion = BinaryCrossEntropy(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.epsilon = epsilon
        self.cls_lmda = cls_lmda if cls_cal else 0
        self.reduction = reduction
        self.cls_cal = cls_cal

    def forward(self, hm_outputs, hm_targets, cls_preds, cls_gts):
        batch_size = hm_outputs.size(0)
        n_classes = hm_outputs.size(1)
        hm_h = hm_outputs.size(2)
        hm_w = hm_outputs.size(3)
        total_pixels = hm_h * hm_w
        poly_loss = torch.zeros(batch_size, 1).cuda()
        hm_outputs_ = _sigmoid(hm_outputs)
        
        for i in range(hm_h):
            for j in range(hm_w):
                ce = binary_cross_entropy(hm_outputs_[:,:,i,j], torch.unsqueeze(hm_targets[:,i,j], -1), reduction='none')
                pt = hm_outputs_[:,:,i,j]
                pt = torch.squeeze(pt)
                pt = torch.where(hm_targets[:,i,j] > 0, pt, 1-pt)
                poly_loss += (ce + self.epsilon * (1.0 - torch.unsqueeze(pt, -1)))

        if self.reduction == 'mean':
            poly_loss = poly_loss.sum()/total_pixels/batch_size
        else:
            poly_loss = poly_loss.sum()
        loss_return = {}
        loss_return['hm'] = poly_loss
        
        loss_cls = self.cls_critetion(cls_preds, cls_gts)
        loss_return['cls'] = loss_cls * self.cls_lmda
        return loss_return
 