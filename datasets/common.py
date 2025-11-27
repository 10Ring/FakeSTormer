#-*- coding: utf-8 -*-
import os
import sys
import simplejson as json
import math
import random

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from box import Box as edict
from natsort import natsorted
from PIL import Image

from .builder import DATASETS
from .utils import _extract_data_based_dist
from package_utils.utils import file_extention
from package_utils.transform import final_transform
from package_utils.image_utils import cal_mask_wh, gaussian_radius


PREFIX_PATH = '/data/deepfake_cluster/datasets_df/FaceForensics++/c0/'


class ParameterStore:
    _instance = None
    _parameters = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def add_parameters(cls, param_name, param_value):
        cls._parameters[param_name] = param_value

    @classmethod
    def get_parameters(cls, param_name):
        return cls._parameters.get(param_name)

    @classmethod
    def del_parameters(cls):
        for k in cls._parameters.keys():
            del cls._parameters[k]
    
    @classmethod
    def has_key(cls, key):
        return key in cls._parameters
    
    @classmethod
    def reset(cls):
        cls._parameters.clear()
        cls._instance = None


@DATASETS.register_module()
class CommonDataset(Dataset, ABC):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self._cfg = edict(cfg) if not isinstance(cfg, edict) else cfg
        self.dataset = self._cfg.DATA[self.split.upper()].NAME
        # self.train = self._cfg["TRAIN"]
        self.train = (self.split != 'test')
        self.final_transforms = final_transform(self._cfg)
        self.sigma_adaptive = self._cfg.ADAPTIVE_SIGMA
        self.sampler_active = self._cfg.DATA.SAMPLES_PER_VIDEO.ACTIVE
        self.samples_per_video = self._cfg.DATA.SAMPLES_PER_VIDEO[self.split.upper()]
        self.sampler_dist = self._cfg.DATA.SAMPLES_PER_VIDEO.DIST # Distribution of [Real, Fake]
        self.heatmap_w = self._cfg.HEATMAP_SIZE[1]
        self.heatmap_h = self._cfg.HEATMAP_SIZE[0]
        self.split_image = self._cfg.SPLIT_IMAGE
        self.compression = self._cfg.COMPRESSION
        self.data_type = self._cfg.DATA_TYPE

        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)

    @abstractmethod
    def _load_from_path(self, split):
        return NotImplemented
    
    def _load_from_file(self, split, anno_file=None):
        """
        @split: train/val
        This function for loading data from file for 4 types of manipulated images FF++ and FaceXray generation data
        """
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be invalid!"
        data_cfg = self._cfg.DATA
        
        if anno_file is None:
            anno_file = data_cfg[split.upper()].ANNO_FILE
        if not os.access(anno_file, os.R_OK):
            anno_file = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, anno_file)
        assert os.access(anno_file, os.R_OK), "Annotation file can not be invalid!!"

        f_name, f_extention = file_extention(anno_file)
        data = None
        image_paths, labels, mask_paths, ot_props = [], [], [], []
        f = open(anno_file)
        if f_extention == '.json':
            data = json.load(f)
            data = edict(data)['data'] #A list of proprocessed data objects containing image properties

            for item in data:
                assert 'image_path' in item.keys(), 'Image path must be available in item dict!'
                image_path = item.image_path
                ot_prop = {}
                
                # Custom base on the specific data structure
                if not 'label' in item.keys():
                    lb = (('fake' in image_path) or (('original' not in image_path) and ('aligned' not in image_path)))
                else:
                    lb = (item.label == 'fake')
                lb_encoded = int(lb)
                labels.append(lb_encoded)
                
                if PREFIX_PATH in item.image_path:
                    image_path = item.image_path.replace(PREFIX_PATH, self._cfg.DATA[self.split.upper()].ROOT)
                else:
                    image_path = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, item.image_path)
                image_paths.append(image_path)

                # Appending more data properties for data loader
                if 'mask_path' in item.keys():
                    mask_path = item.mask_path
                    if PREFIX_PATH in item.mask_path:
                        mask_path = item.mask_path.replace(PREFIX_PATH, self._cfg.DATA[self.split.upper()].ROOT)
                    else:
                        mask_path = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, item.mask_path)
                    mask_paths.append(mask_path)
                if 'best_match' in item.keys():
                    best_match = item.best_match
                    best_match = [os.path.join(self._cfg.DATA[self.split.upper()].ROOT, bm) for bm in best_match if \
                        self._cfg.DATA[self.split.upper()].ROOT not in bm]
                    ot_prop['best_match'] = best_match
                for lms_key in ['aligned_lms', 'orig_lms']:
                    if lms_key in item.keys():
                        f_lms = np.array(item[lms_key])
                        ot_prop[lms_key] = f_lms
                    
                ot_props.append(ot_prop)
        else:
            raise Exception(f'{f_extention} has not been supported yet! Please change to Json file!')
        
        print('{} image paths have been loaded!'.format(len(image_paths)))
        return image_paths, labels, mask_paths, ot_props
    
    def _gen_vul_parts(self, blending_mask):
        H, W, C = blending_mask.shape
        Hp, Wp = self.heatmap_h, self.heatmap_w
        py, px = int(H // Hp), int(W // Wp)

        assert (H // Hp) == (W // Wp)
        vul_parts = np.zeros((Hp, Wp))

        for i in range(0, Hp):
            for j in range(0, Wp):
                blending_part = blending_mask[(py*i):(py*(i+1)), (px*j):(px*(j+1)), 0]
                part_intensity = np.mean(blending_part)
                vul_parts[i, j] = part_intensity
        vul_parts_out = np.tile(vul_parts[:,:, np.newaxis], (1,1,3)).astype(np.uint8)

        return vul_parts_out
    
    def _mask_out_vulnerability(self, input, mask, fake_intensity, mask_prob=0.9):
        if self.dynamic_blending_prob:
            p_h = self._cfg.IMAGE_SIZE[0] // self.heatmap_h
            p_w = self._cfg.IMAGE_SIZE[1] // self.heatmap_w

            max_value = max(0.1, mask[..., 0].max())
            upper_bound_intensity = min(1.0, fake_intensity)
            upper_bound_value = max(0.1, mask[mask[..., 0] < max_value*upper_bound_intensity].max())
            target_mask_ = (mask[..., 0] > upper_bound_value).astype(int)

            # Randomly mask out mask if the input is real
            if np.count_nonzero(target_mask_) == 0:
                pos_matrix = (self.heatmap_h, self.heatmap_w)
                all_indices = [(i, j) for i in range(self.heatmap_h) for j in range(self.heatmap_w)]
                selected_indices = np.random.choice(len(all_indices), size=math.floor(mask_prob*np.prod((pos_matrix))), replace=False)
                selected_indices_2d = [all_indices[i] for i in selected_indices]
                i_indices, j_indices = zip(*selected_indices_2d)
            else:
                all_indices = [(i, j) for i in range(self.heatmap_h) for j in range(self.heatmap_w) if \
                               ((target_mask_[i,j]==0) and (mask[...,0][i,j] < upper_bound_value))]
                idxes = np.where(target_mask_ == 1)
                n_mask_pos_h = len(idxes[0])
                pos_matrix = (self.heatmap_h, self.heatmap_w)

                if len(all_indices) < math.floor(mask_prob*np.prod((pos_matrix))-n_mask_pos_h):
                    size = math.ceil(len(all_indices)*mask_prob)
                else:
                    size = max(0, math.floor(mask_prob*np.prod((pos_matrix)) - n_mask_pos_h))

                selected_indices = np.random.choice(len(all_indices), size=size, replace=False)
                selected_indices_2d = [all_indices[i] for i in selected_indices]
                i_indices_, j_indices_ = zip(*selected_indices_2d)
                i_indices = np.hstack((idxes[0], np.array(i_indices_)))
                j_indices = np.hstack((idxes[1], np.array(j_indices_)))
            
            target_mask_[i_indices, j_indices] = 1
            idxes = np.where(target_mask_ == 1)
            masked_matrix = 1 - target_mask_

            for i, j in zip(idxes[0], idxes[1]):
                input[int(i*p_h):int((i+1)*p_h), int(j*p_w):int((j+1)*p_w), :] = np.zeros((1, 1, input.shape[2]), dtype=input.dtype)
                # mask[i, j] = np.zeros((mask.shape[2]), dtype=mask.dtype)

        return input, mask, masked_matrix, upper_bound_value
    
    def _mask_out_vulnerability2(self, input, mask, fake_intensity, **kwargs):
        mask_prob = kwargs.get('mask_prob')
        param_store_ins = ParameterStore.get_instance()
        masked_matrix = param_store_ins.get_parameters('masked_matrix')

        p_h = self._cfg.IMAGE_SIZE[0] // self.heatmap_h
        p_w = self._cfg.IMAGE_SIZE[1] // self.heatmap_w
        upper_bound_value = None

        if self.dynamic_blending_prob:
            if masked_matrix is not None:
                upper_bound_value = max(1, np.max(mask[...,0]*masked_matrix))
                fake_intensity = upper_bound_value/255
                target_mask_ = 1 - masked_matrix
                idxes = np.where(target_mask_ == 1)
            else:
                max_value = max(1, mask[..., 0].max())
                max_f_intensity = max_value/255
                fake_intensity = min(fake_intensity, max_f_intensity)

                upper_bound_value = max(1, mask[mask[..., 0] < 255*fake_intensity].max())
                target_mask_ = (mask[..., 0] > upper_bound_value).astype(int)

                # Randomly mask out mask if the input is real
                if np.count_nonzero(target_mask_) == 0:
                    pos_matrix = (self.heatmap_h, self.heatmap_w)
                    all_indices = [(i, j) for i in range(self.heatmap_h) for j in range(self.heatmap_w)]
                    selected_indices = np.random.choice(len(all_indices), size=math.ceil(mask_prob*np.prod((pos_matrix))), replace=False)
                    selected_indices_2d = [all_indices[i] for i in selected_indices]
                    i_indices, j_indices = zip(*selected_indices_2d)
                else:
                    all_indices = [(i, j) for i in range(self.heatmap_h) for j in range(self.heatmap_w) if \
                                ((target_mask_[i,j]==0) and (mask[...,0][i,j] < upper_bound_value))]
                    idxes = np.where(target_mask_ == 1)
                    n_mask_pos_h = len(idxes[0])
                    pos_matrix = (self.heatmap_h, self.heatmap_w)

                    if len(all_indices) < math.floor(mask_prob*np.prod((pos_matrix))-n_mask_pos_h):
                        size = math.ceil(len(all_indices)*mask_prob)
                    else:
                        size = max(1, math.ceil(mask_prob*np.prod((pos_matrix))-n_mask_pos_h))

                    selected_indices = np.random.choice(len(all_indices), size=size, replace=False)
                    selected_indices_2d = [all_indices[i] for i in selected_indices]
                    i_indices_, j_indices_ = zip(*selected_indices_2d)
                    i_indices = np.hstack((idxes[0], np.array(i_indices_)))
                    j_indices = np.hstack((idxes[1], np.array(j_indices_)))
                
                target_mask_[i_indices, j_indices] = 1
                idxes = np.where(target_mask_ == 1)
                masked_matrix = 1 - target_mask_
                param_store_ins.add_parameters('masked_matrix', masked_matrix)
            
            for i, j in zip(idxes[0], idxes[1]):
                rand_val = np.random.randint(0,255)
                input[int(i*p_h):int((i+1)*p_h), int(j*p_w):int((j+1)*p_w), :] = np.full((1, 1, input.shape[2]), 0, dtype=input.dtype)
                # mask[i, j] = np.zeros((mask.shape[2]), dtype=mask.dtype)

        return input, mask, masked_matrix, upper_bound_value, fake_intensity

    def _encode_temporal_target(self, target_mask, **kwargs):
        '''
        Adaptively encode targets based on the vulnerability levels for spatial-temporal outputs (3D)
        '''
        assert self.heatmap_type in ['gaussian', 'm_std_normalized', 'max_normalized']

        if isinstance(target_mask, list):
            target_mask = np.array(target_mask)

        saved_params = {}
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        ndim = len(target_mask)
        heatmap = np.zeros((ndim, hm_h, hm_w), dtype=np.float32) # dimension d, h, w
        # cstency_hm = np.zeros((ndim, hm_h, hm_w), dtype=np.float32)

        derivative = np.diff(target_mask[:,:,:,0], axis=0)
        derivative = np.absolute(derivative)
        d_max = max(1, np.max(derivative))

        if bool(derivative.max()) and kwargs.get('vis_derivative'):
            idx = kwargs.get('idx')
            for i in range(len(derivative)):
                di = derivative[i].astype(np.uint8)
                di = np.repeat(di[:,:,np.newaxis], 3, axis=2)
                Image.fromarray(di).save(f'samples/debugs/derivative_f_{idx}_{i}.png')

        if self.heatmap_type == 'gaussian':
            x = np.arange(0, hm_w, 1, float)
            y = np.arange(0, hm_h, 1, float)
            y = np.expand_dims(y, -1)
            z = np.arange(0, ndim, 1, float)
            z = np.expand_dims((np.expand_dims(z, -1)), -1)
            derivative = np.concatenate((np.zeros_like(derivative[0][np.newaxis,...]), derivative), axis=0)
            centers = np.where(derivative == max(0.1, derivative.max()))

            for i,j,k in zip(centers[0], centers[1], centers[2]):
                heatmap_ijk = np.exp(-(((z - i) ** 2) / (2.0 * (self.sigma/2) ** 2) + 
                                    ((y - j) ** 2) / (2.0 * (self.sigma/2) ** 2) + 
                                    ((x - k) ** 2) / (2.0 * (self.sigma/2) ** 2)))
                heatmap = np.maximum(heatmap_ijk, heatmap)
        elif self.heatmap_type == 'm_std_normalized':
            d_m = kwargs.get('d_mean') or np.mean(derivative)
            d_std = kwargs.get('d_std') or np.std(derivative)
            derivative = np.concatenate((np.zeros_like(derivative[0][np.newaxis,...]), derivative), axis=0)
            # Calculating the 3D self-consistency map
            # cstency_hm = 255 - np.absolute(d_max - derivative)

            if d_std != 0:
                heatmap = (derivative - d_m)/d_std

            saved_params = {'d_mean': d_m, 'd_std': d_std}
        elif self.heatmap_type == 'max_normalized':
            derivative = np.concatenate((np.zeros_like(derivative[0][np.newaxis,...]), derivative), axis=0)
            # Calculating the 3D self-consistency map
            # cstency_hm = 255 - np.absolute(d_max - derivative)
            if d_max != 0:
                heatmap[1:] = derivative/d_max
        else:
            raise ValueError('Now only support gaussian or mean std normalization')

        return heatmap, derivative/d_max, saved_params

    def _encode_target(self, target_mask, fake_intensity=0.5):
        '''
        Adaptively encode targets based on the vulnerability levels
        '''
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        heatmap = np.zeros((1, hm_h, hm_w), dtype=np.float32)
        
        # Draw heatmap for blending region
        max_val_all = target_mask[..., 0].max()
        max_val = max_val_all if max_val_all > 0 else 255

        # Select value to draw attention masks
        if self.data_type == "video":
            lower_bound_intensity = max(0.0, (fake_intensity-0.1))
            upper_bound_intensity = min(1.0, (fake_intensity+0.1))
            target_mask_ = ((target_mask[..., 0] >= 255*lower_bound_intensity) & (target_mask[..., 0] < 255*upper_bound_intensity)).astype(np.int8)
        else:
            target_mask_ = (target_mask[..., 0] >= max_val*fake_intensity).astype(np.int8)
            
        points = np.where(target_mask_ == 1)

        for j, i in zip(points[0], points[1]):
            if self.sigma_adaptive:
                w_sbi, h_sbi = cal_mask_wh((j, i), target_mask[..., 0])
                radius = gaussian_radius((h_sbi, w_sbi))
                self.sigma = radius/3 + 1e-4
            tmp = self.sigma * 3
            size = tmp * 2 + 1
            ul = [int(i - tmp), int(j - tmp)]
            br = [int(i + tmp + 1), int(j + tmp + 1)]
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
            
            g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
            
            img_x = max(0, ul[0]), min(br[0], hm_w)
            img_y = max(0, ul[1]), min(br[1], hm_h)
            
            heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]]
            )

        return heatmap, None
    
    def _encode_target_v1(self, target_mask, fake_intensity=0.5):
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        # fake_ratio = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_outputs = 1
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        patches = [[0, 0], [0, 1/2], [1/2, 0], [1/2, 1/2]]
        target_H, target_W = target_mask[..., 0].shape[:2]
        heatmap = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        cstency_hm = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        max_val_all = target_mask[..., 0].max()
        
        # Draw heatmap for blending region
        for fr in range(len(patches)):
            # target_mask_ = np.where(((target_mask[..., 0] > 255*fake_ratio[fr]) & (target_mask[..., 0] <= 255*fake_ratio[fr+1])), 1, 0)
            p_x1, p_y1 = int(target_W * patches[fr][0]), int(target_H * patches[fr][1])
            p_x2, p_y2 = int(target_W * (patches[fr][0] + 1/2)), int(target_H * (patches[fr][1] + 1/2))

            max_value = target_mask[p_y1:p_y2, p_x1:p_x2, 0].max()
            max_value = max_value if max_value > 0 else 1
            target_mask_ = (target_mask[p_y1:p_y2, p_x1:p_x2, 0] == (max_value)).astype(np.uint8)
            points = np.where(target_mask_ == 1)
            
            if len(points[0]):
                p = (points[0] + p_y1, points[1] + p_x1)

                for j, i in zip(p[0], p[1]):
                    if self.sigma_adaptive:
                        w_sbi, h_sbi = cal_mask_wh((j, i), target_mask[..., 0])
                        radius = gaussian_radius((h_sbi, w_sbi))
                        self.sigma = radius/3 + 1e-4
                    tmp = self.sigma * 3
                    size = tmp * 2 + 1
                    ul = [int(i - tmp), int(j - tmp)]
                    br = [int(i + tmp + 1), int(j + tmp + 1)]
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    
                    x0 = y0 = size // 2
                    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
                    
                    g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
                    
                    img_x = max(0, ul[0]), min(br[0], hm_w)
                    img_y = max(0, ul[1]), min(br[1], hm_h)
                    
                    heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                        heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]])
                
                if n_outputs > 1:
                    cstency_hm[fr][p_y1:p_y2, p_x1:p_x2] = 255 - np.absolute(max_value - target_mask[p_y1:p_y2, p_x1:p_x2, 0])
                else:
                    cstency_hm[0][p_y1:p_y2, p_x1:p_x2] = 255 - np.absolute(max_val_all - target_mask[p_y1:p_y2, p_x1:p_x2, 0])
                
        return heatmap, cstency_hm
    
    def _encode_target_v2(self, target_mask, fake_intensity=0.5):
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        n_outputs = 1
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        target_H, target_W = target_mask[..., 0].shape[:2]
        heatmap = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        cstency_hm = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        
        # Draw heatmap for blending region
        target_mask_ = (target_mask[..., 0] > 128).astype(np.uint8)
        points = np.where(target_mask_ == 1)
            
        if len(points[0]):
            p = (int(points[0].mean()), int(points[1].mean()))
            j,i = p
            
            if self.sigma_adaptive:
                w_sbi, h_sbi = cal_mask_wh((j, i), target_mask[..., 0])
                radius = gaussian_radius((h_sbi, w_sbi))
                self.sigma = radius/3 + 1e-4
            tmp = self.sigma * 3
            size = tmp * 2 + 1
            ul = [int(i - tmp), int(j - tmp)]
            br = [int(i + tmp + 1), int(j + tmp + 1)]
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
            
            g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
            
            img_x = max(0, ul[0]), min(br[0], hm_w)
            img_y = max(0, ul[1]), min(br[1], hm_h)
            
            heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]])

            cstency_hm[0] = 255 - np.absolute(target_mask[j,i,0] - target_mask[..., 0])
            
        return heatmap, cstency_hm

    def _sampler(self, image_paths, labels, epoch=0, **params):
        if self.sampler_dist[0] != 1. or self.sampler_dist[1] != 1.:
            image_paths, labels, params = _extract_data_based_dist(self.data_type, image_paths, labels, self.sampler_dist, **params)

        vid_dict = {}
        data = {"image_paths": [], "labels": []}
        
        for k,v in params.items():
            if v is not None and len(v): data[k] = []
        
        for idx, ip in enumerate(image_paths):
            f_name = ip.split('/')[-1]

            if self.compression in ['c0', 'c23', 'c40']:
                vid_id = os.path.dirname(ip)
                if self.dataset == 'FF++' and self.train:
                    f_type = ip.split('/')[-3]
                    vid_id = '_'.join([f_type, vid_id])
            else:
                raise NotImplementedError('Only c23, c40, and c0 compression mode is supported now! Please check again!')
            lb = labels[idx]
            
            data_per_vid = dict(image=ip, label=lb)
            for k,v in params.items():
                if k in data.keys():
                    data_per_vid[k] = v[idx]
            
            if vid_id in vid_dict.keys():
                vid_dict[vid_id].append(data_per_vid)
            else:
                vid_dict[vid_id] = [data_per_vid]
                
        if self.data_type == "image":
            """
            Samples data for the mode of working with single images
            """
            for vid_id in vid_dict.keys():
                if self.train:
                    samples_per_vid = random.choices(vid_dict[vid_id], k=self.samples_per_video)
                else:
                    samples_per_vid = random.sample(vid_dict[vid_id], k=len(vid_dict[vid_id]))
                
                for spl in samples_per_vid:
                    data["image_paths"].append(spl["image"])
                    data["labels"].append(spl["label"])
                    for k in params.keys():
                        if k in data.keys():
                            data[k].append(spl[k])
            return data
        elif self.data_type == "video":
            # Sorting to obtain successive frames for videos, important for temporal modeling
            for vid_id in vid_dict.keys():
                vid_dict[vid_id] = natsorted(vid_dict[vid_id], key=lambda x: x["image"])
            
            # if self.train:
            """
            Generate new video data for training
            """
            assert "NUM_FRAMES" in self._cfg.DATA.SAMPLES_PER_VIDEO
            new_vid_dict = {}
            n_fs = self._cfg.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES

            for vid_id in vid_dict.keys():
                vid_len = len(vid_dict[vid_id])
                start_idx = 0 if epoch == 0 else np.random.randint(0, n_fs-1)

                for k in range(start_idx, vid_len, n_fs):
                    try:
                        if (k + n_fs) <= vid_len:
                            new_vid_id = '+++'.join([vid_id, str(k)]) # Adding index segment to original video to create sub videos
                            new_vid_dict[new_vid_id] = vid_dict[vid_id][k:(k+n_fs)]
                    except:
                        break

            return new_vid_dict
        else:
            raise ValueError(f'{self.data_type} has not been supported! Only "image" or "video" data can be extracted!')

    def select_encode_method(self, version=0, dimension='spatial'):
        if dimension == 'spatial':
            if version==2:
                return self._encode_target_v2
            elif version==1:
                return self._encode_target_v1
            else:
                return self._encode_target
        elif dimension == 'temporal':
            return self._encode_temporal_target
        else:
            raise ValueError(f'The input {dimension} has not been supported yet!')
    
    @abstractmethod
    def __len__(self):
        return NotImplemented

    @abstractmethod
    def __getitem__(self, idx):
        return NotImplemented

    @property
    def __repr__(self):
        return self.__class__.__name__
