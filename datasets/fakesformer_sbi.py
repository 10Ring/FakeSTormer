#-*- coding: utf-8 -*-
import os
import sys
import random

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from numpy.random import randint

from .builder import DATASETS, PIPELINES, build_pipeline
from .master import MasterDataset
from .sbi.utils import *
from .common import ParameterStore
from .pipelines.geo_transform import get_transforms
from package_utils.transform import get_affine_transform, get_center_scale
from package_utils.utils import (
    vis_heatmap, draw_landmarks, 
    draw_most_vul_points, vis_3d_heatmap
)
from package_utils.image_utils import load_image, crop_by_margin


@DATASETS.register_module()
class FakeSFormerSBI(MasterDataset):
    def __init__(self,
                 config,
                 split,
                 **kwargs):
        """
        @params:
        config: Dataset config
        split: train/val/test which directs to the split folders
        """
        self.split = split
        super(FakeSFormerSBI, self).__init__(config, **kwargs)
        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)
        self.rot = 0
        self.pixel_std = 200
        self.target_w = self._cfg.IMAGE_SIZE[1]
        self.target_h = self._cfg.IMAGE_SIZE[0]
        self.aspect_ratio = self.target_w * 1.0 / self.target_h
        self.sigma = self._cfg.SIGMA
        self.heatmap_type = self._cfg.HEATMAP_TYPE
        self.debug = self._cfg.DEBUG
        self.dynamic_blending_prob = self._cfg.DYNAMIC_BLENDING_PROB
        self.dynamic_fxray = self._cfg.DYNAMIC_FXRAY
        self.target_overlap = self._cfg.TARGET_OVERLAP
        if self.data_type == 'video':
            self.mask_prob = self._cfg.MASK_PROB
            self.temp_maskout = self._cfg.TEMP_MASKOUT

        #Load data
        self.data_sampler = self._load_data(split)

        #Parse data
        self._parsing_data()

        #Calling transform methods for inputs
        self.geo_transform = build_pipeline(config.TRANSFORM.geometry, PIPELINES, 
            default_args={"additional_targets": {"image_f": "image", "mask_f": "mask"}})
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        
        self.transforms = get_transforms(data_type=self.data_type)
    
    def __len__(self):
        if self.data_type == 'image':
            assert 'image_paths' in self.data_sampler.keys()
            return len(self.labels_r)
        elif self.data_type == 'video':
            return len(self.data_sampler.keys())
        else:
            raise ValueError(f'{self.data_type} has not been supported. Please use "image" or "video" instead!')
    
    def _load_img(self, img_path):
        return load_image(img_path)

    def _reload_data(self, epoch=0):
        self.data_sampler = self._load_data(self.split, epoch=epoch)

    def _load_data(self, split, anno_file=None, epoch=0):
        from_file = self._cfg.DATA[self.split.upper()].FROM_FILE
        
        if epoch == 0:
            if not from_file:
                self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_from_path(split)
            else:
                self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_from_file(split, anno_file=anno_file)
        
            assert len(self.image_paths) != 0, "Image paths have not been loaded! Please check image directory!"
            assert len(self.labels) != 0, "Labels have not been loaded! Please check annotation file!"
            if not self.dynamic_fxray:
                assert len(self.mask_paths) != 0, "Mask paths have not been loaded! Please check mask directory!"
            
        if self.sampler_active:
            print('Running sampler...')
            params = dict(mask_paths=self.mask_paths, ot_props=self.ot_props, epoch=epoch)
            data_sampler = self._sampler(self.image_paths, self.labels, **params)
        return data_sampler
    
    def _parsing_data(self):
        assert self.data_type in ['image', 'video']
        #Parsing data for training
        if self.data_type == 'video':
            return 

        self.image_paths_r, self.labels_r = self.data_sampler['image_paths'], self.data_sampler['labels']
        if 'mask_paths' in self.data_sampler.keys() and len(self.data_sampler['mask_paths']):
            self.mask_paths_r = self.data_sampler['mask_paths']
        if 'ot_props' in self.data_sampler.keys() and len(self.data_sampler['ot_props']):
            self.ot_props_r = self.data_sampler['ot_props']

    def __getitem_path__(self, idx):
        param_store_ins = ParameterStore.get_instance()
        #Use to store func parameters that can be reused to generate multiple blending, e.g. SBI synthesis frames
        param_store_ins.add_parameters('data_type', self.data_type)
        flag = True
        
        while flag:
            try:
                #Selecting data from data list
                img_path = self.image_paths_r[idx]
                label = self.labels_r[idx]
                vid_id = img_path.split('/')[-2]
                img = self._load_img(img_path)
                if self.split == 'test':
                    # Best is 9,9 and 0.0 and 11,11
                    img = crop_by_margin(img, margin=[9, 9])
                
                img_f = None
                mask = None
                mask_f = None
                
                # if not self.dynamic_fxray or self.split == 'val':
                if not self.dynamic_fxray:
                    if bool(self.mask_paths_r):
                        mask_path = self.mask_paths_r[idx]
                        mask = self._load_img(mask_path)
                    else:
                        mask = np.zeros((img.shape[0], img.shape[1], 3))
                else:
                    if self.train:
                        if len(self.ot_props_r[idx]['aligned_lms']):
                            f_lms = self.ot_props_r[idx]['aligned_lms']
                        elif len(self.ot_props_r[idx]['orig_lms']):
                            f_lms = self.ot_props_r[idx]['orig_lms']
                        else:
                            f_lms = []
                        f_lms = np.array(f_lms)
                        if not f_lms.any():
                            raise ValueError('Can not find fake copy image of empty landmarks!')

                        if len(f_lms) > 68:
                            f_lms = reorder_landmark(f_lms)
                            
                        # if self.debug:
                        #     img_lms_draw = draw_landmarks(img, f_lms)
                        #     Image.fromarray(img_lms_draw).save(f'samples/debugs/orig_{idx}_lms.jpg')
                        
                        if self.split == 'train':
                            if np.random.rand() < 0.5:
                                img, ___, f_lms, __ = sbi_hflip(img, None, f_lms, None)
                    
                        margin = np.random.randint(5, 25)
                        img_f, mask_f, img, mask, fake_intensity = gen_target(img, f_lms, margin=[margin, margin], index=idx, debug=False, \
                                                                              dynamic_blending_prob=self.dynamic_blending_prob)
                target = None
                target_f = None
                
                if mask is not None:
                    assert (mask.shape[:2] == img.shape[:2]), "Color Image and Mask must have the same shape!"
                
                # Applying affine transform
                c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                
                # Applying geo transform to images and masks
                if self.split == 'train':
                    geo_transfomed = self.geo_transform(img, mask=mask, image_f=img_f, mask_f=mask_f)
                    img = geo_transfomed['image']
                    mask = geo_transfomed['mask']
                    img_f = geo_transfomed['image_f']
                    mask_f = geo_transfomed['mask_f']
                
                trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE, pixel_std=self.pixel_std)
                trans_heatmap = get_affine_transform(c, s, self.rot, self._cfg.HEATMAP_SIZE, pixel_std=self.pixel_std)
                
                input = cv2.warpAffine(img,
                                       trans,
                                       (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                       flags=cv2.INTER_LINEAR)
                
                if img_f is not None:
                    input_f = cv2.warpAffine(img_f,
                                             trans,
                                             (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                             flags=cv2.INTER_LINEAR)    
                
                if mask is not None:
                    if self.target_overlap:
                        target = cv2.warpAffine(mask,
                                                trans_heatmap,
                                                (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                                flags=cv2.INTER_LINEAR)
                    else:
                        mask = cv2.warpAffine(mask,
                                              trans,
                                              (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                              flags=cv2.INTER_LINEAR)
                        target = self._gen_vul_parts(blending_mask=mask)
                    
                if mask_f is not None:
                    if self.target_overlap:
                        target_f = cv2.warpAffine(mask_f,
                                                  trans_heatmap,
                                                  (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                                  flags=cv2.INTER_LINEAR)
                    else:
                        mask_f = cv2.warpAffine(mask_f,
                                                trans,
                                                (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                                flags=cv2.INTER_LINEAR)
                        target_f = self._gen_vul_parts(blending_mask=mask_f)
                    
                # Drawing the most vulnerable parts (MVPs)
                if self.debug:
                    mvp_f_drawed = draw_most_vul_points(target_f)
                    mvp_drawed = draw_most_vul_points(target)
                    Image.fromarray(mvp_f_drawed).save(f'samples/fakeformer_debugs/mvp_f_{idx}.jpg')
                    Image.fromarray(mvp_drawed).save(f'samples/fakeformer_debugs/mvp_{idx}.jpg')

                #Target encoding
                #0 for original, 1 for FXRay, 2 for NoFXRay. If 2, comment: mask = (1 - mask) * mask * 4
                heatmap_f, cstency_hm_f = self.select_encode_method(version=0)(target_f, fake_intensity=1.0) if \
                    (target_f is not None and self.heatmap_type=='gaussian' and self.train) else (None, None)
                heatmap, cstency_hm = self.select_encode_method(version=0)(target, fake_intensity=1.0) if \
                    (target is not None and self.heatmap_type=='gaussian' and self.train) else (None, None)
                
                # Applying transform for blending images
                if self.train:
                    if target_f is None:
                        transformed = self.transforms(image=input.astype('uint8'))
                        input=transformed['image']
                    else:
                        transformed = self.transforms(image=input.astype('uint8'),
                                                      image_f=input_f.astype('uint8'))
                        input=transformed['image']
                        input_f=transformed['image_f']
                
                if self.debug:
                    Image.fromarray(input).save(f'samples/fakeformer_debugs/affine_{idx}.jpg')
                    Image.fromarray(input_f).save(f'samples/fakeformer_debugs/affine_f_{idx}.jpg')
                    Image.fromarray(np.tile(target, 3)).save(f'samples/fakeformer_debugs/mask_affine_{idx}.jpg')
                    Image.fromarray(np.tile(target_f, 3)).save(f'samples/fakeformer_debugs/mask_affine_f_{idx}.jpg')
                    Image.fromarray(mask).save(f'samples/fakeformer_debugs/mask_{idx}.jpg')
                    Image.fromarray(mask_f).save(f'samples/fakeformer_debugs/mask_f_{idx}.jpg')
                    if cstency_hm is not None:
                        vis_heatmap(input, cstency_hm_f/255, f'samples/fakeformer_debugs/cstency_mask_f_{idx}.jpg')
                        vis_heatmap(input, cstency_hm/255, f'samples/fakeformer_debugs/cstency_mask_{idx}.jpg')
                    vis_heatmap(input, heatmap, f'samples/fakeformer_debugs/hm_{idx}.jpg')
                    vis_heatmap(input_f, heatmap_f, f'samples/fakeformer_debugs/hm_f_{idx}.jpg')
                
                if self.train:
                    patch_img_trans = self.final_transforms(input/255)
                    patch_img_trans_f = self.final_transforms(input_f/255)
                    patch_heatmap_f = heatmap_f
                    patch_heatmap_r = heatmap
                    patch_target_f = target_f/255
                    patch_target_r = target/255
                    patch_cstency_r = cstency_hm/255 if cstency_hm is not None else None
                    patch_cstency_f = cstency_hm_f/255 if cstency_hm_f is not None else None
                else:
                    #Normalise + Convert numpy array to tensor
                    img_trans = input/255
                    img_trans = self.final_transforms(img_trans)
                    
                label = np.expand_dims(label, axis=-1)
                flag = False
            except Exception as e:
                print(f'There is something wrong! Please check the DataLoader!, {e}')
                flag = True
                idx=torch.randint(low=0, high=self.__len__(), size=(1,)).item()
                
        if self.train:
            return patch_img_trans_f, patch_heatmap_f, patch_target_f, patch_cstency_f, \
                patch_img_trans, patch_heatmap_r, patch_target_r, patch_cstency_r
        else:
            meta = {'vid_id': vid_id, 'img_path': img_path}
            return img_trans, label, meta
        
    def __getitem_video__(self, idx):
        param_store_ins = ParameterStore.get_instance()
        #Use to store func parameters that can be reused to generate multiple blending, e.g. SBI synthesis frames
        param_store_ins.add_parameters('data_type', self.data_type)
        flag = True

        while flag:
            try:
                # Real data section
                inputs = []
                targets = []
                temp_loc = np.zeros(self.samples_per_video)
                masked_matrixes = []

                # Fake data section
                inputs_f = []
                targets_f = []
                labels = []
                temp_loc_f = np.ones(self.samples_per_video)
                masked_matrixes_f = []

                vid_id = [*self.data_sampler.keys()][idx]
                vid_data = self.data_sampler[vid_id]

                # f_idxes = randint(0, len(vid_data), self.samples_per_video) #randint might generate duplicate values, be careful!
                f_idxes = range(0, self.samples_per_video)
                pre_lms = None
                vid_path = None

                if self.train:
                    seq_det = self.distortion.to_deterministic()
                    margin_ = np.random.randint(5, 25)
                else:
                    # Optimal hyper-param for testing
                    margin_ = 15 # 0 for DFW, 5 for DFDCP, 13 for DFD, and 15 for the others

                for ix, f_idx in enumerate(f_idxes):
                    it = vid_data[f_idx]
                    img_path = it["image"]
                    if ix == 0:
                        vid_path = os.path.dirname(img_path)
                    label = it["label"]
                    img = self._load_img(img_path)
                    if self.split == 'test':
                        img = crop_by_margin(img, margin=[margin_, margin_]) # Best is 17,17 and 0.0 and 5,5
                    
                    img_f = None
                    mask = None
                    mask_f = None

                    # if not self.dynamic_fxray or self.split == 'val':
                    if self.train:
                        assert self.dynamic_fxray, "Online blending (dynamic_fxray) is always TRUE when working with SBI!"
                        if "ot_props" in it.keys():
                            ot_props = it["ot_props"]

                            if len(ot_props['aligned_lms']) and 'aligned' in img_path: #only take aligned lms when input already aligned
                                f_lms = ot_props['aligned_lms']
                            elif len(ot_props['orig_lms']):
                                f_lms = ot_props['orig_lms']
                            else:
                                f_lms = []
                            f_lms = np.array(f_lms)
                            if not f_lms.any():
                                raise ValueError('Can not find fake copy image of empty landmarks!')

                            if len(f_lms) > 68:
                                f_lms = reorder_landmark(f_lms)
                                # Compute the variation of lms between each frame
                                if f_idx != 0:
                                    l2_lms_dis = np.linalg.norm(f_lms - pre_lms)/len(f_lms)
                                    # print(f'Change of norm of landmark distance --- {l2_lms_dis}')
                                    if l2_lms_dis > 0.35:
                                        f_lms = pre_lms + (f_lms-pre_lms)/(round(l2_lms_dis/0.2))
                                pre_lms = f_lms
                                
                            if self.debug:
                                img_lms_draw = draw_landmarks(img, f_lms)
                                Image.fromarray(img_lms_draw).save(f'samples/debugs/orig_{idx}_{f_idx}_lms.jpg')
                            
                            if self.split == 'train':
                                rand_flip = param_store_ins.get_parameters('rand_flip')
                                if rand_flip is None:
                                    rand_flip = np.random.rand() < 0.5
                                    param_store_ins.add_parameters('rand_flip', rand_flip)

                                if rand_flip < 0.5:
                                    img, ___, f_lms, __ = sbi_hflip(img, None, f_lms, None)

                            img_f, mask_f, img, mask, fake_intensity = gen_target(img, f_lms, margin=[margin_, margin_], index=f_idx, debug=False, \
                                                                                  dynamic_blending_prob=self.dynamic_blending_prob, distortion=seq_det)
                            
                    target = None
                    target_f = None
                    
                    if mask is not None:
                        assert (mask.shape[:2] == img.shape[:2]), "Color Image and Mask must have the same shape!"
                    
                    # Applying affine transform
                    c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                    
                    # Applying geo transform to images and masks
                    # if self.split == 'train':
                    #     geo_transfomed = self.geo_transform(img, mask=mask, image_f=img_f, mask_f=mask_f)
                    #     img = geo_transfomed['image']
                    #     mask = geo_transfomed['mask']
                    #     img_f = geo_transfomed['image_f']
                    #     mask_f = geo_transfomed['mask_f']
                    
                    trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE, pixel_std=self.pixel_std)
                    trans_heatmap = get_affine_transform(c, s, self.rot, self._cfg.HEATMAP_SIZE, pixel_std=self.pixel_std)
                    
                    input = cv2.warpAffine(img,
                                           trans,
                                           (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                           flags=cv2.INTER_LINEAR)
                    
                    if img_f is not None:
                        input_f = cv2.warpAffine(img_f,
                                                 trans,
                                                 (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                                 flags=cv2.INTER_LINEAR)
                    
                    if mask is not None:
                        if self.target_overlap:
                            target = cv2.warpAffine(mask,
                                                    trans_heatmap,
                                                    (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                                    flags=cv2.INTER_LINEAR)
                        else:
                            mask = cv2.warpAffine(mask,
                                                  trans,
                                                  (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                                  flags=cv2.INTER_LINEAR)
                            target = self._gen_vul_parts(blending_mask=mask)
                        
                    if mask_f is not None:
                        if self.target_overlap:
                            target_f = cv2.warpAffine(mask_f,
                                                      trans_heatmap,
                                                      (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                                      flags=cv2.INTER_LINEAR)
                        else:
                            mask_f = cv2.warpAffine(mask_f,
                                                    trans,
                                                    (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                                    flags=cv2.INTER_LINEAR)
                            target_f = self._gen_vul_parts(blending_mask=mask_f)

                    # Drawing the most vulnerable parts (MVPs)
                    if self.debug:
                        mvp_drawed = draw_most_vul_points(target_f)
                        Image.fromarray(mvp_drawed).save(f'samples/debugs/mvp_{idx}_{f_idx}_{fake_intensity}.png')
                    
                    # Applying transform for blending images
                    if self.train:
                        if target_f is None:
                            if f_idx == 0:
                                transformed = self.transforms(image=input.astype(np.uint8))
                                input=transformed['image']
                                replay_params=transformed['replay']
                                param_store_ins.add_parameters('replay_transform', replay_params)
                            else:
                                replay_params=param_store_ins.get_parameters('replay_transform')
                                data = alb.ReplayCompose.replay(replay_params, image=input.astype(np.uint8))
                                input=data['image']
                        else:
                            if f_idx == 0:
                                transformed = self.transforms(image=input.astype(np.uint8),
                                                              image_f=input_f.astype(np.uint8))
                                input=transformed['image']
                                input_f=transformed['image_f']
                                replay_params=transformed['replay']
                                param_store_ins.add_parameters('replay_transform', replay_params)
                            else:
                                replay_params=param_store_ins.get_parameters('replay_transform')
                                data = alb.ReplayCompose.replay(replay_params, 
                                                                image=input.astype(np.uint8),
                                                                image_f=input_f.astype(np.uint8))
                                input=data['image']
                                input_f=data['image_f']
                        
                        mask_out_rand = param_store_ins.get_parameters('mask_out_rand') or np.random.rand()
                        label = 1
                        if f_idx == 0:
                            param_store_ins.add_parameters('mask_out_rand', mask_out_rand)
                        if mask_out_rand > 0.5 and self.mask_prob > 0:
                            # Mask out vulnerabilities
                            input_f, target_f, masked_matrix_f, upper_bound_value, fake_intensity = self._mask_out_vulnerability2(input_f, 
                                                                                        target_f, 
                                                                                        fake_intensity=fake_intensity, 
                                                                                        mask_prob=self.mask_prob)
                            if upper_bound_value == 1:
                                label = 0

                            temp_loc_f[f_idx] = fake_intensity # Updating temporal location value, default 0
                            input, target, masked_matrix, upper_bound_value, _ = self._mask_out_vulnerability2(input, 
                                                                                        target, 
                                                                                        fake_intensity=fake_intensity,
                                                                                        mask_prob=self.mask_prob)
                        else:
                            masked_matrix = np.ones_like(target[..., 0])
                            masked_matrix_f = np.ones_like(target_f[..., 0])
                        
                        targets.append(target)
                        targets_f.append(target_f)
                        masked_matrixes.append(masked_matrix)
                        masked_matrixes_f.append(masked_matrix_f)
                    
                    if self.debug:
                        # Image.fromarray(mask).save(f'samples/debugs/mask_{fake_intensity}.jpg')
                        # Image.fromarray(mask_f).save(f'samples/debugs/mask_f_{fake_intensity}.jpg')
                        Image.fromarray(input).save(f'samples/debugs/affine_{idx}_{f_idx}_{fake_intensity}.jpg')
                        Image.fromarray(input_f).save(f'samples/debugs/affine_f_{idx}_{f_idx}_{fake_intensity}.png')
                        # Image.fromarray(target).save(f'samples/debugs/mask_affine_{idx}_{f_idx}_{fake_intensity}.jpg')
                        Image.fromarray(target_f).save(f'samples/debugs/mask_affine_f_{idx}_{f_idx}_{fake_intensity}.png')
                        Image.fromarray(mask_f).save(f'samples/debugs/mask_f_{idx}_{f_idx}_{fake_intensity}.png')
                    
                    if self.train:
                        img_trans = self.final_transforms(input/255)
                        img_trans_f = self.final_transforms(input_f/255)
                        inputs_f.append(img_trans_f)
                    else:
                        #Normalise + Convert numpy array to tensor
                        img_trans = input/255
                        img_trans = self.final_transforms(img_trans)
                    
                    inputs.append(img_trans)
                    labels.append(label)
                
                if self.train and self.temp_maskout:
                    if np.random.rand() < 0.5:
                        start_idx = np.random.randint(0, len(inputs)-1)
                        end_idx = min(len(inputs), start_idx+np.random.randint(1, int(len(inputs)/2+1)))
                        targets_f = np.array(targets_f)
                        targets = np.array(targets)

                        if np.random.rand() > 0.5: # Repeat
                            # inputs_f[start_idx:end_idx] = inputs[start_idx:end_idx]
                            inputs_f[start_idx:end_idx] = inputs_f[start_idx].unsqueeze(0).repeat(end_idx-start_idx,1,1,1)
                            targets_f[start_idx:end_idx] = targets_f[np.newaxis, start_idx].repeat(end_idx-start_idx,0)
                            temp_loc_f[start_idx:end_idx] = temp_loc_f[np.newaxis, start_idx].repeat(end_idx-start_idx)
                        else: # Temporal cutout
                            zero_transform = self.final_transforms(np.zeros((int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1]), 3)))

                            # Assigning values for the range from start_idx to end_idx
                            inputs_f.extend(inputs_f[start_idx:end_idx])
                            targets_f = np.append(targets_f, targets_f[start_idx:end_idx], 0)
                            temp_loc_f = np.append(temp_loc_f, temp_loc_f[start_idx:end_idx], 0)

                            inputs_f[start_idx:end_idx] = zero_transform.unsqueeze(0).repeat(end_idx-start_idx,1,1,1)
                            targets_f[start_idx:end_idx] = targets[start_idx:end_idx]
                            temp_loc_f[start_idx:end_idx] = temp_loc[start_idx:end_idx]

                            inputs_f = inputs_f[:self.samples_per_video]
                            targets_f = targets_f[:self.samples_per_video]
                            temp_loc_f = temp_loc_f[:self.samples_per_video]

                        labels[start_idx:end_idx] = [0 for i in range(end_idx-start_idx)]

                #Target encoding
                #0 for original, 1 for FXRay, 2 for NoFXRay. If 2, find and comment the line: mask = (1 - mask) * mask * 4
                heatmap_f, cstency_hm_f, normalized_params = self.select_encode_method(version=0, dimension='temporal')(targets_f, fake_intensity=fake_intensity) if \
                    (len(targets_f) and self.train) else (None, None, None)
                heatmap, cstency_hm, _ = self.select_encode_method(version=0, dimension='temporal')(targets, fake_intensity=fake_intensity, **normalized_params) if \
                    (len(targets) and self.train) else (None, None, None)

                # Debugging 3D heatmap
                if self.debug:
                    vis_3d_heatmap(heatmap, f'samples/debugs/hm_{idx}_{f_idx}.png')
                    vis_3d_heatmap(heatmap_f, f'samples/debugs/hm_f_{idx}_{f_idx}.png')
                    vis_3d_heatmap(cstency_hm, f'samples/debugs/cstency_{idx}_{f_idx}.png')
                    vis_3d_heatmap(cstency_hm_f, f'samples/debugs/cstency_f_{idx}_{f_idx}.png')
                
                # End for loop
                if not self.train:
                    inputs = torch.tensor(np.array([[j.numpy() for j in i] for i in inputs])).transpose(0, 1)
                    labels = torch.tensor(np.array([np.array(it) for it in labels]))
                    label = torch.max(labels).unsqueeze(0)
                else:
                    label = np.max(labels)
                    if label == 0:
                        raise ValueError('There is at least one frame containing artifacts!')

                flag = False
                ParameterStore.reset()
            except Exception as e:
                print(f'There is something wrong! Please check the DataLoader!, {e}')
                flag = True
                idx=torch.randint(low=0, high=self.__len__(), size=(1,)).item()
        
        if self.train:
            return inputs, inputs_f, heatmap, heatmap_f, 0, 1, temp_loc, temp_loc_f, masked_matrixes, masked_matrixes_f, \
                    cstency_hm, cstency_hm_f
        else:
            meta = {'vid_id': vid_id.split('+++')[0], 'vid_path': vid_path}
            return inputs, label, meta
    
    def __getitem__(self, idx):
        if self.data_type == "image":
            return self.__getitem_path__(idx=idx)
        elif self.data_type == "video":
            return self.__getitem_video__(idx=idx)
        else:
            raise ValueError(f'{self.data_type} has not been supported. Only image or video are used for training!')
    
    def train_collate_fn(self, batch):
        batch_data = {}
        
        if self.data_type == 'image':
            img_f, hm_f, target_f, cst_f, img_r, hm_r, target_r, cst_r = zip(*batch)
            
            img = torch.cat([torch.tensor(np.array([it.numpy() for it in img_r])), torch.tensor(np.array([it.numpy() for it in img_f]))], 0)
            heatmap = torch.cat([torch.tensor(np.array(hm_r)).float(), torch.tensor(np.array(hm_f)).float()], 0)
            target = torch.cat([torch.tensor(np.array(target_r)).float(), torch.tensor(np.array(target_f)).float()], 0)
            label = torch.tensor([[0]] * len(img_r) + [[1]]*len(img_f))
            cst = torch.cat([torch.tensor(np.array(cst_r)).float(), torch.tensor(np.array(cst_f)).float()], 0) if \
                None not in cst_r else None

            b_size = label.size(0)
            
            # Permute idxes
            idxes = torch.randperm(b_size)
            img, label, target, heatmap = img[idxes], label[idxes], target[idxes], heatmap[idxes]
            if cst is not None:
                cst = cst[idxes]
            
            batch_data["img"] = img
            batch_data["label"] = label
            batch_data["target"] = target
            batch_data["heatmap"] = heatmap
            batch_data["cstency"] = cst
        else:
            img_r, img_f, hm_r, hm_f, label_r, label_f, temp_loc_r, temp_loc_f, mask_idx_r, mask_idx_f, cst_r, cst_f = zip(*batch)

            img = torch.cat([torch.tensor(np.array([[j.numpy() for j in i] for i in img_r])).transpose(1,2), \
                             torch.tensor(np.array([[j.numpy() for j in i] for i in img_f])).transpose(1,2)], 0)
            label = torch.cat([torch.tensor([i for i in label_r]), torch.tensor([i for i in label_f])], 0).unsqueeze(1)
            hm = torch.cat([torch.tensor(np.array(hm_r)).float(), torch.tensor(np.array(hm_f)).float()], 0).unsqueeze(1)
            temp_loc = torch.cat([torch.tensor(np.array([i for i in temp_loc_r])), torch.tensor(np.array([i for i in temp_loc_f]))], 0)
            mask_idx = torch.cat([torch.tensor(np.array(mask_idx_r)), torch.tensor(np.array(mask_idx_f))], 0).unsqueeze(1)
            cst = torch.cat([torch.tensor(np.array(cst_r)).float(), torch.tensor(np.array(cst_f)).float()], 0).unsqueeze(1)

            b_size = label.size(0)
            
            # Permute idxes
            idxes = torch.randperm(b_size)
            img, label, hm, temp_loc, mask_idx, cst = img[idxes], label[idxes], hm[idxes], temp_loc[idxes], mask_idx[idxes], cst[idxes]

            batch_data["img"] = img
            batch_data["label"] = label
            batch_data["heatmap"] = hm
            batch_data["temp_loc"] = temp_loc
            batch_data["mask_out_p"] = mask_idx
            batch_data["cstency"] = cst
        
        return batch_data
    
    def train_worker_init_fn(self, worker_id):
        # print('Current state {} --- worker id {}'.format(np.random.get_state()[1][0], worker_id))
        np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__=="__main__":
    from pipelines.geo_transform import GeometryTransform
    from torch.utils.data import DataLoader
    from configs.get_config import load_config

    PIPELINES.register_module(module=GeometryTransform)
    
    config = load_config("configs/temporal/ResNet3D_EFPN3D_hm3D_c23.yaml")

    #Seed
    seed = 529
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    hm_ff = DATASETS.build(cfg=config.DATASET, default_args=dict(split='val', config=config.DATASET))
    hm_ff_loader = DataLoader(hm_ff,
                              batch_size=8,
                              shuffle=True,
                              collate_fn=hm_ff.train_collate_fn,
                              worker_init_fn=hm_ff.train_worker_init_fn)

    for b, batch_data in enumerate(hm_ff_loader):
        inputs, labels, heatmaps = batch_data["img"], batch_data["label"], batch_data["heatmap"]
        print(f'X.shape - {inputs.shape}, y shape - {labels.shape}, heatmap shape - {heatmaps.shape}')
        break
