#-*- coding: utf-8 -*-
from abc import abstractmethod
import os
from random import shuffle
import sys

import torch
from PIL import Image
import numpy as np
from torch.utils.data import default_collate

from .builder import DATASETS, PIPELINES, build_pipeline
from .master import MasterDataset
from package_utils.image_utils import load_image


@DATASETS.register_module()
class BinaryFaceForensic(MasterDataset):
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
        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)
        super().__init__(config, **kwargs)

        #Load data
        self.data_sampler = self._load_data(split)

        #Parse data
        self._parsing_data()
        
        #Calling transform methods for inputs
        self.geo_transform = build_pipeline(config.TRANSFORM.geometry, PIPELINES)
        self.colorjitter_transform = build_pipeline(config.TRANSFORM.color, PIPELINES)

    def _load_data(self, split, anno_file=None, epoch=0):
        from_file = self._cfg.DATA[self.split.upper()].FROM_FILE
        
        if epoch == 0:
            if not from_file:
                self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_from_path(split)
            else:
                self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_from_file(split, anno_file=anno_file)
        
            assert len(self.image_paths) != 0, "Image paths have not been loaded! Please check image directory!"
            assert len(self.labels) != 0, "Labels have not been loaded! Please check annotation file!"
            
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
        self.image_paths, self.labels = self.data_sampler['image_paths'], self.data_sampler['labels']

    def _load_img(self, img_path):
        return load_image(img_path)

    def __len__(self):
        if self.data_type == 'image':
            assert 'image_paths' in self.data_sampler.keys()
            return len(self.labels)
        elif self.data_type == 'video':
            return len(self.data_sampler.keys())
        else:
            raise ValueError(f'{self.data_type} has not been supported. Please use "image" or "video" instead!')
    
    def __getitem__(self, idx):
        if self.data_type == "image":
            return self.__getitem_path__(idx=idx)
        elif self.data_type == "video":
            return self.__getitem_video__(idx=idx)
        else:
            raise ValueError(f'{self.data_type} has not been supported. Only image or video are used for training!')

    def __getitem_path__(self, idx):
        img_path = self.image_paths[idx]
        label = np.expand_dims(self.labels[idx], axis=-1)
        img = self._load_img(img_path)

        #Applying geo transform to inputs
        geo_transfomed = self.geo_transform(img)
        img_trans = geo_transfomed['image']
        
        #Applying color transform to inputs
        color_transfomed = self.colorjitter_transform(img_trans)
        img_trans = color_transfomed['image']

        #Normalise + Convert numpy array to tensor
        img_trans = img_trans/255
        img_trans = self.final_transforms(img_trans)
        return img_trans, label

    def __getitem_video__(self, idx):
        inputs = []
        vid_id = [*self.data_sampler.keys()][idx]
        vid_data = self.data_sampler[vid_id]

        label = np.expand_dims(vid_data[0]["label"], axis=-1)

        f_idxes = range(0, self.samples_per_video)
        for ix, f_idx in enumerate(f_idxes):
            it = vid_data[f_idx]
            img_path = it["image"]
            img = self._load_img(img_path)

            if self.train:
                #Applying geo transform to inputs
                geo_transfomed = self.geo_transform(img)
                img_trans = geo_transfomed['image']
                
                #Applying color transform to inputs
                color_transfomed = self.colorjitter_transform(img_trans)
                img_trans = color_transfomed['image']

                #Normalise + Convert numpy array to tensor
                img_trans = img_trans/255
            else:
                img_trans = img/255
            img_trans = self.final_transforms(img_trans)
            inputs.append(img_trans)
        inputs = torch.tensor(np.array([ip.numpy() for ip in inputs]))
        inputs = inputs.transpose(0,1)

        if self.train:
            return inputs, label
        else:
            return inputs, label, vid_id.split('-')[0]

    def train_collate_fn(self, batch):
        return default_collate(batch)


if __name__=="__main__":
    from datasets import *
    from pipelines.geo_transform import GeometryTransform
    from pipelines.color_transform import ColorJitterTransform
    from torch.utils.data import DataLoader
    from configs.get_config import load_config
    
    PIPELINES.register_module(module=GeometryTransform)
    PIPELINES.register_module(module=ColorJitterTransform)

    config = load_config("configs/temporal/bin_cls/TimeSFormer_base_c23.yaml")
    bin_ff = DATASETS.build(cfg=config.DATASET, default_args=dict(split='val', config=config.DATASET))
    bin_ff_loader = DataLoader(bin_ff,
                               batch_size=10,
                               shuffle=True)
    for b, (X, y) in enumerate(bin_ff_loader):
        print(f'X.shape - {X.shape}, y shape - {y.shape}')
        break
