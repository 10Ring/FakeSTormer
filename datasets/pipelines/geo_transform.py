#-*- coding: utf-8 -*-
import os
import sys
from typing import Dict
import random
import math
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import albumentations as A
import numpy as np
from albumentations.augmentations.transforms import DualTransform
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

from datasets.builder import PIPELINES
from .functional import _get_pixels


@PIPELINES.register_module()
class GeometryTransform(object):
    def __init__(self,
                 resize: list,
                 normalize: float,
                 horizontal_flip: float,
                 scale: list,
                 cropping: list,
                 rand_erasing: list,
                 *args,
                 **kwargs):
        super().__init__()
        self.resize = resize #[H, W, p]
        self.normalize = normalize #p
        self.horizontal_flip = horizontal_flip #p
        self.cropping = cropping #[crop_limit, p]
        self.scale = scale #[scale_limit, p]
        self.rand_erasing = rand_erasing #p

        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)

    def _resize(self):
        hr, wr, p = self.resize
        return A.Resize(hr, wr, interpolation=2, p=p)

    # We offen use normalize transform from torch, so set p=0.0
    def _normalize(self, p=0.0):
        return A.Normalize(p=p)

    def _horizontal_flip(self, 
                         p=0.5,
                         always_apply=False):
        return A.HorizontalFlip(always_apply=always_apply,
                                p=p)

    def _random_scale(self,
                      p=0.5,
                      always_apply=False,
                      scale_limit=0.1,
                      interpolation=1):
        return A.RandomScale(scale_limit=scale_limit,
                             interpolation=interpolation,
                             always_apply=always_apply,
                             p=p)
        
    def _random_crop(self,
                     p=0.5,
                     always_apply=False,
                     crop_limit=0.1, 
                     img_h=256,
                     img_w=256):
        crop_h = int((1 - np.random.choice(np.arange(0.0, crop_limit, 0.01))) * img_h)
        crop_w = int((1 - np.random.choice(np.arange(0.0, crop_limit, 0.01))) * img_w)
        return A.RandomCrop(height=crop_h, 
                            width=crop_w,
                            always_apply=always_apply,
                            p=p)

    def _random_erasing(self, 
                        p=0.5,
                        always_apply=False,
                        max_count=3,
                        mode='const'):
        return RandomErasing(p=p,
                             always_apply=always_apply,
                             mode=mode,
                             max_count=max_count)

    def __call__(self, x, mask=None, image_f=None, mask_f=None):
        x_h, x_w = x.shape[:2]

        if hasattr(self, "additional_targets"):
            additional_targets = self.__getattribute__("additional_targets")
        else:
            additional_targets = {}
        
        transform = A.Compose([
            A.OneOf([
                self._random_crop(p=self.cropping[1],
                                  crop_limit=self.cropping[0],
                                  img_h=x_h,
                                  img_w=x_w),
                self._random_scale(p=self.scale[1],
                                   scale_limit=self.scale[0]),
                self._random_erasing(p=self.rand_erasing[0], 
                                     mode='const',
                                     max_count=self.rand_erasing[1]),
            ]),
            A.Compose([
                self._resize(),
                self._normalize(p=self.normalize),
                self._horizontal_flip(p=self.horizontal_flip),
            ]),
        ], additional_targets=additional_targets)

        if mask is not None:
            if mask_f is not None:
                assert image_f is not None, "Image Fake sample can not be None in case of Mask sample!"
                assert len(additional_targets.keys()), "Additional targets for Albumentations can not be None!"
                return transform(image=x, mask=mask, image_f=image_f, mask_f=mask_f)
            else:
                return transform(image=x, mask=mask)
        else:
            return transform(image=x)


class RandomErasing(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, 
                 min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
                 mode='const', min_count=1, max_count=None, num_splits=0,
                 img_h=257, img_w=257, img_chan=3):
        super(RandomErasing, self).__init__(always_apply, p)
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.img_h = img_h
        self.img_w = img_w
        self.img_chan = img_chan
        
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def apply(self, img: np.array, **params):
        return self._erase(img, **params)
    
    def get_params(self) -> Dict:
        area = self.img_h * self.img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        
        tops,lefts,ws,hs = [],[],[],[]
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if w < self.img_w and h < self.img_h:
                    top = random.randint(0, self.img_h - h)
                    left = random.randint(0, self.img_w - w)
                    
                    tops.append(top)
                    lefts.append(left)
                    ws.append(w)
                    hs.append(h)
                    break
        return {"tops": tops, "lefts": lefts, "ws":ws, "hs":hs, "img_chan": self.img_chan}

    def _erase(self, img: np.array, tops: list, lefts: list, hs: list, ws: list, img_chan: int, **params):
        for i in range(len(tops)):
            top = tops[i]
            left = lefts[i]
            w = ws[i]
            h = hs[i]
            
            img[top:top + h, left:left + w, :] = _get_pixels(
                self.per_pixel, self.rand_color, (h, w, img_chan),
                dtype=img.dtype)
        return img


class RandomDownScale(ImageOnlyTransform):
    def __init__(self, 
                 always_apply: bool = False, 
                 p: float = 0.5,
                 ratio_list: list = [2,4]):
        self.ratio_list = ratio_list
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img: np.ndarray, ratio: int, **params):
        return self.randomdownscale(img, ratio, **params)
    
    def get_params(self):
        ratio = self.ratio_list[np.random.randint(len(self.ratio_list))]
        return {"ratio": ratio}
    
    def randomdownscale(self, img, ratio, **kwargs):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        
        # r = np.random.uniform(2, 4)
        img_ds = cv2.resize(img, (int(W/ratio), int(H/ratio)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W,H), interpolation=cv2.INTER_LINEAR)
            
        return img_ds

    def get_transform_init_args_names(self):
        return ("ratio_list",)


def get_source_transforms(data_type='image'):
    '''
    Transforms specially design for SBI synthesis
    '''
    assert data_type in ['image', 'video']
    if data_type == 'image':
        return A.Compose([
            A.Compose([
                A.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
            ],p=1),

            A.OneOf([
                RandomDownScale(p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],p=1),
        ], p=1.)
    else:
        return A.ReplayCompose([
            A.Compose([
                A.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
            ],p=1),

            A.OneOf([
                RandomDownScale(p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],p=1),
        ], p=1.)


def get_transforms(data_type='image'):
    '''
    Transforms specially design for SBI synthesis
    '''
    assert data_type in ['image', 'video']

    if data_type == 'image':
        return A.Compose([
                A.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
                A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ], 
            additional_targets={'image_f': 'image'},
            p=1.)
    else:
        return A.ReplayCompose([
                A.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
                A.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
                A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ], 
            additional_targets={'image_f': 'image'},
            p=1.)
