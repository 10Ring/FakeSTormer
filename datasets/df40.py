#-*- coding: utf-8 -*-
import os
import numpy as np
from glob import glob

from .common import CommonDataset
from .builder import DATASETS


@DATASETS.register_module()
class DF40(CommonDataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
    def _load_from_path(self, split):
        # Check if the root directory exists.
        root_dir = self._cfg.DATA[self.split.upper()].ROOT
        assert os.path.exists(root_dir), "Root path to dataset cannot be None!"
        
        data = self._cfg["DATA"]
        data_type = data.TYPE
        fake_types = self._cfg.DATA[split.upper()]["FAKETYPE"]
        img_paths, labels, mask_paths, ot_props = [], [], [], []

        # Load image data for each type of fake technique.
        for ft in fake_types:
            # Construct path: ROOT/split/fake_type (skipping data_type)
            data_dir = os.path.join(root_dir, self.split, data_type, ft)
            if not os.path.exists(data_dir):
                raise ValueError("Data Directory is invalid!")
            
            # Define common image extensions.
            extensions = ['jpg', 'jpeg', 'png', 'tif', 'webp']
            img_paths_ = []
            # Recursively search for images in the fake type directory.
            for ext in extensions:
                pattern = os.path.join(data_dir, '**', f'*.{ext}')
                img_paths_.extend(glob(pattern, recursive=True))
            
            # Extend the main lists with the images and corresponding labels.
            img_paths.extend(img_paths_)
            labels.extend(np.full(len(img_paths_), int(ft != 'real_videos')))
                
        print('{} image paths have been loaded from DF40!'.format(len(img_paths)))
        return img_paths, labels, mask_paths, ot_props
