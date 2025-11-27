#-*- coding: utf-8 -*-
import os
import numpy as np
from glob import glob

from .common import CommonDataset
from .builder import DATASETS


@DATASETS.register_module()
class DiffSwap(CommonDataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
    def _load_from_path(self, split):
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be None!"
        data = self._cfg["DATA"]
        data_type = data.TYPE
        fake_types = self._cfg.DATA[split.upper()]["FAKETYPE"]
        img_paths, labels, mask_paths, ot_props = [], [], [], []

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            data_dir = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, self.split, data_type, ft)
            if not os.path.exists(data_dir):
                raise ValueError("Data Directory can not be invalid!")
            
            if not os.path.isdir(data_dir): continue

            for sub_dir in os.listdir(data_dir):
                sub_dir_path = os.path.join(data_dir, sub_dir)
                # sub_dir_path = data_dir
                img_paths_ = glob(f'{sub_dir_path}/*.{self._cfg.IMAGE_SUFFIX}')

                img_paths.extend(img_paths_)
                labels.extend(np.full(len(img_paths_), int('Real' not in ft)))
                
        print('{} image paths have been loaded from DiffSwap!'.format(len(img_paths)))          
        return img_paths, labels, mask_paths, ot_props
