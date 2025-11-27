#-*- coding: utf-8 -*-
from .builder import DATASETS
from .celebDF_v1 import CDFV1
from .celebDF_v2 import CDFV2
from .ff import FF
from .dfdcp import DFDCP
from .dfdc import DFDC
from .dfd import DFD
from .dfw import DFW
from .dfo import DFo
from .diffswap import DiffSwap
from .df40 import DF40
from .combine import Combine


@DATASETS.register_module()
class MasterDataset(CDFV1, FF, DFDCP, CDFV2, DFDC, DFD, DFW, DFo, DiffSwap, DF40, Combine):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
    
    def _load_from_path(self, split):
        # Explicitly overide some main methods from the dataset config
        if self.dataset == 'FF++':
            return MasterDataset.__mro__[2]._load_from_path(self, split=split)
        elif self.dataset == 'Celeb-DFv1':
            return MasterDataset.__mro__[1]._load_from_path(self, split=split)
        elif self.dataset == 'DFDCP':
            return MasterDataset.__mro__[3]._load_from_path(self, split=split)
        elif self.dataset == 'Celeb-DFv2':
            return MasterDataset.__mro__[4]._load_from_path(self, split=split)
        elif self.dataset == 'DFDC':
            return MasterDataset.__mro__[5]._load_from_path(self, split=split)
        elif self.dataset == 'DFD':
            return MasterDataset.__mro__[6]._load_from_path(self, split=split)
        elif self.dataset == 'DFW':
            return MasterDataset.__mro__[7]._load_from_path(self, split=split)
        elif self.dataset == 'DFo':
            return MasterDataset.__mro__[8]._load_from_path(self, split=split)
        elif self.dataset == 'DiffSwap':
            return MasterDataset.__mro__[9]._load_from_path(self, split=split)
        elif self.dataset == 'DF40':
            return MasterDataset.__mro__[10]._load_from_path(self, split=split)
        elif self.dataset == 'Combine':
            return MasterDataset.__mro__[11]._load_from_path(self, split=split)
        else:
            return NotImplementedError(f'{self.dataset} has not been supported yet!')
