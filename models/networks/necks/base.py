#-*- coding: utf-8 -*-
from typing import Dict
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BaseNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        return super().__init__()

    @abstractmethod
    def forward(self, x: Dict[str, torch.tensor], **kwargs):
        return NotImplemented
    
    @abstractmethod
    def preprocess_inputs(self, x: Dict[str, torch.tensor], **kwargs):
        return NotImplemented
    
    @abstractmethod
    def init_weights(self, pretrained=None):
        return NotImplemented

    @staticmethod
    def _get_deconv_cfg(deconv_kernel: int):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel in [1, 2]:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
