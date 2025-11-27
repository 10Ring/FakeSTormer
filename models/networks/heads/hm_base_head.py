# Copyright (c) OpenMMLab. All rights reserved.
#-*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn


class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`loss`, supporting to calculate loss.
    Methods:`accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`predict`, supporting to inference model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def predict(self, **kwargs):
        """Inference function."""

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
