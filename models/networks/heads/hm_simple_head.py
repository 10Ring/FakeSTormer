# Copyright (c) OpenMMLab. All rights reserved.
#-*- coding: utf-8 -*-
import os
import sys
from typing import Union, Dict
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from mmcv.cnn import (
    build_conv_layer, build_norm_layer, build_upsample_layer,
    constant_init, normal_init
)
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from models.builder import HEADS
from .hm_base_head import TopdownHeatmapBaseHead
from ..common import resize, conv_block, BN_MOMENTUM
from losses import build_losses, LOSSES
from .head_design import ClassificationHead, RegressionHead, TemporalRegressionHead


@HEADS.register_module()
class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 extra=None,
                 hm_size=[14, 14],
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,
                 conv_2direction=False,
                 use_temp_token=False,
                 features='2D',
                 act='GELU',
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_losses(loss_keypoint, LOSSES)
        self.upsample = upsample
        self.use_temp_token = use_temp_token

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        assert isinstance(heads, dict) and heads is not None, "Head config can not be None!"
        self.heads = heads

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        final_layer = {}
        if not identity_final_layer:
            for head, out_channel in self.heads.items():
                if head == 'hm' or head == 'cstency':
                    if features == '2D':
                        layer = RegressionHead(input_shape=(self.in_channels, hm_size[0], hm_size[1]),
                                               out_planes=out_channel,
                                               kernel_size=kernel_size,
                                               padding=padding,
                                               extra=extra,
                                               conv_2direction=conv_2direction)
                    elif features == '3D':
                        act_func = nn.GELU if act == 'GELU' else nn.ReLU
                        layer = TemporalRegressionHead(inplanes=self.in_channels,
                                                       outplanes=out_channel,
                                                       extra=extra,
                                                       act=act_func)
                    else:
                        raise ValueError('Only support 2D or 3D features, please check your feature dimension!')
                    final_layer[head] = layer
                elif head == 'cls':
                    layer = ClassificationHead(in_planes=self.in_channels,
                                               out_planes=out_channel,
                                               return_prob=False,
                                               last_act='sigmoid',
                                               drop=0.2,
                                               features=features,
                                               avg_pool=kwargs.get('avg_pool'))
                    final_layer[head] = layer
                elif head == 'temp_loc':
                    layer = ClassificationHead(in_planes=self.in_channels,
                                               out_planes=out_channel,
                                               stages=[self.in_channels//out_channel],
                                               return_prob=False,
                                               last_act='sigmoid',
                                               drop=0.2)
                    final_layer[head] = layer
        else:
            final_layer['cls'] = nn.Identity()
        
        for k, l in final_layer.items():
            self.__setattr__(k, l)
    
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels
    
    def _transform_inputs(self, inputs: Union[torch.tensor, Dict[str, torch.tensor]]):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        additional_inputs = {}

        if isinstance(inputs, dict):
            # assert 'embed' in inputs.keys(), "Embed token must be present in the input dict key"
            for k in inputs.keys():
                if k != 'embed':
                    additional_inputs[k] = inputs[k]
            inputs = inputs['embed'] if 'embed' in inputs.keys() else None
        
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs, additional_inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs, additional_inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for head in self.heads.keys():
            for m in self.__getattr__(head).modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
    
    def forward(self, x, **kwargs):
        """ Forward function.
            The input is multiscale feature maps and the output is the heatmap without post processing
        """
        x_embed, additional_inputs = self._transform_inputs(x)
        if 'temp_loc' in self.heads.keys():
            if not self.use_temp_token:
                B, L, T, H, W = x_embed.shape
                x_temp_loc = x_embed.reshape(B, T, L, -1)
                x_temp_loc = torch.max(x_temp_loc, dim=3, keepdim=False)[0]
            else:
                x_temp_loc = additional_inputs['s_cls_token']

        x_outs = {}
        # for head in self.heads.keys():
        if 'cls' in self.heads.keys():
            assert hasattr(self, 'cls'), "There must be a Classification Head, please check the head design!"
            if bool(additional_inputs) and 'cls' in additional_inputs.keys():
                x_outs["cls"] = self.__getattr__('cls')(additional_inputs['cls'])
            else:
                x_outs["cls"] = self.__getattr__('cls')(x_embed)

        # for head in self.heads.keys():
        if 'temp_loc' in self.heads.keys():
            assert hasattr(self, 'temp_loc'), "There must be a head for temporal localization, please check the head design!"
            x_outs["temp_loc"] = self.__getattr__('temp_loc')(x_temp_loc)
            
        if 'hm' in self.heads.keys():
            assert hasattr(self, 'hm'), "There must always be a Heatmap Head in the head!"
            x_outs["hm"] = self.__getattr__('hm')(x_embed)

        if 'cstency' in self.heads.keys():
            assert hasattr(self, 'cstency'), "There must always be a Consistency Head in the head!"
            x_outs["cstency"] = self.__getattr__('cstency')(x_embed)

        return [x_outs]

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)

        return losses


if __name__=='__main__':
    cfg = {}
