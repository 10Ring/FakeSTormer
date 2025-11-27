#-*- coding: utf-8 -*-
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn

from ..common import conv_block, conv3d_block


class ClassificationHead(nn.Module):
    def __init__(self, in_planes, 
                       out_planes, 
                       stages: Optional[List]=[], 
                       return_prob: bool=False, 
                       last_act: str='sigmoid',
                       drop: float=0.,
                       **kwargs):
        '''
        General classification head

        Args:
            stages: indicate a list of outputs of hidden layers between in_planes and out_planes
        '''
        super().__init__()
        self.return_prob = return_prob
        self.drop = drop
        self.in_planes = in_planes
        self.global_avg = kwargs.get('avg_pool') or False

        if self.global_avg:
            if kwargs.get('features') == '3D':
                self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
            else:
                self.avg_pool = nn.AdaptiveAvgPool3d((1,1))

        # Initialize layers
        layers = []
        for stage_plane in stages:
            layers.append(nn.Linear(in_planes, stage_plane))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.drop))
            in_planes = stage_plane

        self.fc1 = nn.Sequential(*layers) if len(layers) else nn.Identity()
        self.fc_out = nn.Linear(self.in_planes, out_planes)

        if last_act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softmax(dim=-1)

    def forward(self, x) -> torch.tensor:
        B = x.shape[0]

        if self.global_avg:
            num_channels = x.shape[1]
            x = self.avg_pool(x).view(B, -1, num_channels)

        assert self.in_planes == x.shape[-1]
        x = self.fc1(x)

        if x.ndim == 3:
            x = x.reshape((B, -1))
        
        x = self.fc_out(x)

        if self.return_prob:
            results = self.act(x)
        else:
            results = x

        return results


class RegressionHead(nn.Module):
    def __init__(self, 
                 input_shape,
                 out_planes,
                 kernel_size,
                 padding,
                 conv_2direction=False,
                 extra=None,
                 **kwargs):
        '''
        General regression head for dense predictions like keypoints regression, etc

        Args:
            input_shape: [C, H, W]
            conv_2direction: Specially design to compute derivative tx, ty from temporal tokens in video input. i.e. FakeSTormer
        '''
        super().__init__()
        self.conv_2direction = conv_2direction

        conv_channels = input_shape[0]

        layers = []
        if extra is not None:
            num_conv_layers = extra.get('num_conv_layers', 0)
            num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)

            for i in range(num_conv_layers):
                layers.append(conv_block(inplanes=conv_channels, 
                                         outplanes=conv_channels, 
                                         kernel_size=kernel_size,
                                         padding=padding))
        self.before_proj = nn.Sequential(*layers) if len(layers) else nn.Identity()

        if not conv_2direction:
            self.proj = nn.Conv2d(in_channels=input_shape[0],
                                  out_channels=out_planes,
                                  kernel_size=kernel_size,
                                  padding=padding)
        else:
            self.d_tx = conv_block(inplanes=input_shape[1],
                                   outplanes=input_shape[1],
                                   kernel_size=kernel_size,
                                   padding=padding)
            
            self.d_ty = conv_block(inplanes=input_shape[2],
                                   outplanes=input_shape[2],
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.proj = nn.Conv2d(in_channels=int(input_shape[0]*2),
                                  out_channels=out_planes,
                                  kernel_size=kernel_size,
                                  padding=padding)
    
    def forward(self, x):
        assert x.ndim == 4
        B, C, H, W = x.shape
        x = self.before_proj(x)

        if self.conv_2direction:
            x_tx = self.d_tx(x.transpose(1, 2)).transpose(2, 1)
            x_ty = self.d_ty(x.transpose(1, 3)).transpose(3, 1)
            x = torch.cat((x_tx, x_ty), 1)
        x = self.proj(x)
        
        return x


class TemporalRegressionHead(nn.Module):
    def __init__(self, 
                 inplanes:int, 
                 outplanes:int,
                 kernel_size:Union[int, Tuple]=(3,1,1),
                 stride:Union[int, Tuple]=(1,1,1),
                 padding:Union[int, Tuple]=(1,0,0),
                 extra:Optional[dict]=None,
                 act: nn.Module=nn.GELU,
                 **kwargs):
        super().__init__(**kwargs)
        '''
        General head design for 3D feature maps
        args:
            inplanes: number of in channels
            outplanes: number of out channels
            extra: doing extra operation before predicting final outputs
        '''
        layers = []
        if extra is not None:
            num_conv_layers = extra.get('num_conv_layers', 0)
            num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)

            for i in range(num_conv_layers):
                layers.append(conv3d_block(inplanes=inplanes, 
                                           outplanes=inplanes, 
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=True,
                                           inplace=True,
                                           act=act))
        self.before_proj = nn.Sequential(*layers) if len(layers) else nn.Identity()
        self.proj = nn.Sequential(conv3d_block(inplanes=inplanes,
                                               outplanes=inplanes//4,
                                               stride=stride,
                                               kernel_size=kernel_size,
                                               padding=padding,
                                               bias=True,
                                               inplace=True,
                                               act=act),
                                  nn.Conv3d(inplanes//4, 
                                            outplanes, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=padding, 
                                            bias=True))

    def forward(self, x):
        assert x.ndim == 5

        x = self.before_proj(x)
        out = self.proj(x)
        return out
