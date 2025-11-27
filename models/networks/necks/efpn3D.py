#-*- coding: utf-8 -*-
from typing import Dict, List, Sequence
import torch
import torch.nn as nn
from mmengine.model import xavier_init

from .base import BaseNeck
from ..common import deconv3d_block, conv3d_block, InceptionBlock3D
from ...builder import NECKS


@NECKS.register_module()
class EFPN3D(BaseNeck):
    def __init__(self, 
                 in_channels: int,
                 num_deconv_layers: int,
                 num_deconv_filters: Sequence[int],
                 num_deconv_kernels: Sequence[int],
                 num_deconv_strides: Sequence[int],
                 efpn: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.efpn = efpn
        self.num_deconv_layers = num_deconv_layers
        
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_3ddeconv_layer(
                num_deconv_filters[0],
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
                num_deconv_strides,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        
        if self.efpn:
            in_filter = self.in_channels
            for idx, out_filter in enumerate(num_deconv_filters):
                deconv_block = nn.Sequential(
                    conv3d_block(inplanes=in_filter,
                                 outplanes=out_filter,
                                 kernel_size=(1,1,1),
                                 stride=(1,1,1),
                                 bias=False),
                    self.deconv_layers[idx],
                )
                in_filter = out_filter*2
                self.__setattr__(f'deconv_block_{idx}', deconv_block)
            self.inception_block3d = InceptionBlock3D(inplanes=num_deconv_filters[-1],
                                                      outplanes=num_deconv_filters[-1],
                                                      stride=1,
                                                      pool_size=3)
            
    def preprocess_inputs(self, 
                          x: Dict[str, torch.tensor], 
                          **kwargs):
        assert 'embed' in x.keys(), 'The input dict must contain embedding features!'
        inputs = x['embed']
        additional_inputs = {}

        for k,v in x.items():
            if k != 'embed':
                additional_inputs[k] = v
        
        return inputs, additional_inputs

    def _make_3ddeconv_layer(self, 
                             in_channels,
                             num_layers: int, 
                             num_filters: List[int], 
                             num_kernels: Sequence[int],
                             num_strides: Sequence[int]) -> list:
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
        in_planes = in_channels
        for i in range(num_layers):
            kernels = []
            paddings = []
            output_paddings = []

            for j in range(len(num_kernels[i])):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i][j])
                kernels.append(kernel)
                paddings.append(padding)
                output_paddings.append(output_padding)

            outplanes = num_filters[i]
            layers.append(deconv3d_block(inplanes=in_planes,
                                         outplanes=outplanes,
                                         kernel_size=kernels,
                                         stride=num_strides[i],
                                         padding=paddings,
                                         bias=False,
                                         out_padding=output_paddings))
            
            # This condition to match n_filters after convolution and optimize number of EFPN parameters
            if self.efpn:
                in_planes = num_filters[i+1] if (i+1 < num_layers) else num_filters[num_layers-1]
            else:
                in_planes = outplanes

        return layers
    
    def forward(self, 
                x: Dict[str, torch.tensor], 
                **kwargs) -> dict:
        inputs, additional_inputs = self.preprocess_inputs(x)
        trapezoids = None
        if 'outputs' in additional_inputs.keys():
            trapezoids = additional_inputs['outputs']
            # x2, x3, x4, x5 = trapezoids[0], trapezoids[1], trapezoids[2], trapezoids[3]

        if not self.efpn:
            x_embed = self.deconv_layers(inputs)
        else:
            assert 'outputs' in additional_inputs.keys(), 'EFPN requires multiscale feature outputs!'
            x_embed = inputs
            for i in range(self.num_deconv_layers):
                x_embed = self.__getattr__(f'deconv_block_{i}')(x_embed)
                if i < self.num_deconv_layers - 1:
                    x_weights = x_embed.sigmoid_()
                    x_inv = torch.sub(1, x_weights, alpha=1)
                    x_ = torch.multiply(x_inv, trapezoids[self.num_deconv_layers-(i+2)])
                    x_embed = torch.cat([x_embed, x_], dim=1)
            
            # Last layer to capture multi-scale artifacts
            x_embed = self.inception_block3d(x_embed)

        res = {}
        res['embed'] = x_embed

        return res

    def init_weights(self) -> None:
        for layer in self.deconv_layers:
            xavier_init(layer, distribution='uniform')
