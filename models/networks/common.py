#-*- coding: utf-8 -*-
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_upsample_layer


BN_MOMENTUM = 0.1
BN3D_MOMENTUM = 0.05 # small for small batch size


def point_wise_block(inplanes, 
                     outplanes):
    return nn.Sequential(
        nn.Conv2d(in_channels=inplanes, 
                  out_channels=outplanes, 
                  kernel_size=1, 
                  padding=0, 
                  stride=1, 
                  bias=False),
        nn.BatchNorm2d(outplanes, 
                       momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )


def conv_block(inplanes, 
               outplanes, 
               kernel_size, 
               stride=1, 
               padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=inplanes, 
                  out_channels=outplanes, 
                  kernel_size=kernel_size, 
                  padding=padding, 
                  stride=stride, 
                  bias=False),
        nn.BatchNorm2d(outplanes, 
                       momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True)
    )


def conv3x3(in_planes, 
            out_planes, 
            stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, 
                     out_planes, 
                     kernel_size=3, 
                     stride=stride,
                     padding=1, 
                     bias=False)


def conv3d_block(inplanes, 
                 outplanes, 
                 kernel_size=(3,1,1), 
                 stride=(1,1,1), 
                 padding=0, 
                 bias=False, 
                 inplace=False, 
                 act=nn.GELU):
    '''
    General conv3d block for handling 3d feature maps
    '''
    return nn.Sequential(
        nn.Conv3d(inplanes, 
                  outplanes, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding, 
                  bias=bias),
        nn.BatchNorm3d(outplanes, momentum=BN3D_MOMENTUM),
        act()
    )


def deconv3d_block(inplanes, 
                   outplanes, 
                   kernel_size=(2,4,4), 
                   stride=(2,2,2), 
                   padding=(0,1,1), 
                   bias=False, 
                   inplace=False, 
                   out_padding=(0,1,1),
                   act=None):
    '''
    General Transpose 3D Convolution for handling 3D feature maps
    '''
    layers = []
    layers.append(
        build_upsample_layer(
            dict(type='deconv3d'),
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            bias=bias))
    layers.append(nn.BatchNorm3d(outplanes))

    if act is not None:
        layers.append(act())

    return nn.Sequential(*layers)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class InceptionBlock(nn.Module):
    def __init__(self, 
                 inplanes, 
                 outplanes, 
                 stride=1, 
                 pool_size=3):
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.pool_size = pool_size
        super(InceptionBlock, self).__init__()

        self.pw_block = point_wise_block(self.inplanes, self.outplanes//4)
        self.mp_layer = nn.MaxPool2d(kernel_size=self.pool_size, stride=stride, padding=1)
        self.conv3_block = conv_block(self.outplanes//4, self.outplanes//4, kernel_size=3, stride=1, padding=1)
        self.conv5_block = conv_block(self.outplanes//4, self.outplanes//4, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x1 = self.pw_block(x)

        x2 = self.pw_block(x)
        x2 = self.conv3_block(x2)

        x3 = self.pw_block(x)
        x3 = self.conv5_block(x3)

        x4 = self.mp_layer(x)
        x4 = self.pw_block(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class InceptionBlock3D(nn.Module):
    def __init__(self, 
                 inplanes, 
                 outplanes, 
                 stride=1, 
                 pool_size=3):
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.pool_size = pool_size
        super(InceptionBlock3D, self).__init__()

        self.pw_block = conv3d_block(self.inplanes, self.outplanes//4, kernel_size=(1,1,1), stride=(1,1,1), act=nn.ReLU)
        self.mp_layer = nn.MaxPool3d(kernel_size=(self.pool_size,1,1), stride=stride, padding=(1,0,0))
        self.conv3_block = conv3d_block(self.outplanes//4, self.outplanes//4, kernel_size=(3,1,1), stride=1, padding=(1,0,0), act=nn.ReLU)
        self.conv5_block = conv3d_block(self.outplanes//4, self.outplanes//4, kernel_size=(5,1,1), stride=1, padding=(2,0,0), act=nn.ReLU)
        
    def forward(self, x):
        x1 = self.pw_block(x)

        x2 = self.pw_block(x)
        x2 = self.conv3_block(x2)

        x3 = self.pw_block(x)
        x3 = self.conv5_block(x3)

        x4 = self.mp_layer(x)
        x4 = self.pw_block(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class SELayer(nn.Module):
    def __init__(self, 
                 channel, 
                 reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Texture_Enhance(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        #self.output_features=num_features
        self.output_features = num_features*4
        self.output_features_d = num_features
        self.conv0 = nn.Conv2d(num_features, num_features, 1)
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features*2, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2*num_features)
        self.conv3 = nn.Conv2d(num_features*3, num_features, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3*num_features)
        self.conv_last = nn.Conv2d(num_features*4 , num_features*4,1)
        self.bn4 = nn.BatchNorm2d(4*num_features)
        self.bn_last = nn.BatchNorm2d(num_features*4)

    def forward(self, feature_maps, attention_maps=(1,1)):
        B, N, H, W = feature_maps.shape

        if type(attention_maps) == tuple:
            attention_size = (int(H*attention_maps[0]), int(W*attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        feature_maps = feature_maps - F.interpolate(feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]), mode='nearest')
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0,feature_maps1], dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_,feature_maps2], dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_,feature_maps3], dim=1)
        feature_maps = self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True)))
        return feature_maps
