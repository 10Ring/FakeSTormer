#-*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange, reduce, repeat

from models.builder import BACKBONES
from .base import BaseBackbone
from .efficientNet import (
    get_model_params
)
from ..pose_efficientNet import EfficientNet
from ..common import BN3D_MOMENTUM


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, att_dimension=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, b_size, **kwargs):
        B, N, C = x.shape
        # mask = None

        # if 'maskout_pes' in kwargs.keys():
        #     mask = kwargs['maskout_pes']
        #     T = mask.shape[1]
        #     num_tokens = mask.shape[2]*mask.shape[3]

        #     if num_tokens == (N-1):
        #         mask = rearrange(mask, 'b t h w -> (b t) (h w)', b=b_size, t=T, h=mask.shape[2], w= mask.shape[3]).unsqueeze(1)
        #         mask = mask.unsqueeze(3)
        #         mask = mask.repeat(1, self.num_heads, 1, num_tokens+1)
        #         mask = torch.cat((torch.ones((b_size*T, self.num_heads, 1, num_tokens+1)).cuda(), mask), 2)
        #     else:
        #         mask = rearrange(mask, 'b t h w -> (b h w) t', b=b_size, t=T, h=mask.shape[2], w= mask.shape[3]).unsqueeze(1)
        #         mask = mask.unsqueeze(3)
        #         mask = mask.repeat(1, self.num_heads, 1, T+1)
        #         mask = torch.cat((torch.ones((b_size*num_tokens, self.num_heads, 1, T+1)).cuda(), mask), 2)
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 3, B (BxT or BxN), H, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        # self.save_v(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, float("-1e20"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # self.save_attn(attn)
        # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, att_dimension='spatial'):
        '''
        Implementation for cross attention through dimension
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        assert (att_dimension in ['spatial', 'temporal'])
        self.att_dimension = att_dimension

    def forward(self, x, b_size):
        B, N, C = x.shape # N can be T or HW // P**2, B can be previous reshaped from self.batch_size * T|B
        T = B // b_size

        qkv = self.qkv(x)
        qkv = qkv.reshape(b_size, T, N, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # Define the window
        window = torch.tensor([-2, -1, 0, 1, 2]).cuda()

        # Start computing locally cross self-attention
        q_scale = q * self.scale

        # Initialize attn weights
        attn = torch.zeros((b_size, self.num_heads, T, N, N), dtype=torch.float).cuda()

        if self.att_dimension == 'spatial':
            sqrt_N = int(math.sqrt(N))
            for w in window:
                i_indices = torch.arange(1, T).cuda()
                j_indices = torch.arange(0, N).cuda()
                
                i_valid = (i_indices + w >= 0) & (i_indices + w < T)
                j_valid = (j_indices + sqrt_N*w >= 0) & (j_indices + sqrt_N*w < N)
                
                i_indices = i_indices[i_valid]
                j_indices = j_indices[j_valid]
                
                attn[:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), i_indices + w))][:,:,:,j_indices][..., j_indices + sqrt_N*w] = \
                    (q_scale[:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), i_indices))][:,:,:,j_indices] \
                     @ k[:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), i_indices + w))][:,:,:,j_indices + sqrt_N*w].transpose(-2, -1))
        else:
            sqrt_T = int(math.sqrt(T))
            
            for w in window:
                i_indices = torch.arange(0, T).cuda()
                j_indices = torch.arange(1, N).cuda()

                i_valid = (i_indices + sqrt_T*w >= 0) & (i_indices + sqrt_T*w < T)
                j_valid = (j_indices + w >= 1) & (j_indices + w < N)

                i_indices = i_indices[i_valid]
                j_indices = j_indices[j_valid]

                attn[:,:,i_indices+sqrt_T*w][:,:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), j_indices))][...,torch.cat((torch.zeros(1, dtype=int).cuda(), j_indices+w))] = \
                    (q_scale[:,:,i_indices][:,:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), j_indices))] \
                     @ k[:,:,i_indices+sqrt_T*w][:,:,:,torch.cat((torch.zeros(1, dtype=int).cuda(), j_indices+w))].transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, b_size, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), b_size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class Block2D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attention_type='divided_space_time',
                 **kwargs):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
        self.register_token = kwargs.get('register_token')
        self.temp_token = kwargs.get('temp_token')
        self.return_s_cls_token = kwargs.get('return_s_cls_token') or False

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, \
                att_dimension='temporal')
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.norm3 = norm_layer(dim)
        # self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W, **kwargs):
        if self.temp_token:
            num_spatial_tokens = (x.size(1) - 2) // T
        else:
            num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, None
        elif self.attention_type == 'divided_space_time':
            # init class token
            init_cls_token = x[:, 0, :].unsqueeze(1)
            if self.temp_token:
                init_temp_token = x[:, -1, :].unsqueeze(1)
            # init_register_token = x[:, -1, :].unsqueeze(1)

            ## Temporal
            if self.temp_token:
                t_cls_token = init_temp_token.repeat(1, num_spatial_tokens, 1)
            else: 
                t_cls_token = init_cls_token.repeat(1, num_spatial_tokens, 1)
            t_cls_token = rearrange(t_cls_token, 'b (h w) m -> (b h w) m', b=B, h=H, w=W).unsqueeze(1)

            if self.temp_token:
                xt = x[:, 1:-1, :]
            else:
                xt = x[:, 1:, :]

            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            xt = torch.cat((xt, t_cls_token), 1)

            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt), b_size=B, **kwargs)) # Processing temporal att.
            res_temporal = self.temporal_fc(res_temporal)
            res_temporal, t_cls_token = res_temporal[:, :-1, :], res_temporal[:, -1, :]
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            t_cls_token = rearrange(t_cls_token, '(b h w) m -> b (h w) m', b=B, h=H, w=W)
            # t_cls_token_avg = torch.mean(t_cls_token, 1, True) ## average for every temporal patch
            xt = x[:, 1:-1, :] + res_temporal

            ## Spatial
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            # register_token = init_register_token.repeat(1, T, 1)
            # register_token = rearrange(register_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            # xs = torch.cat((xs, register_token), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs), b_size=B, **kwargs))

            ### Taking care of TEMP token
            t_cls_token_avg = torch.mean(t_cls_token, 1, True) ## average for every temporal patch

            ### Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token_avg = torch.mean(cls_token, 1, True) ## averaging for every frame
            # register_token = res_spatial[:, -1, :]
            # register_token = rearrange(register_token, '(b t) m -> b t m', b=B, t=T)
            # register_token_avg = torch.mean(register_token, 1, True) ## averaging for every frame

            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            # x = torch.cat((t_cls_token_avg, x), 1) + torch.cat((cls_token_avg, res), 1)
            xt_ = torch.cat((init_cls_token, x), 1)
            if self.temp_token:
                xt_ = torch.cat((xt_, t_cls_token_avg), 1)
            xs_ = torch.cat((cls_token_avg, res), 1)
            if self.temp_token:
                xs_ = torch.cat((xs_, init_temp_token), 1)
            x = xt_ + xs_
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            # Adding MLP for spatial_cls_token, ttemp_cls_token
            # t_cls_token = self.drop_path(self.mlp(self.norm3(t_cls_token)))
            # s_cls_token = self.drop_path(self.mlp(self.norm4(cls_token)))

            if self.return_s_cls_token:
                return x, cls_token, None
            else:
                return x, None, None


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)
    

class PatchEmbed3D(nn.Module):
    """ Images to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1, low_level=False, **override_params):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.low_level = low_level

        if not self.low_level:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            model_name = "efficientnet-b4"
            self.proj = EfficientNet.from_pretrained(model_name, advprop=True, **override_params)
            self.fc = nn.Linear(160, embed_dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        if not self.low_level:
            x = self.proj(x)
        else:
            endpoints = self.proj.extract_endpoints(x)
            x1 = endpoints['reduction_6']
            x2 = endpoints['reduction_5']
            x3 = endpoints['reduction_4']
            x4 = endpoints['reduction_3']
            x5 = endpoints['reduction_2']
            x = x3
            x = x.permute(0,2,3,1)
            x = self.fc(x)
            x = x.permute(0,3,1,2)
            
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, T, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@BACKBONES.register_module()
class ViT(BaseBackbone):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True, class_token=True, attention_type='space',
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, **kwargs,
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.attention_type = attention_type

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                if self.cls_token is not None:
                    nn.init.normal_(self.cls_token, std=1e-6)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, **kwargs):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            # x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
            x = x + self.pos_embed

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, B)

        x = self.norm(x)
        res = {}

        if self.cls_token is not None:
            x_cls = x[:, :1]
        else:
            x_cls = torch.mean(x[:, 1:], 1, False)
        res["cls"] = x_cls

        xp = x[:, 1:]
        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        res["embed"] = xp

        return res

    def forward(self, x, **kwargs) -> Union[torch.tensor, Dict[str, torch.tensor]]:
        x = self.forward_features(x, **kwargs)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class TimeViT(ViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, 
                 drop_path_rate=0, hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True, class_token=True, attention_type='space_only', 
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, num_frames=4, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, 
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, 
                         norm_layer, use_checkpoint, frozen_stages, ratio, last_norm, class_token, attention_type, 
                         patch_padding, freeze_attn, freeze_ffn, **kwargs)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        register_token = kwargs.get('register_token') or False
        temp_token = kwargs.get('temp_token') or False
        self.low_level_enhanced = kwargs.get('low_level_enhanced') or False
        self.patch_size = patch_size

        # Temporary
        # self.patch_embed = PatchEmbed3D(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio, 
        #     low_level=True, include_top=False, include_hm_decoder=False)
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        
        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if register_token else None
        self.temp_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if temp_token else None
        
        if self.attention_type != 'space_only':
            if temp_token:
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames+1, embed_dim))
            else:
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)
        
        self.norm_s_cls = norm_layer(embed_dim)
        self.norm_t_cls = norm_layer(embed_dim)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block2D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type,
                temp_token=temp_token, register_token=register_token, return_s_cls_token=True)
            for i in range(depth)])
        
        if self.low_level_enhanced:
            self.lowconv3d_0 = nn.Conv3d(3, embed_dim//16, kernel_size=(3,4,4), stride=(1,4,4), padding=(1,0,0), bias=False) # B x 48 x 4 x 56 x 56
            self.bn_low0 = nn.BatchNorm3d(embed_dim//16, momentum=BN3D_MOMENTUM)
            self.act_low0 = nn.GELU()
            self.lowconv3d_01 = nn.Conv3d(embed_dim//16, embed_dim//4, kernel_size=(3,4,4), stride=(1,4,4), padding=(1,0,0), bias=False) # B x 48 x 4 x 14 x 14
            self.bn_low01 = nn.BatchNorm3d(embed_dim//4, momentum=BN3D_MOMENTUM)
            self.act_low01 = nn.GELU()
            # self.lowmaxpool3d_0 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0)  # B x 48 x 4 x 14 x 14
            self.lowfc_0 = nn.Linear(embed_dim//4, embed_dim)
            
            self.lowconv3d_1 = nn.Conv3d(embed_dim//16, embed_dim//4, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0), bias=False) # B x 192 x 4 x 28 x 28
            self.bn_low1 = nn.BatchNorm3d(embed_dim//4, momentum=BN3D_MOMENTUM)
            self.act_low1 = nn.GELU()
            self.lowconv3d_11 = nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0), bias=False) # B x 192 x 4 x 14 x 14
            self.bn_low11 = nn.BatchNorm3d(embed_dim, momentum=BN3D_MOMENTUM)
            self.act_low11 = nn.GELU()
            # self.lowmaxpool3d_1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
            self.lowfc_1 = nn.Linear(embed_dim, embed_dim)

            self.lowconv3d_2 = nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0), bias=False) # B x 768 x 4 x 14 x 14
            self.bn_low2 = nn.BatchNorm3d(embed_dim, momentum=BN3D_MOMENTUM)
            self.act_low2 = nn.GELU()
            self.lowconv3d_21 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False) # B x 768 x 4 x 14 x 14
            self.bn_low21 = nn.BatchNorm3d(embed_dim, momentum=BN3D_MOMENTUM)
            self.act_low21 = nn.GELU()
            # self.lowmaxpool3d_2 = nn.MaxPool3d(kernel_size=(1,1,1), stride=(1,1,1), padding=0)
            self.lowfc_2 = nn.Linear(embed_dim, embed_dim)

            self.lowconv3d_3 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False) # B x 768 x 4 x 14 x 14
            self.bn_low3 = nn.BatchNorm3d(embed_dim, momentum=BN3D_MOMENTUM)
            self.act_low3 = nn.GELU()
            self.lowconv3d_31 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False) # B x 768 x 4 x 14 x 14
            self.bn_low31 = nn.BatchNorm3d(embed_dim, momentum=BN3D_MOMENTUM)
            self.act_low31 = nn.GELU()
            # self.lowmaxpool3d_3 = nn.MaxPool3d(kernel_size=(1,1,1), stride=(1,1,1), padding=0)
            self.lowfc_3 = nn.Linear(embed_dim, embed_dim)
    
    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            print('Initializing weights for temporal FC...')
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1
        
        if self.temp_token is not None:
            nn.init.normal_(self.temp_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}
        
    def forward_features(self, x, **kwargs):
        B, C, T, H, W = x.shape

        if self.low_level_enhanced:
            x_low0 = self.lowconv3d_0(x)
            x_low0 = self.bn_low0(x_low0)
            x_low0 = self.act_low0(x_low0)
            # x_low0_ = self.lowmaxpool3d_0(x_low0)
            x_low0_ = self.act_low01(self.bn_low01(self.lowconv3d_01(x_low0)))
            x_low0_ = x_low0_.flatten(2).transpose(1,2)
            x_low0_ = self.lowfc_0(x_low0_)
            x_low0_ = rearrange(x_low0_, 'b (t h w) m -> (b t) (h w) m', b=B, t=T, h=H//self.patch_size, w=W//self.patch_size)
            x_low0_ = x_low0_.sigmoid()
            x_prev = x_low0

        x, T, (Hp, Wp) = self.patch_embed(x)

        if self.low_level_enhanced:
            x = x_low0_ * x

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # if self.register_token is not None:
        #     register_token = self.register_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((x, register_token), dim=1)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            # x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
            if x.size(1) != self.pos_embed.size(1):
                # Resizing the pos embeds in case they do not match the input at inference
                pos_embed = self.pos_embed
                cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
        
        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)

            if self.temp_token is not None:
                temp_token = self.temp_token.expand(x.shape[0], -1, -1)
                x = torch.cat((x, temp_token), dim=1)

            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1) and self.temp_token is None:
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)

            temp_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, :-1]
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
            x = torch.cat((x, temp_tokens), dim=1)

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, s_cls_token, t_cls_token = blk(x, B, T, Wp, **kwargs)
                if self.low_level_enhanced:
                    if (idx+1)%4==0:
                        x_low = self.__getattr__(f'lowconv3d_{(idx+1)//4}')(x_prev)
                        # x_low_ = self.__getattr__(f'lowmaxpool3d_{(idx+1)//4}')(x_low)
                        x_low = self.__getattr__(f'bn_low{(idx+1)//4}')(x_low)
                        x_low = self.__getattr__(f'act_low{(idx+1)//4}')(x_low)
                        x_low_ = self.__getattr__(f'lowconv3d_{(idx+1)//4}1')(x_low)
                        x_low_ = self.__getattr__(f'bn_low{(idx+1)//4}1')(x_low_)
                        x_low_ = self.__getattr__(f'act_low{(idx+1)//4}1')(x_low_)

                        x_low_ = x_low_.flatten(2).transpose(1,2)
                        x_low_ = self.__getattr__(f'lowfc_{(idx+1)//4}')(x_low_)
                        x_low_ = x_low_.sigmoid()
                        x[:,1:-1,:] = x[:,1:-1,:].clone() * x_low_
                        x_prev = x_low
        
        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        res = {}

        if self.temp_token is not None:
            x_cls = x[:, -1:]
        elif self.cls_token is not None:
            x_cls = x[:, :1]
        else:
            x_cls = torch.mean(x[:, 1:-1], 1, False)
        res["cls"] = x_cls

        xp = x[:, 1:-1]
        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        if self.attention_type != 'space_only':
            xp = rearrange(xp, 'b (m t) h w -> b m t h w', b=B, t=T, h=Hp, w=Wp)
        res["embed"] = xp

        if s_cls_token is not None:
            s_cls_token = self.norm_s_cls(s_cls_token)
            res["s_cls_token"] = s_cls_token

        if t_cls_token is not None:
            t_cls_token = self.norm_t_cls(t_cls_token)
            res["t_cls_token"] = t_cls_token

        return res
