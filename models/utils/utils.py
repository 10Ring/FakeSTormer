#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import warnings
import pkgutil
from importlib import import_module
import os.path as osp
from collections import OrderedDict
from tempfile import TemporaryDirectory
import copy
import re

import torchvision
import torch
import torch.nn as nn
from torch.utils import model_zoo
import mmcv
from mmengine.dist import get_dist_info
from mmengine.utils import mkdir_or_exist
from mmengine.fileio import load as load_file
from mmengine.fileio import FileClient
from mmcv.parallel import is_module_wrapper
from ptflops import get_model_complexity_info


ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


layers_position = {
    'PoseResNet_50': 158,
    'PoseResNet_101': 311,
    'PoseEfficientNet_B4': 415,
}


def m_flops(model, cfg):
    if cfg.DATASET.DATA_TYPE != "video":
        macs, params = get_model_complexity_info(model, (3, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[0]), 
                                                 as_strings=True, verbose=True)
    else:
        macs, params = get_model_complexity_info(model, (3, cfg.DATASET.DATA.SAMPLES_PER_VIDEO.TRAIN, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[0]), 
                                                 as_strings=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def n_param_model(model):
    print('Number of parameters', sum(p.numel() for p in model.parameters()))
    print('Number of trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))


def preset_model(cfg, model, optimizer=None, scaler=None):
    #Loading models from config, make sure the pretrained path correct to the model name
    start_epoch = 0
    if 'pretrained' in cfg.TRAIN and os.path.isfile(cfg.TRAIN.pretrained):
        model, optimizer, start_epoch, scaler = load_model(model, 
                                                   cfg.TRAIN.pretrained, 
                                                   optimizer=optimizer, 
                                                   scaler=scaler,
                                                   resume=cfg.TRAIN.resume,
                                                   lr=cfg.TRAIN.lr,
                                                   lr_step=cfg.TRAIN.lr_scheduler.milestones,
                                                   gamma=cfg.TRAIN.lr_scheduler.gamma)
    else:
        model.init_weights(**cfg.MODEL.INIT_WEIGHTS)
    print('Loading model successfully -- {}'.format(cfg.MODEL.type))

    # Showing model FLOPS
    m_flops(model, cfg)
    
    #Freeze backbone if begin_epoch < warm up
    if cfg.TRAIN.freeze_backbone and start_epoch < cfg.TRAIN.warm_up:
        freeze_backbone(cfg.MODEL, model)
    
    n_param_model(model)
    return model, optimizer, start_epoch, scaler


def load_pretrained(model, weight_path):
    '''
    This function only care about state dict of model
    For other modules such as optimizer, resume learning, please refer @load_model
    '''
    state_dict = torch.load(weight_path)['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def freeze_backbone(cfg, model):
    '''
    This func to freeze some specific layers to warm up the models
    '''
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        for i, (n, p) in enumerate(model.named_parameters()):
            if (i <= layers_position[f'{cfg.type}_{cfg.num_layers}']):
                p.requires_grad = False


def unfreeze_backbone(model):
    '''
    This func to unfreeze all model layers
    '''
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True


def load_model(model, model_path, optimizer=None, scaler=None, 
               resume=False, lr=None, lr_step=None, gamma=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                      'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                
            start_epoch = checkpoint['epoch'] + 1
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    
    return model, optimizer, start_epoch, scaler


def save_model(path, epoch, model, optimizer=None, scaler=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    if not (scaler is None):
        data['scaler'] = scaler.state_dict()

    torch.save(data, path)


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))

    mkdir_or_exist(mmcv_home)
    return mmcv_home


def load_url_dist(url, model_dir=None, map_location="cpu"):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir, map_location=map_location)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0],
                                   'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


def load_pavimodel_dist(model_path, map_location=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            downloaded_file = osp.join(tmp_dir, model.name)
            model.download(downloaded_file)
            checkpoint = torch.load(downloaded_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                downloaded_file = osp.join(tmp_dir, model.name)
                model.download(downloaded_file)
                checkpoint = torch.load(
                    downloaded_file, map_location=map_location)
    return checkpoint


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
                      'use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
                          f'of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        # check if is url
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        checkpoint = load_fileclient_dist(
            filename, backend='ceph', map_location=map_location)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None,
                    patch_padding='pad',
                    part_features=None
                    ):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        patch_padding (str): 'pad' or 'bilinear' or 'bicubic', used for interpolate patch embed from 14x14 to 16x16

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    rank, _ = get_dist_info()

    layer_names = [name for name, param in model.named_parameters()]

    if 'patch_embed.proj.weight' in state_dict and 'patch_embed.proj._conv_stem.weight' not in layer_names:
        proj_weight = state_dict['patch_embed.proj.weight']
        orig_size = proj_weight.shape[2:]
        current_size = model.patch_embed.proj.weight.shape[2:]
        padding_size = current_size[0] - orig_size[0]
        padding_l = padding_size // 2
        padding_r = padding_size - padding_l
        if orig_size != current_size:
            if 'pad' in patch_padding:
                proj_weight = torch.nn.functional.pad(proj_weight, (padding_l, padding_r, padding_l, padding_r))
            elif 'bilinear' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bilinear', align_corners=False)
            elif 'bicubic' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bicubic', align_corners=False)
            state_dict['patch_embed.proj.weight'] = proj_weight

    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        # pos_embed_checkpoint = pos_embed_checkpoint[:, :-1, :]
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        if rank == 0:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
    
    new_state_dict = copy.deepcopy(state_dict)
    if part_features is not None:
        current_keys = list(model.state_dict().keys())
        for key in current_keys:
            if "mlp.experts" in key:
                source_key = re.sub(r'experts.\d+.', 'fc2.', key)
                new_state_dict[key] = state_dict[source_key][-part_features:]
            elif 'fc2' in key:
                new_state_dict[key] = state_dict[key][:-part_features]

    # load state_dict
    load_state_dict(model, new_state_dict, strict, logger)
    return checkpoint


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt
