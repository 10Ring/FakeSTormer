#-*-coding: utf-8 -*-
import argparse
import os
import sys
import math
from copy import deepcopy
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize
from torchcam import methods
from image_utils import overlay_mask
import numpy as np
from glob import glob
from natsort import natsorted
import torch.nn.functional as F

from models import MODELS, build_model
from models.utils import load_pretrained
from configs.get_config import load_config
from logs.logger import Logger
from transform import final_transform


def main():
    argparser = argparse.ArgumentParser('Arguments for CAM visualization...')
    argparser.add_argument('--cfg', help='Specify config to load', required=True)
    argparser.add_argument('--target_layer', '-t', help='Specify layer names to visualize CAM', required=False)
    argparser.add_argument("--method", '-m', type=str, default="GradCAM", help="CAM method to use")
    argparser.add_argument('--mode', type=str, choices=['image', 'video'], default='image', help='Mode to visualize gradCAM')
    argparser.add_argument('--image', '-i', help='Specify image to overlay CAM', required=False)
    argparser.add_argument('--video', '-v', help='Specify video to overlay CAM', required=False)
    argparser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    argparser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    argparser.add_argument("--class-idx", type=int, default=0, help="Index of the class to inspect")
    argparser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
    argparser.add_argument('--cuda', action='store_true', help='Running CAM with cuda')
    argparser.add_argument('--save_inverse', action='store_true', help='Saving the inverse of CAM')
    args = argparser.parse_args()
    print(args)
    
    # Loading configs
    cfg = load_config(args.cfg)
    
    # Logger
    logger = Logger(task='CAM_vis')
    
    # Loading model based on the config
    model = build_model(cfg.MODEL, MODELS).to(torch.float)
    logger.info('Loading weight ... {}'.format(cfg.TEST.pretrained))
    model = load_pretrained(model, cfg.TEST.pretrained)

    if args.cuda:
        model = model.cuda()
    model.eval()

    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)
        
    # Loading image
    img_list = []
    if args.mode == 'image':
        assert os.path.exists(args.image), 'Image path must be valid, please check the path again!'
        img = Image.open(args.image)
        H, W = img.size
        img = img.crop((0, 0, W-0, H-0))
        img_list.append(img)
    elif args.mode == 'video':
        assert os.path.exists(args.video), 'Video path must be valid, please check the path again!'
        n_frames = cfg.DATASET.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES
        assert n_frames is not None, 'Number of video frames can not be None!'
        # Load first n_frames inside the video
        img_paths = glob(f'{args.video}/*.png')
        img_paths = natsorted(img_paths) # correct the order of image paths
        img_paths = img_paths[:n_frames]
        for img_path in img_paths:
            img = Image.open(img_path)
            H, W = img.size
            img = img.crop((0, 0, W-0, H-0))
            img_list.append(img)
    else:
        raise ValueError('We only support GradCAM for image or video mode at the moment!')
    
    # Preprocess image
    transform = final_transform(cfg.DATASET)
    image_size = (cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1])

    # Transform images
    transformed_imgs = torch.tensor([])
    for _i in img_list:
        img_resize = _i.resize(image_size)
        img_resize = np.array(img_resize)/255
        img_tensor = transform(img_resize).to(torch.float)
        if args.cuda:
            img_tensor = img_tensor.cuda()
        img_tensor.requires_grad_(True)
        transformed_imgs = torch.cat((transformed_imgs, img_tensor.unsqueeze(0)), 0)
    
    # Hook the corresponding layer in the model
    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    else:
        cam_methods = [
            "CAM",
            "GradCAM",
            "GradCAMpp",
            "SmoothGradCAMpp",
            "ScoreCAM",
            "SSCAM",
            "ISCAM",
            "XGradCAM",
            "LayerCAM",
        ]
    cam_extractors = [
        methods.__dict__[name](model, target_layer=args.target_layer, enable_hooks=False) for name in cam_methods
    ]
    
    if args.mode == 'image':
        num_rows = args.rows
        num_cols = math.ceil((len(cam_extractors)) / num_rows) + 1
    else:
        num_cols = n_frames
        num_rows = len(cam_extractors) + 1
    
    _, axes = plt.subplots(num_rows, num_cols, figsize=(6, 4))
    # Display input
    for idx, _i in enumerate(img_list):
        ax = axes[0][idx] if num_rows > 1 else axes[0] if num_cols > 1 else axes
        ax.imshow(_i)
        ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        if args.mode == 'image':
            scores = model(transformed_imgs)[0]['cls'].sigmoid()
        else:
            transformed_imgs = transformed_imgs.transpose(0,1).unsqueeze(0)
            scores = model(transformed_imgs)[0]['hm'].sigmoid().view(1, -1).max(1, keepdim=True).values
            # output, attn = model(transformed_imgs) # For visualizing the attn scores, will comeback later
            # scores = output[0]['temp_loc'].sigmoid()
        print('Classification Score -- {}'.format(scores))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx
        # class_idx = img_idx

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores)[0].to(torch.float).squeeze(0).cpu()
        # activation_map = torch.cat((activation_map, torch.zeros(4)), 0)
        # activation_map = F.adaptive_avg_pool1d(activation_map.unsqueeze(0), 196).squeeze(0)
        # activation_map = activation_map[class_idx, 1:].reshape((14, 14))
        
        # Clean data
        extractor.remove_hooks()
        extractor._hooks_enabled = False
        
        for img_idx, i_ in enumerate(img_list):
            # Convert it to PL image
            # The indexing below means first image in batch
            heatmap = to_pil_image(activation_map[img_idx].unsqueeze(0), mode='F')
            # activation_map = attn[img_idx].mean(0)[0, 1:]
            # activation_map = activation_map.reshape((14, 14)).detach()
            # activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            # heatmap = to_pil_image(activation_map.unsqueeze(0), mode='F')
            
            # Plot the result
            result = overlay_mask(deepcopy(i_), heatmap, alpha=args.alpha)

            ax = axes[idx][img_idx] if num_rows > 1 else axes[idx] if num_cols > 1 else axes

            ax.imshow(result)
            ax.set_title(extractor.__class__.__name__, size=8)
            
            # Compute the inverse heatmap
            if args.save_inverse:
                inverse_activation_map = torch.sub(1, activation_map[img_idx].unsqueeze(0))
                inverse_heatmap = to_pil_image(inverse_activation_map, mode='F')
                result_inverse = overlay_mask(deepcopy(img), inverse_heatmap, alpha=args.alpha)
                ax = axes[idx][img_idx] if args.rows > 1 else axes[idx] if num_cols > 1 else axes
                ax.imshow(result_inverse)
                ax.set_title(f'{extractor.__class__.__name__}_inverse', size=8)    

    # Clear axes
    if num_cols > 1:
        for _axes in axes:
            if num_rows > 1:
                for ax in _axes:
                    ax.axis("off")
            else:
                _axes.axis("off")

    else:
        axes.axis("off")

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)


if __name__=='__main__':
    main()
