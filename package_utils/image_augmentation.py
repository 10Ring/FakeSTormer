#-*- coding: utf-8 -*-
import os
import glob
import random
import argparse

import torch
from PIL import Image
import numpy as np

from image_utils import (gaussian_noise_color, block_wise, \
    color_saturation, color_contrast, gaussian_blur, \
    jpeg_compression, video_compression, load_image)

# DIST_LEVEL = 3

def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse
    param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse
    param_dict['VC'] = [30, 32, 35, 38, 40]  # larger, worse

    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', dest='path', type=str, default="")
    parser.add_argument('-t', dest='task', choices=["noise", "block", "saturation", 
                                                    "contrast", "blur", "pixel", "compression"], default="noise")
    args=parser.parse_args()
    # Setting device
    device=torch.device('cuda')
    dest = args.path+"_"+args.task+"_"+"random"+'/'
    
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    for (dirpath, dirnames, filenames) in os.walk(args.path):
        possible_files = os.path.join(dirpath, "*.png")
        
        for file in glob.glob(possible_files):
            img = load_image(file)
            dist_level = random.randint(1, 5)

            if args.task == "noise":
                params = get_distortion_parameter("GNC", dist_level)
                img = gaussian_noise_color(img, params)
            elif args.task == "block":
                params = get_distortion_parameter("BW", dist_level)
                img = block_wise(img, params)
            elif args.task == "saturation":
                params = get_distortion_parameter("CS", dist_level)
                img = color_saturation(img, params)
            elif args.task == "contrast":
                params = get_distortion_parameter("CC", dist_level)
                img = color_contrast(img, params)
            elif args.task == "blur":
                params = get_distortion_parameter("GB", dist_level)
                img = gaussian_blur(img, params)
            elif args.task == "pixel":
                params = get_distortion_parameter("JPEG", dist_level)
                img = jpeg_compression(img, params)
            elif args.task == "compression":
                params = get_distortion_parameter("VC", dist_level)
                img = video_compression(img, params)
            res = dest+file.split('/')[-2]
            
            if not os.path.exists(res):
                os.makedirs(res)
            # print(dest+('/').join(file.split('/')[-2:]))
            Image.fromarray(img).save(res + '/' + file.split('/')[-1])
