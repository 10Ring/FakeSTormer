#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/bin_cls/ResNet3D_c23.yaml
