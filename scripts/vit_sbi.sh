#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/spatial/vit_sbi_small.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/spatial/vit_sbi_base.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/spatial/vit_sbi_large.yaml
