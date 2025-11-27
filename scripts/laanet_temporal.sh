#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/ResNet3D_EFPN3D_hm3D_c23.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSwin3D_base_c23.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSwin3D_base_c0.yaml
