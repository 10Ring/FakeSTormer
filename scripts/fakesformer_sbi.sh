#! /bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSFormer_base_c23.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSFormer_base_c0.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSFormer_base_c40.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSFormer_large_c23.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSwin3D_base_c23.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/temporal/FakeSFormer_base_c23_224p8.yaml
