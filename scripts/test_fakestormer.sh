#! /bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/temporal/FakeSFormer_base_c23.yaml \
#                                                     -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/temporal/FakeSFormer_base_c40.yaml \
#                                                     -i samples/debugs/affine_f_2883.jpg

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/temporal/FakeSFormer_base_c0.yaml \
                                              -i samples/debugs/affine_f_2883.jpg
