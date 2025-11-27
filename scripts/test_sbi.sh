#! /bin/bash

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/efn4_fpn_sbi_adv.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/vit_sbi_base.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/vit_sbi_small.yaml \
                                              -i /home/users/XXX/data/FaceForensics++/c0/test/frames/Deepfakes/000_003/000.png

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/spatial/vit_sbi_large.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/swin_sbi_base.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/swin_sbi_small.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/swin_bi_small.yaml \
#                                                     -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/spatial/swin_sbi_tiny.yaml \
#                                                     -i samples/debugs/affine_f_2883.jpg

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/temporal/FakeSFormer_base.yaml \
#                                               -i samples/debugs/affine_f_2883.jpg                                              
