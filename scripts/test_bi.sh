#! /bin/bash

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/vit_bi_small.yaml \
#                                               -i /data/deepfake_cluster/datasets_df/FaceForensics++/c0/test/frames/Deepfakes/000_003/012.png

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/swin_bi_small.yaml \
#                                                     -i ~/data/FaceForensics++/c0/test/frames/NeuralTextures/035_036/000.png

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/efn4_fpn_hm_adv.yaml \
                                              -i ~/data/FaceForensics++/c0/test/frames/NeuralTextures/035_036/000.png
