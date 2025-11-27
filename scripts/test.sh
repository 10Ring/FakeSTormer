#! /bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test.py --cfg configs/temporal/bin_cls/ResNet3D_c23.yaml \
#                                                     -i 447.png

# CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/temporal/FakeSFormer_base_c23.yaml \
#                                               -v /data/deepfake_cluster/datasets_df/DFW/test/frames/fake_test/fake_98_187

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/spatial/xception_sbi.yaml \
                                              -i 447.png
