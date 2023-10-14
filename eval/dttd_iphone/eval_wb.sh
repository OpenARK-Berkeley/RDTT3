#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 eval_wb.py --dataset_root /ssd/zixun/Documents/DigitalTwin-6DPose_cleaned/estimation/dataset/dttd_iphone/DTTD_IPhone_Dataset/root\
                    --model /home/zixun/Documents/RDTT3/result/result_1_m8p4_iphone_1002-1900/checkpoints/epoch_51_dist_0.02677410949266623.pth\
                    --base_latent 256 --embed_dim 512 --fusion_block_num 1 --layer_num_m 8 --layer_num_p 4\
                    --visualize --output eval_results_wb_filter_modelrecon\
                    --filter #--debug