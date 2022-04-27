#!/usr/bin/env bash

## run the training
python vf38/mms_cfd/mms_cfd_git/train.py \
--dataroot ~/vf38_scratch/mmscfd_scratch/dataset_10kfaces \
--ninput_edges 15500 \
--batch_size 2 \
--name mms_cfd_real \
--dataset_mode regression \
--ncf 512 128 \
--pool_res 15000 15000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0 \
--num_aug 20 \
--niter_decay 100 \
--print_freq 10 \
--lr 0.00005 \
--shuffle_dataset \
--log_wandb