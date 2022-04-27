#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/mms_cfd_mock \
--batch_size 2 \
--name mms_cfd_mock \
--dataset_mode regression \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--print_freq 1