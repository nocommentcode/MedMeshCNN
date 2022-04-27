#!/usr/bin/env bash

## run the training
python vf38/mms_cfd/mms_cfd_git/random_search.py \
--dataroot ~/vf38_scratch/mmscfd_scratch/dataset_3kfaces \
--ninput_edges 7000 \
--batch_size 8 \
--name mms_cfd_real \
--dataset_mode regression \
--ncf 512 128 \
--pool_res 5000 4000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0 \
--num_aug 20 \
--niter_decay 100 \
--print_freq 10 \
--test_print_freq 10 \
--lr 0.0006 \
--verbose_test
