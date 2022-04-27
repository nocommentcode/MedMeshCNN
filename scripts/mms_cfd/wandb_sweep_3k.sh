#!/usr/bin/env bash

## run the training
python vf38/mms_cfd/mms_cfd_git/wandb_sweep.py \
--dataroot ~/vf38_scratch/mmscfd_scratch/dataset_3kfaces \
--ninput_edges 7000 \
--name mms_cfd_real \
--dataset_mode regression \
--print_freq 10 \
--test_print_freq 10 \
--shuffle_dataset \
--normalise_targets
