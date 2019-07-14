#! /bin/bash

python test.py \
    --cates all \
    --resume_checkpoint pretrained_models/ae/all/checkpoint.pt \
    --dims 512-512-512 \
    --use_deterministic_encoder \
    --evaluate_recon \
    --resume_dataset_mean pretrained_models/ae/all/train_set_mean.npy \
    --resume_dataset_std pretrained_models/ae/all/train_set_std.npy

