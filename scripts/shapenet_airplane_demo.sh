#! /bin/bash

python demo.py \
    --cates airplane \
    --resume_checkpoint pretrained_models/gen/airplane/checkpoint.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 2048

