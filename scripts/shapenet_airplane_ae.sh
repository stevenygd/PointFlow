#! /bin/bash

cate="airplane"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=16
lr=2e-3
epochs=4000
ds=shapenet15k
log_name="ae/${ds}-cate${cate}"
data_dir="data/ShapeNetCore.v2.PC15k"

python train.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --cates ${cate} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 50 \
    --viz_freq 1 \
    --log_freq 1 \
    --val_freq 10 \
    --use_deterministic_encoder \
    --prior_weight 0 \
    --entropy_weight 0

echo "Done"
exit 0
