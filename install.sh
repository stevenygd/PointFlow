#! /bin/bash

# Install dependecies
# conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
# conda install matplotlib tqdm scikit-learn -y
# pip install pillow==5.0.0
# pip install scipy==1.0.1
# pip install numpy==1.16.4
# pip install tensorflow-gpu==1.13.1
# pip install tensorboardX==1.7
# pip install torchdiffeq==0.0.1

# Compile CUDA kernel for CD/EMD loss
root=`pwd`
cd metrics/pytorch_structural_losses/
make clean
make
cd $root
