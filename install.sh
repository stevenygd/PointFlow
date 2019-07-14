#! /bin/bash

root=`pwd`

# Install dependecies
conda install numpy matplotlib pillow scipy tqdm scikit-learn -y
pip install tensorflow-gpu==1.13.1
pip install tensorboardX==1.7

# Compile CUDA kernel for CD/EMD loss
cd metrics/pytorch_structural_losses/
make clean
make
cd $root

# install torchdiffeq
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e .
