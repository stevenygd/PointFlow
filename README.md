# PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows

This repository contains a PyTorch implementation of the paper:

[PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows](https://arxiv.org/abs/1906.12320). 
<br>
[Guandao Yang*](http://www.guandaoyang.com), 
[Xun Huang*](http://www.cs.cornell.edu/~xhuang/),
[Zekun Hao](http://www.cs.cornell.edu/~zekun/),
[Ming-Yu Liu](http://mingyuliu.net/),
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/),
[Bharath Hariharan](http://home.bharathh.info/)
(* equal contribution)
<br>
ICCV 2019 (**Oral**)


## Introduction

As 3D point clouds become the representation of choice for multiple vision and graphics applications, the ability to synthesize or reconstruct high-resolution, high-fidelity point clouds becomes crucial. Despite the recent success of deep learning models in discriminative tasks of point clouds, generating point clouds remains challenging. This paper proposes a principled probabilistic framework to generate 3D point clouds by modeling them as a distribution of distributions. Specifically, we learn a two-level hierarchy of distributions where the first level is the distribution of shapes and the second level is the distribution of points given a shape. This formulation allows us to both sample shapes and sample an arbitrary number of points from a shape. Our generative model, named PointFlow, learns each level of the distribution with a continuous normalizing flow. The invertibility of normalizing flows enables computation of the likelihood during training and allows us to train our model in the variational inference framework. Empirically, we demonstrate that PointFlow achieves state-of-the-art performance in point cloud generation. We additionally show our model is able to faithfully reconstruct point clouds and learn useful representations in an unsupervised manner. 

## Examples
<p float="left">
    <img src="docs/assets/teaser.gif" height="256"/>
</p>

## Dependencies
* Python 3.6
* CUDA 10.0.
* G++ or GCC 5.
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.0.1
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq).
* (Optional) [Tensorboard](https://www.tensorflow.org/) for visualization of the training process. 

Following is the suggested way to install these dependencies: 
```bash
# Create a new conda environment
conda create -n PointFlow python=3.6
conda activate PointFlow

# Install pytorch (please refer to the commend in the official website)
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch -y

# Install other dependencies such as torchdiffeq, structural losses, etc.
./install.sh
```

## Dataset 

The point clouds are uniformly sampled from meshes from ShapeNetCore dataset (version 2) and use the official split.
Please use this [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing) to download the ShapeNet point cloud.
The point cloud should be placed into `data` directory.
```bash
mv ShapeNetCore.v2.PC15k.zip data/
cd data
unzip ShapeNetCore.v2.PC15k.zip
```

Please contact us if you need point clouds for the ModelNet dataset.

## Training

Example training scripts can be found in `scripts/` folder. 
```bash
# Train auto-encoder (no latent CNF)
./scripts/shapenet_airplane_ae.sh # Train with single GPU, about 7-8 GB GPU memory
./scripts/shapenet_airplane_ae_dist.sh # Train with multiple GPUs

# Train generative model
./scripts/shapenet_airplane_gen.sh # Train with single GPU, about 7-8 GB GPU memory 
./scripts/shapenet_airplane_gen_dist.sh # Train with multiple GPUs 
```

## Pre-trained models and test

Pretrained models can be downloaded from this [link](https://drive.google.com/file/d/1dcxjuuKiAXZxhiyWD_o_7Owx8Y3FbRHG/view?usp=sharing). 
The following is the suggested way to evaluate the performance of the pre-trained models.
```bash
unzip pretrained_models.zip;  # This will create a folder named pretrained_models

# Evaluate the reconstruction performance of an AE trained on the airplane category
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_ae_test.sh; 

# Evaluate the reconstruction performance of an AE trained with the whole ShapeNet
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_all_ae_test.sh;

# Evaluate the generative performance of PointFlow trained on the airplane category.
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_gen_test.sh
```

## Demo

The demo relies on [Open3D](http://www.open3d.org/). The following is the suggested way to install it:
```bash
conda install -c open3d-admin open3d 
```
The demo will sample shapes from a pre-trained model, save those shapes under the `demo` folder, and visualize those point clouds.
Once this dependency is in place, you can use the following script to use the demo for the pre-trained model for airplanes:
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/shapenet_airplane_demo.sh
```

## Point cloud rendering

Please refer to the following github repository for our point cloud rendering code: https://github.com/zekunhao1995/PointFlowRenderer.

## Cite
Please cite our work if you find it useful:
```latex
@article{pointflow,
 title={PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows},
 author={Yang, Guandao and Huang, Xun, and Hao, Zekun and Liu, Ming-Yu and Belongie, Serge and Hariharan, Bharath},
 journal={arXiv},
 year={2019}
}
```
