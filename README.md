# PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows

This repository contains a PyTorch implementation of the paper:

[PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows](www.arxiv.com). 

[Guandao Yang*](http://www.guandaoyang.com), 
[Xun Huang*](http://www.cs.cornell.edu/~xhuang/),
[Zekun Hao](http://www.cs.cornell.edu/~zekun/),
[Ming-Yu Liu](http://mingyuliu.net/),
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/),
[Bharath Hariharan](http://home.bharathh.info/)

**Codes will be available soon!**


## Introduction

As 3D point clouds become the representation of choice for multiple vision and graphics applications, the ability to synthesize or reconstruct high-resolution, high-fidelity point clouds becomes crucial. Despite the recent success of deep learning models in discriminative tasks of point clouds, generating point clouds remains challenging. This paper proposes a principled probabilistic framework to generate 3D point clouds by modeling them as a distribution of distributions. Specifically, we learn a two-level hierarchy of distributions where the first level is the distribution of shapes and the second level is the distribution of points given a shape. This formulation allows us to both sample shapes and sample an arbitrary number of points from a shape. Our generative model, named PointFlow, learns each level of the distribution with a continuous normalizing flow. The invertibility of normalizing flows enables computation of the likelihood during training and allows us to train our model in the variational inference framework. Empirically, we demonstrate that PointFlow achieves state-of-the-art performance in point cloud generation. We additionally show our model is able to faithfully reconstruct point clouds and learn useful representations in an unsupervised manner. 

## Examples
<p float="left">
    <img src="docs/assets/teaser.gif" height="256"/>
</p>

