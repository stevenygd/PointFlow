#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "src/approxmatch.cuh"
#include "src/nndistance.cuh"

#include <vector>
#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
input:
	set1 : batch_size * #dataset_points * 3
	set2 : batch_size * #query_points * 3
returns:
	match : batch_size * #query_points * #dataset_points
*/
//  temp: TensorShape{b,(n+m)*2}
std::vector<at::Tensor> ApproxMatch(at::Tensor set_d, at::Tensor set_q) {
    //std::cout << "[ApproxMatch] Called." << std::endl;
    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m
    //std::cout << "[ApproxMatch] batch_size:" << batch_size << std::endl;
    at::Tensor match = torch::empty({batch_size, n_query_points, n_dataset_points}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor temp = torch::empty({batch_size, (n_query_points+n_dataset_points)*2}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(temp);
    
    approxmatch(batch_size,n_dataset_points,n_query_points,set_d.data<float>(),set_q.data<float>(),match.data<float>(),temp.data<float>(), at::cuda::getCurrentCUDAStream());
    return {match, temp};
}

at::Tensor MatchCost(at::Tensor set_d, at::Tensor set_q, at::Tensor match) {
    //std::cout << "[MatchCost] Called." << std::endl;
    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m
    //std::cout << "[MatchCost] batch_size:" << batch_size << std::endl;
    at::Tensor out = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(out);
    matchcost(batch_size,n_dataset_points,n_query_points,set_d.data<float>(),set_q.data<float>(),match.data<float>(),out.data<float>(),at::cuda::getCurrentCUDAStream());
    return out;
}

std::vector<at::Tensor> MatchCostGrad(at::Tensor set_d, at::Tensor set_q, at::Tensor match) {
    //std::cout << "[MatchCostGrad] Called." << std::endl;
    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m
    //std::cout << "[MatchCostGrad] batch_size:" << batch_size << std::endl;
    at::Tensor grad1 = torch::empty({batch_size,n_dataset_points,3}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor grad2 = torch::empty({batch_size,n_query_points,3}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(grad1);
    CHECK_INPUT(grad2);
    matchcostgrad(batch_size,n_dataset_points,n_query_points,set_d.data<float>(),set_q.data<float>(),match.data<float>(),grad1.data<float>(),grad2.data<float>(),at::cuda::getCurrentCUDAStream());
    return {grad1, grad2};
}


/*
input:
	set_d : batch_size * #dataset_points * 3
	set_q : batch_size * #query_points * 3
returns:
	dist1, idx1 : batch_size * #dataset_points
	dist2, idx2 : batch_size * #query_points
*/
std::vector<at::Tensor> NNDistance(at::Tensor set_d, at::Tensor set_q) {
    //std::cout << "[NNDistance] Called." << std::endl;
    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m
    //std::cout << "[NNDistance] batch_size:" << batch_size << std::endl;
    at::Tensor dist1 = torch::empty({batch_size, n_dataset_points}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor idx1 = torch::empty({batch_size, n_dataset_points}, torch::TensorOptions().dtype(torch::kInt32).device(set_d.device()));
    at::Tensor dist2 = torch::empty({batch_size, n_query_points}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor idx2 = torch::empty({batch_size, n_query_points}, torch::TensorOptions().dtype(torch::kInt32).device(set_d.device()));
    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(dist1);
    CHECK_INPUT(idx1);
    CHECK_INPUT(dist2);
    CHECK_INPUT(idx2);
    // void nndistance(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream);
    nndistance(batch_size,n_dataset_points,set_d.data<float>(),n_query_points,set_q.data<float>(),dist1.data<float>(),idx1.data<int>(),dist2.data<float>(),idx2.data<int>(), at::cuda::getCurrentCUDAStream());
    return {dist1, idx1, dist2, idx2};
}

std::vector<at::Tensor> NNDistanceGrad(at::Tensor set_d, at::Tensor set_q, at::Tensor idx1, at::Tensor idx2, at::Tensor grad_dist1, at::Tensor grad_dist2) {
    //std::cout << "[NNDistanceGrad] Called." << std::endl;
    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m
    //std::cout << "[NNDistanceGrad] batch_size:" << batch_size << std::endl;
    at::Tensor grad1 = torch::empty({batch_size,n_dataset_points,3}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor grad2 = torch::empty({batch_size,n_query_points,3}, torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(idx1);
    CHECK_INPUT(idx2);
    CHECK_INPUT(grad_dist1);
    CHECK_INPUT(grad_dist2);
    CHECK_INPUT(grad1);
    CHECK_INPUT(grad2);
    //void nndistancegrad(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream);
    nndistancegrad(batch_size,n_dataset_points,set_d.data<float>(),n_query_points,set_q.data<float>(),
        grad_dist1.data<float>(),idx1.data<int>(),
        grad_dist2.data<float>(),idx2.data<int>(),
        grad1.data<float>(),grad2.data<float>(),
        at::cuda::getCurrentCUDAStream());
    return {grad1, grad2};
}

