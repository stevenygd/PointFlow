std::vector<at::Tensor> ApproxMatch(at::Tensor in_a, at::Tensor in_b);
at::Tensor MatchCost(at::Tensor set_d, at::Tensor set_q, at::Tensor match);
std::vector<at::Tensor> MatchCostGrad(at::Tensor set_d, at::Tensor set_q, at::Tensor match);

std::vector<at::Tensor> NNDistance(at::Tensor set_d, at::Tensor set_q);
std::vector<at::Tensor> NNDistanceGrad(at::Tensor set_d, at::Tensor set_q, at::Tensor idx1, at::Tensor idx2, at::Tensor grad_dist1, at::Tensor grad_dist2);
