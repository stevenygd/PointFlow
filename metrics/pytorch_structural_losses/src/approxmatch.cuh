/*
template <typename Dtype>
void AddGPUKernel(Dtype *in_a, Dtype *in_b, Dtype *out_c, int N,
                  cudaStream_t stream);
*/
void approxmatch(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp, cudaStream_t stream);
void matchcost(int b,int n,int m,const float * xyz1,const float * xyz2,float * match, float * out, cudaStream_t stream);
void matchcostgrad(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2, cudaStream_t stream);
