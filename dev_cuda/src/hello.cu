#include "semi_global_matching.hpp"



__global__ void cuda_hello(){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from GPU %d!\n", tid);
}

