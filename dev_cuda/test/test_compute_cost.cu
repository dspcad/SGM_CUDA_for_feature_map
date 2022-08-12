#include "semi_global_matching.hpp"

using namespace std;
using namespace sgsm;

int main(int argc, char* argv[]){
    cout << "Test Case for compute_cost_cuda" << endl;

    const unsigned long h = 5;
    const unsigned long w = 5;
    const unsigned long d = 3;
    
    const int kernel_size = 3;

    const int N = h*w*BITMAP_SIZE;
    unsigned int left_census_5x5x32[N]={0}, right_census_5x5x32[N]={0};

    left_census_5x5x32[1+1*w+0*h*w] = 1;
    left_census_5x5x32[1+2*w+0*h*w] = 2;
    left_census_5x5x32[1+3*w+0*h*w] = 3;
    left_census_5x5x32[1+4*w+0*h*w] = 4;
    left_census_5x5x32[1+5*w+0*h*w] = 5;
    left_census_5x5x32[1+6*w+0*h*w] = 6;
    left_census_5x5x32[1+7*w+0*h*w] = 7;
    left_census_5x5x32[1+8*w+0*h*w] = 8;
    left_census_5x5x32[1+9*w+0*h*w] = 9;


    right_census_5x5x32[1+1*w+0*h*w] = 9;
    right_census_5x5x32[1+2*w+0*h*w] = 8;
    right_census_5x5x32[1+3*w+0*h*w] = 7;
    right_census_5x5x32[1+4*w+0*h*w] = 6;
    right_census_5x5x32[1+5*w+0*h*w] = 5;
    right_census_5x5x32[1+6*w+0*h*w] = 4;
    right_census_5x5x32[1+7*w+0*h*w] = 3;
    right_census_5x5x32[1+8*w+0*h*w] = 2;
    right_census_5x5x32[1+9*w+0*h*w] = 1;



    const int out_N = h*w*d;
    unsigned int out[out_N];


    unsigned int *d_left_census, *d_right_census;
    unsigned int *d_out;

    cudaMalloc((void **)&d_left_census,  sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_right_census, sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_out,          sizeof(unsigned int)*out_N);

    cudaMemcpy(d_left_census,  left_census_5x5x32,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_census, right_census_5x5x32, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);


    const dim3 threadsPerBlock(h, w);
    compute_cost_cuda<<<threadsPerBlock,1>>>(d_out,
                                             d_left_census,
                                             d_right_census,
                                             h,
                                             w,
                                             kernel_size,
                                             kernel_size,
                                             d,
                                             REF_IMG::LEFT
                                             );




    cudaMemcpy(out,d_out,sizeof(unsigned int)*out_N,cudaMemcpyDeviceToHost);


   
    for(int i=0;i<w;++i){
        for(int j=0;j<h;++j){
            printf("Pixel (%d,%d): \n",i,j);
            for(int k=0;k<d;++k)
                printf("    d[%d]: %d\n",k,out[i+j*w+k*h*w]);
        }
    }


    cudaFree(d_left_census);
    cudaFree(d_right_census);
    cudaFree(d_out);


    return 0;
}
