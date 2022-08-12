#include "semi_global_matching.hpp"
#include "compute_directional_cost.hpp"

using namespace std;
using namespace sgsm;

int main(int argc, char* argv[]){
    cout << "Test Case for inc_sum_up_cost_cuda" << endl;

    const unsigned long h = 4;
    const unsigned long w = 3;
    const unsigned long d = 2;
    

    const int N = h*w*d;
    unsigned int L0[N]={ 444,   56,  302,  846,  314,  774,  226,  648,  669,  572,  976,  603,  857,   83,  523,  794,   73,  584,  181,  110,  237,  924,  647,  205};
    unsigned int L2[N]={ 875,  868,  976,  743,  385,  962,  548,  639,  368,   30,  197,  725,   19,  839,  553,  825,  585,  911,  591,  791,    7,  533,  342,  262};
    unsigned int L4[N]={ 317,  552,   74,  529,  597,  384,  845,   53,  757,  730,  700,  374,  655,  599,  493,  240,  390,  349,  333,  886,  519,   83,  424,  526};
    unsigned int L6[N]={ 288,  593,  643,  869,  566,   72,  437,   33,  878,  667,  516,  810,  955,  748,  234,   58,   27,  872,  127,  149,  863,  323,   94,  693};
    /*
                        1924  2069  1995  2987  1862  2192  2056  1373  2672  1999  2389  2512  2486  2269  1803  1917  1075  2716  1232  1936  1626  1863  1507  1686  
    */
    


    unsigned int out[N];


    unsigned int *d_L0, *d_L2, *d_L4, *d_L6;
    unsigned int *d_out;

    cudaMalloc((void **)&d_L0,  sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_L2,  sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_L4,  sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_L6,  sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_out, sizeof(unsigned int)*N);
    cudaMemset(d_out, 0, sizeof(unsigned int)*N);

    cudaMemcpy(d_L0,  L0,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L2,  L2,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L4,  L4,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L6,  L6,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);


    const dim3 threadsPerBlock(w, h);
    inc_sum_up_cost_cuda<<<threadsPerBlock,1>>>(d_out,
                                                d_L0,
                                                h,
                                                w,
                                                d,
                                                weight_L0_L4
                                                );
    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);


   
    for(int i=0;i<N;++i){
        //printf("out[%d]: %d\n",i,out[i]);
        printf("%4d  ",out[i]);
    }
    printf("\n");


    inc_sum_up_cost_cuda<<<threadsPerBlock,1>>>(d_out,
                                                d_L2,
                                                h,
                                                w,
                                                d,
                                                weight_L2_L6
                                                );
    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);


   
    for(int i=0;i<N;++i){
        //printf("out[%d]: %d\n",i,out[i]);
        printf("%4d  ",out[i]);
    }
    printf("\n");

    inc_sum_up_cost_cuda<<<threadsPerBlock,1>>>(d_out,
                                                d_L4,
                                                h,
                                                w,
                                                d,
                                                weight_L0_L4
                                                );

    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);


   
    for(int i=0;i<N;++i){
        //printf("out[%d]: %d\n",i,out[i]);
        printf("%4d  ",out[i]);
    }
    printf("\n");

    inc_sum_up_cost_cuda<<<threadsPerBlock,1>>>(d_out,
                                                d_L6,
                                                h,
                                                w,
                                                d,
                                                weight_L2_L6
                                                );

    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);


   
    for(int i=0;i<N;++i){
        //printf("out[%d]: %d\n",i,out[i]);
        printf("%4d  ",out[i]);
    }
    printf("\n");


    cudaFree(d_L0);
    cudaFree(d_L2);
    cudaFree(d_L4);
    cudaFree(d_L6);
    cudaFree(d_out);


    return 0;
}
