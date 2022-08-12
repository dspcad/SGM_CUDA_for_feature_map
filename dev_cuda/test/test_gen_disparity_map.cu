#include "semi_global_matching.hpp"
#include "compute_directional_cost.hpp"

using namespace std;
using namespace sgsm;

int main(int argc, char* argv[]){
    cout << "Test Case for sum_up_cost_cuda" << endl;

    const unsigned long h = 4;
    const unsigned long w = 3;
    const unsigned long d = 2;
    

    const int N = h*w*d;
    unsigned int cost_volume_0[N]={ 444,   56,  302,  846,  314,  774,  226,  648,  669,  572,  976,  603,
                                    857,   83,  523,  794,   73,  584,  181,  110,  237,  924,  647,  205};
//                                    0     0     0     1     1     1     1     1     1     0     1     1

    unsigned int cost_volume_1[N]={ 875,  868,  976,  743,  385,  962,  548,  639,  368,   30,  197,  725,
                                     19,  839,  553,  825,  585,  911,  591,  791,    7,  533,  342,  262};
//                                    1     1     1     0     0     1     0     0     1     0     0     1  

    unsigned int cost_volume_2[N]={ 317,  552,   74,  529,  597,  384,  845,   53,  757,  730,  700,  374,
                                    655,  599,  493,  240,  390,  349,  333,  886,  519,   83,  424,  526};
//                                    0     0     0     1     1     1     1     0     1     1     1     0  


    unsigned int cost_volume_3[N]={ 288,  593,  643,  869,  566,   72,  437,   33,  878,  667,  516,  810,
                                    955,  748,  234,   58,   27,  872,  127,  149,  863,  323,   94,  693};
//                                    0     0     1     1     1     0     1     0     1     1     1     1


    unsigned int out[h*w];


    unsigned int *d_cost_volume;
    unsigned int *d_out;

    cudaMalloc((void **)&d_cost_volume, sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_out,         sizeof(unsigned int)*h*w);

    cudaMemcpy(d_cost_volume,  cost_volume_0,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);


    const dim3 threadsPerBlock(w, h);
    gen_disparity_map_cuda<<<threadsPerBlock,1>>>(d_out,
                                                  d_cost_volume,
                                                  h,
                                                  w,
                                                  d
                                                  );




    cudaMemcpy(out,d_out,sizeof(unsigned int)*h*w,cudaMemcpyDeviceToHost);


   
    printf("Test case 0\n");
    for(int i=0;i<h*w;++i){
        printf("%d  ",out[i]);
    }
    printf("\n");


    cudaMemcpy(d_cost_volume,  cost_volume_1,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    gen_disparity_map_cuda<<<threadsPerBlock,1>>>(d_out,
                                                  d_cost_volume,
                                                  h,
                                                  w,
                                                  d
                                                  );
    cudaMemcpy(out,d_out,sizeof(unsigned int)*h*w,cudaMemcpyDeviceToHost);

    printf("Test case 1\n");
    for(int i=0;i<h*w;++i){
        printf("%d  ",out[i]);
    }
    printf("\n");


    cudaMemcpy(d_cost_volume,  cost_volume_2,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    gen_disparity_map_cuda<<<threadsPerBlock,1>>>(d_out,
                                                  d_cost_volume,
                                                  h,
                                                  w,
                                                  d
                                                  );
    cudaMemcpy(out,d_out,sizeof(unsigned int)*h*w,cudaMemcpyDeviceToHost);

    printf("Test case 2\n");
    for(int i=0;i<h*w;++i){
        printf("%d  ",out[i]);
    }
    printf("\n");


    cudaMemcpy(d_cost_volume,  cost_volume_3,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    gen_disparity_map_cuda<<<threadsPerBlock,1>>>(d_out,
                                                  d_cost_volume,
                                                  h,
                                                  w,
                                                  d
                                                  );
    cudaMemcpy(out,d_out,sizeof(unsigned int)*h*w,cudaMemcpyDeviceToHost);

    printf("Test case 3\n");
    for(int i=0;i<h*w;++i){
        printf("%d  ",out[i]);
    }
    printf("\n");



    cudaFree(d_cost_volume);
    cudaFree(d_out);


    return 0;
}
