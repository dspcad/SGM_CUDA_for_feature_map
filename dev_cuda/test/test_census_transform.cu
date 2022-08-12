#include "semi_global_matching.hpp"

using namespace std;
using namespace sgsm;

int main(int argc, char* argv[]){
    cout << "Test Case 1" << endl;


    //                                             center
    //test_3x3x8[0] = { 2.69, -0.28,  1.62, -0.08,  0.67,  0.35,  1.67, -1.31, -0.62};
    //test_3x3x8[1] = {-1.95,  1.16,  1.46,  0.5,   0.51,  1.68, -0.25,  2.01,  2.64};
    //test_3x3x8[2] = { 0.67, -1.22,  1.75, -0.89, -0.14, -0.37,  0.26,  2.25, -0.48};
    //test_3x3x8[3] = {-1.07,  0.13,  1.64,  0.4,   0.19,  2.85, -0.81,  0.81, -0.88};
    //test_3x3x8[4] = { 1.57,  1.92,  0.18,  0.22, -1.82,  3.08,  0.4,   2.05, -2.43};
    //test_3x3x8[5] = { 3,     0.89,  2.17,  3.11,  0.71,  2.12,  0.35,  1.55, -0.37};
    //test_3x3x8[6] = {-2.52,  0.58,  0.29,  3.38,  0.98, -2.99,  0.14,  0.75,  1.9};
    //test_3x3x8[7] = { 0.35,  2.97,  1.29,  1.78,  0.65,  3.71,  3.65, -1.64, -1.37};

    const unsigned long h = 3;
    const unsigned long w = 3;
    const unsigned long c = 8;
    
    const int kernel_size = 3;

    const int N = h*w*c;
    double feat_map_3x3x8[N] = { 2.69, -0.28,  1.62, -0.08,  0.67,  0.35,  1.67, -1.31, -0.62,
                                -1.95,  1.16,  1.46,  0.5,   0.51,  1.68, -0.25,  2.01,  2.64,
                                 0.67, -1.22,  1.75, -0.89, -0.14, -0.37,  0.26,  2.25, -0.48,                           
                                -1.07,  0.13,  1.64,  0.4,   0.19,  2.85, -0.81,  0.81, -0.88,
                                 1.57,  1.92,  0.18,  0.22, -1.82,  3.08,  0.4,   2.05, -2.43,
                                 3,     0.89,  2.17,  3.11,  0.71,  2.12,  0.35,  1.55, -0.37,
                                -2.52,  0.58,  0.29,  3.38,  0.98, -2.99,  0.14,  0.75,  1.9,
                                 0.35,  2.97,  1.29,  1.78,  0.65,  3.71,  3.65, -1.64, -1.37
                               };
    const int out_N = h*w*BITMAP_SIZE;
    unsigned int out[out_N];


    double       *d_feat_map;
    unsigned int *d_out;

    cudaMalloc((void **)&d_feat_map, sizeof(double)*N); 
    cudaMalloc((void **)&d_out,      sizeof(unsigned int)*out_N);

    cudaMemcpy(d_feat_map, feat_map_3x3x8, sizeof(double)*N, cudaMemcpyHostToDevice);


    const dim3 threadsPerBlock(kernel_size, kernel_size);
    census_transform_cuda<<<threadsPerBlock,1>>>(d_out,
                                                 d_feat_map,
                                                 c,
                                                 h,
                                                 w,
                                                 kernel_size,
                                                 kernel_size
                                                 );

    cudaMemcpy(out,d_out,sizeof(unsigned int)*out_N,cudaMemcpyDeviceToHost);


    /*

    Expect to see only the census transform result of pixel (1,1)

    Example of the channel 0:
        center val = 0.67

                             census transform         1-D pattern

        2.69  -0.28  1.68        1  0  1
       -0.08   0.67  0.35   =>   0  1  0          => 1 0 1 0 1 0 1 0 0
        1.67  -1.31 -0.62        1  0  0  


    Example of the channel 1:
        center val = 0.51

                             census transform         1-D pattern

       -1.95   1.16  1.46        0  1  1
        0.5    0.51  1.68   =>   0  1  1          => 0 1 1 0 1 1 0 1 1
       -0.25   2.01  2.64        0  1  1  


   */


//    for(int i=0;i<w;++i){
//        for(int j=0;j<h;++j){
//            printf("Pixel (%d,%d):\n",i,j);
//            for(int ch=0;ch<BITMAP_SIZE;ch++){
//                printf("  ch %d\n",ch);
//                int idx = i+j*w+ch*h*w;
//
//                for(int k=0;k<32;++k){
//                    int pos = k+ch*32;
//                    if(pos%9==0)printf("\n");
//
//                    int mask = 1 << k;
//                    int v = out[idx] & mask;
//
//                    if(v==0)
//                        printf("0 ");
//                    else 
//                        printf("1 ");
//
//                }
//                printf("\n");
//            }
//            printf("=================\n\n\n");
//        }
//    }

    cudaFree(d_feat_map);
    cudaFree(d_out);


    return 0;
}
