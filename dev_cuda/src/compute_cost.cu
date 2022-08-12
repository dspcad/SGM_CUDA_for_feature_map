#include "semi_global_matching.hpp"




/*

                    -------------
                   /            /
    num_disparity /            / |
                 /            /  |
                 -------------   |
                 |   width   |   |
                 |           |   |
          height |           |   |
                 |           |  /
                 |           | /
                 -------------



    cost: 
        The Hamming distance between the pixel in left and the pixle in right


    Example:
        left:   11101000
        right:  11110100
    XOR -----------------
                00011100  ---> Haming distance is 3
*/

using namespace sgsm;



__device__ int getHammingDist(unsigned int left_pixel_census[BITMAP_SIZE],
                              unsigned int right_pixel_census[BITMAP_SIZE]
                              ){
    int res = 0;
    for(int c=0;c<BITMAP_SIZE;++c){
        unsigned int xor_res = left_pixel_census[c] ^ right_pixel_census[c];
        for(int i=0;i<32;++i){
            if(xor_res & (1<<i))
                ++res;
        }
    }

    return res;
}

__global__ void compute_cost_cuda(unsigned int *out, 
                                  unsigned int *left, 
                                  unsigned int *right, 
                                  unsigned long height,
                                  unsigned long width,
                                  int kernel_h,
                                  int kernel_w,
                                  int num_disparity,
                                  REF_IMG opt
                                  ){

    //printf("GPU: kernel size: %d\n", kernel);

    //__syncthreads();

    //printf("block dim: (%d,%d):\n",blockDim.x, blockDim.y);
    //if(blockIdx.y>10)
    //printf("block (%d,%d):\n",blockIdx.x,blockIdx.y);
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x>=width || y>=height) return;


    //printf("(%d,%d):\n",x,y);

    //The target left pixel census

    if(opt==REF_IMG::LEFT){
        unsigned int left_pixel_census[BITMAP_SIZE];
        for(int c=0;c<BITMAP_SIZE;++c) left_pixel_census[c] = left[x+y*width+c*height*width];



        for(int d=0;d<num_disparity;d++){
            if(x-d-disp_offset<0){
                out[x+y*width+d*height*width] = INF;
            }
            else{
                unsigned int right_pixel_census[BITMAP_SIZE];
                for(int c=0;c<BITMAP_SIZE;++c) right_pixel_census[c]=right[x-d-disp_offset+y*width+c*height*width];

                out[x+y*width+d*height*width] = getHammingDist(left_pixel_census,right_pixel_census);
            }
        }

    }
    else if(opt==REF_IMG::RIGHT){
        unsigned int right_pixel_census[BITMAP_SIZE];
        for(int c=0;c<BITMAP_SIZE;++c) right_pixel_census[c] = right[x+y*width+c*height*width];



        for(int d=0;d<num_disparity;d++){
            if(x+d+disp_offset>=width){
                out[x+y*width+d*height*width] = INF;
            }
            else{
                unsigned int left_pixel_census[BITMAP_SIZE];
                for(int c=0;c<BITMAP_SIZE;++c) left_pixel_census[c]=left[x+d+disp_offset+y*width+c*height*width];

                out[x+y*width+d*height*width] = getHammingDist(right_pixel_census,left_pixel_census);
            }
        }

    }
}


