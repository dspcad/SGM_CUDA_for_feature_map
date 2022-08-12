#include "semi_global_matching.hpp"



/*

                -------------
               /            /
  BITMAP_SIZE /            / |
             /            /  |
             -------------   |
             |   width   |   |
             |           |   |
      height |           |   |
             |           |  /
             |           | /
             -------------


    BITMAP_SIZE is set to 32 by default which means 
    there 1024 bits to store the patterns of census
    transform of each channel

    Along the channel, the pattern is represented by
    binary like 10001010....
*/

__global__ void census_transform_cuda(unsigned int *out, 
                                      const double *feat_map, 
                                      unsigned long channel,
                                      unsigned long height,
                                      unsigned long width,
                                      int kernel_h,
                                      int kernel_w
                                      ){

    //printf("census transform GPU: \n");
    //printf("     channel: %ld    height: %ld    width: %ld\n", channel,height,width);
    //printf("     kernel_h: %d\n", kernel_h);
    //printf("     kernel_w: %d\n", kernel_w);

    //__syncthreads();

    //printf("block dim: (%d,%d):\n",blockDim.x, blockDim.y);
    //if(blockIdx.y>10)
    //printf("block (%d,%d):\n",blockIdx.x,blockIdx.y);
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x<kernel_w/2 || x>=width-kernel_w/2) return;
    if(y<kernel_h/2 || y>=height-kernel_h/2) return;

    //printf("(%d,%d):\n",x,y);


    unsigned int *patch;
    cudaError_t code;
    code = cudaMalloc((void**)&patch,  sizeof(unsigned int)*BITMAP_SIZE);
    if(code==cudaErrorMemoryAllocation) printf("debug: image path is out of memory %ld\n", sizeof(unsigned int)*BITMAP_SIZE);
    //memset(patch, 0 , sizeof(unsigned int)*BITMAP_SIZE);
    for(int c=0;c<BITMAP_SIZE;++c) patch[c]=0;


    //compute the census transform of left image
    const int shift_x = x - kernel_w/2;
    const int shift_y = y - kernel_h/2;
    for(int c=0;c<channel;++c){
        double center_val = feat_map[x+y*width+c*height*width];

        //printf("channel %d    target val: %f\n",c, center_val);
        for(int h=0;h<kernel_h;++h){
            for(int w=0;w<kernel_w;++w){
                double neighbor = feat_map[(shift_x+w)+(shift_y+h)*width+c*height*width];
                
                int idx = w+h*kernel_w+c*kernel_h*kernel_w;
                int q   = idx / UNSIGNED_INT_SIZE;
                int r   = idx % UNSIGNED_INT_SIZE;

                unsigned int bit = neighbor >= center_val ? 1 : 0;

                patch[q] = patch[q] | (bit << r);
                //printf("    (%d %d) = %d (%f)   q:%d   r:%d\n", h,w,bit, neighbor, q,r);
                //printf("              pathc[%d] = %u\n", q, patch[q]);
            }
        }
    }


    //printf("c:%ld  h:%ld   w:%ld    kernel_h:%d   kernel_w:%d\n",channel,height,width,kernel_h,kernel_w);
    for(int c=0;c<BITMAP_SIZE;++c){
        //printf("Writing into %d %d %d = %ld\n", x,y,c, x+y*width+c*height*width);
        out[x+y*width+c*height*width] = patch[c];
    }
    cudaFree(patch);
}


