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
        The SSD/SAD between the pixel in left and the pixle in right


*/

using namespace sgsm;

__device__ double getSumSquareDiff(double *left_patch,
                                   double *right_patch,
                                   unsigned long channel,
                                   int kernel_h,
                                   int kernel_w
                                   ){
    double res = 0;
    for(int c=0;c<channel;++c){
        for(int w=0;w<kernel_w;++w)
            for(int h=0;h<kernel_h;++h){
		double diff = left_patch[w+h*kernel_w+c*kernel_h*kernel_w]-right_patch[w+h*kernel_w+c*kernel_h*kernel_w];
		res += (diff)*(diff);
		//printf("debug: %.2f - %.2f = %.2f   ==>  %.2f\n",left_patch[w+h*kernel_w+c*kernel_h*kernel_w],right_patch[w+h*kernel_w+c*kernel_h*kernel_w],diff,diff*diff);
	    }
    }

    return res;
}

__device__ double getSumAbsoluteDiff(double *left_patch,
                                     double *right_patch,
                                     unsigned long channel,
                                     int kernel_h,
                                     int kernel_w
                                     ){
    double res = 0;
    for(int c=0;c<channel;++c){
        for(int w=0;w<kernel_w;++w)
            for(int h=0;h<kernel_h;++h){
		res += abs(left_patch[w+h*kernel_w+c*kernel_h*kernel_w]-right_patch[w+h*kernel_w+c*kernel_h*kernel_w]);
	    }
    }

    return res;
}


__global__ void sum_diff_transform_cuda(double *out, 
                                        const double *left_feat_map, 
                                        const double *right_feat_map, 
                                        unsigned long channel,
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
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x<kernel_w/2 || x>=width-kernel_w/2) return;
    if(y<kernel_h/2 || y>=height-kernel_h/2) return;


    //printf("(%d,%d):     cheannel: %lu    height: %lu   width: %lu \n",x,y,channel,height,width);

    cudaError_t code;
    double *left_patch, *right_patch;
    code = cudaMalloc((void**)&left_patch,  sizeof(double)*kernel_h*kernel_w*channel);
    if(code==cudaErrorMemoryAllocation) printf("debug: left path is out of memory %ld\n", sizeof(double)*kernel_h*kernel_w*channel);
    code = cudaMalloc((void**)&right_patch, sizeof(double)*kernel_h*kernel_w*channel);
    if(code==cudaErrorMemoryAllocation) printf("debug: right path is out of memory %ld\n", sizeof(double)*kernel_h*kernel_w*channel);




    const int shift_x = x - kernel_w/2;
    const int shift_y = y - kernel_h/2;
    if(opt==REF_IMG::LEFT){
        for(int c=0;c<channel;++c){
            for(int h=0;h<kernel_h;++h){
                for(int w=0;w<kernel_w;++w){
                    left_patch[w+h*kernel_w+c*kernel_h*kernel_w]=left_feat_map[(shift_x+w)+(shift_y+h)*width+c*height*width];

		    //if(x==3 && y==3)
		    //    printf("%.2f ", left_patch[w+h*kernel_w+c*kernel_h*kernel_w]);
                }
	    }
	    //if(x==3 && y==3)
	    //    printf("\n");
	}



        for(int d=0;d<num_disparity;d++){
            if(x - d - kernel_w/2 <0){
                out[x+y*width+d*height*width] = INF;
            }
            else{
                for(int c=0;c<channel;++c){
                    for(int h=0;h<kernel_h;++h){
                        for(int w=0;w<kernel_w;++w){
                            right_patch[w+h*kernel_w+c*kernel_h*kernel_w]=right_feat_map[(shift_x-d+w)+(shift_y+h)*width+c*height*width];


			    //if(x==3 && y==3){
		            //    printf("%.2f ", right_patch[w+h*kernel_w+c*kernel_h*kernel_w]);
			    //}
                        }
		    }

	            //if(x==3 && y==3)
	            //    printf("\n");
		}

                out[x+y*width+d*height*width] = getSumSquareDiff(left_patch,right_patch,channel,kernel_h,kernel_w);
            }
        }

    }
    else if(opt==REF_IMG::RIGHT){

        for(int c=0;c<channel;++c)
            for(int h=0;h<kernel_h;++h)
                for(int w=0;w<kernel_w;++w){
		    //search toward left direction
                    right_patch[w+h*kernel_w+c*kernel_h*kernel_w]=right_feat_map[(shift_x+w)+(shift_y+h)*width+c*height*width];
                }



        for(int d=0;d<num_disparity;d++){
            if(x + kernel_w/2 +d >=width){
                out[x+y*width+d*height*width] = INF;
            }
            else{
                for(int c=0;c<channel;++c)
                    for(int h=0;h<kernel_h;++h)
                        for(int w=0;w<kernel_w;++w){
		            //search toward right direction
                            left_patch[w+h*kernel_w+c*kernel_h*kernel_w]=left_feat_map[(shift_x+d+w)+(shift_y+h)*width+c*height*width];
                        }


                out[x+y*width+d*height*width] = getSumSquareDiff(left_patch,right_patch,channel,kernel_h,kernel_w);

            }
        }
    }

    cudaFree(left_patch);
    cudaFree(right_patch);

}


