/*

    TO DO: 
        - 8 paths: E, SE, S, SW, W, NW, N, NE
        - use r notation to indicate them


    Lr(p,d) = C(p,d) + min(c1,c2,c3,c4)
 
        - c1: Lr(p-r,d)
        - c2: Lr(p-r,d+1) + P1
        - c3: Lr(p-r,d-1) + P1
        - c4: min Lr(p-r,i) + P2, where |d-i|>1
              i


*/

using namespace sgsm;


__device__ __constant__ unsigned int L0_L4_P[3];
__device__ __constant__ unsigned int L2_L6_P[3];
__device__ __constant__ unsigned int L1_L5_P[3];
__device__ __constant__ unsigned int L3_L7_P[3];



template <typename T>
__global__ void compute_directional_cost_L0_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int y = blockIdx.x;
    if(y>=height) return;


    for(int x=0;x<width;++x){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==0) continue;
            T c1 = out[x-1+y*width+d*height*width];
            T c2 = d==0               ? INF : out[x-1+y*width+(d-1)*height*width] + L0_L4_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x-1+y*width+(d+1)*height*width] + L0_L4_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x-1+y*width+j*height*width]+L0_L4_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }

    }
}


template <typename T>
__global__ void compute_directional_cost_L4_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){


    const int y = blockIdx.x;
    if(y>=height) return;


    for(int x=width-1;x>=0;--x){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==width-1) continue;
            T c1 = out[x+1+y*width+d*height*width];
            T c2 = d==0               ? INF : out[x+1+y*width+(d-1)*height*width] + L0_L4_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x+1+y*width+(d+1)*height*width] + L0_L4_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x+1+y*width+j*height*width]+L0_L4_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }

    }
}




template <typename T>
__global__ void compute_directional_cost_L2_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int x = blockIdx.x;
    if(x>=width) return;



    for(int y=0;y<height;++y){
        for(int d=0;d<num_disparity;d++){
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(y==0) continue;
            T c1 = out[x+(y-1)*width+d*height*width];
            T c2 = d==0 ?               INF : out[x+(y-1)*width+(d-1)*height*width] + L2_L6_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x+(y-1)*width+(d+1)*height*width] + L2_L6_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x+(y-1)*width+j*height*width]+L2_L6_P[2]);
            }
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
        }
        
    }
}



template <typename T>
__global__ void compute_directional_cost_L6_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int x = blockIdx.x;
    if(x>=width) return;

    for(int y=height-1;y>=0;--y){
        for(int d=0;d<num_disparity;d++){
        //ignore the pixels on the border
            // Lr(p,d) = C(p,d)
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            if(y==height-1) continue;
            T c1 = out[x+(y+1)*width+d*height*width];
            T c2 = d==0               ? INF : out[x+(y+1)*width+(d-1)*height*width] + L2_L6_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x+(y+1)*width+(d+1)*height*width] + L2_L6_P[1];
    
            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x+(y+1)*width+j*height*width]+L2_L6_P[2]);
            }
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
        }
        
    }
}



template <typename T>
__global__ void compute_directional_cost_L1_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int diag = blockIdx.x;
    if(diag>=height+width-1) return;


    int x;
    int y;
    if(diag>=width){
        x = 0;
	int q = diag/width;
	int r = diag%width;
        y = (q-1)*width+r+1;
    }
    else{
        x = diag;
        y = 0;
    }


    while(x<width && y<height){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==0 || y== 0) continue;
            T c1 = out[x-1+(y-1)*width+d*height*width];
            T c2 = d==0               ? INF : out[x-1+(y-1)*width+(d-1)*height*width] + L1_L5_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x-1+(y-1)*width+(d+1)*height*width] + L1_L5_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x-1+(y-1)*width+j*height*width]+L1_L5_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }
        ++x;
	++y;
    }
}



template <typename T>
__global__ void compute_directional_cost_L5_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int diag = blockIdx.x;
    if(diag>=height+width-1) return;


    int x;
    int y;
    if(diag>=width){
        x = width-1;
	int q = diag/width;
	int r = diag%width;
        y = (q-1)*width+r;

    }
    else{
        x = diag;
        y = height-1;
    }


    while(x>=0 && y>=0){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==width-1 || y== height-1) continue;
            T c1 = out[x+1+(y+1)*width+d*height*width];
            T c2 = d==0               ? INF : out[x+1+(y+1)*width+(d-1)*height*width] + L1_L5_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x+1+(y+1)*width+(d+1)*height*width] + L1_L5_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x+1+(y+1)*width+j*height*width]+L1_L5_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }
        --x;
	--y;
    }
}



template <typename T>
__global__ void compute_directional_cost_L3_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int diag = blockIdx.x;
    if(diag>=height+width-1) return;


    int x;
    int y;
    if(diag>=width){
        x = width-1;
	int q = diag/width;
	int r = diag%width;
        y = (q-1)*width+r+1;
    }
    else{
        x = diag;
        y = 0;
    }


    while(x>=0 && y<height){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==width-1 || y== 0) continue;
            T c1 = out[x+1+(y-1)*width+d*height*width];
            T c2 = d==0               ? INF : out[x+1+(y-1)*width+(d-1)*height*width] + L3_L7_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x+1+(y-1)*width+(d+1)*height*width] + L3_L7_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x+1+(y-1)*width+j*height*width]+L3_L7_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }
        --x;
	++y;
    }
}



template <typename T>
__global__ void compute_directional_cost_L7_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 ){

    const int diag = blockIdx.x;
    if(diag>=height+width-1) return;


    int x;
    int y;
    if(diag>=width){
        x = 0;
	int q = diag/width;
	int r = diag%width;
        y = (q-1)*width+r;
    }
    else{
        x = diag;
        y = height-1;
    }


    while(x<width && y>=0){
        for(int d=0;d<num_disparity;d++){
            //printf("      (x,y,d): %d,%u,%d     cost val: %u\n",x,y,d,cost_volume[x+y*width+d*height*width]);
            out[x+y*width+d*height*width] = cost_volume[x+y*width+d*height*width];

            //ignore the pixels on the border
            if(x==0 || y== height-1) continue;
            T c1 = out[x-1+(y+1)*width+d*height*width];
            T c2 = d==0               ? INF : out[x-1+(y+1)*width+(d-1)*height*width] + L3_L7_P[1];
            T c3 = d==num_disparity-1 ? INF : out[x-1+(y+1)*width+(d+1)*height*width] + L3_L7_P[1];

            T c4 = INF;
            for(int j=0;j<num_disparity;++j){
                if(j==d || j==d-1 || j==d+1)continue;
                c4 = min(c4,out[x-1+(y+1)*width+j*height*width]+L3_L7_P[2]);
            }
        
            out[x+y*width+d*height*width] += min(min(c1,c2),min(c3,c4));
            //printf("        Lr: %u\n", out[x+y*width+d*height*width]);
            //printf("            c1: %u    c2: %u   c3: %u    c4: %u\n", c1,c2,c3,c4);
        }
        ++x;
	--y;
    }
}


template <typename T>
__global__ void inc_sum_up_cost_cuda(T * out,
                                     T * cost_volme_L,
                                     unsigned long height,
                                     unsigned long width,
                                     int num_disparity,
                                     unsigned int weight
                                     ){
    
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x>=width || y>=height) return;

    for(int d=0;d<num_disparity;++d){
        out[x+y*width+d*height*width] += weight*cost_volme_L[x+y*width+d*height*width];
    }


}



template <typename T>
__global__ void gen_disparity_map_cuda(unsigned int * out,
                                       T * cost_volume,
                                       unsigned long height,
                                       unsigned long width,
                                       int num_disparity
                                       ){
    
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x>=width || y>=height) return;

    T min_v = cost_volume[x+y*width];
    int idx = 0;

    for(int d=1;d<num_disparity;++d){
        if(min_v>cost_volume[x+y*width+d*height*width]){
            min_v = cost_volume[x+y*width+d*height*width];
            idx = d;
        }
    }

    out[x+y*width] = idx;

}
