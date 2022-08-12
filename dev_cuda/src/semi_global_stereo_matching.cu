#include "semi_global_matching.hpp"
#include "compute_directional_cost.hpp"
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace sgsm;


// check whether machine is little endian
__host__ int littleendian()
{
    int intval = 1;
    uchar *uval = (uchar *)&intval;
    return uval[0] == 1;
}


__host__ void semi_global_stereo_matching::WriteFilePFM(float *data, int width, int height, const char* filename, float scalefactor=1/255.0)
{
    // Open the file
    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
        exit(1);
    }


    // sign of scalefact indicates endianness, see pfms specs
    if (littleendian())
        scalefactor = -scalefactor;


    // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
    fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

    int n = width;
    // write rows -- pfm stores rows in inverse order!
    for (int y = height-1; y >= 0; y--) {
        float* ptr = data + y * width;
        // change invalid pixels (which seem to be represented as -10) to INF
        for (int x = 0; x < width; x++) {
            if (ptr[x] < 0)
                ptr[x] = INFINITY;
        }
        if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
            fprintf(stderr, "WriteFilePFM: problem writing data\n");
            exit(1);
        }
    }
    
    // close file
    fclose(stream);
}


__host__ void CHECK_LAST_CUDA_ERROR(const char * kernel_name){
    cudaError_t cudaerr {cudaGetLastError()};
    
    printf("----- Kernel \"%s\" ----- \n", kernel_name);
    if (cudaerr != cudaSuccess){
        printf("    CUDA Runtime Error at \"%s\".\n", cudaGetErrorString(cudaerr));
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
    else
        printf("    No CUDA Runtime Error (No Synchronous Error)\n\n\n");

}

__host__ void CHECK_CUDA_ASYNC_ERROR(const char * kernel_name){
    cudaError_t cudaerr {cudaDeviceSynchronize()};
    //printf("Kernel \"%s\": \n", kernel_name);
    printf("----- Kernel \"%s\" ----- \n", kernel_name);
    if (cudaerr != cudaSuccess){
        printf("    kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    else
        printf("    Successfully launch the kernel (No Asynchronous Error)\n\n\n");

}



__host__ void semi_global_stereo_matching::inconsistence_check(unsigned long height,
                                                               unsigned long width,
                                                               int num_disparity
                                                               ){
    float total = height*width;
    //printf("h: %lu    w: %lu   disp: %d\n", height, width, num_disparity);
    //printf("Total number of pixels: %f\n",total);

    printf("------------------------------------\n");
    float cnt = 0;
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            if(x>=left_out[x+y*width] && left_out[x+y*width]!=right_out[x-left_out[x+y*width]+y*width])
                cnt++; 
        }

    printf("    left inconsisitence:  %f\n", cnt/total);
 
    cnt = 0;
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            if(x+right_out[x+y*width]<width && right_out[x+y*width]!=left_out[x+right_out[x+y*width]+y*width])
                cnt++; 
        }

    printf("    right inconsisitence: %f\n", cnt/total);

}





__global__ void sum_up_cost_cuda(unsigned int * out,
                                 unsigned int * cost_volme_L0,
                                 unsigned int * cost_volme_L2,
                                 unsigned int * cost_volme_L4,
                                 unsigned int * cost_volme_L6,
                                 unsigned long height,
                                 unsigned long width,
                                 int num_disparity
                                 ){
    
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x>=width || y>=height) return;


    for(int d=0;d<num_disparity;++d){
        out[x+y*width+d*height*width] = weight_L0_L4*cost_volme_L0[x+y*width+d*height*width] +
                                        weight_L2_L6*cost_volme_L2[x+y*width+d*height*width] +
                                        weight_L0_L4*cost_volme_L4[x+y*width+d*height*width] +
                                        weight_L2_L6*cost_volme_L6[x+y*width+d*height*width];
    }

}









__host__ void semi_global_stereo_matching::init_CUDA_HEAP(){
    size_t setsz = 20LL*1024LL*1024LL*1024LL;
    size_t getsz;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, setsz);
    cudaDeviceGetLimit(&getsz, cudaLimitMallocHeapSize);
    printf("Heap requested %ld got %ld\n", setsz, getsz);


    setsz = 1024LL*1024LL*1024LL;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, setsz);
    cudaDeviceGetLimit(&getsz, cudaLimitPrintfFifoSize);
    printf("Print buffer requested %ld got %ld\n", setsz, getsz);


}



__host__ void semi_global_stereo_matching::execute(const char * left_path,
                                                   const char * right_path,
                                                   int kernel_size,
                                                   int num_disparity,
                                                   REF_IMG opt
                                                   ){


    double *left  = nullptr;
    double *right = nullptr;

    std::vector<unsigned long> left_shape  = featAllocation(left_path, &left);
    std::vector<unsigned long> right_shape = featAllocation(right_path, &right);

    const unsigned long channel = left_shape[0];
    const unsigned long height  = left_shape[1];
    const unsigned long width   = left_shape[2];

    img_height = height;
    img_width  = width;


    unsigned long N     = getSize(left_shape);
    unsigned long out_census_transform_N = height*width*BITMAP_SIZE;



    //print_feat(left_shape,left);
    //print_feat(right_shape,right);

    double       *d_left,     *d_right;
    unsigned int *d_left_census, *d_right_census;


    cudaMalloc((void**)&d_left,      sizeof(double) * N);
    cudaMalloc((void**)&d_right,     sizeof(double) * N);
    cudaMalloc((void**)&d_left_census,  sizeof(int) * out_census_transform_N);
    cudaMalloc((void**)&d_right_census, sizeof(int) * out_census_transform_N);


    cudaMemcpy(d_left,  left,  sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, sizeof(double) * N, cudaMemcpyHostToDevice);


    constexpr int block_size {32};



    printf("DIM: %ld %ld\n",(block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 numBlocks((block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 threadsPerBlock(block_size, block_size);


    //cuda_hello<<<1,1>>>();
    //cudaDeviceSynchronize();

    cout << "Calling census transform for left" << endl;
    census_transform_cuda<<<numBlocks,threadsPerBlock>>>(d_left_census,
                                                         d_left,
                                                         channel,
                                                         height,
                                                         width,
                                                         kernel_size,
                                                         kernel_size
                                                         );

    CHECK_LAST_CUDA_ERROR("census_transform_cuda for left feat map");
    CHECK_CUDA_ASYNC_ERROR("census_transform_cuda for left feat map");




    cout << "Calling census transform for right" << endl;
    census_transform_cuda<<<numBlocks,threadsPerBlock>>>(d_right_census,
                                                         d_right,
                                                         channel,
                                                         height,
                                                         width,
                                                         kernel_size,
                                                         kernel_size
                                                         );

    CHECK_LAST_CUDA_ERROR("census_transform_cuda for right feat map");
    CHECK_CUDA_ASYNC_ERROR("census_transform_cuda for right feat map");




    cudaFree(d_left);
    cudaFree(d_right);

    size_t free_sz, total_sz;
    cudaMemGetInfo(&free_sz, &total_sz);
    cout << "free:  " << free_sz/1000000 << "\ntotal: " << total_sz/1000000 << endl;



    cout << "Finish GPU kernel" << endl;

    //cudaMemcpy(left_census,  d_left_out,  sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(right_census, d_right_out, sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);


    unsigned int *d_cost_volume;
    cudaMalloc((void **)&d_cost_volume, sizeof(unsigned int)*num_disparity*height*width);

    compute_cost_cuda<<<numBlocks,threadsPerBlock>>>(d_cost_volume,
                                                     d_left_census,
                                                     d_right_census,
                                                     height,
                                                     width,
                                                     kernel_size,
                                                     kernel_size,
                                                     num_disparity,
                                                     opt
                                                     );


    CHECK_LAST_CUDA_ERROR("compute cost with left census and right census");
    CHECK_CUDA_ASYNC_ERROR("compute cost with left census and right census");


    cudaFree(d_left_census);
    cudaFree(d_right_census);



    unsigned int *d_opt_cost_volume_L0, *d_opt_cost_volume_L4, *d_opt_cost_volume_L2, *d_opt_cost_volume_L6;
    cudaMalloc((void **)&d_opt_cost_volume_L0, sizeof(unsigned int)*num_disparity*height*width);
    cudaMalloc((void **)&d_opt_cost_volume_L4, sizeof(unsigned int)*num_disparity*height*width);
    cudaMalloc((void **)&d_opt_cost_volume_L2, sizeof(unsigned int)*num_disparity*height*width);
    cudaMalloc((void **)&d_opt_cost_volume_L6, sizeof(unsigned int)*num_disparity*height*width);



    const dim3 numHorizontalScanLines(height);
    compute_directional_cost_L0_cuda<unsigned int><<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L0,
                                                                   d_cost_volume,
                                                                   height,
                                                                   width,
                                                                   num_disparity
                                                                   );

    CHECK_LAST_CUDA_ERROR("compute L0 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L0 directional cost");


    compute_directional_cost_L4_cuda<<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L4,
                                                                   d_cost_volume,
                                                                   height,
                                                                   width,
                                                                   num_disparity
                                                                   );
    CHECK_LAST_CUDA_ERROR("compute L4 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L4 directional cost");



    const dim3 numVerticalScanLines(width);
    compute_directional_cost_L2_cuda<<<numVerticalScanLines,1>>>(d_opt_cost_volume_L2,
                                                                 d_cost_volume,
                                                                 height,
                                                                 width,
                                                                 num_disparity
                                                                 );

    CHECK_LAST_CUDA_ERROR("compute L2 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L2 directional cost");

    compute_directional_cost_L6_cuda<<<numVerticalScanLines,1>>>(d_opt_cost_volume_L6,
                                                                 d_cost_volume,
                                                                 height,
                                                                 width,
                                                                 num_disparity
                                                                 );

    CHECK_LAST_CUDA_ERROR("compute L6 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L6 directional cost");


    //reuse d_cost_volume
    sum_up_cost_cuda<<<numBlocks,threadsPerBlock>>>(d_cost_volume,
                                                    d_opt_cost_volume_L0,
                                                    d_opt_cost_volume_L2,
                                                    d_opt_cost_volume_L4,
                                                    d_opt_cost_volume_L6,
                                                    height,
                                                    width,
                                                    num_disparity
                                                    );

    CHECK_LAST_CUDA_ERROR("sum up 4 directional costs");
    CHECK_CUDA_ASYNC_ERROR("sum up 4 directional costs");
    cudaFree(d_opt_cost_volume_L0);
    cudaFree(d_opt_cost_volume_L4);
    cudaFree(d_opt_cost_volume_L2);
    cudaFree(d_opt_cost_volume_L6);





    int out_N         = height*width;
    unsigned int *out = (unsigned int *)malloc(sizeof(unsigned int)*out_N);

    unsigned int * d_out;
    cudaMalloc((void**)&d_out,   sizeof(unsigned int) * out_N);


    gen_disparity_map_cuda<<<numBlocks,threadsPerBlock>>>(d_out,
                                                          d_cost_volume,
                                                          height,
                                                          width,
                                                          num_disparity
                                                          );

    CHECK_LAST_CUDA_ERROR("generate the disparity map");
    CHECK_CUDA_ASYNC_ERROR("generate the disparity map");


    cudaMemcpy(out, d_out, sizeof(unsigned int) * out_N, cudaMemcpyDeviceToHost);
    Mat disparity_img(height,width, CV_8U);
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            //printf("r:%d   c:%d = %d\n",row,col,res.at<int>(row,col));
            //printf("x:%d   y:%d = %d\n",x,y, out[x+y*width]);
            //res.at<int>(y,x)=out[x+y*width];
            disparity_img.at<uint8_t>(y,x)=out[x+y*width]*256/num_disparity;
        }

    Mat color_disparity;
    applyColorMap(disparity_img, color_disparity, COLORMAP_JET);

    Mat median_filter_res = disparity_img.clone();
    medianBlur (disparity_img, median_filter_res, 7);

    Mat color_median_filter;
    applyColorMap(median_filter_res, color_median_filter, COLORMAP_JET);
    //GaussianBlur( res, dst, Size( kernel_size, kernel_size ), 0, 0 );
    //bilateralFilter ( res, dst, kernel_size, kernel_size*2, kernel_size/2 );


    if(opt==REF_IMG::LEFT){
        left_out = out;
        imwrite("./left_semi_global_disparity.png", color_disparity);
        imwrite("./left_semi_global_median_filter.png", color_median_filter);
    }
    else{
        right_out = out;
        imwrite("./right_semi_global_disparity.png", color_disparity);
        imwrite("./right_semi_global_median_filter.png", color_median_filter);
    }




    cudaFree(d_cost_volume);



    free(left);
    free(right);

}





__host__ void semi_global_stereo_matching::inc_execute(const char * left_path,
                                                       const char * right_path,
                                                       int kernel_size,
                                                       int num_disparity,
						       unsigned int P1,
						       unsigned int P2,
                                                       REF_IMG opt
                                                       ){

    unsigned int penality[3] = {0,100,1000};
    penality[1] = P1;
    penality[2] = P2;
    cudaMemcpyToSymbol(L0_L4_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L2_L6_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L1_L5_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L3_L7_P, penality, 3*sizeof(unsigned int));

    double *left  = nullptr;
    double *right = nullptr;

    std::vector<unsigned long> left_shape  = featAllocation(left_path, &left);
    std::vector<unsigned long> right_shape = featAllocation(right_path, &right);

    const unsigned long channel = left_shape[0];
    const unsigned long height  = left_shape[1];
    const unsigned long width   = left_shape[2];

    img_height = height;
    img_width  = width;

    unsigned long N     = getSize(left_shape);
    unsigned long out_census_transform_N = height*width*BITMAP_SIZE;
    unsigned int *left_census   = (unsigned int *)malloc(sizeof(unsigned int)*out_census_transform_N);
    unsigned int *right_census  = (unsigned int *)malloc(sizeof(unsigned int)*out_census_transform_N);


    

    //print_feat(left_shape,left);
    //print_feat(right_shape,right);

    double       *d_left,        *d_right;
    unsigned int *d_left_census, *d_right_census;


    cudaMalloc((void**)&d_left,      sizeof(double) * N);
    cudaMalloc((void**)&d_right,     sizeof(double) * N);
    cudaMalloc((void**)&d_left_census,  sizeof(int) * out_census_transform_N);
    cudaMalloc((void**)&d_right_census, sizeof(int) * out_census_transform_N);


    cudaMemcpy(d_left,  left,  sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, sizeof(double) * N, cudaMemcpyHostToDevice);


    constexpr int block_size {32};



    printf("DIM: %ld %ld\n",(block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 numBlocks((block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 threadsPerBlock(block_size, block_size);


    //cuda_hello<<<1,1>>>();
    //cudaDeviceSynchronize();

    cout << "Calling census transform for left" << endl;
    census_transform_cuda<<<numBlocks,threadsPerBlock>>>(d_left_census,
                                                         d_left,
                                                         channel,
                                                         height,
                                                         width,
                                                         kernel_size,
                                                         kernel_size
                                                         );
  
    CHECK_LAST_CUDA_ERROR("census_transform_cuda for left feat map");
    CHECK_CUDA_ASYNC_ERROR("census_transform_cuda for left feat map");




    cout << "Calling census transform for right" << endl;
    census_transform_cuda<<<numBlocks,threadsPerBlock>>>(d_right_census,
                                                         d_right,
                                                         channel,
                                                         height,
                                                         width,
                                                         kernel_size,
                                                         kernel_size
                                                         );

    CHECK_LAST_CUDA_ERROR("census_transform_cuda for right feat map");
    CHECK_CUDA_ASYNC_ERROR("census_transform_cuda for right feat map");





    size_t free_sz, total_sz;
    cudaMemGetInfo(&free_sz, &total_sz);
    cout << "free:  " << free_sz/1000000 << "\ntotal: " << total_sz/1000000 << endl;


    cudaFree(d_left);
    cudaFree(d_right);


    cout << "Finish GPU kernel" << endl;

    //cudaMemcpy(left_census,  d_left_out,  sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(right_census, d_right_out, sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);

    
    unsigned int *d_cost_volume;
    cudaMalloc((void **)&d_cost_volume, sizeof(unsigned int)*num_disparity*height*width);


    compute_cost_cuda<<<numBlocks,threadsPerBlock>>>(d_cost_volume,
                                                     d_left_census,
                                                     d_right_census,
                                                     height,
                                                     width,
                                                     kernel_size,
                                                     kernel_size,
                                                     num_disparity,
                                                     opt
                                                     );

    cudaFree(d_left_census);
    cudaFree(d_right_census);

    CHECK_LAST_CUDA_ERROR("compute cost with left census and right census");
    CHECK_CUDA_ASYNC_ERROR("compute cost with left census and right census");

    
    unsigned int *d_sum_up_cost_volume;
    cudaMalloc((void **)&d_sum_up_cost_volume, sizeof(unsigned int)*num_disparity*height*width);
    cudaMemset(d_sum_up_cost_volume, 0, sizeof(unsigned int)*num_disparity*height*width);


    unsigned int *d_opt_cost_volume_L;
    cudaMalloc((void **)&d_opt_cost_volume_L, sizeof(unsigned int)*num_disparity*height*width);



    const dim3 numHorizontalScanLines(height);
    compute_directional_cost_L0_cuda<unsigned int><<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L,
                                                                                 d_cost_volume,
                                                                                 height,
                                                                                 width,
                                                                                 num_disparity
                                                                                 );

    CHECK_LAST_CUDA_ERROR("compute L0 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L0 directional cost");

    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L0_L4
                                                        );

    CHECK_LAST_CUDA_ERROR("incrementally sum up the L0 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L0 directional costs");



    compute_directional_cost_L4_cuda<unsigned int><<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L,
                                                                                 d_cost_volume,
                                                                                 height,
                                                                                 width,
                                                                                 num_disparity
                                                                                 );
    CHECK_LAST_CUDA_ERROR("compute L4 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L4 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L0_L4
                                                        );
    CHECK_LAST_CUDA_ERROR("incrementally sum up the L4 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L4 directional costs");


    const dim3 numVerticalScanLines(width);
    compute_directional_cost_L2_cuda<unsigned int><<<numVerticalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L2 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L2 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L2_L6
                                                        );
    CHECK_LAST_CUDA_ERROR("incrementally sum up the L2 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L2 directional costs");


    compute_directional_cost_L6_cuda<unsigned int><<<numVerticalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L6 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L6 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L2_L6
                                                        );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L6 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L6 directional costs");


    const dim3 numDiagonalScanLines(height+width-1);
    compute_directional_cost_L1_cuda<unsigned int><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L1 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L1 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L1_L5
                                                        );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L1 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L1 directional costs");


    compute_directional_cost_L5_cuda<unsigned int><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L5 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L5 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L1_L5
                                                        );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L5 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L5 directional costs");


    compute_directional_cost_L3_cuda<unsigned int><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L3 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L3 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L3_L7
                                                        );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L3 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L3 directional costs");


    compute_directional_cost_L7_cuda<unsigned int><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                               d_cost_volume,
                                                                               height,
                                                                               width,
                                                                               num_disparity
                                                                               );

    CHECK_LAST_CUDA_ERROR("compute L7 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L7 directional cost");


    inc_sum_up_cost_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                        d_opt_cost_volume_L,
                                                        height,
                                                        width,
                                                        num_disparity,
                                                        weight_L3_L7
                                                        );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L7 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L7 directional costs");

    cudaFree(d_opt_cost_volume_L);




    int out_N         = height*width;
    unsigned int *out = (unsigned int *)malloc(sizeof(unsigned int)*out_N);
    float *f_out      = (float *)malloc(sizeof(float)*out_N);

    unsigned int * d_out;
    cudaMalloc((void**)&d_out,   sizeof(unsigned int) * out_N);


    gen_disparity_map_cuda<unsigned int><<<numBlocks,threadsPerBlock>>>(d_out,
                                                                        d_sum_up_cost_volume,
                                                                        height,
                                                                        width,
                                                                        num_disparity
                                                                        );

    CHECK_LAST_CUDA_ERROR("generate the disparity map");
    CHECK_CUDA_ASYNC_ERROR("generate the disparity map");


    cudaMemcpy(out, d_out, sizeof(unsigned int) * out_N, cudaMemcpyDeviceToHost);
    Mat disparity_img(height,width, CV_8U);
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            //printf("r:%d   c:%d = %d\n",row,col,res.at<int>(row,col));
            //printf("x:%d   y:%d = %d\n",x,y, out[x+y*width]);
            //res.at<int>(y,x)=out[x+y*width];

            disparity_img.at<uint8_t>(y,x)=out[x+y*width]*255/num_disparity;
            //disparity_img.at<uint8_t>(y,x)=out[x+y*width];
	    f_out[x+y*width] = (float)out[x+y*width];
            //disparity_img.at<uint8_t>(y,x)=out[x+y*width];
        }

    Mat color_disparity;
    applyColorMap(disparity_img, color_disparity, COLORMAP_JET);

    Mat median_filter_res = disparity_img.clone();
    medianBlur (disparity_img, median_filter_res, kernel_size);

    Mat color_median_filter;
    applyColorMap(median_filter_res, color_median_filter, COLORMAP_JET);
    //GaussianBlur( res, dst, Size( kernel_size, kernel_size ), 0, 0 );
    //bilateralFilter ( res, dst, kernel_size, kernel_size*2, kernel_size/2 );


    if(opt==REF_IMG::LEFT){
        left_out = out;
        imwrite("./left_semi_global_disparity.png", color_disparity);
        imwrite("./left_disparity.png", disparity_img);
	WriteFilePFM(f_out, width, height, "./left_semi_global_disparity.pfm", 1.0/num_disparity);
    }
    else{
        right_out = out;
        imwrite("./right_semi_global_disparity.png", color_disparity);
        imwrite("./right_disparity.png", disparity_img);
	WriteFilePFM(f_out, width, height, "./right_semi_global_disparity.pfm", 1.0/num_disparity);
    }





    cudaFree(d_cost_volume);
    cudaFree(d_sum_up_cost_volume);


    free(left);
    free(right);

}


__host__ void semi_global_stereo_matching::inc_ssd_execute(const char * left_path,
                                                           const char * right_path,
                                                           int kernel_size,
                                                           int num_disparity,
						           unsigned int P1,
						           unsigned int P2,
                                                           REF_IMG opt
                                                           ){

    unsigned int penality[3] = {0,100,1000};
    penality[1] = P1;
    penality[2] = P2;
    cudaMemcpyToSymbol(L0_L4_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L2_L6_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L1_L5_P, penality, 3*sizeof(unsigned int));
    cudaMemcpyToSymbol(L3_L7_P, penality, 3*sizeof(unsigned int));

    double *left  = nullptr;
    double *right = nullptr;

    std::vector<unsigned long> left_shape  = featAllocation(left_path, &left);
    std::vector<unsigned long> right_shape = featAllocation(right_path, &right);

    const unsigned long channel = left_shape[0];
    const unsigned long height  = left_shape[1];
    const unsigned long width   = left_shape[2];

    img_height = height;
    img_width  = width;

    unsigned long N     = getSize(left_shape);



    double *d_left, *d_right;
    cudaMalloc((void**)&d_left,      sizeof(double) * N);
    cudaMalloc((void**)&d_right,     sizeof(double) * N);


    cudaMemcpy(d_left,  left,  sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, sizeof(double) * N, cudaMemcpyHostToDevice);


    constexpr int block_size {4};



    printf("DIM: %ld %ld\n",(block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 numBlocks((block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 threadsPerBlock(block_size, block_size);


    //cuda_hello<<<1,1>>>();
    //cudaDeviceSynchronize();



    size_t free_sz, total_sz;
    cudaMemGetInfo(&free_sz, &total_sz);
    cout << "free:  " << free_sz/1000000 << "\ntotal: " << total_sz/1000000 << endl;




    cout << "Finish GPU kernel" << endl;

    //cudaMemcpy(left_census,  d_left_out,  sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(right_census, d_right_out, sizeof(unsigned int) * out_census_transform_N, cudaMemcpyDeviceToHost);

    
    double *d_cost_volume;
    cudaMalloc((void **)&d_cost_volume, sizeof(double)*num_disparity*height*width);




    sum_diff_transform_cuda<<<numBlocks,threadsPerBlock>>>(d_cost_volume,
                                                           d_left,
                                                           d_right,
                                                           channel,
                                                           height,
                                                           width,
                                                           kernel_size,
                                                           kernel_size,
                                                           num_disparity,
                                                           opt
                                                           );


    CHECK_LAST_CUDA_ERROR("compute cost using sum difference bewteen the pixels");
    CHECK_CUDA_ASYNC_ERROR("compute cost using sum difference bewteen the pixels");

    cudaFree(d_left);
    cudaFree(d_right);


    
    double *d_sum_up_cost_volume;
    cudaMalloc((void **)&d_sum_up_cost_volume, sizeof(double)*num_disparity*height*width);
    cudaMemset(d_sum_up_cost_volume, 0, sizeof(double)*num_disparity*height*width);


    double *d_opt_cost_volume_L;
    cudaMalloc((void **)&d_opt_cost_volume_L, sizeof(double)*num_disparity*height*width);



    const dim3 numHorizontalScanLines(height);
    compute_directional_cost_L0_cuda<double><<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L,
                                                                           d_cost_volume,
                                                                           height,
                                                                           width,
                                                                           num_disparity
                                                                           );

    CHECK_LAST_CUDA_ERROR("compute L0 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L0 directional cost");

    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L0_L4
                                                                );

    CHECK_LAST_CUDA_ERROR("incrementally sum up the L0 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L0 directional costs");



    compute_directional_cost_L4_cuda<double><<<numHorizontalScanLines,1>>>(d_opt_cost_volume_L,
                                                                           d_cost_volume,
                                                                           height,
                                                                           width,
                                                                           num_disparity
                                                                           );
    CHECK_LAST_CUDA_ERROR("compute L4 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L4 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L0_L4
                                                                );
    CHECK_LAST_CUDA_ERROR("incrementally sum up the L4 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L4 directional costs");


    const dim3 numVerticalScanLines(width);
    compute_directional_cost_L2_cuda<double><<<numVerticalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L2 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L2 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L2_L6
                                                                );
    CHECK_LAST_CUDA_ERROR("incrementally sum up the L2 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L2 directional costs");


    compute_directional_cost_L6_cuda<double><<<numVerticalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L6 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L6 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L2_L6
                                                                );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L6 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L6 directional costs");


    const dim3 numDiagonalScanLines(height+width-1);
    compute_directional_cost_L1_cuda<double><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L1 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L1 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L1_L5
                                                                );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L1 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L1 directional costs");


    compute_directional_cost_L5_cuda<double><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L5 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L5 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L1_L5
                                                                );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L5 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L5 directional costs");


    compute_directional_cost_L3_cuda<double><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L3 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L3 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L3_L7
                                                                );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L3 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L3 directional costs");


    compute_directional_cost_L7_cuda<double><<<numDiagonalScanLines,1>>>(d_opt_cost_volume_L,
                                                                         d_cost_volume,
                                                                         height,
                                                                         width,
                                                                         num_disparity
                                                                         );

    CHECK_LAST_CUDA_ERROR("compute L7 directional cost");
    CHECK_CUDA_ASYNC_ERROR("compute L7 directional cost");


    inc_sum_up_cost_cuda<double><<<numBlocks,threadsPerBlock>>>(d_sum_up_cost_volume,
                                                                d_opt_cost_volume_L,
                                                                height,
                                                                width,
                                                                num_disparity,
                                                                weight_L3_L7
                                                                );



    CHECK_LAST_CUDA_ERROR("incrementally sum up the L7 directional costs");
    CHECK_CUDA_ASYNC_ERROR("incrementally sum up the L7 directional costs");

    cudaFree(d_opt_cost_volume_L);




    int out_N         = height*width;
    unsigned int *out = (unsigned int *)malloc(sizeof(unsigned int)*out_N);
    float *f_out      = (float *)malloc(sizeof(float)*out_N);

    unsigned int * d_out;
    cudaMalloc((void**)&d_out,   sizeof(unsigned int) * out_N);


    gen_disparity_map_cuda<double><<<numBlocks,threadsPerBlock>>>(d_out,
                                                                  d_sum_up_cost_volume,
                                                                  height,
                                                                  width,
                                                                  num_disparity
                                                                  );

    CHECK_LAST_CUDA_ERROR("generate the disparity map");
    CHECK_CUDA_ASYNC_ERROR("generate the disparity map");


    cudaMemcpy(out, d_out, sizeof(unsigned int) * out_N, cudaMemcpyDeviceToHost);
    Mat disparity_img(height,width, CV_8U);
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            //printf("r:%d   c:%d = %d\n",row,col,res.at<int>(row,col));
            //printf("x:%d   y:%d = %d\n",x,y, out[x+y*width]);
            //res.at<int>(y,x)=out[x+y*width];

            disparity_img.at<uint8_t>(y,x)=out[x+y*width]*255/num_disparity;
	    f_out[x+y*width] = (float)out[x+y*width];
            //disparity_img.at<uint8_t>(y,x)=out[x+y*width];
        }

    Mat color_disparity;
    applyColorMap(disparity_img, color_disparity, COLORMAP_JET);



    if(opt==REF_IMG::LEFT){
        left_out = out;
        imwrite("./left_semi_global_disparity.png", color_disparity);
	WriteFilePFM(f_out, width, height, "./left_semi_global_disparity.pfm", 1.0/num_disparity);
	//WriteFilePFM(f_out, width, height, "./left_semi_global_disparity.pfm");
    }
    else{
        right_out = out;
        imwrite("./right_semi_global_disparity.png", color_disparity);
	WriteFilePFM(f_out, width, height, "./right_semi_global_disparity.pfm", 1.0/num_disparity);
	//WriteFilePFM(f_out, width, height, "./right_semi_global_disparity.pfm");
    }





    cudaFree(d_cost_volume);
    cudaFree(d_sum_up_cost_volume);


    free(left);
    free(right);

}


