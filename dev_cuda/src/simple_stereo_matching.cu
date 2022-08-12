#include "simple_stereo_matching.hpp"
#include "npy.hpp"
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;
using namespace ssm;

__host__ void printStats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    //printf("free:  %ld     total:  %ld\n", free, total);
    std::cout << "free:  " << free/1000000 << "\ntotal: " << total/1000000 << std::endl;
}



template <MATCHING_COST mode>
__global__ void simple_stereo_matching_cuda(int *out, 
                                            const double *left, 
                                            const double *right,
                                            unsigned long channel,
                                            unsigned long height,
                                            unsigned long width,
                                            int kernel,
                                            int num_disparity
                                            ){

    //printf("GPU: kernel size: %d\n", kernel);

    //__syncthreads();

    //printf("block dim: (%d,%d):\n",blockDim.x, blockDim.y);
    //if(blockIdx.y>10)
    //printf("block (%d,%d):\n",blockIdx.x,blockIdx.y);
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(x<kernel/2 || x>=width-kernel/2) return;
    if(y<kernel/2 || y>=height-kernel/2) return;

    //printf("(%d,%d):\n",x,y);
    //printf("c:%ld  h:%ld   w:%ld    kernel:%d\n",channel,height,width,kernel);


    double *left_patch, *right_patch;
    cudaError_t code;
    //code = cudaMalloc((void**)&left_patch,  sizeof(double)*64);
    code = cudaMalloc((void**)&left_patch,  sizeof(double)*kernel*kernel*channel);
    if(code==cudaErrorMemoryAllocation) printf("debug: left path is out of memory %ld\n", sizeof(double)*kernel*kernel*channel);
    //printf("debug: %s\n", cudaGetErrorString(code));
    //code = cudaMalloc((void**)&right_patch, sizeof(double)*64);
    code = cudaMalloc((void**)&right_patch, sizeof(double)*kernel*kernel*channel);
    if(code==cudaErrorMemoryAllocation) printf("debug: right path is out of memory %ld\n", sizeof(double)*kernel*kernel*channel);


    //double *left_patch  = (double *)malloc(sizeof(double)*kernel*kernel*channel);
    //double *right_patch = (double *)malloc(sizeof(double)*kernel*kernel*channel);


    //shift (x,y) from the center to the top-left
    const int shift_x = x - kernel/2;
    const int shift_y = y - kernel/2;
    //printf("    shifted: (%d,%d)\n", shift_x,shift_y);
    //printf("        kernel: %d   channel: %ld\n", kernel, channel);
    for(int c=0;c<channel;++c)
        for(int h=0;h<kernel;++h)
            for(int w=0;w<kernel;++w){
                //printf("%d %d %d\n",c,w,h);
                left_patch[w+h*kernel+c*kernel*kernel]=left[(shift_x+w)+(shift_y+h)*width+c*height*width];
                //left_patch[h+w*kernel+c*kernel*kernel]=1;
            }


    //printf("GPU Mode: %d\n", mode);
    //printf("    Mode SSD: %d\n", MATCHING_COST::SSD);
    //printf("    Mode ASD: %d\n", MATCHING_COST::ASD);
    int idx=0;
    double res = 1000000000;

    for(int i=0;i<num_disparity;++i){
        if(shift_x-i<0)continue;


        for(int c=0;c<channel;++c)
            for(int h=0;h<kernel;++h)
                for(int w=0;w<kernel;++w)
                    right_patch[w+h*kernel+c*kernel*kernel]=right[(shift_x+w-i)+(shift_y+h)*width+c*height*width];

        double diff = 0;

        if(mode == MATCHING_COST::SSD){
             for(int j=0;j<kernel*kernel*channel;++j)
                diff += pow((left_patch[j]-right_patch[j]),2);
        }
        else{
             for(int j=0;j<kernel*kernel*channel;++j)
                diff += abs(left_patch[j]-right_patch[j]);
        }
       


        //for(int j=0;j<kernel*kernel*channel;++j)
        //    diff += abs(left_patch[j]-right_patch[j]);
            //diff += pow((left_patch[j]-right_patch[j]),2);

        if(diff<res){
            //printf("    idx: %d    %f\n",i, diff);
            idx = i;
            res = diff;
        }

    }
    //printf("END\n");

    //printf("(%d %d) disparity map: %d\n", x,y,idx);
    out[x+y*width] = idx;
    //printf("pixel: %d %d = %d\n", x,y,out[x+y*width]);
    cudaFree(left_patch);
    cudaFree(right_patch);
}

__host__ void simple_stereo_matching::print_feat(const vector<unsigned long> &shape, double *feat_map){
    for(int c=0;c<shape[0];++c){
        for(int h=0;h<shape[1];++h){
            for(int w=0;w<shape[2];++w)
                cout << feat_map[w+w*shape[1]+c*shape[1]*shape[2]] << " ";
            cout << endl;
        }
    }
}

__host__ unsigned long simple_stereo_matching::getSize(const vector<unsigned long> &shape){
    unsigned long N=1;
    cout << "C, H, W: ";
    for(auto v : shape){
        cout << v << ", ";
        N*=v;
    }
    cout << "   N: " << N << endl;
   
    return N;
}

__host__ vector<unsigned long> simple_stereo_matching::featAllocation(const char *path, double **res){
    vector<unsigned long> shape;
    bool fortran_order;
    vector<double> data;

    cout << "path: " << path << endl;
    cout << data.size() << endl;
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    cout << "Load " << path << " done." << endl;

    unsigned long N= getSize(shape);
    double *feat_map   = (double *)malloc(sizeof(double)*N);
    const unsigned long channel = shape[0];
    const unsigned long height  = shape[1];
    const unsigned long width   = shape[2];


    for(int c=0;c<channel;++c)
        for(int h=0;h<height;++h)
            for(int w=0;w<width;++w){
                //cout << h+w*shape[0]+c*shape[0]*shape[1] << endl;
                feat_map[w+h*width+c*height*width]=data[w+h*width+c*height*width];
            }
            
    *res = feat_map;

    return shape;
}


__host__ void simple_stereo_matching::execute(const char * left_path,
                                              const char * right_path,
                                              int kernel,
                                              int num_disparity,
                                              MATCHING_COST mode
                                              ){

    double *left  = nullptr;
    double *right = nullptr;

    std::vector<unsigned long> left_shape  = featAllocation(left_path, &left);
    std::vector<unsigned long> right_shape = featAllocation(right_path, &right);

    const unsigned long channel = left_shape[0];
    const unsigned long height  = left_shape[1];
    const unsigned long width   = left_shape[2];

    unsigned long N     = getSize(left_shape);
    unsigned long out_N = height*width;
    int *out            = (int *)malloc(sizeof(int)*out_N);


    size_t setsz = 2LL*1024LL*1024LL*1024LL;
    size_t getsz;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, setsz);
    cudaDeviceGetLimit(&getsz, cudaLimitMallocHeapSize);
    printf("Heap requested %ld got %ld\n", setsz, getsz);


    setsz = 128LL*1024LL*1024LL;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, setsz);
    cudaDeviceGetLimit(&getsz, cudaLimitPrintfFifoSize);
    printf("Print buffer requested %ld got %ld\n", setsz, getsz);


    //print_feat(left_shape,left);
    //print_feat(right_shape,right);

    double *d_left, *d_right;
    int *d_out;


    cudaMalloc((void**)&d_left,  sizeof(double) * N);
    cudaMalloc((void**)&d_right, sizeof(double) * N);
    cudaMalloc((void**)&d_out,   sizeof(int) * out_N);


    cudaMemcpy(d_left,  left,  sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, sizeof(double) * N, cudaMemcpyHostToDevice);



    constexpr int block_size {32};




    printf("DIM: %ld %ld\n",(block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 numBlocks((block_size+width-1)/block_size, (block_size+height-1)/block_size);
    const dim3 threadsPerBlock(block_size, block_size);



    cout << "Call GPU kernel" << endl;
    if(mode == MATCHING_COST::SSD){
        cout << "    Mode SSD" << endl;
        simple_stereo_matching_cuda<MATCHING_COST::SSD><<<numBlocks,threadsPerBlock>>>(d_out,
                                                                                       d_left,
                                                                                       d_right,
                                                                                       channel,
                                                                                       height,
                                                                                       width,
                                                                                       kernel,
                                                                                       num_disparity);
    }
    else if(mode == MATCHING_COST::SAD){
        cout << "    Mode SAD" << endl;
        simple_stereo_matching_cuda<MATCHING_COST::SAD><<<numBlocks,threadsPerBlock>>>(d_out,
                                                                                       d_left,
                                                                                       d_right,
                                                                                       channel,
                                                                                       height,
                                                                                       width,
                                                                                       kernel,
                                                                                       num_disparity);

    }




    cout << "Finish GPU kernel" << endl;

    cudaMemcpy(out, d_out, sizeof(int) * out_N, cudaMemcpyDeviceToHost);
    Mat res(height,width, CV_8U);
    for(int x=0;x<width;++x)
        for(int y=0;y<height;++y){
            //printf("r:%d   c:%d = %d\n",row,col,res.at<int>(row,col));
            //printf("r:%d   c:%d = %d\n",row,col, out[row+col*height]);
            //res.at<int>(y,x)=out[x+y*width];
            res.at<uint8_t>(y,x)=out[x+y*width]*255/num_disparity;
        }

    Mat color_disparity;
    applyColorMap(res, color_disparity, COLORMAP_JET);
    imwrite("./color_disp_image.png", color_disparity);
    //cv::namedWindow("image", WINDOW_AUTOSIZE);
    //cv::imshow("image", res);
    //cv::waitKey(3000000);    


    //size_t free, total;
    //cudaMemGetInfo(&free, &total);
    //std::cout << "free:  " << free << "\ntotal: " << total << std::endl;

    printStats();
    //for(int i = 0; i < N; i++) printf("out[%d]: %f\n",i,out[i]);


    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_out);

    free(left);
    free(right);
    free(out);

}



