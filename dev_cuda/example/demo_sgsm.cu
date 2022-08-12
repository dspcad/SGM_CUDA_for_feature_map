#include<bits/stdc++.h>
#include "semi_global_matching.hpp"

using namespace sgsm;
using namespace std;

//__global__ void cuda_hello(){
//    //int a;
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("Hello World from GPU %d!\n", tid);
//    //printf("size of int: %d\n", sizeof(a));
//}



int main(int argc, char* argv[]){
    if (argc!=7){
        printf("Usage: %s left.npy right.npy kernel_size num_disparity P1 P2\n", argv[0] );
        return 1;
    }


//    cuda_hello<<<1,1>>>();
//    cudaDeviceSynchronize();


    //const char * left_path  = "data/left.npy";
    //const char * right_path = "data/right.npy";
    const char * left_path  = argv[1];
    const char * right_path = argv[2];

    const int kernel        = stoi(argv[3]);
    const int num_disparity = stoi(argv[4]);

    const unsigned int P1   = stoi(argv[5]);
    const unsigned int P2   = stoi(argv[6]);
    printf("-----------------------\n");
    printf("Kernel size: %d\n", kernel);
    printf("Number of Disparity: %d\n", num_disparity);
    printf("Disparity Offset:    %d\n", disp_offset);
    printf("Effective Offset:    %d\n", num_disparity+disp_offset);
    printf("P1: %d\n", P1);
    printf("P2: %d\n", P2);
    printf("-----------------------\n");
    //constexpr int kernel {15};
    //constexpr int num_disparity {64};


    semi_global_stereo_matching engine;

    engine.init_CUDA_HEAP();

    engine.inc_execute(left_path,
                       right_path,
                       kernel,
                       num_disparity,
		       P1,
		       P2,
                       REF_IMG::LEFT);

    engine.inc_execute(left_path,
                       right_path,
                       kernel,
                       num_disparity,
		       P1,
		       P2,
                       REF_IMG::RIGHT);


    //engine.inc_ssd_execute(left_path,
    //                   right_path,
    //                   kernel,
    //                   num_disparity,
    //                   P1,
    //                   P2,
    //                   REF_IMG::LEFT);

    //engine.inc_ssd_execute(left_path,
    //                   right_path,
    //                   kernel,
    //                   num_disparity,
    //                   P1,
    //                   P2,
    //                   REF_IMG::RIGHT);



    if(engine.left_out!=nullptr) printf("left_out might have some data\n");
    if(engine.right_out!=nullptr) printf("right_out might have some data\n");

    if(engine.left_out!=nullptr && engine.right_out!=nullptr) engine.inconsistence_check(engine.img_height, engine.img_width,num_disparity);


    return 0;
}
