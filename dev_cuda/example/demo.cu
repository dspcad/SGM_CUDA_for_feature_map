#include<bits/stdc++.h>
#include "simple_stereo_matching.hpp"

using namespace ssm;
using namespace std;


int main(int argc, char* argv[]){
    if (argc!=4){
        printf("Usage: %s left.npy right.npy mode\n", argv[0] );
        printf("       mode could be either 0 (SAD) or 1 (SSD)\n");
        return 1;
    }



    //const char * left_path  = "data/left.npy";
    //const char * right_path = "data/right.npy";
    const char * left_path  = argv[1];
    const char * right_path = argv[2];


    constexpr int kernel {3};
    constexpr int num_disparity {64};
    const MATCHING_COST mode = static_cast<MATCHING_COST>(stoi(argv[3]));


    cout << "Mode: " << static_cast<int>(mode) << endl;
    simple_stereo_matching engine;
    engine.execute(left_path,
                   right_path,
                   kernel,
                   num_disparity,
                   mode);


    return 0;
}
