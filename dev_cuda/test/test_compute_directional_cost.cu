#include "semi_global_matching.hpp"
#include "compute_directional_cost.hpp"

using namespace std;
using namespace sgsm;

int main(int argc, char* argv[]){
    cout << "Test Case for compute_cost_cuda" << endl;

/*
    Example: one scan line


      ______________________________
      \ 487   432   37   628   312  \
      |\  721   38   403   611   597 \   disparity = 3
      | \  999   183  366    925   858\
      |  ------------------------------
      \  |                            |  height = 1
        \|____________________________|
                  width = 5

*/

   

//    constexpr unsigned long height    = 1;
//    constexpr unsigned long width     = 5;
//    constexpr unsigned long disparity = 3;
//    
//    const int N = height*width*disparity;
//    unsigned int cost_volume[N] = {999, 183, 366, 925, 858, 
//                                   721,  38, 403, 611, 597, 
//                                   487, 432,  37, 628, 312};
//
//    unsigned int out[N];



/*
    Example: a toy example of cost volume


   ___          ______________________________
    |           \ 487   432   37   628   312  \
    |           |\  721   38   403   611   597 \   disparity = 3
    |           | \  999   183  366    925   858\
    |           |  ------------------------------
    |           \  |                            |  
                  \|____________________________|

                ______________________________
                \ 543   612   578   729  630  \
                |\  329  317   234   473  987  \   
                | \  249  920   711   218   466 \
                |  ------------------------------
                \  |                            |  
                  \|____________________________|

height=4        ______________________________
                \ 470   641   656   754  459  \
                |\  28   828   555   950  887  \   
                | \  883   80    22   491   97  \
                |  ------------------------------
                \  |                            |  
                  \|____________________________|

                ______________________________
                \ 633   649   575   350  158  \
                |\  743   37   464   521  332  \   
    |           | \  388  258   60    477  223  \
    |           |  ------------------------------
    |           \  |                            |  
    |             \|____________________________|
   _|_
                            width = 5
*/


    constexpr unsigned long height    = 4;
    constexpr unsigned long width     = 5;
    constexpr unsigned long disparity = 3;
    
    const int N = height*width*disparity;
    unsigned int cost_volume[N] = {999, 183, 366, 925, 858, 
                                   249, 920, 711, 218, 466, 
                                   883,  80,  22, 491,  97, 
                                   388, 258,  60, 477, 223,

                                   721,  38, 403, 611, 597, 
                                   329, 317, 234, 473, 987,
                                    28, 828, 555, 950, 887, 
                                   743,  37, 464, 521, 332,

                                   487, 432,  37, 628, 312, 
                                   543, 612, 578, 729, 630,
                                   470, 641, 656, 754, 459, 
                                   633, 649, 575, 350, 158};

    unsigned int out[N];







    unsigned int *d_cost_volume;
    unsigned int *d_out;

    cudaMalloc((void **)&d_cost_volume, sizeof(unsigned int)*N); 
    cudaMalloc((void **)&d_out,         sizeof(unsigned int)*N);

    cudaMemcpy(d_cost_volume,  cost_volume,  sizeof(unsigned int)*N, cudaMemcpyHostToDevice);


    const dim3 numScanLines(height);
    compute_directional_cost_L0_cuda<unsigned int><<<numScanLines,1>>>(d_out,
                                                         d_cost_volume,
                                                         height,
                                                         width,
                                                         disparity
                                                         );




    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);


   
    for(int d=0;d<disparity;++d){
        printf("Disparity %d: \n", d);
        for(int j=0;j<height;++j){
            for(int i=0;i<width;++i){
                printf("%d ", out[i+j*width+d*height*width]);
            }
            printf("\n");
        }
    }


    cudaFree(d_cost_volume);
    cudaFree(d_out);


    return 0;
}
