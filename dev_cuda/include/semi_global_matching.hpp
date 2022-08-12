#include "simple_stereo_matching.hpp"
#define BITMAP_SIZE 50
#define UNSIGNED_INT_SIZE 32
using namespace ssm;



namespace sgsm{
    //constexpr unsigned int P1_L0_L4 = 100;
    //constexpr unsigned int P2_L0_L4 = 1000;

    //constexpr unsigned int P1_L2_L6 = 100;
    //constexpr unsigned int P2_L2_L6 = 1000;

    //constexpr unsigned int P1_L1_L5 = 100;
    //constexpr unsigned int P2_L1_L5 = 1000;

    //constexpr unsigned int P1_L3_L7 = 100;
    //constexpr unsigned int P2_L3_L7 = 1000;


    constexpr int disp_offset = 0;

    constexpr unsigned int weight_L0_L4= 1;
    constexpr unsigned int weight_L2_L6= 1;
    constexpr unsigned int weight_L1_L5= 1;
    constexpr unsigned int weight_L3_L7= 1;

    constexpr unsigned int INF = 1000000000;

    enum class REF_IMG {LEFT=0,RIGHT=1};

    class semi_global_stereo_matching: public simple_stereo_matching{
        public:
            unsigned int *left_out;
            unsigned int *right_out;
            unsigned int img_height;
            unsigned int img_width;
            semi_global_stereo_matching(): left_out{nullptr}, right_out{nullptr}, img_height{1}, img_width{1} {}

            __host__ void init_CUDA_HEAP();

            __host__ void execute(const char * left_path,
                                  const char * right_path,
                                  int kernel,
                                  int num_disparity,
                                  REF_IMG opt
                                  );  

            __host__ void inc_execute(const char * left_path,
                                      const char * right_path,
                                      int kernel_size,
                                      int num_disparity,
				      unsigned int P1,
				      unsigned int P2,
                                      REF_IMG opt
                                      );  

            __host__ void inc_ssd_execute(const char * left_path,
                                          const char * right_path,
                                          int kernel_size,
                                          int num_disparity,
				          unsigned int P1,
				          unsigned int P2,
                                          REF_IMG opt
                                          );  



            __host__ void inconsistence_check(unsigned long height,
                                              unsigned long width,
                                              int num_disparity
                                              );  


            void WriteFilePFM(float *data, int width, int height, const char* filename, float scalefactor);
            ~semi_global_stereo_matching()=default;
    };

 

    


}//end of namespace sgsm


__device__ double getSumSquareDiff(double *left_patch,
                                   double *right_path,
                                   unsigned long channel,
                                   int kernel_h,
                                   int kernel_w
                                   );


__device__ double getSumAbsoluteDiff(double *left_patch,
                                     double *right_path,
                                     unsigned long channel,
                                     int kernel_h,
                                     int kernel_w
                                     );

__global__ void sum_diff_transform_cuda(double *out, 
                                        const double *left_feat_map, 
                                        const double *right_feat_map, 
                                        unsigned long channel,
                                        unsigned long height,
                                        unsigned long width,
                                        int kernel_h,
                                        int kernel_w,
                                        int num_disparity,
                                        sgsm::REF_IMG opt
                                        );



__device__ int getHammingDist(unsigned int left_pixel_census[BITMAP_SIZE],
                              unsigned int right_pixel_census[BITMAP_SIZE]
                              );


__global__ void compute_cost_cuda(unsigned int *out,
                                  unsigned int *left,
                                  unsigned int *right,
                                  unsigned long height,
                                  unsigned long width,
                                  int kernel_h,
                                  int kernel_w,
                                  int num_disparity,
                                  sgsm::REF_IMG opt
                                  );


__global__ void census_transform_cuda(unsigned int  *out,
                                      const double  *feat_map,
                                      unsigned long channel,
                                      unsigned long height,
                                      unsigned long width,
                                      int kernel_h,
                                      int kernel_w
                                      );

__global__ void sum_up_cost_cuda(unsigned int * out,
                                 unsigned int * cost_volme_L0,
                                 unsigned int * cost_volme_L2,
                                 unsigned int * cost_volme_L4,
                                 unsigned int * cost_volme_L6,
                                 unsigned long height,
                                 unsigned long width,
                                 int num_disparity
                                 );




__global__ void cuda_hello();



