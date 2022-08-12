#include<bits/stdc++.h>



namespace ssm{
    enum class MATCHING_COST {SAD=0,SSD=1};

    class simple_stereo_matching{
        public:
            simple_stereo_matching()=default;

            __host__ void print_feat(const std::vector<unsigned long> &shape, double *feat_map);
            __host__ void execute(const char * left_path,
                                  const char * right_path,
                                  int kernel,
                                  int num_disparity,
                                  MATCHING_COST mode
                                  );  



            ~simple_stereo_matching()=default;
        protected:
            __host__ unsigned long getSize(const std::vector<unsigned long> &shape);
            __host__ std::vector<unsigned long> featAllocation(const char *path, double **res);
    };

    
}//end of namespace ssm

template <ssm::MATCHING_COST mode>
__global__ void simple_stereo_matching_cuda(int *out, 
                                            const double *left, 
                                            const double *right,
                                            unsigned long channel,
                                            unsigned long height,
                                            unsigned long width,
                                            int kernel,
                                            int num_disparity
                                            );



