template <typename T>
__global__ void compute_directional_cost_L0_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L4_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L1_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L5_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L3_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L7_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );


template <typename T>
__global__ void compute_directional_cost_L2_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );

template <typename T>
__global__ void compute_directional_cost_L6_cuda(T *out, 
                                                 T *cost_volume, 
                                                 unsigned long height,
                                                 unsigned long width,
                                                 unsigned long num_disparity
                                                 );


template <typename T>
__global__ void inc_sum_up_cost_cuda(T * out,
                                     T * cost_volme_L,
                                     unsigned long height,
                                     unsigned long width,
                                     int num_disparity,
                                     unsigned int weight
                                    );

template <typename T>
__global__ void gen_disparity_map_cuda(unsigned int * out,
                                       T * cost_volme,
                                       unsigned long height,
                                       unsigned long width,
                                       int num_disparity
                                       );



 

#include "compute_directional_cost_cuda.hpp"
