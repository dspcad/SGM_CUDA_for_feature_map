find_package(OpenCV REQUIRED)
set_source_files_properties(simple_stereo_matching.cu census_transform.cu compute_cost.cu semi_global_stereo_matching.cu compute_sum_diff_cost.cu hello.cu PROPERTIES LANGUAGE CUDA)

add_library(stereo_matching
            STATIC simple_stereo_matching.cu census_transform.cu compute_cost.cu semi_global_stereo_matching.cu compute_sum_diff_cost.cu hello.cu
           )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_compile_features(stereo_matching 
                        PUBLIC cxx_std_11)
set_target_properties( stereo_matching
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
