find_package(OpenCV REQUIRED)

set_source_files_properties(test_census_transform.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_census_transform test_census_transform.cu )
target_link_libraries( test_census_transform stereo_matching ${OpenCV_LIBS})
add_test(NAME TestCensusTransform COMMAND test_census_transform)


set_source_files_properties(test_compute_cost.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_compute_cost test_compute_cost.cu )
target_link_libraries( test_compute_cost stereo_matching ${OpenCV_LIBS})
add_test(NAME TestComputeCost COMMAND test_compute_cost)


set_source_files_properties(test_compute_directional_cost.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_compute_directional_cost test_compute_directional_cost.cu )
target_link_libraries( test_compute_directional_cost stereo_matching ${OpenCV_LIBS})
add_test(NAME TestComputeDirectionalCost COMMAND test_compute_directional_cost)

set_source_files_properties(test_sum_up_cost.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_sum_up_cost test_sum_up_cost.cu )
target_link_libraries( test_sum_up_cost stereo_matching ${OpenCV_LIBS})
add_test(NAME TestSumUpCost COMMAND test_sum_up_cost)

set_source_files_properties(test_inc_sum_up_cost.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_inc_sum_up_cost test_inc_sum_up_cost.cu )
target_link_libraries( test_inc_sum_up_cost stereo_matching ${OpenCV_LIBS})
add_test(NAME TestIncSumUpCost COMMAND test_inc_sum_up_cost)


set_source_files_properties(test_gen_disparity_map.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_gen_disparity_map test_gen_disparity_map.cu )
target_link_libraries( test_gen_disparity_map stereo_matching ${OpenCV_LIBS})
add_test(NAME TestGenDisparityMap COMMAND test_gen_disparity_map)


set_source_files_properties(test_compute_sum_diff_cost.cu PROPERTIES LANGUAGE CUDA)
add_executable( test_compute_sum_diff_cost test_compute_sum_diff_cost.cu )
target_link_libraries( test_compute_sum_diff_cost stereo_matching ${OpenCV_LIBS})
add_test(NAME TestComputeSumDiffCost COMMAND test_compute_sum_diff_cost)
