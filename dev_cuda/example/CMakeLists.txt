find_package(OpenCV REQUIRED)
set_source_files_properties(demo.cu PROPERTIES LANGUAGE CUDA)
add_executable( demo demo.cu )
target_link_libraries( demo stereo_matching ${OpenCV_LIBS})

set_source_files_properties(demo_sgsm.cu PROPERTIES LANGUAGE CUDA)
add_executable( demo_sgsm demo_sgsm.cu )
target_link_libraries( demo_sgsm stereo_matching ${OpenCV_LIBS})

