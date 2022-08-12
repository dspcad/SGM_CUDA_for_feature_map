cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project(simple_stereo_matching LANGUAGES CXX CUDA)
include(CTest)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include_directories( include )

add_subdirectory (src)
add_subdirectory (example)
add_subdirectory (test)

