cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
project(cuda_example)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Set the CUDA compiler flags
set(CUDA_NVCC_FLAGS "-arch=sm_30")

# Add the CUDA include directories and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARIES})
cuda_add_library(cuda_lib kernel.cu)

# Add the executable target
add_executable(cuda_example main.cpp kernel.cu)
target_link_libraries(cuda_example cuda_lib)