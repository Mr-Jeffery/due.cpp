cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)

set(dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
set(CMAKE_BUILD_FILES_DIRECTORY ${dir})
set(CMAKE_BUILD_DIRECTORY ${dir})
set(CMAKE_BINARY_DIR  ${dir})
SET(EXECUTABLE_OUTPUT_PATH ${dir})
SET(LIBRARY_OUTPUT_PATH ${dir}lib)
SET(CMAKE_CACHEFILE_DIR ${dir})
# set(Python_EXECUTABLE /home/jeffery/miniconda3/bin/python)
# Set the CUDA compiler
# set(CMAKE_CUDA_COMPILER /home/jeffery/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/cuda-11.8.0-njuq5zswd2ene635o4nk2rhwrq3pv3gw/bin/nvcc)

project(env_test LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find CUDA
find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# find_package(nvtx3 REQUIRED)
# include_directories(${nvtx3_INCLUDE_DIRS})

# Find Torch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Add executable
add_executable(env_test src/tests/env_test.cpp)
add_executable(config_test src/tests/config_test.cpp)


# Link libraries
# target_link_libraries(envtest PUBLIC CUDA::cublas)
target_link_libraries(env_test PUBLIC CUDA::cublas ${TORCH_LIBRARIES})
target_link_libraries(config_test PUBLIC CUDA::cublas ${TORCH_LIBRARIES} yaml-cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found")
    target_link_libraries(env_test PUBLIC OpenMP::OpenMP_CXX)
endif()
