# Install libtorch
mkdir third_party
cd third_party
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip

# Install NVTX
# git clone https://github.com/NVIDIA/NVTX.git

# Install yaml-cpp
wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
tar -xvf 0.8.0.tar.gz

spack install cuda # Comment this line if you have installed cuda
spack load cuda
rm 0.8.0.tar.gz 
mkdir yaml-cpp-0.8.0/build
cd yaml-cpp-0.8.0/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 16
cd ../.. 
cd .. # go back to the root directory

# Comment the above lines if you have installed libtorch and yaml-cpp

export CUDACXX=$(which nvcc)
export CUDAToolkit_ROOT=$(dirname $(dirname $CUDACXX))
export PROJECT_DIR=$(pwd)
export Torch_DIR=$PROJECT_DIR/third_party/libtorch/share/cmake/Torch
export PATH=$CUDAToolkit_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAToolkit_ROOT/lib64:$PROJECT_DIR/third_party/libtorch/lib:$LD_LIBRARY_PATH

cmake -DCMAKE_BUILD_TYPE=Release\
    -DCAFFE2_USE_CUDNN=1\
    -DCMAKE_BUILD_TYPE=Realease\
    -DPYTHON_EXECUTABLE=$(which python)\
    -DCUDAToolkit_ROOT=CUDAToolkit_ROOT\
    -DTorch_DIR=Torch_DIR . 
make -j 16 # Number of threads to use for compilation