# Check if libtorch and yaml-cpp are already installed
if [ -d "third_party/libtorch" ] && [ -d "third_party/yaml-cpp-0.8.0/build" ]; then
    echo "libtorch and yaml-cpp already installed. Skipping installation."
else
    # Install libtorch
    mkdir -p third_party
    cd third_party
    if [ ! -d "libtorch" ]; then
        wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip
        unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
        rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
    fi

    # Install NVTX
    # git clone https://github.com/NVIDIA/NVTX.git

    # Install yaml-cpp
    if [ ! -d "yaml-cpp-0.8.0/build" ]; then
        wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
        tar -xvf 0.8.0.tar.gz
        rm 0.8.0.tar.gz
        mkdir -p yaml-cpp-0.8.0/build
        cd yaml-cpp-0.8.0/build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j 16
        cd ../..
    fi
    cd .. # go back to the root directory

    spack install cuda cudnn # Comment this line if you have installed cuda and cudnn
fi

spack load cuda cudnn
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