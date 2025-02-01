# Install libtorch
# mkdir third_party
# cd third_party
# wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip
# unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
# rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
# git clone https://github.com/NVIDIA/NVTX.git
spack install libyaml cuda
spack load libyaml cuda
export CUDACXX=$(which nvcc)
# export CUDA_DIR=
# export Torch_DIR=/home/jeffery/grad/cpp/libtorch/share/cmake/Torch
# export PATH=/home/jeffery/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/cuda-11.8.0-njuq5zswd2ene635o4nk2rhwrq3pv3gw/bin:$PATH
# export LD_LIBRARY_PATH=/home/jeffery/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/cuda-11.8.0-njuq5zswd2ene635o4nk2rhwrq3pv3gw/lib64:/home/jeffery/grad/cpp/third_party/libtorch/lib:$LD_LIBRARY_PATH
export PROJECT_DIR=/home/jeffery/grad/cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release\
    -DCAFFE2_USE_CUDNN=1\
    -DPYTHON_EXECUTABLE=$(which python)\
    -DCUDAToolkit_ROOT=/home/jeffery/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/cuda-11.8.0-njuq5zswd2ene635o4nk2rhwrq3pv3gw\
    -DTorch_DIR=$PROJECT_DIR/third_party/libtorch/share/cmake/Torch .. 
make -j 16
./envtest
    # -Dnvtx2_DIR=$PROJECT_DIR/third_party/NVTX/c\