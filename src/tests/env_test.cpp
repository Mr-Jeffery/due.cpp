// #include <Eigen/Dense>
#include <cublas_v2.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    // Test Eigen
    // Eigen::Matrix3d mat;
    // mat << 1, 2, 3,
    //        4, 5, 6,
    //        7, 8, 9;

    // std::cout << "Eigen Matrix:\n" << mat << std::endl;

    // Test CUDA cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "cuBLAS initialized successfully!" << std::endl;
    }
    // Test libtorch with CUDA
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Testing libtorch with CUDA..." << std::endl;
        torch::Tensor tensor = torch::rand({2, 3}, torch::device(torch::kCUDA));
        std::cout << "Tensor on GPU:\n" << tensor << std::endl;
    } else {
        std::cerr << "CUDA is not available!" << std::endl;
    }
    // Clean up cuBLAS
    cublasDestroy(handle);

    return 0;
}