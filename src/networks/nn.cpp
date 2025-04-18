#include "nn.hpp"

void set_seed(unsigned int seed) {
    setenv("PYTHONHASHSEED", std::to_string(seed).c_str(), 1);
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);
}