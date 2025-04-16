#ifndef FCN_HPP
#define FCN_HPP

#include "../utils/utils.h"
#include "../utils/config.hpp"
#include "../utils/trainer.hpp"
#include "nn.hpp"

// Affine class
class Affine : public NN {
public:
    torch::Tensor vmin, vmax;
    std::string dtype;
    int memory, output_dim, input_dim;
    torch::nn::Linear mDMD{nullptr};

    Affine(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor predict(torch::Tensor x, int steps, torch::Device device);
};

// MLP class
class MLP : public torch::nn::Module {
public:
    std::string dtype;
    int output_dim, memory, input_dim, depth, width;
    torch::nn::ModuleList layers;
    std::function<torch::Tensor(torch::Tensor)> activation;

    MLP(const ConfigNet& config);
    torch::Tensor forward(torch::Tensor x);

private:
    void set_seed(int seed);
};

// ResNet class
class ResNet : public Affine {
public:
    MLP mlp;

    ResNet(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config);
    torch::Tensor forward(torch::Tensor x);
};

// Uncomment and define these classes if needed
// class GResNetImpl : public AffineImpl { ... };
// TORCH_MODULE(GResNet);

// class OSGNetImpl : public torch::nn::Module { ... };
// TORCH_MODULE(OSGNet);

// class DualOSGNetImpl : public OSGNetImpl { ... };
// TORCH_MODULE(DualOSGNet);

#endif // FCN_HPP