#include <due/utils/config.hpp>
#include <due/utils/trainer.hpp>
#include <due/networks/nn.hpp>

#pragma once
// Affine class
struct Affine : torch::nn::Module {
    torch::Tensor vmin, vmax;
    std::string dtype;
    int memory, output_dim, input_dim;
    torch::nn::Linear mDMD{nullptr};

    Affine(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config);
    virtual torch::Tensor forward(torch::Tensor x);
    torch::Tensor predict(torch::Tensor x, int steps, torch::Device device);

    virtual ~Affine() = default;
};

// MLP class
struct MLP : torch::nn::Module {
    std::string dtype;
    int output_dim, memory, input_dim, depth, width;
    torch::nn::ModuleList layers;
    std::function<torch::Tensor(torch::Tensor)> activation;

    MLP(const ConfigNet& config);
    torch::Tensor forward(torch::Tensor x);

    // void set_seed(unsigned int seed);
};

// ResNet class
struct ResNet : Affine {
    std::shared_ptr<MLP> mlp;

    ResNet(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config);
    torch::Tensor forward(torch::Tensor x) override;
};