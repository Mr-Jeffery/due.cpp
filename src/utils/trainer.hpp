#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>

torch::nn::AnyModule get_activation(const std::string& name);

std::unique_ptr<torch::optim::Optimizer> get_optimizer(
    const std::string& name,
    torch::nn::Module& model,
    double lr
);

std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> get_loss(const std::string& name);

#endif // TRAINER_HPP