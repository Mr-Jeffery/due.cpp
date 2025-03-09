#pragma once

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
// #include <matio.h> // For loading .mat files

class ODEDataset {
public:
    ODEDataset(const std::map<std::string, torch::IValue>& config)
        : problem_dim_(config.at("problem_dim").toInt()), memory_steps_(config.at("memory").toInt()),
          multi_steps_(config.at("multi_steps").toInt()), nbursts_(config.at("nbursts").toInt()),
          dtype_(config.at("dtype").toStringRef()) {
        assert(memory_steps_ >= 0);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> load(const std::string& file_path_train, const std::string& file_path_test = "") {
        
    }

private:
    torch::Tensor normalize(torch::Tensor data) {
        vmax_ = std::get<0>(torch::max(data, 0, true));
        vmin_ = std::get<0>(torch::min(data, 0, true));
        auto target = 2 * (data - 0.5 * (vmax_ + vmin_)) / (vmax_ - vmin_);
        return torch::clamp(target, -1, 1);
    }


    int64_t problem_dim_;
    int64_t memory_steps_;
    int64_t multi_steps_;
    int64_t nbursts_;
    std::string dtype_;
    torch::Tensor vmin_, vmax_;
};