#pragma once

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>

#include "../utils/config.hpp"
// #include <matio.h> // For loading .mat files

class ODEDataset {
public:
    int problem_dim;
    int memory_steps;
    int multi_steps;
    int nbursts;
    std::string dtype;
    torch::Tensor vmin, vmax;

    ODEDataset(const ConfigData& config)
        : problem_dim(config.problem_dim),
          memory_steps(config.memory),
          multi_steps(config.multi_steps),
          nbursts(config.nbursts),
          dtype(config.dtype)
    {
        assert(problem_dim > 0);
        assert(memory_steps >= 0);
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> load(const std::string& file_path_train, const std::string& file_path_test = "") {
        int N = 1000;
        int d = this->problem_dim;
        int T = 1001;
        // auto target = torch::zeros({N*this->nbursts, d, this->memory_steps + this->multi_steps + 2});
        // for (int i = 0; i < N; i++) {
        //     auto inits = torch::randint(0, T-this->memory_steps-this->multi_steps-1, {this->nbursts}); // (0, K-subset+1, J0)

        // }
        
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
        auto trainX = torch::randn({10000, 2});
        auto trainY = torch::randn({10000, 2, 11});

        return std::make_tuple(trainX, trainY, torch::zeros({1,2,1}), torch::ones({1,2,1}));
    }

private:
    torch::Tensor normalize(torch::Tensor data) {
        vmax = std::get<0>(torch::max(data, 0, true));
        vmin = std::get<0>(torch::min(data, 0, true));
        auto target = 2 * (data - 0.5 * (vmax + vmin)) / (vmax - vmin);
        return torch::clamp(target, -1, 1);
    }


};