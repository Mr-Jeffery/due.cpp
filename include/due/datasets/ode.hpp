
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <random>
#include <algorithm>

#include <due/utils/config.hpp>

#pragma once

class ODEDataset : public torch::data::Dataset<ODEDataset> {
public:
    torch::Tensor data, targets;
    uint problem_dim;
    uint memory_steps;
    uint multi_steps;

    // Constructor
    ODEDataset();
    // Constructor with data and targets
    ODEDataset(torch::Tensor _data, torch::Tensor _targets);

    torch::optional<size_t> size() const override;

    // Returns a single sample at index
    torch::data::Example<> get(size_t index) override;
};


class RawDataLoader{
private:
    auto normalize_(torch::Tensor data);
public:
    // number of free dimentions
    uint problem_dim;
    // number of time steps in the past
    uint memory_steps;
    // number of time steps in the future
    uint multi_steps;
    // number of bursts selected from every trajectory
    uint nbursts;
    // data type
    std::string dtype;
    // min and max values of the data
    torch::Tensor vmin, vmax;

    RawDataLoader(const ConfigData& config);

    // Load data from .pt files
    ODEDataset load(const std::string& file_path_train = "", const std::string& file_path_test = "");
};

