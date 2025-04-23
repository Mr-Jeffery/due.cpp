#include <string>
#include <tuple>
#include <fstream>
#include <sstream>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#pragma once
struct ConfigData {
    std::string problem_type;
    uint nbursts;
    uint memory;
    uint multi_steps;
    uint problem_dim;

    uint seed;
    std::string dtype;
};

struct ConfigNet {
    uint depth;
    uint width;
    std::string activation;

    uint problem_dim;
    uint memory;

    std::string device;

    uint seed;
    std::string dtype;
};

struct ConfigTrain {
    std::string device;
    float valid;
    uint epochs;
    uint batch_size;
    std::string optimizer;
    std::string scheduler;
    double learning_rate;
    int verbose;
    std::string loss;
    std::string save_path;

    uint seed;
    std::string dtype;
};
std::tuple<ConfigData, ConfigNet, ConfigTrain> read_config(const std::string& config_path);
torch::Tensor read_csv(const std::string& path, const std::string& dtype);