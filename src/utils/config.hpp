#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <tuple>
#include <torch/torch.h>
// #include <yaml-cpp/yaml.h>

struct ConfigData {
    std::string problem_type;
    int nbursts;
    int memory;
    int multi_steps;
    uint problem_dim;

    uint seed;
    std::string dtype;
};

struct ConfigNet {
    int depth;
    int width;
    std::string activation;

    uint problem_dim;

    std::string device;

    uint seed;
    std::string dtype;
};

struct ConfigTrain {
    std::string device;
    int valid;
    int epochs;
    int batch_size;
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

#endif // CONFIG_HPP