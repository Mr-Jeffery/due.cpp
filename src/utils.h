#include <torch/torch.h>
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

// Simple structs to hold config data
// ...existing code...


struct ConfigData {
    std::string problem_type;
    int nbursts;
    int memory;
    int multi_steps;
    int problem_dim;
    int seed;
    std::string dtype;
};

struct ConfigNet {
    int depth;
    int width;
    std::string activation;
    int seed;
    std::string dtype;
    std::string device;
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
    int seed;
    std::string dtype;
};

std::tuple<ConfigData, ConfigNet, ConfigTrain> read_config(const std::string& config_path) {
    YAML::Node config = YAML::LoadFile(config_path);

    ConfigData confData;
    confData.seed         = config["seed"].as<int>();
    confData.dtype        = config["dtype"].as<std::string>();
    confData.problem_type = config["data"]["problem_type"].as<std::string>();
    confData.nbursts      = config["data"]["nbursts"].as<int>();
    confData.memory       = config["data"]["memory"].as<int>();
    confData.multi_steps  = config["data"]["multi_steps"].as<int>();
    confData.problem_dim  = config["data"]["problem_dim"].as<int>();

    ConfigNet confNet;
    confNet.depth      = config["network"]["depth"].as<int>();
    confNet.width      = config["network"]["width"].as<int>();
    confNet.activation = config["network"]["activation"].as<std::string>();
    confNet.seed       = confData.seed;
    confNet.dtype      = confData.dtype;
    confNet.device     = config["training"]["device"].as<std::string>();

    ConfigTrain confTrain;
    confTrain.device          = config["training"]["device"].as<std::string>();
    confTrain.valid           = config["training"]["valid"].as<int>();
    confTrain.epochs          = config["training"]["epochs"].as<int>();
    confTrain.batch_size      = config["training"]["batch_size"].as<int>();
    confTrain.optimizer       = config["training"]["optimizer"].as<std::string>();
    confTrain.scheduler       = config["training"]["scheduler"].as<std::string>();
    confTrain.learning_rate   = config["training"]["learning_rate"].as<double>();
    confTrain.verbose         = config["training"]["verbose"].as<int>();
    confTrain.loss            = config["training"]["loss"].as<std::string>();
    confTrain.save_path       = config["training"]["save_path"].as<std::string>();
    confTrain.seed            = confData.seed;
    confTrain.dtype           = confData.dtype;

    return std::make_tuple(confData, confNet, confTrain);
}


std::tuple<ConfigData, ConfigNet, ConfigTrain> read_config(const std::string& config_path) {
    YAML::Node config = YAML::LoadFile(config_path);

    ConfigData confData;
    confData.problem_type = config["data"]["problem_type"].as<std::string>();
    confData.seed         = config["seed"].as<int>();
    confData.dtype        = config["dtype"].as<std::string>();
    // If these keys can be absent, add safety checks.
    confData.memory       = config["data"]["memory"] ? config["data"]["memory"].as<int>() : 0;
    confData.problem_dim  = config["data"]["problem_dim"] ? config["data"]["problem_dim"].as<int>() : 0;

    ConfigNet confNet;
    confNet.seed   = confData.seed;
    confNet.dtype  = confData.dtype;
    confNet.memory = confData.memory;
    confNet.device = config["training"]["device"].as<std::string>();
    confNet.problem_dim = 0;

    ConfigTrain confTrain;
    confTrain.seed   = confData.seed;
    confTrain.dtype  = confData.dtype;
    confTrain.device = config["training"]["device"].as<std::string>();
    confTrain.modes  = config["training"]["modes"].as<int>();

    // Logic to populate problem_dim
    if (confData.problem_type == "1d_irregular" || confData.problem_type == "1d_regular") {
        confNet.problem_dim = 2 * confTrain.modes + 1;
    } else if (confData.problem_type == "2d_irregular" || confData.problem_type == "2d_regular") {
        int val = 2 * confTrain.modes + 1;
        confNet.problem_dim = val * val;
    } else {
        confNet.problem_dim = confData.problem_dim;
    }

    return std::make_tuple(confData, confNet, confTrain);
}

torch::Tensor read_csv(const std::string& path, const std::string& dtype) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    int cols = -1, rows = 0;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int currentCols = 0;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
            currentCols++;
        }
        if (cols < 0) cols = currentCols;
        rows++;
    }
    file.close();

    auto options = torch::TensorOptions().dtype(
        (dtype == "single") ? torch::kFloat32 : torch::kFloat64);

    torch::Tensor temp = torch::from_blob(values.data(), {rows, cols}, options).clone();
    return temp;
}


// Equivalent of rel_l1_norm(true, pred)
torch::Tensor rel_l1_norm(const torch::Tensor& truth, const torch::Tensor& pred) {
    auto bsize = truth.size(0);
    auto diff = torch::norm(truth.reshape({bsize, -1}) - pred.reshape({bsize, -1}), 1, /*dim=*/1);
    auto base = torch::norm(truth.reshape({bsize, -1}), 1, /*dim=*/1);
    return torch::mean(diff / base);
}

// Equivalent of rel_l2_norm(true, pred)
torch::Tensor rel_l2_norm(const torch::Tensor& truth, const torch::Tensor& pred) {
    auto bsize = truth.size(0);
    auto diff = torch::norm(truth.reshape({bsize, -1}) - pred.reshape({bsize, -1}), 2, /*dim=*/1);
    auto base = torch::norm(truth.reshape({bsize, -1}), 2, /*dim=*/1);
    return torch::mean(diff / base);
}

// Equivalent of rel_l2_norm_pde(true, pred)
torch::Tensor rel_l2_norm_pde(const torch::Tensor& truth, const torch::Tensor& pred) {
    // (N,L,D,T) -> reshape to (N, -1, D, T)
    auto t_reshaped = truth.reshape({truth.size(0), -1, truth.size(-2), truth.size(-1)});
    auto p_reshaped = pred.reshape({pred.size(0), -1, pred.size(-2), pred.size(-1)});
    auto diff = torch::norm(t_reshaped - p_reshaped, 2, /*dim=*/1);
    auto base = torch::norm(t_reshaped, 2, /*dim=*/1);
    return torch::mean(diff / base);
}

// Equivalent of rel_l1_norm_pde(true, pred)
torch::Tensor rel_l1_norm_pde(const torch::Tensor& truth, const torch::Tensor& pred) {
    auto diff = torch::norm(truth - pred, 1, /*dim=*/1);
    auto base = torch::norm(truth, 1, /*dim=*/1);
    return torch::mean(diff / base);
}

// Equivalent of get_activation(name)
torch::nn::AnyModule get_activation(const std::string& name) {
    if (name == "tanh" || name == "Tanh")
        return torch::nn::AnyModule(torch::nn::Tanh());
    else if (name == "relu" || name == "ReLU")
        return torch::nn::AnyModule(torch::nn::ReLU());
    else if (name == "leaky_relu" || name == "LeakyReLU")
        return torch::nn::AnyModule(torch::nn::LeakyReLU());
    else if (name == "sigmoid" || name == "Sigmoid")
        return torch::nn::AnyModule(torch::nn::Sigmoid());
    else if (name == "softplus" || name == "Softplus")
        return torch::nn::AnyModule(torch::nn::Softplus());
    else if (name == "gelu" || name == "Gelu")
        // gelu in C++ is available via functional::gelu or a custom module
        // For example:
        return torch::nn::AnyModule(torch::nn::Functional(torch::nn::functional::gelu));
    else
        throw std::runtime_error("unknown or unsupported activation function: " + name);
}

// Equivalent of get_optimizer(name, model, lr)
std::unique_ptr<torch::optim::Optimizer> get_optimizer(
    const std::string& name,
    torch::nn::Module& model,
    double lr
) {
    if (name == "adam" || name == "Adam" || name == "ADAM") {
        return std::make_unique<torch::optim::Adam>(model.parameters(), lr);
    } else if (name == "nadam" || name == "NAdam" || name == "NADAM") {
        return std::make_unique<torch::optim::NAdam>(model.parameters(), lr);
    } else if (name == "adamw" || name == "AdamW" || name == "ADAMW") {
        return std::make_unique<torch::optim::AdamW>(model.parameters(), lr);
    } else if (name == "SGD" || name == "sgd" || name == "Sgd") {
        return std::make_unique<torch::optim::SGD>(model.parameters(), lr);
    }
    throw std::runtime_error("unknown or unsupported optimizer: " + name);
}

// Equivalent of get_schedule(...)
std::shared_ptr<torch::optim::LRScheduler> get_schedule(
    torch::optim::Optimizer& optimizer,
    const std::string& name,
    int epochs,
    int batch_size,
    int ntrain
) {
    auto steps_per_epoch = ntrain / batch_size;
    if (steps_per_epoch <= 0) {
        return nullptr;
    }

    if (name == "cyclic_cosine" || name == "Cyclic_cosine" || name == "Cyclic_Cosine") {
        // CosineAnnealingWarmRestarts is not yet in the C++ API in older versions.
        // Direct usage if available:
        return std::make_shared<torch::optim::CosineAnnealingWarmRestarts>(
            dynamic_cast<torch::optim::Adam&>(optimizer),
            /*T_0=*/(epochs / 5) * steps_per_epoch
        );
    } else if (name == "cosine" || name == "Cosine") {
        return std::make_shared<torch::optim::CosineAnnealingLR>(
            dynamic_cast<torch::optim::Adam&>(optimizer),
            epochs * steps_per_epoch
        );
    } else if (name == "one_cycle" || name == "One_Cycle" || name == "OneCycle") {
        // OneCycleLR is also not in older C++ APIs. Example usage:
        return std::make_shared<torch::optim::OneCycleLR>(
            dynamic_cast<torch::optim::Adam&>(optimizer),
            /*max_lr=*/optimizer.param_groups().front().options().get<torch::optim::AdamOptions>()->lr(),
            /*steps=*/epochs * steps_per_epoch
        );
    } else if (name == "none" || name == "None") {
        return nullptr;
    }
    throw std::runtime_error("unknown or unsupported learning schedule: " + name);
}

// Equivalent of get_loss(name)
std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> get_loss(const std::string& name) {
    if (name == "mse" || name == "Mse" || name == "MSE") {
        return [](const auto& t, const auto& p) {
            return torch::mse_loss(p, t);
        };
    } else if (name == "mae" || name == "Mae" || name == "MAE") {
        return [](const auto& t, const auto& p) {
            return torch::l1_loss(p, t);
        };
    } else if (name == "rel_l2" || name == "Rel_l2" || name == "Rel_L2") {
        return [](const auto& t, const auto& p) {
            return rel_l2_norm(t, p);
        };
    } else if (name == "rel_l2_pde" || name == "Rel_l2_pde" || name == "Rel_L2_pde") {
        return [](const auto& t, const auto& p) {
            return rel_l2_norm_pde(t, p);
        };
    } else if (name == "rel_l1_pde" || name == "Rel_l1_pde" || name == "Rel_L1_pde") {
        return [](const auto& t, const auto& p) {
            return rel_l1_norm_pde(t, p);
        };
    } else if (name == "rel_l1" || name == "Rel_l1" || name == "Rel_L1") {
        return [](const auto& t, const auto& p) {
            return rel_l1_norm(t, p);
        };
    }
    throw std::runtime_error("unknown or unsupported loss function: " + name);
}