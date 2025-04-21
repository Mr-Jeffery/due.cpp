#include "config.hpp"

std::tuple<ConfigData, ConfigNet, ConfigTrain> read_config(const std::string& config_path) {
    YAML::Node config = YAML::LoadFile(config_path);

    ConfigData confData;
    confData.problem_type = config["data"]["problem_type"].as<std::string>();
    confData.nbursts      = config["data"]["nbursts"].as<uint>();
    confData.memory       = config["data"]["memory"].as<uint>(0); // Default value
    confData.multi_steps  = config["data"]["multi_steps"].as<uint>();
    confData.problem_dim  = config["data"]["problem_dim"].as<uint>();

    confData.seed         = config["seed"].as<int>();
    confData.dtype        = config["dtype"].as<std::string>();

    ConfigNet confNet;
    confNet.depth      = config["network"]["depth"].as<int>();
    confNet.width      = config["network"]["width"].as<int>();
    confNet.activation = config["network"]["activation"].as<std::string>();
    confNet.device     = config["training"]["device"].as<std::string>();

    confNet.problem_dim  = confData.problem_dim;
    confNet.memory       = confData.memory;
    confNet.seed       = confData.seed;
    confNet.dtype      = confData.dtype;

    ConfigTrain confTrain;
    confTrain.device          = config["training"]["device"].as<std::string>();
    confTrain.valid           = config["training"]["valid"].as<float>();
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
    
    // // Logic to populate problem_dim
    // if (confData.problem_type == "1d_irregular" || confData.problem_type == "1d_regular") {
    //     confNet.problem_dim = 2 * confTrain.modes + 1;
    // } else if (confData.problem_type == "2d_irregular" || confData.problem_type == "2d_regular") {
    //     int val = 2 * confTrain.modes + 1;
    //     confNet.problem_dim = val * val;
    // } else {
    //     confNet.problem_dim = confData.problem_dim;
    // }

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

