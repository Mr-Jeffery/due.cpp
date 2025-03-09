#include "../utils/utils.hpp"
#include "../utils/config.cpp"

int main(){
    std::string config_path = "config.yaml";
    auto [confData, confNet, confTrain] = read_config(config_path);
    std::cout << "confData.problem_type: " << confData.problem_type << std::endl;
    std::cout << "confData.nbursts: " << confData.nbursts << std::endl;
    std::cout << "confData.memory: " << confData.memory << std::endl;
    std::cout << "confData.multi_steps: " << confData.multi_steps << std::endl;
    std::cout << "confData.problem_dim: " << confData.problem_dim << std::endl;
    std::cout << "confData.seed: " << confData.seed << std::endl;
    std::cout << "confData.dtype: " << confData.dtype << std::endl;
    std::cout << "confNet.depth: " << confNet.depth << std::endl;
    std::cout << "confNet.width: " << confNet.width << std::endl;
    std::cout << "confNet.activation: " << confNet.activation << std::endl;
    std::cout << "confNet.problem_dim: " << confNet.problem_dim << std::endl;
    std::cout << "confNet.device: " << confNet.device << std::endl;
    std::cout << "confNet.seed: " << confNet.seed << std::endl;
    std::cout << "confNet.dtype: " << confNet.dtype << std::endl;
    std::cout << "confTrain.device: " << confTrain.device << std::endl;
    std::cout << "confTrain.valid: " << confTrain.valid << std::endl;
    std::cout << "confTrain.epochs: " << confTrain.epochs << std::endl;
    std::cout << "confTrain.batch_size: " << confTrain.batch_size << std::endl;
    std::cout << "confTrain.optimizer: " << confTrain.optimizer << std::endl;
    std::cout << "confTrain.scheduler: " << confTrain.scheduler << std::endl;
    std::cout << "confTrain.learning_rate: " << confTrain.learning_rate << std::endl;
    std::cout << "confTrain.verbose: " << confTrain.verbose << std::endl;
    std::cout << "confTrain.loss: " << confTrain.loss << std::endl;
    std::cout << "confTrain.save_path: " << confTrain.save_path << std::endl;
    std::cout << "confTrain.seed: " << confTrain.seed << std::endl;
    std::cout << "confTrain.dtype: " << confTrain.dtype << std::endl;
    return 0;
}