#include "../../src/due.hpp"

// ConfigData conf_data;
// ConfigNet conf_net;
// ConfigTrain conf_train;
int main(){
    torch::Device device(torch::kCUDA); 
    // Load the configuration for the modules: datasets, networks, and models
    auto [conf_data, conf_net, conf_train] = read_config("config.yaml");

    // Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
    auto raw_data_loader = RawDataLoader(conf_data);
    auto train_dataset = raw_data_loader.load("/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");

    std::cout << "vmin: " << raw_data_loader.vmin.sizes() << std::endl;
    std::cout << "vmax: " << raw_data_loader.vmax.sizes() << std::endl;

    // Construct a neural network
    auto mynet = ResNet(raw_data_loader.vmin, raw_data_loader.vmax, conf_net);
    mynet.to(device);

    // Define and train a model, save necessary information of the training history
    auto model = ODE(train_dataset, (Affine*)&mynet, conf_train);

    std::cout << model.train_dataset.data.sizes() << std::endl;

    model.train();
}