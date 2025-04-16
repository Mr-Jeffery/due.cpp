#include "../../src/due.hpp"

// ConfigData conf_data;
// ConfigNet conf_net;
// ConfigTrain conf_train;
torch::Device device(torch::kCUDA); 
// Load the configuration for the modules: datasets, networks, and models
auto [conf_data, conf_net, conf_train] = read_config("config.yaml");

// Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles
auto dataset = ODEDataset(conf_data);
auto [trainX, trainY, vmin, vmax] = dataset.load("train.mat", "test.mat");

// Construct a neural network
auto mynet = ResNet(vmin, vmax, conf_net);

// Define and train a model, save necessary information of the training history
auto model = ODE(&trainX, &trainY, (Affine*)&mynet, conf_train);