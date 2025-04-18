#include <torch/torch.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>

#include "datasets/ode_parody.hpp"
#include "models/ode.cpp"
#include "networks/fcn.hpp"
#include "networks/nn.hpp"
#include "utils/utils.hpp"

// ConfigNet conf_net;
// ConfigTrain conf_train;
// ConfigData conf_data;
// torch::Device device;
// torch::Tensor vmin, vmax;

// auto mynet = std::make_shared<ResNet>(vmin, vmax, conf_net);
// auto model = std::make_shared<ODE>(mynet, conf_data.problem_dim, conf_data.memory, conf_data.nbursts);