#include <torch/torch.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>

#include "datasets/ode.cpp"
#include "models/ode.cpp"
#include "networks/fcn.hpp"
#include "networks/nn.hpp"
#include "utils/utils.hpp"