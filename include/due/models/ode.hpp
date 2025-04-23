#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <due/utils/utils.hpp>
#include <due/datasets/ode.hpp>
#include <due/networks/fcn.hpp>

class ODE {
public:
    ODEDataset train_dataset, valid_dataset;
    Affine *net;
    int64_t multi_steps;
    int64_t memory_steps;
    int64_t problem_dim;
    int64_t nepochs;
    int64_t bsize;
    double lr;
    bool do_validation;
    float valid;
    int verbose;
    torch::Device device;
    std::string save_path;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::shared_ptr<torch::optim::StepLR> scheduler;

    ODE(
        ODEDataset dataset,
        Affine *net_,
        ConfigTrain config
    );

    void train();

    double validate();

    void summary();

    void set_seed(unsigned int seed);

};
