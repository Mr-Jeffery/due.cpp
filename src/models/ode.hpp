#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include "../utils/utils.hpp"
#include "../datasets/ode_parody.hpp"
#include "../networks/fcn.hpp"




class ODE {
private:
    std::shared_ptr<torch::Tensor> trainX, trainY, validX, validY;
    std::shared_ptr<Affine> net;
    int64_t multi_steps;
    int64_t memory_steps;
    int64_t nepochs;
    int64_t bsize;
    double lr;
    bool do_validation;
    float valid;
    bool verbose;
    torch::Device device;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::TensorDataset,
        torch::data::samplers::RandomSampler>> train_loader;
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::TensorDataset,
        torch::data::samplers::SequentialSampler>> valid_loader;

public:
    ODE(at::Tensor *trainX_,
        at::Tensor *trainY_,
        Affine *net_,
        ConfigTrain config
    );

    void train();

    double validate();

    void summary();

    void set_seed(unsigned int seed);

};
