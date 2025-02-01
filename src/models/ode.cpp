#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

// Placeholder network module
struct MyNetImpl : torch::nn::Module {
    int64_t output_dim;
    MyNetImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim_)
      : fc1(register_module("fc1", torch::nn::Linear(input_dim, hidden_dim))),
        fc2(register_module("fc2", torch::nn::Linear(hidden_dim, output_dim_))),
        output_dim(output_dim_) {}

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = torch::relu(fc1->forward(x));
        return fc2->forward(out);
    }

    torch::nn::Linear fc1, fc2;
};
TORCH_MODULE(MyNet);

class ODE {
public:
    ODE(const torch::Tensor& trainX_,
        const torch::Tensor& trainY_,
        MyNet& network,
        int64_t epochs,
        int64_t batch_size,
        double lr,
        bool do_validation,
        float valid_ratio,
        bool verbose,
        torch::DeviceType device,
        unsigned seed)
      : trainX(trainX_.clone()),
        trainY(trainY_.clone()),
        net(network),
        nepochs(epochs),
        bsize(batch_size),
        learning_rate(lr),
        doValidation(do_validation),
        validRatio(valid_ratio),
        verboseN(verbose),
        device_(torch::Device(device)) 
    {
        set_seed(seed);
        trainX = trainX.to(device_);
        trainY = trainY.to(device_);
        multi_steps = trainY.size(-1);
        memory_steps = (trainX.size(1) > trainY.size(1))
            ? trainX.size(1) / trainY.size(1)
            : 1;

        net->to(device_);

        optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(lr));
        if (doValidation && validRatio > 0.f) {
            auto splitSize = static_cast<int64_t>(validRatio * trainX.size(0));
            validX = trainX.slice(0, trainX.size(0) - splitSize, trainX.size(0)).clone();
            validY = trainY.slice(0, trainY.size(0) - splitSize, trainY.size(0)).clone();
            trainX = trainX.slice(0, 0, trainX.size(0) - splitSize).clone();
            trainY = trainY.slice(0, 0, trainY.size(0) - splitSize).clone();
        }
        auto datasetTrain = torch::data::datasets::TensorDataset(trainX, trainY).map(torch::data::transforms::Stack<>());
        auto trainLoader = std::make_unique<torch::data::StatelessDataLoader<decltype(datasetTrain)>>(
            datasetTrain, bsize);

        if (doValidation && validRatio > 0.f) {
            auto datasetValid = torch::data::datasets::TensorDataset(validX, validY).map(torch::data::transforms::Stack<>());
            auto validLoader = std::make_unique<torch::data::StatelessDataLoader<decltype(datasetValid)>>(
                datasetValid, bsize);
        }
    }

    void train() {
        summary();
        double minLoss = 1.0e15;
        auto startOverall = std::chrono::steady_clock::now();
        auto start = startOverall;

        for (int64_t ep = 0; ep < nepochs; ++ep) {
            net->train();
            double trainLoss = 0.0;
            int64_t batchCount = 0;

            for (auto& batch : *trainLoader) {
                auto xx = batch.data.to(device_);
                auto yy = batch.target.to(device_);

                optimizer->zero_grad();
                auto pred = torch::zeros_like(yy);
                for (int64_t t = 0; t < multi_steps; ++t) {
                    auto out = net->forward(xx);
                    pred.slice(-1, t, t+1) = out;
                    xx = torch::cat({xx.slice(-1, net->output_dim, xx.size(-1)), out}, -1);
                }
                auto loss = torch::mse_loss(pred, yy);
                loss.backward();
                optimizer->step();
                trainLoss += loss.item<double>();
                batchCount++;
            }
            trainLoss /= static_cast<double>(batchCount);

            if (doValidation && validRatio > 0.f) {
                double validLoss = validate();
                if (validLoss < minLoss) {
                    minLoss = validLoss;
                    // Save model here if needed
                }
                if (verboseN && (ep + 1) % 10 == 0) {
                    auto end = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                    std::cout << "Epoch " << ep+1 << " --- Time: " << elapsed
                              << "s --- Training loss: " << trainLoss
                              << " --- Validation loss: " << validLoss << "\n";
                    start = end;
                }
            } else {
                if (trainLoss < minLoss) {
                    minLoss = trainLoss;
                }
                if (verboseN && (ep + 1) % 10 == 0) {
                    auto end = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                    std::cout << "Epoch " << ep+1 << " --- Time: " << elapsed
                              << "s --- Training loss: " << trainLoss << "\n";
                    start = end;
                }
            }
        }
    }

    double validate() {
        net->eval();
        torch::NoGradGuard no_grad;
        double totalLoss = 0.0;
        int64_t count = 0;
        for (auto& batch : *validLoader) {
            auto xx = batch.data.to(device_);
            auto yy = batch.target.to(device_);

            auto pred = torch::zeros_like(yy);
            for (int64_t t = 0; t < multi_steps; ++t) {
                auto out = net->forward(xx);
                pred.slice(-1, t, t+1) = out;
                xx = torch::cat({xx.slice(-1, net->output_dim, xx.size(-1)), out}, -1);
            }
            auto loss = torch::mse_loss(pred, yy);
            totalLoss += loss.item<double>();
            count++;
        }
        return totalLoss / static_cast<double>(count);
    }

    void summary() {
        std::cout << "Number of trainable parameters: "
                  << count_params()     << "\n";
        std::cout << "Number of epochs: " << nepochs  << "\n";
        std::cout << "Batch size: "      << bsize    << "\n";
    }

    void set_seed(unsigned seed) {
        torch::manual_seed(seed);
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
        at::globalContext().setDeterministicCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(false);
    }

    int64_t count_params() {
        int64_t total = 0;
        for (const auto& p : net->parameters()) {
            total += p.numel();
        }
        return total;
    }

private:
    torch::Tensor trainX, trainY, validX, validY;
    MyNet net;
    int64_t multi_steps;
    int64_t memory_steps;
    int64_t nepochs;
    int64_t bsize;
    double learning_rate;
    bool doValidation;
    float validRatio;
    bool verboseN;
    torch::Device device_;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::TensorDataset,
        torch::data::samplers::RandomSampler>> trainLoader;
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::TensorDataset,
        torch::data::samplers::SequentialSampler>> validLoader;
};
