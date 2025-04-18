#include "ode.hpp"

static void ode_debug() {
    static int count = 0;
    std::cout << "ODE debug count: " << count++ << std::endl;
}

ODE::ODE(at::Tensor *trainX_,
    at::Tensor *trainY_,
    Affine *net_,
    ConfigTrain config
) :
    trainX(trainX_),
    trainY(trainY_),
    net(net_),
    nepochs(config.epochs),
    bsize(config.batch_size),
    lr(config.learning_rate),
    do_validation(config.valid > 0),
    valid(config.valid > 0 ? config.valid : 0.f),
    verbose(config.verbose),
    device(config.device)
{


    set_seed(config.seed);
    *trainX = trainX->to(device);
    *trainY = trainY->to(device);
    multi_steps = trainY->size(-1);
    memory_steps = (trainX->size(1) > trainY->size(1))
        ? trainX->size(1) / trainY->size(1)
        : 1;

    net->to(device);

    optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(lr));
    if (do_validation) {
        int64_t split_size = static_cast<int64_t>(valid * trainX->size(0));
        *validX = trainX->slice(0, trainX->size(0) - split_size, trainX->size(0));
        *validY = trainY->slice(0, trainY->size(0) - split_size, trainY->size(0));
        *trainX = trainX->slice(0, 0, trainX->size(0) - split_size);
        *trainY = trainY->slice(0, 0, trainY->size(0) - split_size);

        auto valid_loader = std::make_unique<torch::data::StatelessDataLoader<
            sampleDataLoader,
            torch::data::samplers::SequentialSampler>>(
            sampleDataLoader(validX, validY),
            torch::data::samplers::SequentialSampler(validX->size(0)),
            torch::data::DataLoaderOptions().batch_size(bsize));
    }
    auto train_loader = std::make_unique<torch::data::StatelessDataLoader<
        sampleDataLoader,
        torch::data::samplers::RandomSampler>>(
        sampleDataLoader(trainX, trainY),
        torch::data::samplers::RandomSampler(trainX->size(0)),
        torch::data::DataLoaderOptions().batch_size(bsize));
};

void ODE::train() {
    summary();
    double minLoss = 1.0e15;
    auto startOverall = std::chrono::steady_clock::now();
    auto start = startOverall;

    for (int64_t ep = 0; ep < nepochs; ++ep) {
        net->train();
        double trainLoss = 0.0;
        int64_t batchCount = 0;

        for (auto& batch : *train_loader) {
            auto xx = batch[0].data.to(device);
            auto yy = batch[1].data.to(device);

            ode_debug();

            optimizer->zero_grad();
            auto pred = torch::zeros_like(yy);
            for (int64_t t = 0; t < multi_steps; ++t) {
                auto out = net->forward(xx);
                pred.slice(-1, t, t+1) = out.unsqueeze(-1);
                xx = torch::cat({xx.slice(-1, net->output_dim, xx.size(-1)), out}, -1);
            }
            auto loss = torch::mse_loss(pred, yy);
            loss.backward();
            optimizer->step();
            trainLoss += loss.item<double>();
            batchCount++;
        }
        trainLoss /= static_cast<double>(batchCount);

        if (do_validation) {
            double validLoss = validate();
            if (validLoss < minLoss) {
                minLoss = validLoss;
                // Save model here if needed
            }
            if (verbose && (ep + 1) % 10 == 0) {
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
            if (verbose && (ep + 1) % 10 == 0) {
                auto end = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
                std::cout << "Epoch " << ep+1 << " --- Time: " << elapsed
                            << "s --- Training loss: " << trainLoss << "\n";
                start = end;
            }
        }
    }
}

double ODE::validate() {
    net->eval();
    torch::NoGradGuard no_grad;
    double totalLoss = 0.0;
    int64_t count = 0;
    for (auto& batch : *valid_loader) {
        auto xx = batch[0].data.to(device);
        auto yy = batch[1].data.to(device);

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

void ODE::summary() {
    std::cout << "Number of trainable parameters: "
                << count_params(*net)     << "\n";
    std::cout << "Number of epochs: " << nepochs  << "\n";
    std::cout << "Batch size: "      << bsize    << "\n";
}

void ODE::set_seed(unsigned int seed) {
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
}

