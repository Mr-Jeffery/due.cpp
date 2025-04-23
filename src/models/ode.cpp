#include <due/models/ode.hpp>
#define Slice torch::indexing::Slice
#define None torch::indexing::None

static void ode_debug() {
    static int count = 0;
    std::cout << "ODE debug count: " << count++ << std::endl;
}

ODE::ODE(
    ODEDataset dataset,
    Affine *net_,
    ConfigTrain config
) :
    nepochs(config.epochs),
    bsize(config.batch_size),
    lr(config.learning_rate),
    do_validation(config.valid > 0),
    valid(config.valid > 0 ? config.valid : 0.f),
    verbose(config.verbose),
    device(config.device),
    save_path(config.save_path),
    train_dataset(std::move(dataset)),
    net(net_)
{
    multi_steps = dataset.multi_steps;
    memory_steps = dataset.memory_steps;
    problem_dim = dataset.problem_dim;

    set_seed(config.seed);
    net->to(device);

    optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(lr));
    // optimizer = get_optimizer(config.optimizer, *net, lr);

    if (optimizer == nullptr) {
        throw std::runtime_error("Optimizer not found");
    }
    // scheduler = std::make_shared<torch::optim::StepLR>(*optimizer, /*step_size=*/10, /*gamma=*/0.1);
};

void ODE::train() {
    std::cout << "Training started..." << std::endl;
    summary();
    double minLoss = 1.0e15;
    auto startOverall = std::chrono::steady_clock::now();
    auto start = startOverall;

    if (do_validation) {
        int64_t total_size = train_dataset.size().value();
        int64_t split_size = static_cast<int64_t>(valid * total_size);
        

        auto valid_data = train_dataset.data.index({Slice(total_size - split_size, total_size),Slice(),Slice()});
        auto valid_targets = train_dataset.targets.index({Slice(total_size - split_size, total_size),Slice(),Slice()});
        std::cout << "Validation data shape: " << valid_data.sizes() << std::endl;
        std::cout << "Validation target shape: " << valid_targets.sizes() << std::endl;
        valid_dataset = ODEDataset(valid_data, valid_targets);

        train_dataset.data = train_dataset.data.index({Slice(0, total_size - split_size),Slice(),Slice()});
        train_dataset.targets = train_dataset.targets.index({Slice(0, total_size - split_size),Slice(),Slice()});
    }
    auto train_loader = torch::data::make_data_loader(train_dataset.map(torch::data::transforms::Stack<>()), bsize);
    auto new_optimizer = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(lr));

    for (int64_t ep = 0; ep < nepochs; ++ep) {
        net->train();
        double trainLoss = 0.0;
        int64_t batchCount = 0;

        for (auto it = train_loader->begin(); it != train_loader->end(); ++it) {
            auto batch = *it;
            auto xx = batch.data.to(device).detach();
            auto yy = batch.target.to(device).detach();
            auto pred = torch::zeros_like(yy);
            for (int64_t t = 0; t < multi_steps; ++t) {
                auto out = net->forward(xx);
                // std::cout << "slice1: \n" << xx.index({Slice(), Slice(), Slice(1,memory_steps+1)}) << std::endl;
                pred.slice(-1, t, t+1) = out;
                xx = torch::cat({xx.slice(-1, 1, xx.size(-1)), out}, -1);
            }
            auto loss = torch::mse_loss(pred, yy);
            optimizer->zero_grad();
            loss.backward();
            new_optimizer->step();
            // scheduler->step();
            trainLoss += loss.item<double>();
            batchCount++;
        }
        trainLoss /= static_cast<double>(batchCount);

        if (do_validation) {
            double validLoss = validate();
            if (validLoss < minLoss) {
                minLoss = validLoss;
                // torch::save(*net, "best_model.pt");
                // Save model here if needed
            }
            if (verbose && (ep + 1) % verbose == 0) {
                auto end = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
                std::cout << "Epoch " << ep+1 << " --- Time: " << elapsed
                            << "s --- Training loss: " << trainLoss
                            << " --- Validation loss: " << validLoss << "\n";
                start = end;
            }
        } else {
            if (trainLoss < minLoss) {
                minLoss = trainLoss;
                // torch::save(*net, "best_model.pt");
            }
            if (verbose && (ep + 1) % verbose == 0) {
                auto end = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
                std::cout << "Epoch " << ep+1 << " --- Time: " << elapsed
                            << "s --- Training loss: " << trainLoss << "\n";
                start = end;
            }
        }
    }
}

double ODE::validate() {
    // net->eval();
    // torch::NoGradGuard no_grad;
    double totalLoss = 0.0;
    int64_t count = 0;
    auto valid_loader = torch::data::make_data_loader(valid_dataset.map(torch::data::transforms::Stack<>()), bsize);
    for (auto it = valid_loader->begin(); it != valid_loader->end(); ++it) {
        auto batch = *it;
        auto xx = batch.data.to(device);
        auto yy = batch.target.to(device);

        auto pred = torch::zeros_like(yy);
        for (int64_t t = 0; t < multi_steps; ++t) {
            auto out = net->forward(xx);
            pred.slice(-1, t, t+1) = out;
            xx = torch::cat({xx.slice(-1, 1, xx.size(-1)), out}, -1);
        }
        auto loss = torch::mse_loss(pred, yy);
        totalLoss += loss.item<double>();
        count++;
    }
    return totalLoss / static_cast<double>(count);
}

void ODE::summary() {
    std::cout << "Number of trainable parameters: " << count_params(*this->net) << std::endl;
    std::cout << "Number of epochs: " << nepochs << std::endl;
    std::cout << "Batch size: "      << bsize << std::endl;
}

void ODE::set_seed(unsigned int seed) {
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
}

