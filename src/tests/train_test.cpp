#include <torch/torch.h>
#include <iostream>
#include "../utils/utils.hpp"
#include "../utils/config.cpp"

struct MLP : torch::nn::Module {
    int output_dim = 2, 
    memory = 0, 
    input_dim = 2,
    depth = 3, 
    width = 10;
    torch::nn::ModuleList layers;
    std::function<torch::Tensor(torch::Tensor)> activation = torch::nn::GELU();

    MLP(){
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));
        layers->push_back(torch::nn::Linear(this->input_dim, this->width));
        for (int i = 1; i < this->depth; ++i) {
                layers->push_back(torch::nn::Linear(this->width, this->width));
        }
        layers->push_back(torch::nn::Linear(this->width, this->output_dim));
        register_module("layers", layers);
    }

    torch::Tensor forward(torch::Tensor x) {
        for (size_t i = 0; i < layers->size() - 1; ++i) {
            x = layers[i]->as<torch::nn::Linear>()->forward(x);
            x = activation(x);
        }
        x = layers[layers->size() - 1]->as<torch::nn::Linear>()->forward(x);
        return x;
    }
};

struct ResNet : torch::nn::Module {
    std::shared_ptr<MLP> mlp;
    int output_dim = 2, 
    memory = 0, 
    input_dim = 2;

    ResNet() {
        mlp = std::make_shared<MLP>();
        register_module("mlp", mlp);
    }

    torch::Tensor forward(torch::Tensor x) {
        return mlp->forward(x) + x.index({torch::indexing::Slice(), torch::indexing::Slice(-this->output_dim, torch::indexing::None)});
    }
};

template <typename NN>
int count_params (NN& nn){
    int total_params = 0;
    for (const auto& p : nn.parameters()) {
        total_params += p.requires_grad() ? p.numel() : 0;
    }
    return total_params;
}

class ODEDataset : public torch::data::Dataset<ODEDataset> {
public:
    int problem_dim;
    int memory_steps;
    int multi_steps;
    int nbursts;
    std::string dtype;
    torch::Tensor vmin, vmax;

    ODEDataset(const ConfigData& config)
        : problem_dim(config.problem_dim),
          memory_steps(config.memory),
          multi_steps(config.multi_steps),
          nbursts(config.nbursts),
          dtype(config.dtype)
    {
        assert(problem_dim > 0);
        assert(memory_steps >= 0);
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> load(const std::string& file_path_train, const std::string& file_path_test = "") {
        int N = 1000;
        int d = this->problem_dim;
        int T = 1001;
        // auto target = torch::zeros({N*this->nbursts, d, this->memory_steps + this->multi_steps + 2});
        // for (int i = 0; i < N; i++) {
        //     auto inits = torch::randint(0, T-this->memory_steps-this->multi_steps-1, {this->nbursts}); // (0, K-subset+1, J0)

        // }
        
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
        auto trainX = torch::randn({10000, 2});
        auto trainY = torch::randn({10000, 2, 11});

        return std::make_tuple(trainX, trainY, torch::zeros({1,2,1}), torch::ones({1,2,1}));
    }

private:
    torch::Tensor normalize(torch::Tensor data) {
        vmax = std::get<0>(torch::max(data, 0, true));
        vmin = std::get<0>(torch::min(data, 0, true));
        auto target = 2 * (data - 0.5 * (vmax + vmin)) / (vmax - vmin);
        return torch::clamp(target, -1, 1);
    }


};

class sampleDataLoader : public torch::data::Dataset<sampleDataLoader> {
public:
    sampleDataLoader(const torch::Tensor& data_, const torch::Tensor& target_)
      : data(data_), target(target_) {}

    torch::data::Example<> get(size_t index) override {
        return {data[index], target[index]};
    }

    torch::optional<size_t> size() const override {
        return data.size(0);
    }
    torch::Tensor data, target;
};



int main(){
    auto device = torch::Device(torch::kCUDA);
    ResNet mynet;
    mynet.to(device);
    torch::Tensor trainX = torch::rand({100, 2}).to(device);
    torch::Tensor trainY = torch::rand({100, 2, 10}).to(device);

    int multi_steps = 10;
    auto pred = torch::zeros_like(trainY);
    auto optimizer = std::make_unique<torch::optim::Adam>(mynet.parameters(), torch::optim::AdamOptions(0.001));

    auto train_loader = std::make_unique<torch::data::StatelessDataLoader<
    sampleDataLoader,
    torch::data::samplers::RandomSampler>>(
    sampleDataLoader(trainX, trainY),
    torch::data::samplers::RandomSampler(trainX.size(0)),
    torch::data::DataLoaderOptions().batch_size(10));

    auto dataset = torch::data::datasets::MNIST("data/mnist")
        .map(torch::data::transforms::Stack<>());

auto data_loader = torch::data::make_data_loader(dataset, 10);

    auto batch = *data_loader->begin();
    auto xx = batch.data.to(device);
    auto yy = batch.target.to(device);

    for (int64_t t = 0; t < multi_steps; ++t) {
        auto out = mynet.forward(xx);
        pred.slice(-1, t, t+1) = out.unsqueeze(-1);
        xx = torch::cat({xx.slice(-1, mynet.output_dim, xx.size(-1)), out}, -1);
    }
    auto loss = torch::mse_loss(pred, yy);
    loss.backward();
    std::cout << "Loss: " << loss.item<double>() << std::endl;
    std::cout << "Output: " << yy << std::endl;
    std::cout << "Total parameters: " << count_params(mynet) << std::endl;
}