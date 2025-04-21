#include "../due.hpp"
#define Slice torch::indexing::Slice
#define None torch::indexing::None

int main(){
    auto device = torch::Device(torch::kCUDA);
    std::string config_path = "config.yaml";
    auto [confData, confNet, confTrain] = read_config(config_path);

    auto raw_data_loader = RawDataLoader(confData);
    auto train_dataset = raw_data_loader.load("/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");
    auto data_loader = torch::data::make_data_loader(train_dataset.map(torch::data::transforms::Stack<>()), confTrain.batch_size);

    ResNet mynet = ResNet(raw_data_loader.vmin, raw_data_loader.vmax, confNet);
    mynet.to(device);

    int multi_steps = 10;
    auto optimizer = std::make_unique<torch::optim::Adam>(mynet.parameters(), torch::optim::AdamOptions(confTrain.learning_rate));

for (int64_t ep = 0; ep < confTrain.epochs; ++ep) {
    for (auto it = data_loader->begin(); it != data_loader->end(); ++it) {
        auto batch = *it;
        auto xx = batch.data.to(device).detach();
        auto yy = batch.target.to(device).detach();
        auto pred = torch::zeros_like(yy).to(device);

        for (int64_t t = 0; t < multi_steps; ++t) {
            auto out = mynet.forward(xx);
            pred.slice(-1, t, t+1) = out;
            xx = torch::cat({xx.slice(-1, 1, xx.size(-1)), out}, -1);
        }
        auto loss = torch::mse_loss(pred, yy);
        optimizer->zero_grad();
        loss.backward();
        std::cout << "Loss: " << loss.item<double>() << std::endl;
        optimizer->step();
    }
}
    std::cout << "Total parameters: " << count_params(mynet) << std::endl;
}