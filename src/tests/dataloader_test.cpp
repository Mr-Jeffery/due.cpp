#include <torch/torch.h>
#include <iostream>
#include <due/utils/utils.hpp>
#include <due/datasets/ode.hpp>


int main() {
    std::string config_path = "config.yaml";
    auto [confData, confNet, confTrain] = read_config(config_path);

    // Create dataset and dataloader
    auto raw_data_loader = RawDataLoader(confData);
    auto train_dataset = raw_data_loader.load("/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");
    auto dataloader = torch::data::make_data_loader(train_dataset.map(torch::data::transforms::Stack<>()), confTrain.batch_size);


    // Iterate over batches
    for (auto& batch : *dataloader) {
        std::cout << "Data: \n" << batch.data << std::endl;
        std::cout << "Target shape: " << batch.target.sizes() << std::endl;   // (batch_size, 2, 11)
        break; // Just show the first batch
    }
    return 0;
}