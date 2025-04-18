#include <torch/torch.h>
#include <iostream>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
    torch::Tensor data_, targets_;
public:
    CustomDataset(torch::Tensor data, torch::Tensor targets)
        : data_(std::move(data)), targets_(std::move(targets)) {}

    // Returns the number of samples in the dataset
    torch::optional<size_t> size() const override {
        return data_.size(0);
    }

    // Returns a single sample at index
    torch::data::Example<> get(size_t index) override {
        return {data_[index], targets_[index]};
    }
};


int main() {
    const int64_t num_samples = 100;
    const int64_t batch_size = 16;

    // Create random data and targets
    auto data = torch::rand({num_samples, 2});
    auto targets = torch::rand({num_samples, 2, 11});

    // Create dataset and dataloader
    auto dataset = CustomDataset(data, targets)
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(dataset, batch_size);

    // Iterate over batches
    for (auto& batch : *dataloader) {
        std::cout << "Data shape: " << batch.data.sizes() << std::endl;       // (batch_size, 2)
        std::cout << "Target shape: " << batch.target.sizes() << std::endl;   // (batch_size, 2, 11)
        break; // Just show the first batch
    }
    return 0;
}