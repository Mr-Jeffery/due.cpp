#include <torch/torch.h>
#include <string>

#pragma once
template <typename NN>
int count_params (NN& nn){
    int total_params = 0;
    for (const auto& p : nn.parameters()) {
        total_params += p.requires_grad() ? p.numel() : 0;
    }
    return total_params;
}

template <typename NN>
void load_params(const std::string& save_path , NN& nn) {
    torch::serialize::InputArchive archive;
    archive.load_from(save_path);
    nn->load(archive);
}

void set_seed(unsigned int seed);