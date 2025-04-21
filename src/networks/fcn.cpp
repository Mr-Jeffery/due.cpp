#include "fcn.hpp"
#define Slice torch::indexing::Slice
#define None torch::indexing::None

Affine::Affine(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config) {
    this->vmin = vmin;
    this->vmax = vmax;
    this->dtype = config.dtype;
    this->memory = config.memory;
    this->output_dim = config.problem_dim;
    this->input_dim = this->output_dim * (this->memory + 1);

    mDMD = torch::nn::Linear(this->input_dim, this->output_dim);
    if (this->dtype == "double") {
        mDMD->to(torch::kFloat64);
    } else if (this->dtype == "single") {
        mDMD->to(torch::kFloat32);
    } else {
        std::cerr << "self.dtype error. The self.dtype must be either single or double." << std::endl;
        exit(1);
    }
    register_module("mDMD", mDMD);
}

torch::Tensor Affine::forward(torch::Tensor x) {
    return mDMD->forward(x);
}

torch::Tensor Affine::predict(torch::Tensor x, int steps, torch::Device device) {
    this->to(device);
    assert(x.size(1) == this->output_dim);
    assert(x.size(2) == this->memory + 1);

    torch::Tensor xx = 2 * (x - 0.5 * (this->vmax + this->vmin)) / (this->vmax - this->vmin);
    xx = xx.to(device);

    torch::Tensor yy = torch::zeros({xx.size(0), this->output_dim, steps + this->memory + 1}, torch::TensorOptions().device(device).dtype(xx.dtype()));
    yy.index_put_({Slice(), Slice(), Slice(0, this->memory + 1)}, xx);

    this->eval();
    torch::NoGradGuard no_grad;
    for (int t = 0; t < steps; ++t) {
        yy.index_put_({Slice(), Slice(), this->memory + 1 + t},
                        this->forward(yy.index({Slice(), Slice(), Slice(t, this->memory + 1 + t)})
                                    .permute({0, 2, 1}).reshape({-1, this->input_dim})));
    }

    // yy = yy.cpu();
    yy = yy * 0.5 * (this->vmax - this->vmin) + 0.5 * (this->vmax + this->vmin);

    return yy;
}

MLP::MLP(const ConfigNet& config) {
    this->dtype = config.dtype;
    this->output_dim = config.problem_dim;
    this->memory = config.memory;
    this->input_dim = this->output_dim * (this->memory + 1);
    this->depth = config.depth;
    this->width = config.width;
    this->activation = get_activation(config.activation);

    if (this->dtype == "double") {
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));
        layers->push_back(torch::nn::Linear(this->input_dim, this->width));
        for (int i = 1; i < this->depth; ++i) {
                layers->push_back(torch::nn::Linear(this->width, this->width));
        }
        layers->push_back(torch::nn::Linear(this->width, this->output_dim));
    } else if (this->dtype == "single") {
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat32));
        layers->push_back(torch::nn::Linear(this->input_dim, this->width));
        for (int i = 1; i < this->depth; ++i) {
            layers->push_back(torch::nn::Linear(this->width, this->width));
        }
        layers->push_back(torch::nn::Linear(this->width, this->output_dim));
    } else {
        std::cerr << "self.dtype error. The self.dtype must be either single or double." << std::endl;
        exit(1);
    }
    register_module("layers", layers);
}

torch::Tensor MLP::forward(torch::Tensor x) {
    for (size_t i = 0; i < layers->size() - 1; ++i) {
        x = layers[i]->as<torch::nn::Linear>()->forward(x);
        x = activation(x);
    }
    x = layers[layers->size() - 1]->as<torch::nn::Linear>()->forward(x);
    return x;
}

ResNet::ResNet(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config)
    : Affine(vmin, vmax, config)
{
    mlp = std::make_shared<MLP>(config);
    register_module("mlp", mlp);
}

torch::Tensor ResNet::forward(torch::Tensor x) {
    return mlp->forward(x.reshape({x.size(0), -1})).unsqueeze(-1) + x.index({Slice(), Slice(), Slice(-1,None)});
}