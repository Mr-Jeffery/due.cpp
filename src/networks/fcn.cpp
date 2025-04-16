#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "../utils/config.hpp"
#include "../utils/trainer.hpp"
#include "nn.hpp"

class Affine : public NN {
public:
    torch::Tensor vmin, vmax;
    std::string dtype;
    int memory, output_dim, input_dim;
    torch::nn::Linear mDMD{nullptr};

    Affine(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config) {
        this->vmin = vmin;
        this->vmax = vmax;
        this->dtype = config.dtype;
        this->memory = config.memory;
        this->output_dim = config.problem_dim;
        this->input_dim = this->output_dim * (this->memory + 1);

        set_seed(config.seed);

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

    torch::Tensor forward(torch::Tensor x) {
        return mDMD->forward(x);
    }

    torch::Tensor predict(torch::Tensor x, int steps, torch::Device device) {
        this->to(device);
        assert(x.size(1) == this->output_dim);
        assert(x.size(2) == this->memory + 1);

        torch::Tensor xx = 2 * (x - 0.5 * (this->vmax + this->vmin)) / (this->vmax - this->vmin);
        xx = xx.to(device);

        torch::Tensor yy = torch::zeros({xx.size(0), this->output_dim, steps + this->memory + 1}, torch::TensorOptions().device(device).dtype(xx.dtype()));
        yy.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, this->memory + 1)}, xx);

        this->eval();
        torch::NoGradGuard no_grad;
        for (int t = 0; t < steps; ++t) {
            yy.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), this->memory + 1 + t},
                          this->forward(yy.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(t, this->memory + 1 + t)})
                                        .permute({0, 2, 1}).reshape({-1, this->input_dim})));
        }

        // yy = yy.cpu();
        yy = yy * 0.5 * (this->vmax - this->vmin) + 0.5 * (this->vmax + this->vmin);

        return yy;
    }

};

class MLP : public torch::nn::Module {
public:
    std::string dtype;
    int output_dim, memory, input_dim, depth, width;
    torch::nn::ModuleList layers;
    std::function<torch::Tensor(torch::Tensor)> activation;

    MLP(const ConfigNet& config) {
        this->dtype = config.dtype;
        this->output_dim = config.problem_dim;
        this->memory = config.memory;
        this->input_dim = this->output_dim * (this->memory + 1);
        this->depth = config.depth;
        this->width = config.width;
        this->activation = get_activation(config.activation);


        set_seed(config.seed);

        if (this->dtype == "double") {
            torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));
            for (int i = 0; i < this->depth; ++i) {
                if (i == 0) {
                    layers->push_back(torch::nn::Linear(this->input_dim, this->width));
                } else {
                    layers->push_back(torch::nn::Linear(this->width, this->width));
                }
            }
            layers->push_back(torch::nn::Linear(this->width, this->output_dim));
        } else if (this->dtype == "single") {
            torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat32));
            for (int i = 0; i < this->depth; ++i) {
                if (i == 0) {
                    layers->push_back(torch::nn::Linear(this->input_dim, this->width));
                } else {
                    layers->push_back(torch::nn::Linear(this->width, this->width));
                }
            }
            layers->push_back(torch::nn::Linear(this->width, this->output_dim));
        } else {
            std::cerr << "self.dtype error. The self.dtype must be either single or double." << std::endl;
            exit(1);
        }
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

private:
    void set_seed(int seed) {
        torch::manual_seed(seed);
    }
};

class ResNet : public Affine {
public:
    MLP mlp;

    ResNet(torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config)
        : Affine(vmin, vmax, config), mlp(config) {
        
    }

    torch::Tensor forward(torch::Tensor x) {
        return mlp.forward(x) + x.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(-this->output_dim)});
    }
};

// class GResNetImpl : public AffineImpl {
// public:
//     MLP mlp;
//     torch::nn::Module prior;

//     GResNetImpl(torch::nn::Module prior, torch::Tensor vmin, torch::Tensor vmax, const ConfigNet& config)
//         : AffineImpl(vmin, vmax, config), mlp(config), prior(prior) {
//         for (auto& param : prior->parameters()) {
//             param.requires_grad_(false);
//         }
//         register_module("mlp", mlp);
//         register_module("prior", prior);
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         return prior->forward(x) + mlp->forward(x);
//     }
// };
// TORCH_MODULE(GResNet);

// class OSGNetImpl : public torch::nn::Module {
// public:
//     torch::Tensor vmin, vmax;
//     double tmin, tmax;
//     std::string dtype;
//     int input_dim, output_dim, depth, width;
//     bool multiscale;
//     torch::nn::ModuleList layers;
//     std::function<torch::Tensor(torch::Tensor)> activation;

//     OSGNetImpl(torch::Tensor vmin, torch::Tensor vmax, double tmin, double tmax, const ConfigNet& config, bool multiscale = true)
//         : vmin(vmin), vmax(vmax), tmin(tmin), tmax(tmax), multiscale(multiscale) {
//         this->dtype = config.at("dtype").toStringRef();
//         this->input_dim = config.at("problem_dim").toInt() + 1;
//         this->output_dim = config.at("problem_dim").toInt();
//         this->depth = config.at("depth").toInt();
//         this->width = config.at("width").toInt();
//         this->activation = get_activation(config.at("activation").toStringRef());

//         set_seed(config.at("seed").toInt());

//         if (this->dtype == "double") {
//             for (int i = 0; i < this->depth; ++i) {
//                 if (i == 0) {
//                     layers->push_back(torch::nn::Linear(this->input_dim, this->width).to(torch::kFloat64));
//                 } else {
//                     layers->push_back(torch::nn::Linear(this->width, this->width).to(torch::kFloat64));
//                 }
//             }
//             layers->push_back(torch::nn::Linear(this->width, this->output_dim).to(torch::kFloat64));
//         } else if (this->dtype == "single") {
//             for (int i = 0; i < this->depth; ++i) {
//                 if (i == 0) {
//                     layers->push_back(torch::nn::Linear(this->input_dim, this->width));
//                 } else {
//                     layers->push_back(torch::nn::Linear(this->width, this->width));
//                 }
//             }
//             layers->push_back(torch::nn::Linear(this->width, this->output_dim));
//         } else {
//             std::cerr << "self.dtype error. The self.dtype must be either single or double." << std::endl;
//             exit(1);
//         }
//         register_module("layers", layers);
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         torch::Tensor dt = x.index({torch::indexing::Slice(), -1}) * 0.5 * (this->tmax - this->tmin) + 0.5 * (this->tmax + this->tmin);
//         if (this->multiscale) {
//             dt = torch::pow(10, dt);
//         }

//         torch::Tensor xx = x.clone();
//         for (size_t i = 0; i < layers->size() - 1; ++i) {
//             xx = layers[i]->as<torch::nn::Linear>()->forward(xx);
//             xx = activation(xx);
//         }
//         xx = layers[layers->size() - 1]->as<torch::nn::Linear>()->forward(xx);
//         return x.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)}) + xx * dt;
//     }

//     torch::Tensor predict(torch::Tensor x, torch::Tensor dt, torch::Device device) {
//         this->to(device);
//         assert(x.size(1) == this->output_dim);
//         int steps = dt.size(1);

//         dt = dt.to(device);
//         if (this->multiscale) {
//             dt = torch::log10(dt);
//         }
//         dt = 2 * (dt - 0.5 * (this->tmax + this->tmin)) / (this->tmax - this->tmin);

//         x = x.to(device);
//         x = 2 * (x - 0.5 * (this->vmax + this->vmin)) / (this->vmax - this->vmin);

//         torch::Tensor y = x.unsqueeze(-1);
//         this->eval();
//         torch::NoGradGuard no_grad;
//         for (int t = 0; t < steps; ++t) {
//             torch::Tensor xx = torch::cat({y.index({torch::indexing::Slice(), torch::indexing::Slice(), -1}), dt.index({torch::indexing::Slice(), t}).unsqueeze(1)}, -1);
//             torch::Tensor pred = this->forward(xx);
//             y = torch::cat({y, pred.unsqueeze(-1)}, -1);
//         }

//         y = y.cpu();
//         y = y * 0.5 * (this->vmax.unsqueeze(-1) - this->vmin.unsqueeze(-1)) + 0.5 * (this->vmax.unsqueeze(-1) + this->vmin.unsqueeze(-1));

//         return y;
//     }

// private:
//     void set_seed(int seed) {
//         torch::manual_seed(seed);
//     }
// };
// TORCH_MODULE(OSGNet);

// class DualOSGNetImpl : public OSGNetImpl {
// public:
//     OSGNet osgnet1, osgnet2;
//     torch::nn::ModuleList gate;

//     DualOSGNetImpl(torch::Tensor vmin, torch::Tensor vmax, double tmin, double tmax, const ConfigNet& config, bool multiscale = true)
//         : OSGNetImpl(vmin, vmax, tmin, tmax, config, multiscale), osgnet1(vmin, vmax, tmin, tmax, config, multiscale), osgnet2(vmin, vmax, tmin, tmax, config, multiscale) {
//         if (osgnet1->dtype == "double") {
//             gate->push_back(torch::nn::Linear(1, osgnet1->width).to(torch::kFloat64));
//             gate->push_back(torch::nn::Linear(osgnet1->width, 2).to(torch::kFloat64));
//         } else if (osgnet1->dtype == "single") {
//             gate->push_back(torch::nn::Linear(1, osgnet1->width));
//             gate->push_back(torch::nn::Linear(osgnet1->width, 2));
//         }
//         register_module("osgnet1", osgnet1);
//         register_module("osgnet2", osgnet2);
//         register_module("gate", gate);
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         torch::Tensor p = torch::nn::functional::softmax(gate[1]->as<torch::nn::Linear>()->forward(osgnet1->activation(gate[0]->as<torch::nn::Linear>()->forward(x.index({torch::indexing::Slice(), -1}).unsqueeze(1)))), -1);
//         torch::Tensor y1 = osgnet1->forward(x);
//         torch::Tensor y2 = osgnet2->forward(x);
//         return p.index({torch::indexing::Slice(), 0}).unsqueeze(1) * y1 + p.index({torch::indexing::Slice(), 1}).unsqueeze(1) * y2;
//     }
// };
// TORCH_MODULE(DualOSGNet);