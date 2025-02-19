#include <torch/torch.h>
#include <ATen/ATen.h> 
#include <ATen/Tensor.h>
#include <string>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>

// Example of a base "NN" class in C++. 
// If you already have a base class, inherit from it. 
// Here, we show a simple version so that pit_fixdt can extend it.
class NN : public torch::nn::Module {
public:
    NN() {}

    virtual int64_t count_params() {
        int64_t total_params = 0;
        for (auto& p : this->parameters()) {
            if (p.requires_grad()) {
                total_params += p.numel();
            }
        }
        return total_params;
    }

    virtual void load_params(const std::string& save_path) {
        torch::load(*this, save_path);
    }

    virtual void set_seed(unsigned seed) {
        setenv("PYTHONHASHSEED", std::to_string(seed).c_str(), 1);
        torch::manual_seed(seed);
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setDeterministicCuDNN(true);
    }
};

// Class pit_fixdt, inheriting from NN
/*Base class for Position-induced Transformers.
    
    Args:
        mesh1 (ndarray): The first mesh array of shape (L1, d).
        mesh2 (ndarray): The second mesh array of shape (L2, d).
        device (str): The device to run the computations on.
        config (dict): Configuration parameters for the class.
        
    Attributes:
        device (str): The device to run the computations on.
        msh_qry (ndarray): The mesh for training/testing data, array of shape (L_qry, d).
        msh_ltt (ndarray): The pre-fixed latent mesh for pit, array of shape (L_ltt, d).
        m_cross (ndarray): Pairwise distance between msh_qry and msh_ltt.
        m_latent (ndarray): Pairwise distance between msh_ltt and msh_ltt.
        npoints (int): Number of points in msh_qry.
        memory (int): Memory step.
        input_dim (int): Input channels.
        output_dim (int): Output channels.
        activation (function): Activation function.
        hid_dim (int): Hidden channels/network width/lifting dimension.
        n_head (int): Number of attention heads.
        n_blocks (int): Number of attention blocks.
        en_local (float): quantile for local attention in the encoder.
        de_local (float): quantile for local attention in the decoder.*/
class pit_fixdt : public NN {
public:
    pit_fixdt(const torch::Tensor& mesh1,
              const torch::Tensor& mesh2,
              const std::string& device_str,
              const std::map<std::string, torch::IValue>& config)
        // Convert device string to torch::Device
        // Initialize device_ here because torch::Device has no default constructor
        : device_((device_str == "cuda") ? torch::kCUDA : torch::kCPU) 
    {

        // Store the mesh data on device_
        msh_qry_  = mesh1.to(device_);
        msh_ltt_  = mesh2.to(device_);

        // Compute pairwise distances, normalized
        m_cross_  = pairwise_dist(msh_qry_, msh_ltt_);
        m_latent_ = pairwise_dist(msh_ltt_, msh_ltt_);

        npoints_ = msh_qry_.size(0);

        // Retrieve config settings. Adjust as needed if they differ in type.
        memory_      = config.at("memory").toInt();
        problem_dim_ = config.at("problem_dim").toInt();
        input_dim_   = problem_dim_ * (memory_ + 1);
        output_dim_  = problem_dim_;
        hid_dim_     = config.at("width").toInt();
        n_head_      = config.at("n_head").toInt();
        n_blocks_    = config.at("depth").toInt();
        en_local_    = config.at("locality_encoder").toDouble();
        de_local_    = config.at("locality_decoder").toDouble();

        // Activation is stored as a string in Python; in C++ we'd 
        // implement or retrieve a function ptr / module. 
        // For example, you might have a get_activation() function:
        activation_name_ = config.at("activation").toStringRef();

        // Set random seed
        auto seed = config.at("seed").toInt();
        set_seed(seed);
    }

    // Predict function. Instead of returning NumPy data, 
    // we return a torch::Tensor. 
    // Steps: number of time steps to predict into the future.
    // x shape: (N, L, d, memory)
    torch::Tensor predict(const torch::Tensor& x, int steps) {
        auto xx = x.to(device_);

        // Make an output container: (N, L, d, steps+memory+1)
        // Python code did: 
        // yy[..., :memory+1] = xx and then loop
        // We'll do something similar in C++ 
        auto sizes = xx.sizes().vec();
        // last dimension is memory, we enlarge it for steps + memory + 1
        sizes[sizes.size()-1] = steps + memory_ + 1;
        auto yy = torch::zeros(sizes, xx.options());

        // Copy xx into the first memory+1 slices
        yy.index_put_({torch::indexing::Ellipsis, 
                       torch::indexing::Slice(0, memory_ + 1)}, xx);

        // For each step, call forward and set the next slice
        this->eval();
        {
            torch::NoGradGuard no_grad;
            for (int t = 0; t < steps; ++t) {
                // forward(...) must be implemented by the user 
                // For now we call a placeholder forward(...) 
                auto next_out = forward(
                    yy.index({torch::indexing::Ellipsis, 
                              torch::indexing::Slice(t, memory_ + t + 1)}));
                yy.index_put_({torch::indexing::Ellipsis, memory_ + t + 1}, next_out);
            }
        }
        return yy.to(torch::kCPU);
    }

    // Return a repeated version of the mesh for each batch
    torch::Tensor get_mesh(const torch::Tensor& inputs) {
        auto batch_size = inputs.size(0);
        // Unsqueeze mesh, then repeat
        // This is like: torch.tile(mesh.unsqueeze(0), [batch_size, 1, 1])
        auto expanded = msh_qry_.unsqueeze(0).repeat({batch_size, 1, 1});
        return expanded;
    }

    // Simple pairwise distance function like Python's cdist
    torch::Tensor pairwise_dist(const torch::Tensor& A, const torch::Tensor& B) {
        auto dist = torch::cdist(A, B, /*p=*/2.0); 
        auto dist2 = dist.pow(2);
        auto max_val = dist2.max();
        // Avoid div by zero if needed
        if (max_val.item<double>() > 0.0) {
            dist2 = dist2 / max_val;
        }
        return dist2;
    }

    // We need a forward(...) method to define how the net processes data.
    // For demonstration, we just define a placeholder returning zeros
    torch::Tensor forward(const torch::Tensor& x) {
        // Replace with real forward logic
        // Must match (N, L, d, memory+1) slicing from 'predict'
        auto out_shape = x.sizes().vec();
        // The last dimension is memory+1, but we want output_dim_ in shape
        // e.g. (N, L, d) in PDE? Up to your design. 
        // We'll do a naive example of shape cloning:
        out_shape[out_shape.size() - 1] = output_dim_;
        return torch::zeros(out_shape, x.options());
    }

private:
    // Device
    torch::Device device_;

    // Storing relevant tensors
    torch::Tensor msh_qry_, msh_ltt_;
    torch::Tensor m_cross_, m_latent_;
    int64_t npoints_ = 0;

    // Model hyperparameters
    int64_t memory_      = 0;
    int64_t problem_dim_ = 0;
    int64_t input_dim_   = 0;
    int64_t output_dim_  = 0;
    int64_t hid_dim_     = 0;
    int64_t n_head_      = 0;
    int64_t n_blocks_    = 0;
    double  en_local_    = 0.0;
    double  de_local_    = 0.0;
    std::string activation_name_;
};