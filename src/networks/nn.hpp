#ifndef NN_HPP
#define NN_HPP

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <string>
#include <vector>
#include <map>

class NN : public torch::nn::Module {
public:
    NN() {}

    virtual int64_t count_params();
    virtual void load_params(const std::string& save_path);
    virtual void set_seed(unsigned seed);
};

class pit_fixdt : public NN {
public:
    pit_fixdt(const torch::Tensor& mesh1,
              const torch::Tensor& mesh2,
              const std::string& device_str,
              const std::map<std::string, torch::IValue>& config);

    torch::Tensor predict(const torch::Tensor& x, int steps);
    torch::Tensor get_mesh(const torch::Tensor& inputs);
    torch::Tensor pairwise_dist(const torch::Tensor& A, const torch::Tensor& B);
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::Device device_;
    torch::Tensor msh_qry_, msh_ltt_;
    torch::Tensor m_cross_, m_latent_;
    int64_t npoints_ = 0;

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

#endif // NN_HPP