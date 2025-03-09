#pragma once

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
// #include <matio.h> // For loading .mat files

class ODEDataset {
public:
    ODEDataset(const std::map<std::string, torch::IValue>& config)
        : problem_dim_(config.at("problem_dim").toInt()), memory_steps_(config.at("memory").toInt()),
          multi_steps_(config.at("multi_steps").toInt()), nbursts_(config.at("nbursts").toInt()),
          dtype_(config.at("dtype").toStringRef()) {
        assert(memory_steps_ >= 0);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> load(const std::string& file_path_train, const std::string& file_path_test = "") {
        auto data = load_mat(file_path_train);
        auto trajectories = data["trajectories"];

        int64_t N = trajectories.size(0);
        int64_t T = trajectories.size(2);
        if (trajectories.size(1) != problem_dim_) {
            throw std::runtime_error("Only support data arrays with size (N,d,T), N being number of trajectories, d being the number of state variables, T being the number of time instances.");
        }
        std::cout << "Dataset loaded, " << N << " trajectories, " << problem_dim_ << " variables, " << T << " time instances" << std::endl;

        if (nbursts_ > T - multi_steps_ - memory_steps_ - 1) {
            nbursts_ = T - multi_steps_ - memory_steps_ - 1;
        }

        auto target = torch::zeros({N * nbursts_, problem_dim_, memory_steps_ + multi_steps_ + 2}, torch::kFloat64);
        if (nbursts_ > T - multi_steps_ - memory_steps_ - 1) {
            auto inits = torch::arange(T - multi_steps_ - memory_steps_ - 1);
            for (int64_t i = 0; i < N; ++i) {
                auto selected = trajectories.index({i, torch::indexing::Ellipsis, torch::indexing::Slice()}).index({torch::indexing::None, torch::indexing::None, inits});
                target.index_put_({torch::indexing::Slice(i * nbursts_, (i + 1) * nbursts_), torch::indexing::Ellipsis}, selected);
            }
        } else {
            for (int64_t i = 0; i < N; ++i) {
                auto inits = torch::randint(0, T - multi_steps_ - memory_steps_ - 1, {nbursts_}, torch::kInt64);
                while (torch::unique(inits).size(0) != inits.size(0)) {
                    inits = torch::randint(0, T - multi_steps_ - memory_steps_ - 1, {nbursts_}, torch::kInt64);
                }
                auto selected = trajectories.index({i, torch::indexing::Ellipsis, torch::indexing::Slice()}).index({torch::indexing::None, torch::indexing::None, inits});
                target.index_put_({torch::indexing::Slice(i * nbursts_, (i + 1) * nbursts_), torch::indexing::Ellipsis}, selected);
            }
        }

        std::cout << "Dataset regrouped, " << target.size(0) << " bursts, " << problem_dim_ << " variables, " << target.size(2) << " time instances" << std::endl;

        target = normalize(target);
        std::cout << "Training data is normalized" << std::endl;

        auto trainX = target.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, memory_steps_ + 1)}).permute({0, 2, 1}).reshape({target.size(0), -1});
        auto trainY = target.index({torch::indexing::Ellipsis, torch::indexing::Slice(memory_steps_ + 1, torch::indexing::None)});

        std::cout << "Input shape " << trainX.sizes() << ". Output shape " << trainY.sizes() << std::endl;

        if (file_path_test.empty()) {
            return std::make_tuple(trainX.to(torch::kFloat32), trainY.to(torch::kFloat32), vmin_.to(torch::kFloat32), vmax_.to(torch::kFloat32));
        } else {
            auto test_data = load_mat(file_path_test);
            auto test_trajectories = test_data["trajectories"];
            return std::make_tuple(trainX.to(torch::kFloat32), trainY.to(torch::kFloat32), test_trajectories.to(torch::kFloat32), vmin_.to(torch::kFloat32), vmax_.to(torch::kFloat32));
        }
    }

private:
    torch::Tensor normalize(torch::Tensor data) {
        vmax_ = std::get<0>(torch::max(data, 0, true));
        vmin_ = std::get<0>(torch::min(data, 0, true));
        auto target = 2 * (data - 0.5 * (vmax_ + vmin_)) / (vmax_ - vmin_);
        return torch::clamp(target, -1, 1);
    }

    std::map<std::string, torch::Tensor> load_mat(const std::string& file_path) {
        mat_t* matfp = Mat_Open(file_path.c_str(), MAT_ACC_RDONLY);
        if (matfp == nullptr) {
            throw std::runtime_error("Error opening MAT file");
        }

        std::map<std::string, torch::Tensor> data;
        matvar_t* matvar = nullptr;
        while ((matvar = Mat_VarReadNext(matfp)) != nullptr) {
            if (matvar->data_type == MAT_T_DOUBLE) {
                auto tensor = torch::from_blob(matvar->data, {matvar->dims[0], matvar->dims[1], matvar->dims[2]}, torch::kFloat64).clone();
                data[matvar->name] = tensor;
            }
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        return data;
    }

    int64_t problem_dim_;
    int64_t memory_steps_;
    int64_t multi_steps_;
    int64_t nbursts_;
    std::string dtype_;
    torch::Tensor vmin_, vmax_;
};