#include <due/datasets/ode.hpp>

#define Slice torch::indexing::Slice
#define None torch::indexing::None

// Constructor
ODEDataset::ODEDataset() : data(), targets(), problem_dim(0), memory_steps(0), multi_steps(0) {}
// Constructor with data and targets
ODEDataset::ODEDataset(torch::Tensor _data, torch::Tensor _targets)
    : data(std::move(_data)), targets(std::move(_targets)) 
{
    problem_dim = data.size(1);
    multi_steps = targets.size(2);
    memory_steps = data.size(2) - 1;
}

torch::optional<size_t> ODEDataset::size() const {
    return data.size(0);
}

// Returns a single sample at index
torch::data::Example<> ODEDataset::get(size_t index) {
    return {data.index({(int) index, Slice(), Slice()}), targets.index({(int) index, Slice(), Slice()})};
}


auto RawDataLoader::normalize_(torch::Tensor data) {
    vmax = data.amax({0, 2}, true);
    vmin = data.amin({0, 2}, true);
    auto processed_data = 1 * (data - 0.5 * (vmax + vmin)) / (vmax - vmin);
    return torch::clip(processed_data, -1, 1);
}

RawDataLoader::RawDataLoader(const ConfigData& config)
    : problem_dim(config.problem_dim),
        memory_steps(config.memory),
        multi_steps(config.multi_steps),
        nbursts(config.nbursts),
        dtype(config.dtype)
{
    if (dtype == "float32" || dtype == "float") {
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat32));
    } else if (dtype == "float64" || dtype == "double") {
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
    } else {
        std::cerr << "Unsupported dtype: " << dtype << ", defaulting to float32." << std::endl;
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat32));
    }
}

// Load data from .pt files
ODEDataset RawDataLoader::load(const std::string& file_path_train, const std::string& file_path_test) {

    // Load data from .pt files
    torch::Tensor raw_data;
    torch::load(raw_data, file_path_train);
    uint N = raw_data.size(0);
    uint d = raw_data.size(1);
    uint T = raw_data.size(2);

    assert(d == this->problem_dim && "Only support data arrays with size (N,d,T), N being number of trajectories, d being the number of state variables, T being the number of time instances.");
    assert(T > this->memory_steps + this->multi_steps + 2 && "T must be greater than memory_steps + multi_steps + 2 .");

    printf("Dataset loaded with %d trajectories, %d variables, and %d time instances.\n", N, d, T);

    // Normalize the data
    raw_data = normalize_(raw_data);

    uint max_bursts = T - multi_steps - memory_steps - 1;
    int burst_len = memory_steps + multi_steps + 2;

    nbursts = std::min(nbursts, max_bursts);

    // Create a tensor to hold the processed_data
    auto processed_data = torch::zeros({N * nbursts, d, burst_len});
    for (int i = 0; i < N; ++i) {
        torch::Tensor inits;
        if (nbursts == max_bursts) {
            inits = torch::arange(0, max_bursts, torch::kLong);
        } else {
            inits = torch::randperm(max_bursts, torch::kLong).slice(0, 0, nbursts);
        }
        for (int j = 0; j < nbursts; ++j) {
            int64_t init = inits[j].item<int64_t>();
            processed_data.index_put_(
                {i * (int64_t) nbursts + j, Slice(), Slice(0, burst_len)},
                raw_data.index({i, Slice(), Slice(init, init + burst_len)})
            );
        }
    }
    // std::cout << "Processed data shape: " << processed_data.sizes() << std::endl;
    torch::Tensor trainX = processed_data.index({Slice(), Slice(), Slice(0, memory_steps + 1)}); //.permute({0,2,1}).reshape({processed_data.size(0), -1});
    torch::Tensor trainY = processed_data.index({Slice(), Slice(), Slice(memory_steps + 1, None)});      
    // std::cout << "TrainX sample: \n" << trainX.index({0, Slice(), Slice()}) << std::endl;

    return ODEDataset(trainX, trainY);

}

