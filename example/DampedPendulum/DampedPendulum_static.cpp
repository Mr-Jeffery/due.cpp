#include "../../src/due.cpp"
#define Slice torch::indexing::Slice
#define None torch::indexing::None

int main(){
    torch::Device device(torch::kCUDA); 
    // Load the configuration for the modules: datasets, networks, and models
    auto conf_data = ConfigData();
    auto conf_net = ConfigNet();
    auto conf_train = ConfigTrain();

    conf_data.problem_dim = 2;
    conf_data.memory = 0;
    conf_data.multi_steps = 10;
    conf_data.nbursts = 10;
    conf_data.dtype = "double";

    conf_net.problem_dim = 2;
    conf_net.memory = 0;
    conf_net.depth = 2;
    conf_net.width = 10;
    conf_net.dtype = "double";
    conf_net.activation = "relu";

    conf_train.epochs = 500;
    conf_train.batch_size = 2048;
    conf_train.learning_rate = 0.001;
    conf_train.valid = 0;
    conf_train.verbose = 10;
    conf_train.device = "cuda";
    conf_train.seed = 42;
    conf_train.save_path = "best_model.pt";
    conf_train.loss = "mse";
    conf_train.optimizer = "adam";

    // Load the (measurement) data, slice them into short bursts, apply normalization, and store the minimum and maximum values of the state varaibles

    torch::Tensor data = torch::zeros({10000, conf_data.problem_dim, conf_data.memory + 1}, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor target = torch::zeros({10000, conf_data.problem_dim, conf_data.multi_steps}, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor vmin = torch::zeros({1, conf_data.problem_dim, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor vmax = torch::zeros({1, conf_data.problem_dim, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    auto my_dataset = ODEDataset(data, target);

    auto raw_data_loader = RawDataLoader(conf_data);
    auto train_dataset = raw_data_loader.load("/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");
    my_dataset.data = train_dataset.data.index({Slice(0, 10000), Slice(0, conf_data.problem_dim), Slice(0, conf_data.memory + 1)});
    my_dataset.targets = train_dataset.targets.index({Slice(0, 10000), Slice(0, conf_data.problem_dim), Slice(0, conf_data.multi_steps)});
    vmin = raw_data_loader.vmin.index({Slice(0, 1), Slice(0, conf_data.problem_dim), Slice(0, 1)});
    vmax = raw_data_loader.vmax.index({Slice(0, 1), Slice(0, conf_data.problem_dim), Slice(0, 1)});

    std::cout << "vmin: " << raw_data_loader.vmin.sizes() << std::endl;
    std::cout << "vmax: " << raw_data_loader.vmax.sizes() << std::endl;

    // Construct a neural network
    // auto mynet = ResNet(raw_data_loader.vmin, raw_data_loader.vmax, conf_net);
    auto mynet = ResNet(vmin, vmax, conf_net);
    mynet.to(device);

    // Define and train a model, save necessary information of the training history
    auto model = ODE(train_dataset, (Affine*)&mynet, conf_train);

    std::cout << model.train_dataset.data.sizes() << std::endl;

    model.train();
}