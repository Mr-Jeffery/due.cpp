#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>

int main() {
    torch::jit::script::Module m;
    torch::Tensor tensor;
    torch::load(tensor, "/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");

    // m = torch::jit::load("/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");
    // tensor = m.attr("trajectories").toTensor();
    std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
}