#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor;
    torch::load(tensor, "/home/jeffery/grad/py/examples/DampedPendulum/DampedPendulum_train.pt");
    std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
}