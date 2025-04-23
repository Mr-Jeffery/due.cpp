#include <cmath>
#include <torch/optim/schedulers/lr_scheduler.h>

#pragma once
namespace torch {
namespace optim {

class TORCH_API CosineAnnealingLR : public LRScheduler {
public:

  CosineAnnealingLR(torch::optim::Optimizer& optimizer,
         const unsigned initial_lr,
         const double eta_min,
         const int last_step = -1);

private:
  std::vector<double> get_lrs() override;
    const double initial_lr_;
    const int last_step_;
    const double eta_min_;

};
} // namespace optim
} // namespace torch