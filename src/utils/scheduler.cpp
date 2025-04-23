#include <due/utils/scheduler.hpp>
// #pragma once
namespace torch {
namespace optim {
CosineAnnealingLR::CosineAnnealingLR(
    torch::optim::Optimizer& optimizer,
    const unsigned initial_lr,
    const double eta_min,
    const int last_step
) :
    LRScheduler(optimizer),
    initial_lr_(initial_lr),
    eta_min_(eta_min),
    last_step_(last_step)
{}

std::vector<double> CosineAnnealingLR::get_lrs() {
    if(last_step_ == -1)
        return get_current_lrs();
    else {
        std::vector<double> lrs = get_current_lrs();
        std::transform(lrs.begin(), lrs.end(), lrs.begin(),
                    [this](const double& v){ return initial_lr_ * (1 + std::cos(M_PI * v / last_step_)) / 2; });
        return lrs;
    }
}

} // namespace optim
} // namespace torch