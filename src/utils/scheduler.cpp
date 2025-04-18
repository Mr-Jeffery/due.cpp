#include "scheduler.hpp"

CosineAnnealingLR::CosineAnnealingLR(double initial_lr, int T_max)
    : initial_lr(initial_lr), T_max(T_max), T_cur(0) {}

double CosineAnnealingLR::get_lr() {
    return initial_lr * (1 + std::cos(M_PI * T_cur / T_max)) / 2;
}

void CosineAnnealingLR::step() {
    T_cur++;
    if (T_cur > T_max) {
        T_cur = 0;
    }
}
