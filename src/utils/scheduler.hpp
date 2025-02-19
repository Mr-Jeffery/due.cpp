#include <cmath>

class CosineAnnealingLR {
public:
    CosineAnnealingLR(double initial_lr, int T_max)
        : initial_lr_(initial_lr), T_max_(T_max), T_cur_(0) {}

    double get_lr() {
        return initial_lr_ * (1 + std::cos(M_PI * T_cur_ / T_max_)) / 2;
    }

    void step() {
        T_cur_++;
        if (T_cur_ > T_max_) {
            T_cur_ = 0;
        }
    }

private:
    double initial_lr_;
    int T_max_;
    int T_cur_;
};