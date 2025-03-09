#pragma once

#include <cmath>

class CosineAnnealingLR {
public:
    CosineAnnealingLR(double initial_lr, int T_max);

    double get_lr();
    void step();

private:
    double initial_lr_;
    int T_max_;
    int T_cur_;
};