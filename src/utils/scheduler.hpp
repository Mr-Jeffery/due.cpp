#include <cmath>

#pragma once
class CosineAnnealingLR {
public:
    CosineAnnealingLR(double initial_lr_, int T_max_);

    double get_lr();
    void step();

private:
    double initial_lr;
    int T_max;
    int T_cur;
};