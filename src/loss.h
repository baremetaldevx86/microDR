#ifndef LOSS_H
#define LOSS_H

#include "engine.h"

// mse loss
static inline Tensor* mse(Tensor* a, Tensor* b) {
    Tensor* diff = tensor_sub(a, b);
    Tensor* sq   = tensor_mul(diff, diff);
    return tensor_mean(sq);
}

#endif

