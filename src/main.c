#include <stdio.h>
#include "engine.h"
#include "vec3.h"
#include "renderer.h"
#include "loss.h"

int main() {

    // -------------------------------------------------
    // Learnable scene parameters
    // -------------------------------------------------

    Tensor* cx = tensor_create(0.5f);   // wrong initial x
    Tensor* cy = tensor_create(0.0f);
    Tensor* cz = tensor_create(-1.0f);

    // radius parameterization: r = exp(r_raw) > 0
    Tensor* r_raw = tensor_create(-1.0f);   // exp(-1) ~ 0.37

    Vec3 sphere_center = { cx, cy, cz };

    // -------------------------------------------------
    // Fixed ray
    // -------------------------------------------------

    Vec3 ray_origin = {
        tensor_create(0.0f),
        tensor_create(0.0f),
        tensor_create(0.0f)
    };

    Vec3 ray_dir = {
        tensor_create(0.0f),
        tensor_create(0.0f),
        tensor_create(-1.0f)
    };

    // -------------------------------------------------
    // Target (want pixel to be white)
    // -------------------------------------------------

    Tensor* target = tensor_create(1.0f);

    // soft visibility sharpness
    Tensor* k = tensor_create(5.0f);

    float lr = 0.01f;

    // -------------------------------------------------
    // Optimization loop
    // -------------------------------------------------

    for (int iter = 0; iter < 100; iter++) {

        // ---- IMPORTANT: zero gradients ----
        tensor_zero_grad(cx);
        tensor_zero_grad(cy);
        tensor_zero_grad(cz);
        tensor_zero_grad(r_raw);

        // positive radius
        Tensor* r = tensor_expn(r_raw);

        // forward render
        Tensor* pred = render_pixel(
            ray_origin,
            ray_dir,
            sphere_center,
            r,
            k
        );

        // loss
        Tensor* loss = mse(pred, target);

        // backward
        tensor_backward(loss);

        // SGD update
        cx->data[0] -= lr * cx->grad[0];
        cy->data[0] -= lr * cy->grad[0];
        cz->data[0] -= lr * cz->grad[0];
        r_raw->data[0] -= lr * r_raw->grad[0];

        printf(
            "iter %3d | loss %.6f | cx %.4f | r %.4f\n",
            iter,
            loss->data[0],
            cx->data[0],
            r->data[0]
        );
    }

    return 0;
}

