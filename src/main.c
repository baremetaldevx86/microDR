#include <stdio.h>
#include "engine.h"
#include "vec3.h"
#include "renderer.h"

#define W 16
#define H 16

int main() {

    Tensor* cx = tensor_create(0.4f);
    Tensor* cy = tensor_create(0.3f);
    Tensor* cz = tensor_create(-1.2f);
    Tensor* r_raw = tensor_create(-1.0f);

    Vec3 sphere_center = { cx, cy, cz };

    Vec3 gt_center = {
        tensor_create(0.0f),
        tensor_create(0.0f),
        tensor_create(-1.0f)
    };
    Tensor* gt_radius = tensor_create(0.5f);

    tensor_retain(gt_center.x);
    tensor_retain(gt_center.y);
    tensor_retain(gt_center.z);
    tensor_retain(gt_radius);

    Vec3 ray_origin = {
        tensor_create(0.0f),
        tensor_create(0.0f),
        tensor_create(0.0f)
    };

    tensor_retain(ray_origin.x);
    tensor_retain(ray_origin.y);
    tensor_retain(ray_origin.z);

    Tensor* k = tensor_create(1.0f);
    tensor_retain(k);

    float lr = 0.001f;

    for (int iter = 0; iter < 200; iter++) {

        tensor_zero_grad(cx);
        tensor_zero_grad(cy);
        tensor_zero_grad(cz);
        tensor_zero_grad(r_raw);

        Tensor* r = tensor_expn(r_raw);
        Tensor* loss_sum = NULL;

        for (int j = 0; j < H; j++) {
            for (int i = 0; i < W; i++) {

                float x = (2.0f * i - W) / W;
                float y = (2.0f * j - H) / H;

                Tensor* rx = tensor_create(x);
                Tensor* ry = tensor_create(y);
                Tensor* rz = tensor_create(-1.0f);

                Vec3 ray_dir = { rx, ry, rz };

                Tensor* pred = render_pixel(
                    ray_origin, ray_dir,
                    sphere_center, r, k
                );

                Tensor* target = render_pixel(
                    ray_origin, ray_dir,
                    gt_center, gt_radius, k
                );

                Tensor* diff = tensor_sub(pred, target);
                Tensor* sq   = tensor_mul(diff, diff);

                if (loss_sum == NULL) {
                    loss_sum = sq;
                } else {
                    Tensor* new_sum = tensor_add(loss_sum, sq);
                    tensor_release(loss_sum);
                    tensor_release(sq);
                    loss_sum = new_sum;
                }
                
                tensor_release(diff);
                tensor_release(pred);
                tensor_release(target);

                tensor_release(rx);
                tensor_release(ry);
                tensor_release(rz);
            }
        }

        Tensor* loss = tensor_div(
            loss_sum,
            tensor_create((float)(W * H))
        );

        tensor_backward(loss);

        cx->data[0] -= lr * cx->grad[0];
        cy->data[0] -= lr * cy->grad[0];
        cz->data[0] -= lr * cz->grad[0];
        r_raw->data[0] -= lr * r_raw->grad[0];

        if (iter % 10 == 0) {
            printf(
                "iter %3d | loss %.6f | cx %.4f | cy %.4f | r %.4f\n",
                iter,
                loss->data[0],
                cx->data[0],
                cy->data[0],
                r->data[0]
            );
        }

        tensor_release(loss);
        tensor_release(r);
    }

    return 0;
}

