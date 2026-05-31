#include <stdio.h>
#include <math.h>
#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "image.h"

#define W 48
#define H 48
#define ITERS 400
#define LR 0.2f

int main(void) {
    int N = W * H;
    Rays rays = camera_rays(W, H);

    /* fixed (non-learned) scene constants */
    Vec3 light_dir = { tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    Tensor* k       = tensor_create(8.0f);
    Tensor* ambient = tensor_create(0.08f);
    Vec3 bg = { tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };

    /* ground-truth render -> constant target image */
    Tensor* tr = tensor_create_matrix(N, 1);
    Tensor* tg = tensor_create_matrix(N, 1);
    Tensor* tb = tensor_create_matrix(N, 1);
    {
        Scene gt;
        gt.center    = (Vec3){ tensor_create(0.0f), tensor_create(0.0f), tensor_create(-3.0f) };
        gt.radius    = tensor_create(1.0f);
        gt.albedo    = (Vec3){ tensor_create(0.85f), tensor_create(0.35f), tensor_create(0.30f) };
        gt.light_dir = light_dir; gt.k = k; gt.ambient = ambient; gt.bg = bg;

        Image3 gt_img = render_image(rays, gt);
        write_ppm("target.ppm", gt_img, W, H);
        for (int i = 0; i < N; i++) {
            tr->data[i] = gt_img.r->data[i];
            tg->data[i] = gt_img.g->data[i];
            tb->data[i] = gt_img.b->data[i];
        }
        tensor_release(gt_img.r);
        tensor_release(gt_img.g);
        tensor_release(gt_img.b);
        vec3_release(gt.center);
        tensor_release(gt.radius);
        vec3_release(gt.albedo);
    }
    Image3 target = { tr, tg, tb };

    /* learnable parameters (perturbed init) */
    Tensor* cx = tensor_create(0.30f);
    Tensor* cy = tensor_create(-0.20f);
    Tensor* cz = tensor_create(-3.40f);
    Tensor* r_raw = tensor_create(logf(0.60f));   /* radius = exp(r_raw) keeps it positive */
    Tensor* ar = tensor_create(0.50f);
    Tensor* ag = tensor_create(0.50f);
    Tensor* ab = tensor_create(0.50f);
    Tensor* params[7] = { cx, cy, cz, r_raw, ar, ag, ab };

    float first_loss = -1.0f, last_loss = 0.0f;

    for (int iter = 0; iter < ITERS; iter++) {
        Tensor* radius = tensor_expn(r_raw);

        Scene s;
        s.center = (Vec3){ cx, cy, cz };
        s.radius = radius;
        s.albedo = (Vec3){ ar, ag, ab };
        s.light_dir = light_dir; s.k = k; s.ambient = ambient; s.bg = bg;

        Image3 pred = render_image(rays, s);
        Tensor* loss = image_mse(pred, target);

        tensor_backward(loss);   /* zeroes all grads in the graph, then accumulates */

        for (int p = 0; p < 7; p++)
            params[p]->data[0] -= LR * params[p]->grad[0];

        last_loss = loss->data[0];
        if (first_loss < 0.0f) first_loss = last_loss;

        if (iter % 50 == 0 || iter == ITERS - 1) {
            printf("iter %4d | loss %.6f | c (%.3f, %.3f, %.3f) | r %.3f | albedo (%.2f, %.2f, %.2f)\n",
                   iter, last_loss, cx->data[0], cy->data[0], cz->data[0],
                   radius->data[0], ar->data[0], ag->data[0], ab->data[0]);
            char path[64];
            snprintf(path, sizeof(path), "pred_%04d.ppm", iter);
            write_ppm(path, pred, W, H);
        }

        tensor_release(loss);
        tensor_release(pred.r);
        tensor_release(pred.g);
        tensor_release(pred.b);
        tensor_release(radius);
    }

    printf("\nfinal loss %.6f (started %.6f, %.1fx reduction)\n",
           last_loss, first_loss, first_loss / (last_loss + 1e-9f));
    printf("%s", last_loss < first_loss * 0.1f
                 ? "CONVERGED: loss reduced by >10x\n"
                 : "WARNING: loss did not reduce by 10x\n");

    for (int p = 0; p < 7; p++) tensor_release(params[p]);
    tensor_release(target.r);
    tensor_release(target.g);
    tensor_release(target.b);
    vec3_release(light_dir);
    tensor_release(k);
    tensor_release(ambient);
    vec3_release(bg);
    vec3_release(rays.origin);
    vec3_release(rays.dir);
    return 0;
}
