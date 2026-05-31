#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "tests/test_util.h"

#define W 6
#define H 6

/* Build a scene from plain floats. Caller owns all tensors and must release
   them with free_scene(). Tensors are leaves (no graph) until render. */
static Scene make_scene(float cx, float cy, float cz, float radius,
                        float ar, float ag, float ab) {
    Scene s;
    s.center    = (Vec3){ tensor_create(cx), tensor_create(cy), tensor_create(cz) };
    s.radius    = tensor_create(radius);
    s.albedo    = (Vec3){ tensor_create(ar), tensor_create(ag), tensor_create(ab) };
    s.light_dir = (Vec3){ tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    s.k         = tensor_create(8.0f);
    s.ambient   = tensor_create(0.08f);
    s.bg        = (Vec3){ tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };
    return s;
}

static void free_scene(Scene s) {
    vec3_release(s.center);
    tensor_release(s.radius);
    vec3_release(s.albedo);
    vec3_release(s.light_dir);
    tensor_release(s.k);
    tensor_release(s.ambient);
    vec3_release(s.bg);
}

/* Render `s` against the fixed target image, return scalar loss VALUE only
   (releases the whole graph; does not touch param grads). */
static float loss_value(Rays rays, Scene s, Image3 target) {
    Image3 pred = render_image(rays, s);
    Tensor* loss = image_mse(pred, target);
    float v = loss->data[0];
    tensor_release(loss);
    tensor_release(pred.r);
    tensor_release(pred.g);
    tensor_release(pred.b);
    return v;
}

/* central finite difference of loss w.r.t. *param (a scalar leaf) */
static float fd_grad(Rays rays, Scene s, Image3 target, Tensor* param, float h) {
    float orig = param->data[0];
    param->data[0] = orig + h;
    float lp = loss_value(rays, s, target);
    param->data[0] = orig - h;
    float lm = loss_value(rays, s, target);
    param->data[0] = orig;
    return (lp - lm) / (2.0f * h);
}

int main(void) {
    Rays rays = camera_rays(W, H);

    /* ground-truth render -> copy into constant target tensors */
    Scene gt = make_scene(0.0f, 0.0f, -3.0f, 1.0f, 0.85f, 0.35f, 0.30f);
    Image3 gt_img = render_image(rays, gt);
    int N = W * H;
    Image3 target;
    target.r = tensor_create_matrix(N, 1);
    target.g = tensor_create_matrix(N, 1);
    target.b = tensor_create_matrix(N, 1);
    for (int i = 0; i < N; i++) {
        target.r->data[i] = gt_img.r->data[i];
        target.g->data[i] = gt_img.g->data[i];
        target.b->data[i] = gt_img.b->data[i];
    }
    tensor_release(gt_img.r);
    tensor_release(gt_img.g);
    tensor_release(gt_img.b);
    free_scene(gt);

    /* perturbed scene where gradients are non-trivial */
    Scene s = make_scene(0.25f, -0.15f, -3.2f, 0.8f, 0.5f, 0.5f, 0.5f);

    /* analytic gradients via one backward pass */
    Image3 pred = render_image(rays, s);
    Tensor* loss = image_mse(pred, target);
    tensor_backward(loss);
    float g_cx = s.center.x->grad[0];
    float g_r  = s.radius->grad[0];
    float g_ar = s.albedo.x->grad[0];
    tensor_release(loss);
    tensor_release(pred.r);
    tensor_release(pred.g);
    tensor_release(pred.b);

    /* numeric gradients */
    float h = 1e-3f;
    float n_cx = fd_grad(rays, s, target, s.center.x, h);
    float n_r  = fd_grad(rays, s, target, s.radius,   h);
    float n_ar = fd_grad(rays, s, target, s.albedo.x, h);

    printf("cx: analytic % .5f numeric % .5f\n", g_cx, n_cx);
    printf("r : analytic % .5f numeric % .5f\n", g_r,  n_r);
    printf("ar: analytic % .5f numeric % .5f\n", g_ar, n_ar);

    CHECK_GRAD(g_cx, n_cx, 0.05f, 5e-3f, "grad center.x");
    CHECK_GRAD(g_r,  n_r,  0.05f, 5e-3f, "grad radius");
    CHECK_GRAD(g_ar, n_ar, 0.05f, 5e-3f, "grad albedo.x");

    free_scene(s);
    tensor_release(target.r);
    tensor_release(target.g);
    tensor_release(target.b);
    vec3_release(rays.origin);
    vec3_release(rays.dir);

    TEST_PASS();
    return 0;
}
