#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "image.h"
#include "tests/test_util.h"

int main(void) {
    int W = 32, H = 32, N = W * H;
    Rays rays = camera_rays(W, H);

    Scene s;
    s.center    = (Vec3){ tensor_create(0.0f), tensor_create(0.0f), tensor_create(-3.0f) };
    s.radius    = tensor_create(1.0f);
    s.albedo    = (Vec3){ tensor_create(0.85f), tensor_create(0.35f), tensor_create(0.30f) };
    s.light_dir = (Vec3){ tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    s.k         = tensor_create(8.0f);
    s.ambient   = tensor_create(0.08f);
    s.bg        = (Vec3){ tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };

    Image3 img = render_image(rays, s);

    /* center pixel should be on the sphere and lit; a corner is background */
    int center = (H/2) * W + (W/2);
    int corner = 0;
    CHECK(img.r->data[center] > img.r->data[corner],
          "center brighter than corner (sphere is rendered)");
    CHECK(img.r->data[center] > 0.2f, "center has lit albedo");
    CHECK(img.r->data[corner] < 0.2f, "corner is background");
    for (int i = 0; i < N; i++) {
        CHECK(isfinite(img.r->data[i]), "pixel finite r");
        CHECK(isfinite(img.g->data[i]), "pixel finite g");
        CHECK(isfinite(img.b->data[i]), "pixel finite b");
    }

    write_ppm("tests/forward.ppm", img, W, H);
    printf("wrote tests/forward.ppm (%dx%d)\n", W, H);

    tensor_release(img.r);
    tensor_release(img.g);
    tensor_release(img.b);
    tensor_release(s.radius);
    tensor_release(s.k);
    tensor_release(s.ambient);
    vec3_release(s.center);
    vec3_release(s.albedo);
    vec3_release(s.light_dir);
    vec3_release(s.bg);
    vec3_release(rays.origin);
    vec3_release(rays.dir);

    TEST_PASS();
    return 0;
}
