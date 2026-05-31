#ifndef CAMERA_H
#define CAMERA_H

#include "engine.h"
#include "vec3.h"

/* origin: scalar [1] components (shared by all pixels)
   dir:    field  [N] components (one normalized ray per pixel) */
typedef struct {
    Vec3 origin;
    Vec3 dir;
} Rays;

/* Pinhole camera at the world origin looking down -z.
   Image plane spans [-aspect, aspect] x [-1, 1] at focal distance. */
static inline Rays camera_rays(int W, int H) {
    int N = W * H;
    float aspect = (float)W / (float)H;
    float focal  = 1.5f;

    Tensor* ox = tensor_create(0.0f);
    Tensor* oy = tensor_create(0.0f);
    Tensor* oz = tensor_create(0.0f);

    Tensor* dx = tensor_create_matrix(N, 1);
    Tensor* dy = tensor_create_matrix(N, 1);
    Tensor* dz = tensor_create_matrix(N, 1);

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int idx = j * W + i;
            float u = (2.0f * ((i + 0.5f) / (float)W) - 1.0f) * aspect;
            float v = 1.0f - 2.0f * ((j + 0.5f) / (float)H);
            float fz = -focal;
            float len = sqrtf(u*u + v*v + fz*fz);
            dx->data[idx] = u  / len;
            dy->data[idx] = v  / len;
            dz->data[idx] = fz / len;
        }
    }

    Rays r = { { ox, oy, oz }, { dx, dy, dz } };
    return r;
}

#endif
