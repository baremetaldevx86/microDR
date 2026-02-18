#ifndef RENDERER_H
#define RENDERER_H

#include "sdf.h"


static inline Tensor* render_pixel(
    Vec3 ray_origin,
    Vec3 ray_dir,
    Vec3 sphere_center,
    Tensor* radius,
    Tensor* k
) {
    // Fixed depth sample(generalize later) 
    Tensor* t = tensor_create(1.0f);

    Vec3 p = vec3_add(ray_origin, vec3_scale(ray_dir, t));

    Tensor* d = sdf_sphere(p, sphere_center, radius);

    Tensor* neg = tensor_create(-1.0f);
    Tensor* neg_kd = tensor_mul(neg, tensor_mul(k, d));

    return tensor_expn(neg_kd);   // soft visibility
}

#endif

