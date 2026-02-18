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
    Tensor* t = tensor_create(0.5f);

    Vec3 ray_scaled = vec3_scale(ray_dir, t);
    Vec3 p = vec3_add(ray_origin, ray_scaled);
    
    vec3_release(ray_scaled);

    Tensor* d = sdf_sphere(p, sphere_center, radius);

    vec3_release(p);

    Tensor* neg = tensor_create(-1.0f);
    Tensor* kd = tensor_mul(k, d);
    tensor_release(d);

    Tensor* neg_kd = tensor_mul(neg, kd);
    tensor_release(neg);
    tensor_release(kd);

    Tensor* res = tensor_expn(neg_kd);   // soft visibility
    
    tensor_release(neg_kd);
    tensor_release(t);
    
    return res;
}

#endif

