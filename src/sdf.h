#ifndef SDF_H
#define SDF_H

#include "engine.h"
#include "vec3.h"

// sphere
static inline Tensor* sdf_sphere(Vec3 p, Vec3 center, Tensor* radius) {
    Vec3 d = vec3_sub(p, center);
    Tensor* dist = vec3_length(d);        /* epsilon-safe */
    vec3_release(d);

    Tensor* res = tensor_sub(dist, radius);
    tensor_release(dist);
    return res;
}

#endif
