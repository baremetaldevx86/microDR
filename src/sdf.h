#ifndef SDF_H
#define SDF_H

#include "engine.h"
#include "vec3.h"

static inline Tensor* sdf_sphere(Vec3 p, Vec3 center, Tensor* radius) {
    Vec3 d = vec3_sub(p, center);
    Tensor* dist2 = vec3_dot(d, d);
    Tensor* dist  = tensor_sqrt(dist2);
    return tensor_sub(dist, radius);
}

#endif
