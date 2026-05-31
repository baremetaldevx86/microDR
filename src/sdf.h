#ifndef SDF_H
#define SDF_H

#include "engine.h"
#include "vec3.h"

// sphere
static inline Tensor* sdf_sphere(Vec3 p, Vec3 center, Tensor* radius) {
    Vec3 d = vec3_sub(p, center);
    Tensor* dist2 = vec3_dot(d, d);
    
    // vec3_dot consumes d if it was moved? No, vec3_dot retains d components.
    // So we must release d because vec3_sub created it with ref=1.
    // And vec3_dot retained references.
    
    vec3_release(d);

    Tensor* dist  = tensor_sqrt(dist2);
    
    // dist2 is retained by dist
    tensor_release(dist2);

    Tensor* res = tensor_sub(dist, radius);
    
    // dist is retained by res
    tensor_release(dist);

    return res;
}

#endif
