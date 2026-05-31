#ifndef VEC3_H
#define VEC3_H

#include "engine.h"

#define VEC_EPS 1e-8f

//vector
typedef struct {
   Tensor* x; 
   Tensor* y;
   Tensor* z;
} Vec3;

static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){
        tensor_add(a.x, b.x),
        tensor_add(a.y, b.y),
        tensor_add(a.z, b.z)
    };
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){
        tensor_sub(a.x,b.x),
        tensor_sub(a.y,b.y),
        tensor_sub(a.z,b.z),
    };
}

static inline Vec3 vec3_scale(Vec3 v, Tensor* s) {
    return (Vec3) {
        tensor_mul(v.x, s),
        tensor_mul(v.y, s),
        tensor_mul(v.z, s),
    };
}

static inline void vec3_release(Vec3 v) {
    tensor_release(v.x);
    tensor_release(v.y);
    tensor_release(v.z);
}

static inline Tensor* vec3_dot(Vec3 a, Vec3 b) {
    Tensor* xx = tensor_mul(a.x, b.x); 
    Tensor* yy = tensor_mul(a.y, b.y); 
    Tensor* zz = tensor_mul(a.z, b.z); 

    Tensor* tmp = tensor_add(xx, yy);
    Tensor* res = tensor_add(tmp, zz);

    tensor_release(xx);
    tensor_release(yy);
    tensor_release(zz);
    tensor_release(tmp);

    return res;
}

static inline Tensor* vec3_length(Vec3 v) {
    Tensor* d2  = vec3_dot(v, v);            /* x^2+y^2+z^2 */
    Tensor* eps = tensor_create(VEC_EPS);
    Tensor* d2e = tensor_add(d2, eps);       /* scalar broadcast if field */
    Tensor* len = tensor_sqrt(d2e);
    tensor_release(d2);
    tensor_release(eps);
    tensor_release(d2e);
    return len;
}

static inline Vec3 vec3_normalize(Vec3 v) {
    Tensor* len = vec3_length(v);
    Vec3 r = {
        tensor_div(v.x, len),
        tensor_div(v.y, len),
        tensor_div(v.z, len)
    };
    tensor_release(len);
    return r;
}

#endif
