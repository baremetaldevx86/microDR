#ifndef RENDERER_H
#define RENDERER_H

#include "engine.h"
#include "vec3.h"
#include "camera.h"

typedef struct {
    Tensor* r;
    Tensor* g;
    Tensor* b;
} Image3;

typedef struct {
    Vec3    center;     /* scalar [1] components */
    Tensor* radius;     /* scalar [1], positive */
    Vec3    albedo;     /* scalar [1] components */
    Vec3    light_dir;  /* scalar [1] components (normalized inside) */
    Tensor* k;          /* scalar [1], edge sharpness */
    Tensor* ambient;    /* scalar [1] */
    Vec3    bg;         /* scalar [1] components */
} Scene;

/* shaded_c = mask*(albedo_c*lambert + ambient) + (1-mask)*bg_c, all fields */
static inline Tensor* shade_channel(Tensor* albedo, Tensor* lambert,
                                    Tensor* ambient, Tensor* mask,
                                    Tensor* inv_mask, Tensor* bg) {
    Tensor* al     = tensor_mul(albedo, lambert);   /* [1]*[N] -> [N] */
    Tensor* shaded = tensor_add(al, ambient);       /* [N]+[1] -> [N] */
    tensor_release(al);

    Tensor* fg     = tensor_mul(mask, shaded);      /* [N]*[N] */
    tensor_release(shaded);

    Tensor* bgterm = tensor_mul(inv_mask, bg);      /* [N]*[1] -> [N] */
    Tensor* out    = tensor_add(fg, bgterm);
    tensor_release(fg);
    tensor_release(bgterm);
    return out;
}

static inline Image3 render_image(Rays rays, Scene s) {
    /* oc = center - origin  (scalar [1]) */
    Vec3 oc = vec3_sub(s.center, rays.origin);

    /* tca = dot(oc, dir)  ([1]-broadcast over [N]) -> [N] */
    Tensor* tca = vec3_dot(oc, rays.dir);

    /* oc2 = dot(oc, oc)  ([1]) */
    Tensor* oc2 = vec3_dot(oc, oc);
    vec3_release(oc);

    /* d2 = oc2 - tca^2  ([1]-[N]) -> [N] (perpendicular distance^2) */
    Tensor* tca2 = tensor_mul(tca, tca);
    Tensor* d2   = tensor_sub(oc2, tca2);
    tensor_release(oc2);
    tensor_release(tca2);

    /* perp = sqrt(d2 + eps) */
    Tensor* eps1 = tensor_create(1e-8f);
    Tensor* d2e  = tensor_add(d2, eps1);
    Tensor* perp = tensor_sqrt(d2e);
    tensor_release(eps1);
    tensor_release(d2e);

    /* sil = perp - radius  ([N]-[1]) -> [N] */
    Tensor* sil = tensor_sub(perp, s.radius);
    tensor_release(perp);

    /* mask = sigmoid(-k * sil) */
    Tensor* ks     = tensor_mul(s.k, sil);          /* [1]*[N] -> [N] */
    tensor_release(sil);
    Tensor* negone = tensor_create(-1.0f);
    Tensor* nks    = tensor_mul(negone, ks);
    tensor_release(negone);
    tensor_release(ks);
    Tensor* mask   = tensor_sigmoid(nks);
    tensor_release(nks);

    /* thc = sqrt(relu(radius^2 - d2) + eps) */
    Tensor* r2   = tensor_mul(s.radius, s.radius);  /* [1] */
    Tensor* rd   = tensor_sub(r2, d2);              /* [1]-[N] -> [N] */
    tensor_release(r2);
    tensor_release(d2);
    Tensor* rdr  = tensor_relu(rd);
    tensor_release(rd);
    Tensor* eps2 = tensor_create(1e-8f);
    Tensor* rdre = tensor_add(rdr, eps2);
    tensor_release(eps2);
    tensor_release(rdr);
    Tensor* thc  = tensor_sqrt(rdre);
    tensor_release(rdre);

    /* t_hit = tca - thc  ([N]) */
    Tensor* t_hit = tensor_sub(tca, thc);
    tensor_release(tca);
    tensor_release(thc);

    /* p = origin + dir * t_hit */
    Vec3 scaled = vec3_scale(rays.dir, t_hit);      /* [N]*[N] */
    tensor_release(t_hit);
    Vec3 p = vec3_add(rays.origin, scaled);         /* [1]+[N] -> [N] */
    vec3_release(scaled);

    /* normal = normalize(p - center) */
    Vec3 pc = vec3_sub(p, s.center);                /* [N]-[1] -> [N] */
    vec3_release(p);
    Vec3 n  = vec3_normalize(pc);
    vec3_release(pc);

    /* lambert = relu(dot(n, normalize(light))) */
    Vec3 Ln       = vec3_normalize(s.light_dir);
    Tensor* ndotl = vec3_dot(n, Ln);                /* [N].[1] -> [N] */
    vec3_release(n);
    vec3_release(Ln);
    Tensor* lambert = tensor_relu(ndotl);
    tensor_release(ndotl);

    /* inv_mask = 1 - mask */
    Tensor* one      = tensor_create(1.0f);
    Tensor* inv_mask = tensor_sub(one, mask);       /* [1]-[N] -> [N] */
    tensor_release(one);

    Image3 img;
    img.r = shade_channel(s.albedo.x, lambert, s.ambient, mask, inv_mask, s.bg.x);
    img.g = shade_channel(s.albedo.y, lambert, s.ambient, mask, inv_mask, s.bg.y);
    img.b = shade_channel(s.albedo.z, lambert, s.ambient, mask, inv_mask, s.bg.z);

    tensor_release(lambert);
    tensor_release(mask);
    tensor_release(inv_mask);
    return img;
}

/* mean over all elements of (a-b)^2 */
static inline Tensor* channel_mse(Tensor* a, Tensor* b) {
    Tensor* diff = tensor_sub(a, b);
    Tensor* sq   = tensor_mul(diff, diff);
    Tensor* m    = tensor_mean(sq);
    tensor_release(diff);
    tensor_release(sq);
    return m;
}

/* mean over the 3 channels */
static inline Tensor* image_mse(Image3 a, Image3 b) {
    Tensor* mr = channel_mse(a.r, b.r);
    Tensor* mg = channel_mse(a.g, b.g);
    Tensor* mb = channel_mse(a.b, b.b);
    Tensor* s1 = tensor_add(mr, mg);
    Tensor* s  = tensor_add(s1, mb);
    Tensor* three = tensor_create(3.0f);
    Tensor* loss  = tensor_div(s, three);
    tensor_release(mr);
    tensor_release(mg);
    tensor_release(mb);
    tensor_release(s1);
    tensor_release(s);
    tensor_release(three);
    return loss;
}

#endif
