#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "tests/test_util.h"

/* ---- test functions are appended in later tasks ---- */

static void test_sqrt_safe(void) {
    /* forward: sqrt(0) = 0, no NaN; backward grad finite */
    Tensor* a = tensor_create(0.0f);
    Tensor* y = tensor_sqrt(a);
    CHECK_CLOSE(y->data[0], 0.0f, 1e-6f, "sqrt(0) forward");
    tensor_backward(y);
    CHECK(isfinite(a->grad[0]), "sqrt(0) backward must be finite");
    tensor_release(y);
    tensor_release(a);

    /* forward: negative input clamps to 0 (no NaN) */
    Tensor* n = tensor_create(-4.0f);
    Tensor* yn = tensor_sqrt(n);
    CHECK(isfinite(yn->data[0]), "sqrt(negative) forward must be finite");
    tensor_release(yn);
    tensor_release(n);

    /* gradient correctness at a=4: d/da sqrt(a) = 1/(2*2) = 0.25 */
    Tensor* p = tensor_create(4.0f);
    Tensor* yp = tensor_sqrt(p);
    CHECK_CLOSE(yp->data[0], 2.0f, 1e-6f, "sqrt(4) forward");
    tensor_backward(yp);
    CHECK_CLOSE(p->grad[0], 0.25f, 1e-5f, "sqrt(4) backward");
    tensor_release(yp);
    tensor_release(p);
}

static void test_sub_mul_div_broadcast(void) {
    Tensor* f = tensor_create_matrix(3, 1);
    f->data[0] = 2.0f; f->data[1] = 4.0f; f->data[2] = 6.0f;
    Tensor* s = tensor_create(2.0f);

    /* sub: field - scalar */
    Tensor* d = tensor_sub(f, s);
    CHECK_CLOSE(d->data[1], 2.0f, 1e-6f, "sub bc fwd");
    tensor_backward(d);
    CHECK_CLOSE(f->grad[1], 1.0f, 1e-6f, "sub bc grad f");
    CHECK_CLOSE(s->grad[0], -3.0f, 1e-6f, "sub bc grad s (sum of -1)");
    tensor_release(d);

    /* mul: scalar * field */
    Tensor* m = tensor_mul(s, f);
    CHECK_CLOSE(m->data[2], 12.0f, 1e-6f, "mul bc fwd");
    tensor_backward(m);
    /* d/df = s = 2 each; d/ds = sum(f) = 12 */
    CHECK_CLOSE(f->grad[0], 2.0f, 1e-6f, "mul bc grad f");
    CHECK_CLOSE(f->grad[1], 2.0f, 1e-6f, "mul bc grad f1");
    CHECK_CLOSE(f->grad[2], 2.0f, 1e-6f, "mul bc grad f2");
    CHECK_CLOSE(s->grad[0], 12.0f, 1e-6f, "mul bc grad s");
    tensor_release(m);

    /* div: field / scalar */
    Tensor* q = tensor_div(f, s);
    CHECK_CLOSE(q->data[0], 1.0f, 1e-6f, "div bc fwd");
    tensor_backward(q);
    /* d/df = 1/s = 0.5 each; d/ds = sum(-f/s^2) = -(2+4+6)/4 = -3 */
    CHECK_CLOSE(f->grad[0], 0.5f, 1e-6f, "div bc grad f");
    CHECK_CLOSE(f->grad[1], 0.5f, 1e-6f, "div bc grad f1");
    CHECK_CLOSE(f->grad[2], 0.5f, 1e-6f, "div bc grad f2");
    CHECK_CLOSE(s->grad[0], -3.0f, 1e-6f, "div bc grad s");
    tensor_release(q);

    tensor_release(f);
    tensor_release(s);
}

static void test_add_broadcast(void) {
    /* field [3x1] + scalar [1] */
    Tensor* f = tensor_create_matrix(3, 1);
    f->data[0] = 1.0f; f->data[1] = 2.0f; f->data[2] = 3.0f;
    Tensor* s = tensor_create(10.0f);

    Tensor* c = tensor_add(f, s);     /* field + scalar */
    CHECK_CLOSE(c->data[0], 11.0f, 1e-6f, "add bc 0");
    CHECK_CLOSE(c->data[1], 12.0f, 1e-6f, "add bc 1");
    CHECK_CLOSE(c->data[2], 13.0f, 1e-6f, "add bc 2");

    tensor_backward(c);
    /* each field elem gets grad 1; scalar gets sum = 3 */
    CHECK_CLOSE(f->grad[0], 1.0f, 1e-6f, "add bc grad f0");
    CHECK_CLOSE(s->grad[0], 3.0f, 1e-6f, "add bc grad s");
    tensor_release(c);

    /* scalar [1] + field [3x1] (scalar first) must also work */
    Tensor* c2 = tensor_add(s, f);
    CHECK_CLOSE(c2->data[2], 13.0f, 1e-6f, "add bc scalar-first");
    tensor_release(c2);

    tensor_release(f);
    tensor_release(s);
}

static void test_sigmoid(void) {
    /* sigmoid(0) = 0.5 */
    Tensor* z = tensor_create(0.0f);
    Tensor* s = tensor_sigmoid(z);
    CHECK_CLOSE(s->data[0], 0.5f, 1e-6f, "sigmoid(0)");
    /* d/dz at 0 = 0.25 */
    tensor_backward(s);
    CHECK_CLOSE(z->grad[0], 0.25f, 1e-5f, "sigmoid' at 0");
    tensor_release(s);
    tensor_release(z);

    /* large positive saturates to ~1, finite gradient */
    Tensor* big = tensor_create(40.0f);
    Tensor* sb = tensor_sigmoid(big);
    CHECK_CLOSE(sb->data[0], 1.0f, 1e-6f, "sigmoid(+big)");
    CHECK(isfinite(sb->data[0]), "sigmoid(+big) finite");
    tensor_release(sb);
    tensor_release(big);

    /* large negative saturates to ~0, no overflow */
    Tensor* sm = tensor_create(-40.0f);
    Tensor* ss = tensor_sigmoid(sm);
    CHECK_CLOSE(ss->data[0], 0.0f, 1e-6f, "sigmoid(-big)");
    CHECK(isfinite(ss->data[0]), "sigmoid(-big) finite");
    tensor_release(ss);
    tensor_release(sm);
}

static void test_vec3_length_normalize(void) {
    Vec3 v = { tensor_create(3.0f), tensor_create(0.0f), tensor_create(4.0f) };
    Tensor* len = vec3_length(v);
    CHECK_CLOSE(len->data[0], 5.0f, 1e-4f, "vec3_length 3-4-5");
    tensor_release(len);

    Vec3 n = vec3_normalize(v);
    Tensor* nlen = vec3_length(n);
    CHECK_CLOSE(nlen->data[0], 1.0f, 1e-4f, "normalized length 1");
    tensor_release(nlen);
    vec3_release(n);

    /* zero vector must not produce NaN */
    Vec3 z = { tensor_create(0.0f), tensor_create(0.0f), tensor_create(0.0f) };
    Tensor* zlen = vec3_length(z);
    CHECK(isfinite(zlen->data[0]), "length(0) finite");
    Vec3 zn = vec3_normalize(z);
    CHECK(isfinite(zn.x->data[0]), "normalize(0) finite");
    tensor_release(zlen);
    vec3_release(zn);
    vec3_release(z);

    vec3_release(v);
}

static void test_camera(void) {
    int W = 4, H = 4, N = W * H;
    Rays r = camera_rays(W, H);

    /* origin is scalar [1] components at the camera position (0,0,0) */
    CHECK(r.origin.x->size == 1, "origin scalar");
    CHECK_CLOSE(r.origin.x->data[0], 0.0f, 1e-6f, "origin x");

    /* directions are a field [N] */
    CHECK(r.dir.x->size == N, "dir field size");

    /* every direction is unit length and points into -z */
    for (int i = 0; i < N; i++) {
        float dx = r.dir.x->data[i];
        float dy = r.dir.y->data[i];
        float dz = r.dir.z->data[i];
        float len = sqrtf(dx*dx + dy*dy + dz*dz);
        CHECK_CLOSE(len, 1.0f, 1e-4f, "dir unit length");
        CHECK(dz < 0.0f, "dir points into -z");
    }

    vec3_release(r.origin);
    vec3_release(r.dir);
}

int main(void) {
    /* smoke: scalar add still works */
    Tensor* a = tensor_create(2.0f);
    Tensor* b = tensor_create(3.0f);
    Tensor* c = tensor_add(a, b);
    CHECK_CLOSE(c->data[0], 5.0f, 1e-6f, "scalar add");
    tensor_release(c);
    tensor_release(a);
    tensor_release(b);

    test_sqrt_safe();
    test_sigmoid();
    test_add_broadcast();
    test_sub_mul_div_broadcast();
    test_vec3_length_normalize();
    test_camera();

    TEST_PASS();
    return 0;
}
