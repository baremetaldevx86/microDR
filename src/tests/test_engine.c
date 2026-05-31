#include "engine.h"
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

    TEST_PASS();
    return 0;
}
