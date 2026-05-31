#include "engine.h"
#include "tests/test_util.h"

/* ---- test functions are appended in later tasks ---- */

int main(void) {
    /* smoke: scalar add still works */
    Tensor* a = tensor_create(2.0f);
    Tensor* b = tensor_create(3.0f);
    Tensor* c = tensor_add(a, b);
    CHECK_CLOSE(c->data[0], 5.0f, 1e-6f, "scalar add");
    tensor_release(c);
    tensor_release(a);
    tensor_release(b);

    TEST_PASS();
    return 0;
}
