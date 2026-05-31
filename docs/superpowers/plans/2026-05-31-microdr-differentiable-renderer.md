# microDR Differentiable Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vectorized, analytic differentiable renderer in pure C on the existing autograd engine that recovers a single sphere's geometry and appearance from a target image by gradient descent.

**Architecture:** Each image field (per-pixel X/Y/Z/channel) is an `[N×1]` tensor where `N=H·W`; scene/material params are scalar `[1]` tensors that broadcast across pixels. One forward pass renders the whole image through one autograd graph; one backward pass produces all parameter gradients. Geometry is closed-form ray–sphere intersection (no march loop); the normal is analytic `normalize(p−center)`; shading is Lambertian with a soft `sigmoid` silhouette mask.

**Tech Stack:** C (C99), gcc, `-lm`. No third-party libraries. PPM image output. Tests are standalone C programs run via Makefile targets, using assert-style macros and finite-difference gradient checks.

**Working directory for all commands:** `src/` (the Makefile lives there). All paths below are relative to `src/` unless noted.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `engine.h` / `engine.c` | autograd: add scalar broadcasting to add/sub/mul/div; ε-safe `sqrt`; new `tensor_sigmoid` | Modify |
| `vec3.h` | add `vec3_length`, `vec3_normalize` (ε-safe) | Modify |
| `sdf.h` | `sdf_sphere` uses `vec3_length` (kept for future generic-SDF upgrade; not used by the analytic renderer) | Modify |
| `camera.h` | `Rays` struct + `camera_rays()` pinhole generator | Rewrite |
| `renderer.h` | `Scene`/`Image3` structs, analytic `render_image()`, `image_mse()` | Rewrite |
| `image.h` | `write_ppm()` | Create |
| `loss.h` | keep existing scalar `mse` | Keep |
| `main.c` | inverse-rendering loop | Rewrite |
| `tests/test_util.h` | assert/gradcheck macros | Create |
| `tests/test_engine.c` | unit + gradient checks for engine ops | Create |
| `tests/test_render.c` | forward-render smoke test (milestone 2) | Create |
| `tests/test_gradcheck.c` | end-to-end loss gradient check (milestone 3) | Create |
| `Makefile` | test targets + main build | Modify |

**Type contract (used across tasks — names are fixed):**

```c
// engine.h
Tensor* tensor_sigmoid(Tensor* a);

// vec3.h
Tensor* vec3_length(Vec3 v);      // sqrt(dot(v,v)+eps)
Vec3    vec3_normalize(Vec3 v);   // v / vec3_length(v)

// camera.h
typedef struct { Vec3 origin; Vec3 dir; } Rays;   // origin: scalar [1]; dir: field [N]
Rays camera_rays(int W, int H);

// renderer.h
typedef struct { Tensor* r; Tensor* g; Tensor* b; } Image3;   // each [N×1]
typedef struct {
    Vec3    center;     // scalar [1] components
    Tensor* radius;     // scalar [1], already positive
    Vec3    albedo;     // scalar [1] components
    Vec3    light_dir;  // scalar [1] components (will be normalized inside)
    Tensor* k;          // scalar [1], edge sharpness
    Tensor* ambient;    // scalar [1]
    Vec3    bg;         // scalar [1] components (background color)
} Scene;
Image3  render_image(Rays rays, Scene s);
Tensor* image_mse(Image3 a, Image3 b);   // mean over all channels of (a-b)^2
```

---

## Task 1: Test harness

**Files:**
- Create: `tests/test_util.h`
- Create: `tests/test_engine.c`
- Modify: `Makefile`

- [ ] **Step 1: Create the assert/gradcheck helper header**

Create `tests/test_util.h`:

```c
#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_check_count = 0;

#define CHECK(cond, msg) do {                                            \
    g_check_count++;                                                     \
    if (!(cond)) {                                                       \
        fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);   \
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) do {                                 \
    g_check_count++;                                                     \
    float _a = (float)(a), _b = (float)(b);                             \
    if (fabsf(_a - _b) > (tol)) {                                        \
        fprintf(stderr, "FAIL [%s:%d] %s: %f != %f (tol %g)\n",         \
                __FILE__, __LINE__, msg, _a, _b, (double)(tol));        \
        exit(1);                                                         \
    }                                                                    \
} while (0)

/* relative-or-absolute closeness, for noisy finite-difference comparisons */
#define CHECK_GRAD(analytic, numeric, rtol, atol, msg) do {              \
    g_check_count++;                                                     \
    float _an = (float)(analytic), _nu = (float)(numeric);             \
    float _ad = fabsf(_an - _nu);                                        \
    float _rd = _ad / (fabsf(_nu) + 1e-6f);                             \
    if (_ad > (atol) && _rd > (rtol)) {                                 \
        fprintf(stderr,                                                  \
            "FAIL [%s:%d] %s: analytic %f vs numeric %f (abs %g, rel %g)\n", \
            __FILE__, __LINE__, msg, _an, _nu, (double)_ad, (double)_rd);\
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define TEST_PASS() printf("PASS %s (%d checks)\n", __FILE__, g_check_count)

#endif
```

- [ ] **Step 2: Create a smoke test that compiles against the engine**

Create `tests/test_engine.c`:

```c
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
```

- [ ] **Step 3: Add test infrastructure to the Makefile**

Replace the entire `Makefile` with:

```make
CC = gcc
CFLAGS = -O0 -g -I.
LDFLAGS = -lm

TARGET = dr_test
SRCS = main.c engine.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- tests ----
TEST_DEPS = engine.c engine.h tests/test_util.h

tests/test_engine: tests/test_engine.c $(TEST_DEPS)
	$(CC) $(CFLAGS) tests/test_engine.c engine.c -o $@ $(LDFLAGS)

tests/test_render: tests/test_render.c $(TEST_DEPS) vec3.h camera.h renderer.h image.h
	$(CC) $(CFLAGS) tests/test_render.c engine.c -o $@ $(LDFLAGS)

tests/test_gradcheck: tests/test_gradcheck.c $(TEST_DEPS) vec3.h camera.h renderer.h
	$(CC) $(CFLAGS) tests/test_gradcheck.c engine.c -o $@ $(LDFLAGS)

test-engine: tests/test_engine
	./tests/test_engine

test-render: tests/test_render
	./tests/test_render

test-gradcheck: tests/test_gradcheck
	./tests/test_gradcheck

test: test-engine

clean:
	rm -f $(OBJS) $(TARGET) tests/test_engine tests/test_render tests/test_gradcheck

.PHONY: all clean test test-engine test-render test-gradcheck
```

(The `test` target only runs `test-engine` for now; later tasks expand it to `test-engine test-render test-gradcheck`.)

- [ ] **Step 4: Run the smoke test**

Run: `cd src && make test-engine`
Expected: compiles, prints `PASS tests/test_engine.c (1 checks)`.

- [ ] **Step 5: Commit**

```bash
git add src/tests/test_util.h src/tests/test_engine.c src/Makefile
git commit -m "test: add C test harness and engine smoke test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: ε-safe `sqrt`

The current `sqrt_backward` divides by `2·y` where `y=sqrt(a)`, producing `inf/NaN` when `a=0`. Make the forward clamp negatives and the backward divide-safe.

**Files:**
- Modify: `tests/test_engine.c`
- Modify: `engine.c:494-514` (forward `tensor_sqrt`), `engine.c:326-335` (`sqrt_backward`)

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add this function above `main`:

```c
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
```

Call it in `main` before `TEST_PASS()`:

```c
    test_sqrt_safe();
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL on `sqrt(0) backward must be finite` (current backward divides by zero).

- [ ] **Step 3: Implement the ε-safe sqrt**

In `engine.c`, near the top after the includes, add the epsilon constant:

```c
#define SQRT_EPS 1e-7f
```

Replace `sqrt_backward` (currently around `engine.c:326`) with:

```c
static void sqrt_backward(Tensor* self) {
    Tensor* a = self->parents[0];

    for (int i = 0; i < self->size; i++) {
        float grad = self->grad[i];
        float y = self->data[i];                 // y = sqrt(a)
        a->grad[i] += grad / (2.0f * y + SQRT_EPS);
    }
}
```

In the forward `tensor_sqrt` (around `engine.c:494`), replace the data loop:

```c
    for (int i = 0; i < a->size; i++) {
        float v = a->data[i];
        c->data[i] = sqrtf(v > 0.0f ? v : 0.0f);
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: `PASS tests/test_engine.c (...)`.

- [ ] **Step 5: Commit**

```bash
git add src/engine.c src/tests/test_engine.c
git commit -m "fix: make tensor_sqrt epsilon-safe at zero and negatives

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `tensor_sigmoid`

**Files:**
- Modify: `engine.h`, `engine.c`, `tests/test_engine.c`

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add above `main`:

```c
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
```

Call `test_sigmoid();` in `main`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL to compile/link — `tensor_sigmoid` not declared.

- [ ] **Step 3: Implement sigmoid**

In `engine.h`, add after the `tensor_sqrt` declaration:

```c
Tensor* tensor_sigmoid(Tensor* a);
```

In `engine.c`, add the backward kernel near the other `static void *_backward` definitions:

```c
static void sigmoid_backward(Tensor* self) {
    Tensor* a = self->parents[0];

    for (int i = 0; i < self->size; i++) {
        float y = self->data[i];                  // y = sigmoid(a)
        a->grad[i] += y * (1.0f - y) * self->grad[i];
    }
}
```

And add the forward op in the "Forward ops" section:

```c
static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

Tensor* tensor_sigmoid(Tensor* a) {
    Tensor* c;
    if (a->ndim == 0) {
        c = tensor_create(sigmoid_stable(*(a->data)));
    } else {
        c = tensor_create_matrix(a->shape[0], a->shape[1]);
        for (int i = 0; i < a->size; i++) {
            c->data[i] = sigmoid_stable(a->data[i]);
        }
    }

    c->n_parents = 1;
    c->parents = (Tensor**)malloc(sizeof(Tensor*));
    c->parents[0] = a;
    tensor_retain(a);

    c->backward = sigmoid_backward;
    return c;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine.c src/engine.h src/tests/test_engine.c
git commit -m "feat: add numerically-stable tensor_sigmoid op

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Scalar broadcasting for `add`

Generalize `tensor_add` so a size-1 operand broadcasts across a size-N operand, while preserving the existing equal-size and bias-row paths. Backward accumulates the scalar operand's gradient as the sum of incoming grads.

**Files:**
- Modify: `engine.c` (`tensor_add` ~line 341, `add_backward` ~line 157)
- Modify: `tests/test_engine.c`

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add above `main`:

```c
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
```

Call `test_add_broadcast();` in `main`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL — `tensor_add(s, f)` (scalar first) dereferences `a->shape` which is NULL → crash/segfault, or wrong result.

- [ ] **Step 3: Implement broadcasting add**

First add a static allocation helper near the top of `engine.c` (after the `TensorList` helpers, before the backward kernels):

```c
/* Allocate the broadcast-result tensor for an elementwise binary op.
   Both scalar -> scalar; otherwise a matrix shaped like the larger operand. */
static Tensor* alloc_broadcast(Tensor* a, Tensor* b) {
    if (a->size == 1 && b->size == 1) {
        return tensor_create(0.0f);
    }
    Tensor* big = (a->size >= b->size) ? a : b;
    return tensor_create_matrix(big->shape[0], big->shape[1]);
}
```

Replace `tensor_add` (around `engine.c:341`) with:

```c
Tensor* tensor_add(Tensor* a, Tensor* b) {
    int an = a->size, bn = b->size;
    Tensor* c;

    if (an == bn) {
        c = alloc_broadcast(a, b);
        for (int i = 0; i < an; i++) c->data[i] = a->data[i] + b->data[i];
    } else if (an == 1 || bn == 1) {
        c = alloc_broadcast(a, b);
        int n = c->size;
        for (int i = 0; i < n; i++) {
            float av = (an == 1) ? a->data[0] : a->data[i];
            float bv = (bn == 1) ? b->data[0] : b->data[i];
            c->data[i] = av + bv;
        }
    } else if (a->shape && bn == a->shape[1]) {     /* bias row broadcast */
        int rows = a->shape[0], cols = a->shape[1];
        c = tensor_create_matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                c->data[i*cols + j] = a->data[i*cols + j] + b->data[j];
    } else {
        printf("tensor_add shape mismatch\n");
        exit(1);
    }

    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    tensor_retain(a);
    tensor_retain(b);
    c->n_parents = 2;
    c->backward = add_backward;
    return c;
}
```

Replace `add_backward` (around `engine.c:157`) with:

```c
static void add_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];
    int an = a->size, bn = b->size, n = self->size;

    if (an == bn) {
        for (int i = 0; i < n; i++) {
            a->grad[i] += self->grad[i];
            b->grad[i] += self->grad[i];
        }
    } else if (an == 1 || bn == 1) {
        for (int i = 0; i < n; i++) {
            int ai = (an == 1) ? 0 : i;
            int bi = (bn == 1) ? 0 : i;
            a->grad[ai] += self->grad[i];
            b->grad[bi] += self->grad[i];
        }
    } else {                                        /* bias row broadcast */
        int rows = a->shape[0], cols = a->shape[1];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                a->grad[i*cols + j] += self->grad[i*cols + j];
                b->grad[j]          += self->grad[i*cols + j];
            }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine.c src/tests/test_engine.c
git commit -m "feat: scalar broadcasting for tensor_add (preserves bias path)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Scalar broadcasting for `sub`, `mul`, `div`

Same broadcasting rule (equal-size, else size-1 broadcast) for the other three elementwise ops, each with a gradient check.

**Files:**
- Modify: `engine.c` (`tensor_sub`/`sub_backward`, `tensor_mul`/`mul_backward`, `tensor_div`/`div_backward`)
- Modify: `tests/test_engine.c`

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add above `main`:

```c
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
    CHECK_CLOSE(s->grad[0], 12.0f, 1e-6f, "mul bc grad s");
    tensor_release(m);

    /* div: field / scalar */
    Tensor* q = tensor_div(f, s);
    CHECK_CLOSE(q->data[0], 1.0f, 1e-6f, "div bc fwd");
    tensor_backward(q);
    /* d/df = 1/s = 0.5 each; d/ds = sum(-f/s^2) = -(2+4+6)/4 = -3 */
    CHECK_CLOSE(f->grad[0], 0.5f, 1e-6f, "div bc grad f");
    CHECK_CLOSE(s->grad[0], -3.0f, 1e-6f, "div bc grad s");
    tensor_release(q);

    tensor_release(f);
    tensor_release(s);
}
```

Call `test_sub_mul_div_broadcast();` in `main`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL — `tensor_sub`/`tensor_mul`/`tensor_div` currently `exit(1)` on size mismatch or index out of bounds.

- [ ] **Step 3: Implement broadcasting for the three ops**

Replace `tensor_sub` (around `engine.c:384`) with:

```c
Tensor* tensor_sub(Tensor* a, Tensor* b) {
    int an = a->size, bn = b->size;
    if (an != bn && an != 1 && bn != 1) {
        printf("tensor_sub shape mismatch\n");
        exit(1);
    }
    Tensor* c = alloc_broadcast(a, b);
    int n = c->size;
    for (int i = 0; i < n; i++) {
        float av = (an == 1) ? a->data[0] : a->data[i];
        float bv = (bn == 1) ? b->data[0] : b->data[i];
        c->data[i] = av - bv;
    }

    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    tensor_retain(a);
    tensor_retain(b);
    c->n_parents = 2;
    c->backward = sub_backward;
    return c;
}
```

Replace `sub_backward` (around `engine.c:183`) with:

```c
static void sub_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];
    int an = a->size, bn = b->size, n = self->size;

    for (int i = 0; i < n; i++) {
        int ai = (an == 1) ? 0 : i;
        int bi = (bn == 1) ? 0 : i;
        a->grad[ai] += self->grad[i];
        b->grad[bi] -= self->grad[i];
    }
}
```

Replace `tensor_mul` (around `engine.c:429`) with:

```c
Tensor* tensor_mul(Tensor* a, Tensor* b) {
    int an = a->size, bn = b->size;
    if (an != bn && an != 1 && bn != 1) {
        printf("tensor_mul shape mismatch\n");
        exit(1);
    }
    Tensor* c = alloc_broadcast(a, b);
    int n = c->size;
    for (int i = 0; i < n; i++) {
        float av = (an == 1) ? a->data[0] : a->data[i];
        float bv = (bn == 1) ? b->data[0] : b->data[i];
        c->data[i] = av * bv;
    }

    c->n_parents = 2;
    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    tensor_retain(a);
    tensor_retain(b);
    c->backward = mul_backward;
    return c;
}
```

Replace `mul_backward` (around `engine.c:196`) with:

```c
static void mul_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];
    int an = a->size, bn = b->size, n = self->size;

    for (int i = 0; i < n; i++) {
        int ai = (an == 1) ? 0 : i;
        int bi = (bn == 1) ? 0 : i;
        a->grad[ai] += b->data[bi] * self->grad[i];
        b->grad[bi] += a->data[ai] * self->grad[i];
    }
}
```

Replace `tensor_div` (around `engine.c:461`) with:

```c
Tensor* tensor_div(Tensor* a, Tensor* b) {
    int an = a->size, bn = b->size;
    if (an != bn && an != 1 && bn != 1) {
        printf("tensor_div shape mismatch\n");
        exit(1);
    }
    Tensor* c = alloc_broadcast(a, b);
    int n = c->size;
    for (int i = 0; i < n; i++) {
        float av = (an == 1) ? a->data[0] : a->data[i];
        float bv = (bn == 1) ? b->data[0] : b->data[i];
        c->data[i] = av / bv;
    }

    c->n_parents = 2;
    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    tensor_retain(a);
    tensor_retain(b);
    c->backward = div_backward;
    return c;
}
```

Replace `div_backward` (around `engine.c:311`) with:

```c
static void div_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];
    int an = a->size, bn = b->size, n = self->size;

    for (int i = 0; i < n; i++) {
        int ai = (an == 1) ? 0 : i;
        int bi = (bn == 1) ? 0 : i;
        float av = a->data[ai];
        float bv = b->data[bi];
        float g  = self->grad[i];
        a->grad[ai] += g / bv;
        b->grad[bi] += -av * g / (bv * bv);
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine.c src/tests/test_engine.c
git commit -m "feat: scalar broadcasting for tensor_sub/mul/div with grads

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `vec3_length` and `vec3_normalize`

**Files:**
- Modify: `vec3.h`
- Modify: `sdf.h` (use `vec3_length`)
- Modify: `tests/test_engine.c`

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add `#include "vec3.h"` at the top (after `#include "engine.h"`), and add above `main`:

```c
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
```

Call `test_vec3_length_normalize();` in `main`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL — `vec3_length`/`vec3_normalize` not declared.

- [ ] **Step 3: Implement the vec3 helpers**

In `vec3.h`, add `#define VEC_EPS 1e-8f` near the top (after the includes), and add before the closing `#endif`:

```c
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
```

- [ ] **Step 4: Simplify `sdf_sphere` to use `vec3_length`**

Replace the body of `sdf_sphere` in `sdf.h` with:

```c
static inline Tensor* sdf_sphere(Vec3 p, Vec3 center, Tensor* radius) {
    Vec3 d = vec3_sub(p, center);
    Tensor* dist = vec3_length(d);        /* epsilon-safe */
    vec3_release(d);

    Tensor* res = tensor_sub(dist, radius);
    tensor_release(dist);
    return res;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vec3.h src/sdf.h src/tests/test_engine.c
git commit -m "feat: add epsilon-safe vec3_length/vec3_normalize; tidy sdf_sphere

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Camera ray generation

**Files:**
- Rewrite: `camera.h`
- Modify: `tests/test_engine.c`

- [ ] **Step 1: Add the failing test**

In `tests/test_engine.c`, add `#include "camera.h"` at the top, and add above `main`:

```c
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
```

Call `test_camera();` in `main`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd src && make test-engine`
Expected: FAIL — `Rays`/`camera_rays` not declared.

- [ ] **Step 3: Implement the camera**

Replace the entire `camera.h` with:

```c
#ifndef CAMERA_H
#define CAMERA_H

#include "engine.h"
#include "vec3.h"

/* origin: scalar [1] components (shared by all pixels)
   dir:    field  [N] components (one normalized ray per pixel) */
typedef struct {
    Vec3 origin;
    Vec3 dir;
} Rays;

/* Pinhole camera at the world origin looking down -z.
   Image plane spans [-aspect, aspect] x [-1, 1] at focal distance. */
static inline Rays camera_rays(int W, int H) {
    int N = W * H;
    float aspect = (float)W / (float)H;
    float focal  = 1.5f;

    Tensor* ox = tensor_create(0.0f);
    Tensor* oy = tensor_create(0.0f);
    Tensor* oz = tensor_create(0.0f);

    Tensor* dx = tensor_create_matrix(N, 1);
    Tensor* dy = tensor_create_matrix(N, 1);
    Tensor* dz = tensor_create_matrix(N, 1);

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int idx = j * W + i;
            float u = (2.0f * ((i + 0.5f) / (float)W) - 1.0f) * aspect;
            float v = 1.0f - 2.0f * ((j + 0.5f) / (float)H);
            float fz = -focal;
            float len = sqrtf(u*u + v*v + fz*fz);
            dx->data[idx] = u  / len;
            dy->data[idx] = v  / len;
            dz->data[idx] = fz / len;
        }
    }

    Rays r = { { ox, oy, oz }, { dx, dy, dz } };
    return r;
}

#endif
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src && make test-engine`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/camera.h src/tests/test_engine.c
git commit -m "feat: pinhole camera ray generation (scalar origin, field dir)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Analytic renderer (`render_image`, `image_mse`)

**Files:**
- Rewrite: `renderer.h`

This task has a syntax-only compile check of its own — its behavior is exercised by Task 9 (forward smoke test) and Task 10 (gradient check). The implementation must be correct and self-contained here.

- [ ] **Step 1: Implement the renderer**

Replace the entire `renderer.h` with:

```c
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cd src && gcc -O0 -g -I. -fsyntax-only -x c renderer.h && echo OK`
Expected: prints `OK` (no errors). (This only checks syntax; behavior is verified in Tasks 9 and 10.)

- [ ] **Step 3: Commit**

```bash
git add src/renderer.h
git commit -m "feat: analytic ray-sphere differentiable renderer + image MSE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: PPM writer + forward-render smoke test (milestone 2)

**Files:**
- Create: `image.h`
- Create: `tests/test_render.c`
- Modify: `Makefile` (`test` target)

- [ ] **Step 1: Implement the PPM writer**

Create `image.h`:

```c
#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include "renderer.h"

static inline unsigned char clamp_u8(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (unsigned char)(v * 255.0f + 0.5f);
}

/* binary P6 PPM */
static inline void write_ppm(const char* path, Image3 img, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "write_ppm: cannot open %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    int N = W * H;
    for (int i = 0; i < N; i++) {
        fputc(clamp_u8(img.r->data[i]), f);
        fputc(clamp_u8(img.g->data[i]), f);
        fputc(clamp_u8(img.b->data[i]), f);
    }
    fclose(f);
}

#endif
```

- [ ] **Step 2: Write the forward-render smoke test**

Create `tests/test_render.c`:

```c
#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "image.h"
#include "tests/test_util.h"

int main(void) {
    int W = 32, H = 32, N = W * H;
    Rays rays = camera_rays(W, H);

    Scene s;
    s.center    = (Vec3){ tensor_create(0.0f), tensor_create(0.0f), tensor_create(-3.0f) };
    s.radius    = tensor_create(1.0f);
    s.albedo    = (Vec3){ tensor_create(0.85f), tensor_create(0.35f), tensor_create(0.30f) };
    s.light_dir = (Vec3){ tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    s.k         = tensor_create(8.0f);
    s.ambient   = tensor_create(0.08f);
    s.bg        = (Vec3){ tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };

    Image3 img = render_image(rays, s);

    /* center pixel should be on the sphere and lit; a corner is background */
    int center = (H/2) * W + (W/2);
    int corner = 0;
    CHECK(img.r->data[center] > img.r->data[corner],
          "center brighter than corner (sphere is rendered)");
    CHECK(img.r->data[center] > 0.2f, "center has lit albedo");
    CHECK(img.r->data[corner] < 0.2f, "corner is background");
    for (int i = 0; i < N; i++) {
        CHECK(isfinite(img.r->data[i]), "pixel finite r");
        CHECK(isfinite(img.g->data[i]), "pixel finite g");
        CHECK(isfinite(img.b->data[i]), "pixel finite b");
    }

    write_ppm("tests/forward.ppm", img, W, H);
    printf("wrote tests/forward.ppm (%dx%d)\n", W, H);

    tensor_release(img.r);
    tensor_release(img.g);
    tensor_release(img.b);
    tensor_release(s.radius);
    tensor_release(s.k);
    tensor_release(s.ambient);
    vec3_release(s.center);
    vec3_release(s.albedo);
    vec3_release(s.light_dir);
    vec3_release(s.bg);
    vec3_release(rays.origin);
    vec3_release(rays.dir);

    TEST_PASS();
    return 0;
}
```

- [ ] **Step 3: Add `image.h` to the render test deps and run it**

In the `Makefile`, the `tests/test_render` rule already lists `image.h` indirectly via `renderer.h`; update its prerequisite line to include `image.h` explicitly:

```make
tests/test_render: tests/test_render.c $(TEST_DEPS) vec3.h camera.h renderer.h image.h
	$(CC) $(CFLAGS) tests/test_render.c engine.c -o $@ $(LDFLAGS)
```

Run: `cd src && make test-render`
Expected: PASS, prints `wrote tests/forward.ppm (32x32)`.

- [ ] **Step 4: Eyeball the image**

Run: `cd src && python3 -c "import sys; print(open('tests/forward.ppm','rb').read(20))"`
Expected: header begins with `b'P6\n32 32\n255\n'`. (Optionally open `tests/forward.ppm` in an image viewer — it should show a reddish shaded sphere on a dark background.)

- [ ] **Step 5: Expand the `test` target**

In the `Makefile`, change the `test` target to:

```make
test: test-engine test-render
```

- [ ] **Step 6: Commit**

```bash
git add src/image.h src/tests/test_render.c src/Makefile
git commit -m "feat: PPM writer + forward-render smoke test (milestone 2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: End-to-end gradient check (milestone 3)

Verify that the analytic loss gradient (from `tensor_backward`) matches central finite differences for `center.x`, `radius`, and `albedo.x`.

**Files:**
- Create: `tests/test_gradcheck.c`
- Modify: `Makefile` (`test` target)

- [ ] **Step 1: Write the gradient-check test**

Create `tests/test_gradcheck.c`:

```c
#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "tests/test_util.h"

#define W 6
#define H 6

/* Build a scene from plain floats. Caller owns all tensors and must release
   them with free_scene(). Tensors are leaves (no graph) until render. */
static Scene make_scene(float cx, float cy, float cz, float radius,
                        float ar, float ag, float ab) {
    Scene s;
    s.center    = (Vec3){ tensor_create(cx), tensor_create(cy), tensor_create(cz) };
    s.radius    = tensor_create(radius);
    s.albedo    = (Vec3){ tensor_create(ar), tensor_create(ag), tensor_create(ab) };
    s.light_dir = (Vec3){ tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    s.k         = tensor_create(8.0f);
    s.ambient   = tensor_create(0.08f);
    s.bg        = (Vec3){ tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };
    return s;
}

static void free_scene(Scene s) {
    vec3_release(s.center);
    tensor_release(s.radius);
    vec3_release(s.albedo);
    vec3_release(s.light_dir);
    tensor_release(s.k);
    tensor_release(s.ambient);
    vec3_release(s.bg);
}

/* Render `s` against the fixed target image, return scalar loss VALUE only
   (releases the whole graph; does not touch param grads). */
static float loss_value(Rays rays, Scene s, Image3 target) {
    Image3 pred = render_image(rays, s);
    Tensor* loss = image_mse(pred, target);
    float v = loss->data[0];
    tensor_release(loss);
    tensor_release(pred.r);
    tensor_release(pred.g);
    tensor_release(pred.b);
    return v;
}

/* central finite difference of loss w.r.t. *param (a scalar leaf) */
static float fd_grad(Rays rays, Scene s, Image3 target, Tensor* param, float h) {
    float orig = param->data[0];
    param->data[0] = orig + h;
    float lp = loss_value(rays, s, target);
    param->data[0] = orig - h;
    float lm = loss_value(rays, s, target);
    param->data[0] = orig;
    return (lp - lm) / (2.0f * h);
}

int main(void) {
    Rays rays = camera_rays(W, H);

    /* ground-truth render -> copy into constant target tensors */
    Scene gt = make_scene(0.0f, 0.0f, -3.0f, 1.0f, 0.85f, 0.35f, 0.30f);
    Image3 gt_img = render_image(rays, gt);
    int N = W * H;
    Image3 target;
    target.r = tensor_create_matrix(N, 1);
    target.g = tensor_create_matrix(N, 1);
    target.b = tensor_create_matrix(N, 1);
    for (int i = 0; i < N; i++) {
        target.r->data[i] = gt_img.r->data[i];
        target.g->data[i] = gt_img.g->data[i];
        target.b->data[i] = gt_img.b->data[i];
    }
    tensor_release(gt_img.r);
    tensor_release(gt_img.g);
    tensor_release(gt_img.b);
    free_scene(gt);

    /* perturbed scene where gradients are non-trivial */
    Scene s = make_scene(0.25f, -0.15f, -3.2f, 0.8f, 0.5f, 0.5f, 0.5f);

    /* analytic gradients via one backward pass */
    Image3 pred = render_image(rays, s);
    Tensor* loss = image_mse(pred, target);
    tensor_backward(loss);
    float g_cx = s.center.x->grad[0];
    float g_r  = s.radius->grad[0];
    float g_ar = s.albedo.x->grad[0];
    tensor_release(loss);
    tensor_release(pred.r);
    tensor_release(pred.g);
    tensor_release(pred.b);

    /* numeric gradients */
    float h = 1e-3f;
    float n_cx = fd_grad(rays, s, target, s.center.x, h);
    float n_r  = fd_grad(rays, s, target, s.radius,   h);
    float n_ar = fd_grad(rays, s, target, s.albedo.x, h);

    printf("cx: analytic % .5f numeric % .5f\n", g_cx, n_cx);
    printf("r : analytic % .5f numeric % .5f\n", g_r,  n_r);
    printf("ar: analytic % .5f numeric % .5f\n", g_ar, n_ar);

    CHECK_GRAD(g_cx, n_cx, 0.05f, 5e-3f, "grad center.x");
    CHECK_GRAD(g_r,  n_r,  0.05f, 5e-3f, "grad radius");
    CHECK_GRAD(g_ar, n_ar, 0.05f, 5e-3f, "grad albedo.x");

    free_scene(s);
    tensor_release(target.r);
    tensor_release(target.g);
    tensor_release(target.b);
    vec3_release(rays.origin);
    vec3_release(rays.dir);

    TEST_PASS();
    return 0;
}
```

- [ ] **Step 2: Build and run**

Run: `cd src && make test-gradcheck`
Expected: prints the three analytic/numeric pairs (each pair close), then `PASS tests/test_gradcheck.c (...)`.

- [ ] **Step 3: If a check fails, diagnose (do not loosen tolerances blindly)**

If `CHECK_GRAD` fails, the analytic and numeric values will be printed. A *sign* or *order-of-magnitude* mismatch indicates a real backward bug (revisit Tasks 4–5 broadcasting grads, or the op chain in `render_image`). A small mismatch only just over tolerance on `radius` is most likely float-precision noise — confirm by re-running with `h = 5e-3f`; the numeric value should move toward the analytic one. Only then adjust `h`, never the analytic code, to chase the numeric value.

- [ ] **Step 4: Expand the `test` target**

In the `Makefile`, change the `test` target to:

```make
test: test-engine test-render test-gradcheck
```

- [ ] **Step 5: Commit**

```bash
git add src/tests/test_gradcheck.c src/Makefile
git commit -m "test: end-to-end finite-difference gradient check (milestone 3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Inverse-rendering loop (milestone 4)

Recover sphere center, radius, and albedo from a target image by gradient descent; dump PPMs as it converges.

**Files:**
- Rewrite: `main.c`

- [ ] **Step 1: Write the inverse-rendering program**

Replace the entire `main.c` with:

```c
#include <stdio.h>
#include <math.h>
#include "engine.h"
#include "vec3.h"
#include "camera.h"
#include "renderer.h"
#include "image.h"

#define W 48
#define H 48
#define ITERS 400
#define LR 0.05f

int main(void) {
    int N = W * H;
    Rays rays = camera_rays(W, H);

    /* fixed (non-learned) scene constants */
    Vec3 light_dir = { tensor_create(0.4f), tensor_create(0.5f), tensor_create(0.8f) };
    Tensor* k       = tensor_create(8.0f);
    Tensor* ambient = tensor_create(0.08f);
    Vec3 bg = { tensor_create(0.04f), tensor_create(0.04f), tensor_create(0.06f) };

    /* ground-truth render -> constant target image */
    Tensor* tr = tensor_create_matrix(N, 1);
    Tensor* tg = tensor_create_matrix(N, 1);
    Tensor* tb = tensor_create_matrix(N, 1);
    {
        Scene gt;
        gt.center    = (Vec3){ tensor_create(0.0f), tensor_create(0.0f), tensor_create(-3.0f) };
        gt.radius    = tensor_create(1.0f);
        gt.albedo    = (Vec3){ tensor_create(0.85f), tensor_create(0.35f), tensor_create(0.30f) };
        gt.light_dir = light_dir; gt.k = k; gt.ambient = ambient; gt.bg = bg;

        Image3 gt_img = render_image(rays, gt);
        write_ppm("target.ppm", gt_img, W, H);
        for (int i = 0; i < N; i++) {
            tr->data[i] = gt_img.r->data[i];
            tg->data[i] = gt_img.g->data[i];
            tb->data[i] = gt_img.b->data[i];
        }
        tensor_release(gt_img.r);
        tensor_release(gt_img.g);
        tensor_release(gt_img.b);
        vec3_release(gt.center);
        tensor_release(gt.radius);
        vec3_release(gt.albedo);
    }
    Image3 target = { tr, tg, tb };

    /* learnable parameters (perturbed init) */
    Tensor* cx = tensor_create(0.30f);
    Tensor* cy = tensor_create(-0.20f);
    Tensor* cz = tensor_create(-3.40f);
    Tensor* r_raw = tensor_create(logf(0.60f));   /* radius = exp(r_raw) keeps it positive */
    Tensor* ar = tensor_create(0.50f);
    Tensor* ag = tensor_create(0.50f);
    Tensor* ab = tensor_create(0.50f);
    Tensor* params[7] = { cx, cy, cz, r_raw, ar, ag, ab };

    float first_loss = -1.0f, last_loss = 0.0f;

    for (int iter = 0; iter < ITERS; iter++) {
        Tensor* radius = tensor_expn(r_raw);

        Scene s;
        s.center = (Vec3){ cx, cy, cz };
        s.radius = radius;
        s.albedo = (Vec3){ ar, ag, ab };
        s.light_dir = light_dir; s.k = k; s.ambient = ambient; s.bg = bg;

        Image3 pred = render_image(rays, s);
        Tensor* loss = image_mse(pred, target);

        tensor_backward(loss);   /* zeroes all grads in the graph, then accumulates */

        for (int p = 0; p < 7; p++)
            params[p]->data[0] -= LR * params[p]->grad[0];

        last_loss = loss->data[0];
        if (first_loss < 0.0f) first_loss = last_loss;

        if (iter % 50 == 0 || iter == ITERS - 1) {
            printf("iter %4d | loss %.6f | c (%.3f, %.3f, %.3f) | r %.3f | albedo (%.2f, %.2f, %.2f)\n",
                   iter, last_loss, cx->data[0], cy->data[0], cz->data[0],
                   radius->data[0], ar->data[0], ag->data[0], ab->data[0]);
            char path[64];
            snprintf(path, sizeof(path), "pred_%04d.ppm", iter);
            write_ppm(path, pred, W, H);
        }

        tensor_release(loss);
        tensor_release(pred.r);
        tensor_release(pred.g);
        tensor_release(pred.b);
        tensor_release(radius);
    }

    printf("\nfinal loss %.6f (started %.6f, %.1fx reduction)\n",
           last_loss, first_loss, first_loss / (last_loss + 1e-9f));
    printf("%s", last_loss < first_loss * 0.1f
                 ? "CONVERGED: loss reduced by >10x\n"
                 : "WARNING: loss did not reduce by 10x\n");

    for (int p = 0; p < 7; p++) tensor_release(params[p]);
    tensor_release(target.r);
    tensor_release(target.g);
    tensor_release(target.b);
    vec3_release(light_dir);
    tensor_release(k);
    tensor_release(ambient);
    vec3_release(bg);
    vec3_release(rays.origin);
    vec3_release(rays.dir);
    return 0;
}
```

- [ ] **Step 2: Build**

Run: `cd src && make clean && make`
Expected: builds `dr_test` with no errors.

- [ ] **Step 3: Run the optimizer**

Run: `cd src && ./dr_test`
Expected: loss prints decreasing across iterations; final line reports `CONVERGED: loss reduced by >10x`. Files `target.ppm` and several `pred_XXXX.ppm` are written. The recovered `c` should approach `(0, 0, -3)`, `r` approach `1.0`, and albedo approach `(0.85, 0.35, 0.30)`.

- [ ] **Step 4: Eyeball convergence**

Run: `cd src && ls -1 target.ppm pred_0000.ppm pred_0399.ppm`
Expected: all three exist. Open them: `pred_0000.ppm` is an off-position/off-color sphere; `pred_0399.ppm` closely matches `target.ppm`.

- [ ] **Step 5: Commit**

```bash
git add src/main.c
git commit -m "feat: inverse-rendering loop recovers sphere geometry+albedo (milestone 4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: README + ignore generated images

**Files:**
- Modify: `README.md`
- Create/Modify: `.gitignore`

- [ ] **Step 1: Update the README**

Replace `src/README.md` with:

```markdown
# microDR

A tiny differentiable renderer in pure C, built on a hand-written autograd
engine (micrograd-style, generalized to tensors).

## What it does

Renders a single sphere with Lambertian shading via analytic ray–sphere
intersection, fully vectorized over pixels (one autograd graph per frame), and
performs **inverse rendering**: recovering the sphere's center, radius, and
albedo from a target image by gradient descent.

## Build & run

```sh
cd src
make            # builds ./dr_test
./dr_test       # runs inverse rendering, writes target.ppm + pred_*.ppm
```

## Tests

```sh
cd src
make test       # engine unit/gradient checks, forward render, end-to-end gradcheck
```

## Layout

- `engine.c/.h` — autograd tensors, ops, reverse-mode backward
- `vec3.h`, `sdf.h` — vector + SDF helpers
- `camera.h` — pinhole ray generation
- `renderer.h` — analytic sphere intersection + shading
- `image.h` — PPM output
- `main.c` — inverse-rendering loop

## Roadmap

Generic SDF sphere tracing (with closest-approach silhouette), multiple
primitives via `min`-composition, soft shadows, and an Adam optimizer.
```

- [ ] **Step 2: Ignore generated artifacts**

Create (or append to) `src/.gitignore`:

```
dr_test
*.o
*.ppm
tests/test_engine
tests/test_render
tests/test_gradcheck
```

- [ ] **Step 3: Commit**

```bash
git add src/README.md src/.gitignore
git commit -m "docs: update README; ignore build + image artifacts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the full suite: `cd src && make clean && make test`
  Expected: `PASS tests/test_engine.c`, `PASS tests/test_render.c`, `PASS tests/test_gradcheck.c`.
- [ ] Run the app: `cd src && make && ./dr_test`
  Expected: loss decreases; `CONVERGED: loss reduced by >10x`.
- [ ] (Optional, recommended) Run under valgrind to confirm the leak discipline:
  `cd src && valgrind --leak-check=summary ./dr_test 2>&1 | tail -5`
  Expected: no growth in "definitely lost" proportional to iteration count. (A small fixed amount from one-time constants is acceptable.)
