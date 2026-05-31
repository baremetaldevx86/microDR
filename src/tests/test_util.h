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
