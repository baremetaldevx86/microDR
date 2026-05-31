# microDR

A tiny differentiable 3D renderer written from scratch in pure C, built using [minigrad](https://github.com/baremetaldevx86/minigrad) - a custom reverse-mode automatic differentiation engine also written from scratch in C.


It builds a scalar/tensor autograd engine (reverse-mode autodiff, C99, no
dependencies) and uses it to render a sphere with Lambertian shading — then
runs **inverse rendering**: starting from a perturbed guess, gradient descent
recovers the sphere's position, size, and color to match a target image.

---

## What is differentiable rendering?

A normal renderer takes scene parameters (where objects are, what color they
are) and produces an image. A **differentiable** renderer does the same, but
also computes the gradient of any loss on the output image with respect to the
scene parameters. This means you can run gradient descent to find the scene
that best explains an observed image — recovering geometry and appearance from
pixels.

microDR is a minimal proof-of-concept of this idea. It is intentionally small
so every piece of the pipeline is readable.

---

## How it works

### 1. Autograd engine (`engine.c` / `engine.h`)

A micrograd-style reverse-mode autodiff engine, generalized from scalars to
tensors. Every operation (add, sub, mul, div, sqrt, exp, relu, sigmoid, tanh,
matmul, mean, pow) records its parents in a computation graph. Calling
`tensor_backward` on any scalar loss:

1. Builds a topological order of the graph
2. Zeros all gradients
3. Seeds the output gradient to 1
4. Walks the graph in reverse, calling each node's backward kernel

Memory is managed with reference counting (`tensor_retain` / `tensor_release`).

**Key additions over a basic scalar autograd:**
- Scalar broadcasting: a `[1]` tensor broadcasts across any `[N]` tensor in all
  elementwise ops (critical for scene params that apply to all pixels at once)
- Numerically stable `sqrt` (ε-safe forward + backward, no NaN at zero)
- Numerically stable `sigmoid` (two-branch to avoid exp overflow)

### 2. Vectorized rendering

Instead of looping over pixels one at a time, the entire image is a single
forward pass through the autograd graph. Each image channel (R, G, B) is an
`[N×1]` tensor where `N = W × H`. Scene parameters (sphere center, radius,
albedo) are scalar `[1]` tensors that broadcast to all `N` pixels via the
engine's broadcasting ops.

One call to `render_image` → one autograd graph → one call to
`tensor_backward` → gradients for all scene parameters.

### 3. Analytic ray–sphere intersection (`renderer.h`)

Given a pinhole camera and a sphere, each pixel's ray is tested against the
sphere analytically:

```
oc  = center − origin
tca = dot(oc, ray_dir)          # closest approach depth
d²  = |oc|² − tca²              # perpendicular distance squared

sil = sqrt(d² + ε) − radius     # silhouette distance (negative inside the disk)
mask = sigmoid(−k · sil)        # soft occupancy in [0, 1]

thc   = sqrt(relu(r² − d²) + ε) # half-chord length
t_hit = tca − thc               # depth to front surface
p     = origin + t_hit · dir    # hit point
n     = normalize(p − center)   # analytic surface normal
```

Shading:

```
lambert = relu(dot(n, normalize(light_dir)))
color_c = mask · (albedo_c · lambert + ambient) + (1 − mask) · bg_c
```

The soft mask (`sigmoid` of the silhouette distance) gives a differentiable
silhouette edge — gradients flow through it to the sphere radius and center.
The Lambertian shading gives gradients through the normal to the center and
radius, and directly to the albedo.

### 4. Inverse rendering (`main.c`)

```
ground truth params → render target image (fixed, no grad)

perturbed init params → render predicted image
                      → MSE loss vs target
                      → tensor_backward
                      → SGD step on [cx, cy, cz, r_raw, ar, ag, ab]

repeat 400 iterations
```

`r_raw` is the raw radius parameter; the actual radius is `exp(r_raw)`, which
keeps it strictly positive throughout optimization.

---

## Results

Starting from a perturbed sphere (offset center, wrong radius, grey albedo),
the optimizer converges to the ground truth in 400 iterations:

| Parameter | Ground truth | Initial | Recovered |
|-----------|-------------|---------|-----------|
| center x  | 0.00        | 0.30    | ~0.00     |
| center y  | 0.00        | -0.20   | ~0.00     |
| center z  | -3.00       | -3.40   | ~-3.23    |
| radius    | 1.00        | 0.60    | ~1.08     |
| albedo R  | 0.85        | 0.50    | ~0.84     |
| albedo G  | 0.35        | 0.50    | ~0.35     |
| albedo B  | 0.30        | 0.50    | ~0.30     |

Loss is reduced by ~2000× over 400 iterations.

---

## Build & run

Requires: `gcc`, `make`, `-lm`. No other dependencies.

```sh
cd src
make          # builds ./dr_test
./dr_test     # runs inverse rendering
              # writes target.ppm + pred_0000.ppm, pred_0050.ppm, ...
```

Output images are written as PNG files (`target.png`, `pred_0000.png`, …)
and can be opened directly in any image viewer.

---

## Tests

```sh
cd src
make test
```

Three test programs:

| Test | What it checks |
|------|----------------|
| `tests/test_engine` | Unit tests for each op (forward + gradient via backward). 70 checks. |
| `tests/test_render` | Forward renders a 32×32 sphere; asserts center pixel is brighter than corner, all pixels finite. 3075 checks. |
| `tests/test_gradcheck` | Compares analytic gradients (from `tensor_backward`) against central finite differences for `center.x`, `radius`, `albedo.x`. Agreement to 5 significant figures. |

---

## Code layout

```
src/
├── engine.c / engine.h   # autograd: tensors, ops, backward, memory
├── vec3.h                # Vec3 (three Tensor* components), dot, length, normalize
├── sdf.h                 # sphere SDF (kept for future generic sphere tracing)
├── camera.h              # pinhole camera → Rays (scalar origin, field[N] dirs)
├── renderer.h            # analytic intersection, shading, image MSE
├── image.h               # write_ppm (binary P6)
├── loss.h                # scalar MSE helper
├── main.c                # inverse-rendering loop
├── Makefile
└── tests/
    ├── test_util.h       # CHECK / CHECK_CLOSE / CHECK_GRAD macros
    ├── test_engine.c
    ├── test_render.c
    └── test_gradcheck.c
```

---

## Roadmap

- **Generic SDF sphere tracing** — replace analytic intersection with a fixed-step
  march that tracks the closest-approach distance (proper min-tracking silhouette),
  enabling scenes not covered by closed-form intersection
- **Multiple primitives** — SDF composition via `min`, differentiable w.r.t. all
  primitive parameters
- **Soft shadows** — secondary march toward the light
- **Adam optimizer** — faster convergence than vanilla SGD
- **Hash-set topo sort** — the current O(N²) visited check in `tensor_backward`
  is fine for small graphs but becomes the bottleneck at large image sizes
