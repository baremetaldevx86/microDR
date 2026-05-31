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
