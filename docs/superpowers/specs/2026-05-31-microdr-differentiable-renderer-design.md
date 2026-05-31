# microDR — Differentiable Renderer Design

**Date:** 2026-05-31
**Status:** Approved (brainstorming) — pending implementation plan

## Goal

Build a *capable* tiny differentiable renderer in pure C on top of the existing
hand-written autograd engine (`engine.c`/`engine.h`). The renderer performs
sphere tracing of a signed-distance scene, shades the surface with a Lambertian
model, and supports **inverse rendering**: recovering scene geometry and
appearance (sphere center, radius, albedo, and optionally light direction) from
a target image by gradient descent.

The first capable version targets a **single sphere**, designed so that
multi-primitive scenes (`min`-composition), soft shadows, and a faster topo sort
can be added later without a rewrite.

## Scoping decisions (from brainstorming)

1. **Ambition:** capable renderer (real sphere tracing + shading + inverse
   rendering), not a minimal toy.
2. **Data flow:** fully **vectorized** over rays — the whole image renders
   through one computation graph.
3. **Shading:** **Lambertian RGB**. Geometry via **analytic ray–sphere
   intersection** (see "Rendering pipeline"); the surface normal is the
   analytic `normalize(p − center)`.
4. **Scene scope:** **single sphere**; optimize center (3), radius (1), albedo
   RGB (3), optionally light direction (3). Interfaces structured for later
   multi-primitive composition.
5. **Output:** **PPM** images (zero dependencies), dumped periodically.
6. **Inverse setup:** render a **synthetic target** from known ground-truth
   params, perturb, and recover — enabling finite-difference gradient checks.

## Architecture & data layout

The pivotal design choice: **keep the `Vec3` struct unchanged** (three
`Tensor*` components), but let each component be an `[N×1]` column tensor where
`N = H·W`, instead of a scalar. A `Vec3` thus becomes a *field* of vectors — one
3D vector per pixel — and every existing `vec3_add/sub/scale/dot` operates over
the whole image element-wise, unchanged.

Scene and material parameters (center, radius, albedo, light direction) stay as
**scalar `[1]` tensors** and **broadcast** across the N pixels.

The entire image is rendered through **one computation graph**; backward runs
once per iteration.

```
ray_origin[N], ray_dir[N]  ──► march (K steps) ──► hit point p[N]
                                      │
                                      ├─► finite-diff normal n[N]
                                      ├─► soft mask = sigmoid(-k·sdf_final)[N]
                                      └─► Lambertian shade · albedo ─► color[N per channel]
                                                                          │
target image ───────────────────────────────────────────► MSE loss ─► backward
```

## Engine changes (foundation — implement and verify first)

1. **Generalized broadcasting** for `mul`, `sub`, `div` (and tidy up `add`):
   - Rule: if operand sizes are equal → element-wise; else if one operand has
     `size == 1` → broadcast that scalar across the field; output takes the
     larger shape.
   - **Backward must accumulate the scalar operand's gradient as the sum of all
     incoming grads.**
   - This is what makes `dist[N] − radius[1]`, `k[1]·d[N]`, and
     `albedo[1]·shade[N]` work with correct gradients. **This is the single most
     important addition.**
2. **Fix `sqrt_backward` divide-by-zero** (`engine.c:333`): use an epsilon —
   `grad / (2·sqrt(a + ε))` — and clamp the forward similarly. Critical for SDF
   and vector-length numerical stability (a ray sample at the sphere center
   otherwise produces NaN gradients).
3. **`tensor_sigmoid`** (forward + numerically-stable backward) for the soft
   silhouette mask.
4. **Fix `main.c` memory leaks** (the `W*H` constant and the `loss_sum` chain in
   the current loop); maintain release discipline in the rewritten `main.c`.
5. *(Deferred — not in this milestone set)* `min`/`max` ops for multi-primitive
   scenes. A single sphere does not need them.

**Known limitation (accepted, not fixed now):** the topo-sort visited check in
`tensor_backward` is O(N²) in the number of graph nodes. It is acceptable at the
image sizes targeted here; the eventual fix is a hash set. Noted, not addressed
in this milestone set.

## Rendering pipeline

Replaces the per-pixel `renderer.h` with a vectorized, **analytic ray–sphere**
pipeline. This was chosen over a generic SDF sphere-trace because the naive
marched silhouette (`sigmoid(−k·sdf_final)`) collapses to a flat 0.5-alpha disk
— the march stops at the surface and never produces the negative interior
distances that fill the mask. The analytic form is closed-form, fully
differentiable, and gives a correct filled silhouette + soft edge. Generic SDF
sphere tracing (with closest-approach `min`-tracking to fix the silhouette
properly) is the documented upgrade path, alongside multi-primitive `min`.

Let camera origin `o` be **scalar** `[1]` components (same for all pixels), ray
direction `v` be a **field** `[N]`, sphere `center` and `radius` scalar `[1]`.

- **Camera** (`camera.h`): pinhole. Origin = 3 scalar `[1]` tensors; direction =
  3 field `[N]` tensors, one normalized ray per pixel.
- **Intersection + silhouette** (`renderer.h`):
  - `oc = center − origin` (scalar `[1]`); `tca = dot(oc, v)` (`[N]`);
    `d2 = dot(oc,oc) − tca²` (perpendicular distance², `[N]`).
  - `perp = sqrt(d2 + ε)`; silhouette distance `sil = perp − radius` (negative
    *inside* the disk). `mask = sigmoid(−k·sil)` → correct filled silhouette
    with a soft edge.
  - hit depth `t_hit = tca − sqrt(relu(radius² − d2) + ε)`; hit point
    `p = origin + t_hit·v`.
- **Normal**: `n = normalize(p − center)` (analytic, ε-safe).
- **Shade**: `lambert = relu(dot(n, light_dir))`; per channel
  `shaded_c = albedo_c · lambert + ambient`.
- **Composite**: `color_c = mask·shaded_c + (1−mask)·background_c`. The mask
  provides silhouette/shape gradients; shading provides interior shape and
  albedo gradients.

## Inverse-rendering loop (`main.c`)

- Build **ground-truth params** → render once → store as the **target image**
  (constant; built from `tensor_create` leaves that are never updated).
- Initialize **learnable params** perturbed from ground truth: `center`,
  `radius` (parametrized as `exp(raw)` to stay positive), `albedo` (3),
  optionally `light_dir`.
- Each iteration: render → `loss = mean((pred − target)²)` over all RGB
  elements (`tensor_mean`) → `tensor_backward(loss)` → manual SGD step (Adam is
  a later upgrade) → release the per-iteration graph.
- Print loss and params; dump images every N iterations.

## Visualization & verification

- **PPM writer** (`image.h`): write `pred`, `target`, and an `|error|` map as
  `.ppm` every N iterations. Clamp to [0,1] only at write time — this clamp is
  non-differentiable and output-only.
- **Gradient check** (`gradcheck`): compare analytic parameter gradients against
  central finite differences of the loss for a few parameters. This is the
  **primary correctness gate** before trusting convergence results.

## Module plan

| File | Role |
|---|---|
| `engine.c` / `engine.h` | broadcasting for `mul`/`sub`/`div`, `sqrt` ε-fix, `sigmoid` |
| `vec3.h` | unchanged ops + add `vec3_length` and `vec3_normalize` (ε-safe) |
| `sdf.h` | `sdf_sphere` using ε-safe length; kept for the future generic-SDF upgrade (not used by the analytic renderer) |
| `camera.h` | pinhole ray generation (`Rays`: scalar origin, field dir) |
| `renderer.h` | analytic intersection + normal + shade + composite (header-only) |
| `image.h` | PPM output (pred, target, error map) |
| `loss.h` | keep `mse` (mean of squared difference) |
| `main.c` | inverse-rendering loop + gradient check |

## Milestones (each independently verifiable)

1. **Engine:** broadcasting + `sqrt` ε-fix + `sigmoid`, with a small unit test
   and a finite-difference gradient check on each new/changed op.
2. **Forward render:** camera + analytic intersection + shade + composite → dump
   a single PPM of a hard-coded sphere. *Verify by eye it looks like a shaded
   sphere; assert a center pixel is brighter than a corner pixel.*
3. **End-to-end gradient check:** loss gradient w.r.t. center, radius, and
   albedo matches central finite differences.
4. **Inverse rendering:** perturb ground truth, recover params, confirm loss → 0
   and the dumped PPMs converge to the target.
5. *(Later, out of scope now)* generic SDF sphere tracing with closest-approach
   `min`-tracking, multi-primitive `min`-composition, soft shadows, faster
   (hash-set) topo sort, Adam optimizer.

## Out of scope (explicitly deferred)

- Generic SDF sphere tracing (march loop) and multi-primitive scenes
  (`min`/`max`). The single-sphere milestone uses analytic intersection.
- Soft shadows and a secondary march toward the light.
- Adam / momentum optimizers.
- Hash-set topo sort (O(N²) accepted for now).
- PNG output (PPM only; convert externally if desired).
