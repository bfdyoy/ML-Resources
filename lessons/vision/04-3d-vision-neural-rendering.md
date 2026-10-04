# CV-04: 3D Vision & Neural Rendering

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Computer Vision | ~8 h | L3 | DL-04 · math: [MATH-01](../math/01-linear-algebra.md) (all blocks) |

## Why this matters
Robotics, AR/VR, mapping, and 3D content creation all need geometry: cameras, depth, and 3D reconstruction. Neural radiance fields
(NeRF) and 3D Gaussian splatting changed the field by learning a scene from photos. They're also a great exercise in combining
classical geometry with differentiable optimization.

## Learning goals
By the end you can:
- Explain the pinhole camera model, intrinsics/extrinsics, and how a 3D point projects to a pixel.
- Describe structure-from-motion and multi-view stereo at a conceptual level (where camera poses come from).
- Explain NeRF: a positional-encoded MLP from (x, y, z, direction) to (density, colour), plus differentiable volume rendering.
- Explain 3D Gaussian splatting, and why it renders in real time while NeRF doesn't.
- Train a small NeRF on a synthetic scene.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: 3D Vision & Neural Rendering](../../notes/vision/04-3d-vision-neural-rendering.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed.)](https://szeliski.org/Book/) | Ch. 2 §2.1 (geometric primitives and transformations, 3D→2D projection). Then skim Ch. 11 (structure from motion and SLAM) for the big picture. | 2 h |
| 2 | **Read** | [Szeliski](https://szeliski.org/Book/) | Ch. 14 "Image-based rendering", including the neural rendering section | 1 h |
| 3 | **Read** | [NeRF](https://arxiv.org/abs/2003.08934) | §3–5: the scene representation, volume rendering, positional encoding, and hierarchical sampling | 1.5 h |
| 4 | **Read** | [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | §1, §3–5: the representation and the fast rasterizer | 1 h |
| 5 | **Build** | Tiny NeRF | Implement a minimal NeRF (no hierarchical sampling) on a synthetic scene at low resolution | 1.5 h |

**Notes for the learner:** This lesson leans on papers because the field is young. Szeliski gives you the classical background, so
read §2.1 carefully. Every later equation depends on the projection model.

## Check your understanding
1. Write the projection of a 3D point to pixel coordinates with intrinsics K and extrinsics [R | t].
2. Why does NeRF need positional encoding? What happens to renders without it?
3. What does the volume-rendering integral accumulate along a ray, and why is it differentiable?
4. Where do NeRF's training camera poses come from in practice?
5. Why are Gaussian splats faster to render than NeRF's ray marching?
6. *(debug)* Your NeRF renders blurry, "foggy" images. What are two likely causes?

## Mini-project
**Task:** Train a tiny NeRF on a synthetic scene, and render a 360° video. Ablate positional encoding (on/off) and the number of samples per ray.
**Deliverable:** Rendered frames, and a PSNR table for the ablations.

## Go deeper
- [Foundations of Computer Vision](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/) (Torralba, Isola & Freeman): the geometry and 3D chapters.
- [Szeliski](https://szeliski.org/Book/) Ch. 12 (depth estimation) and Ch. 13 (3D reconstruction).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 06: Computer vision](../../toolbox/06-computer-vision.md) (3D vision & neural rendering).
- **Papers:** [Computer vision](../../papers/02-computer-vision.md) (3D & neural rendering section).
- **Implement it yourself:** camera projection plus ray generation from scratch in NumPy. Verify that reprojecting known 3D points lands on the right pixels.
- **Drills:** more in [exercises/](../../exercises/README.md).
