# CV-04 notes: 3D Vision & Neural Rendering

[← Lesson CV-04](../../lessons/vision/04-3d-vision-neural-rendering.md) · [All notes](../README.md) · [← CV-03 notes](03-clip-vision-language-models.md) · Next: [EL-01 notes →](../electives/01-time-series-forecasting.md)

> **Reading time** ≈ 60 min. **You need:** [MATH-01 notes](../math/01-linear-algebra.md) §A (matrices as maps), [DL-01 notes](../deep-learning/01-neural-networks-from-scratch.md) (MLPs, backprop), and [DL-08 notes](../deep-learning/08-modern-architectures-moe-ssm.md) §1 (sinusoidal encodings, which reappear here).

---

## Where we are

Images are 2-D projections of a 3-D world. This lesson goes backwards, from photos to a 3-D scene you can render from new viewpoints. It takes three pieces:

1. **Camera geometry:** how 3-D points land on pixels.
2. **Structure from motion:** where the cameras were.
3. **Neural scene representations** (NeRF, Gaussian splats): fit by gradient descent so that *rendering* them reproduces the photos.

---

## 1. The pinhole camera

### 1.1 Projection

A world point $X_w$ is first moved into the camera's coordinate frame by the **extrinsics**, a rotation $R$ and translation $t$: $X_c = R X_w + t$. It's then projected by the **intrinsics** $K$:

```math
\lambda\begin{bmatrix}u\\v\\1\end{bmatrix} = K\,[\,R \mid t\,]\begin{bmatrix}X_w\\1\end{bmatrix},\qquad K = \begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix}.
```

- $f_x, f_y$ are the focal lengths in pixels, and $(c_x, c_y)$ is the principal point (about the image centre).
- $\lambda$ is the depth $Z_c$. Dividing by it is the **perspective division** that makes distant things small: $u = f_x X_c/Z_c + c_x$ and $v = f_y Y_c/Z_c + c_y$.
- Projection **loses depth**: every point along a ray through the camera centre lands on the same pixel. That's why one image can't determine 3-D structure, and why we need several views.

*Worked example:* $f_x = f_y = 500$, $(c_x, c_y) = (320, 240)$, an identity rotation, $t = 0$, and the point $(0.2, -0.1, 2)$. Then $u = 500\cdot0.1 + 320 = 370$ and $v = 500\cdot(-0.05) + 240 = 215$.

### 1.2 Rays

To go from a pixel back into the world: the ray starts at the camera centre and points along $d = R^\top K^{-1}[u, v, 1]^\top$, normalized. NeRF renders an image by casting one such ray per pixel.

---

## 2. Where camera poses come from: SfM and MVS

NeRF and splats need each training photo's $K$, $R$, and $t$. In practice they come from **structure-from-motion** (COLMAP is the standard tool):

1. Detect keypoints (SIFT and similar) in every image, and **match** them across images.
2. Estimate the relative poses of image pairs from the matches (epipolar geometry), and **triangulate** the matched points into 3-D.
3. Add images incrementally, and refine everything with **bundle adjustment**: a big non-linear least-squares problem minimizing the **reprojection error** (the distance between each observed keypoint and the projection of its 3-D point) over all the poses and points.

The output is the camera poses plus a sparse point cloud. **Multi-view stereo** then densifies the geometry.
SfM fails on textureless surfaces, reflective or transparent objects, repeated patterns, or too little overlap between views, and **bad poses ruin everything downstream**.

---

## 3. NeRF: a scene as a neural field

### 3.1 The representation

An MLP $F_\theta$ maps a 3-D position $\mathbf{x}$ and a viewing direction $\mathbf{d}$ to a **volume density** $\sigma \ge 0$ ("how much stuff is here") and an **RGB colour** $\mathbf{c}$ (allowed to depend on direction, for reflections):

```math
F_\theta(\gamma(\mathbf{x}), \gamma(\mathbf{d})) \to (\sigma, \mathbf{c}).
```

### 3.2 Volume rendering (the differentiable renderer)

Sample points $t_1 < \dots < t_M$ along a pixel's ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$, with spacings $\delta_i = t_{i+1} - t_i$:

```math
\hat C(\mathbf r) = \sum_{i=1}^{M} T_i\,\big(1 - e^{-\sigma_i\delta_i}\big)\,\mathbf c_i,\qquad T_i = \exp\Big(-\sum_{j \lt i}\sigma_j\delta_j\Big).
```

- $\alpha_i = 1 - e^{-\sigma_i\delta_i}$ is the **opacity** of segment $i$: the chance the ray stops there.
- $T_i$ is the **transmittance**: the chance the ray got that far without being stopped.
- So $w_i = T_i\alpha_i$ is the probability the ray **terminates at sample $i$**, and the pixel colour is the expected colour along the ray. The weights sum to at most 1 (the remainder is the background). The same weights also give an expected **depth**, $\sum_i w_i t_i$.

**Why it's differentiable:** every operation (exponentials, products, sums) is smooth in $\sigma_i$ and $\mathbf c_i$, which are smooth outputs of the MLP. So the photometric loss $\lVert\hat C(\mathbf r) - C(\mathbf r)\rVert^2$ backpropagates to $\theta$. No 3-D supervision is needed, only posed photos.
NeRF also uses **hierarchical sampling**: a coarse pass finds where the weights are large, and a fine pass samples densely there.

### 3.3 Positional encoding: beating spectral bias

Plain MLPs on raw coordinates have **spectral bias**: they learn low-frequency (smooth) functions easily, and high-frequency detail very slowly. Fed $(x, y, z)$ directly, NeRF renders blurry scenes. The fix is to map each coordinate to multi-frequency sinusoids:

```math
\gamma(p) = \big(\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\big).
```

With $L = 10$ for positions, the MLP can represent fine detail, because high-frequency variation in space becomes a *simple* function of these features. (Same idea as Fourier features and the sinusoidal positions in transformers.) The code fits a 1-D signal with and without it.

### 3.4 NeRF's cost

Rendering one pixel means evaluating the MLP at 64–192 points along the ray. Training and rendering are slow (minutes to hours per scene in the original; seconds per frame).
Hash-grid encodings (Instant-NGP) make it orders of magnitude faster, by moving most of the capacity into a lookup table.

---

## 4. 3D Gaussian splatting

Represent the scene explicitly, as **millions of 3-D Gaussians**. Each has a position, a covariance (shape and orientation), an opacity, and a view-dependent colour (spherical harmonics). To render:

1. **Project** each Gaussian to a 2-D Gaussian on the image plane (a "splat"). The covariance transforms through the projection's Jacobian.
2. **Sort** the splats by depth, per screen tile.
3. **Alpha-composite** them front to back, with the *same* compositing equation as NeRF ($\sum_i T_i\alpha_i c_i$), but over the splats overlapping a pixel, not over samples along a ray.

**Why it's real-time:** it's **rasterization**, which is what GPUs are built for. There's no neural network queried at hundreds of points per pixel. Each pixel just blends a few sorted splats, in parallel tiles.
It's still differentiable, so the Gaussians are fitted to the photos by gradient descent, starting from the SfM point cloud. They're adaptively split, cloned, and pruned during training. The costs are memory (many Gaussians) and occasional artifacts in sparsely observed areas.

**Debug: a "foggy", blurry NeRF.** Two likely causes:

1. **inaccurate camera poses** (SfM errors, or mixing different intrinsics), so rays disagree and the model explains the inconsistency with semi-transparent fog;
2. **insufficient high-frequency capacity or sampling:** positional encoding missing or too few frequencies, too few samples per ray, or wrong near/far bounds.

Also check: too few or poorly distributed views ("floaters" in unobserved space), lighting or exposure changes between photos, moving objects, and an untreated background.

```python
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0)

# --- Projecting a point with K [R|t] -------------------------------------------------------------
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
R, t = np.eye(3), np.zeros(3)
X = np.array([0.2, -0.1, 2.0])
uvw = K @ (R @ X + t); u, v = uvw[:2] / uvw[2]
print(f"pixel: ({u:.0f}, {v:.0f})")
X_far = X * 3                                                   # 3x farther along the same ray
uvw2 = K @ X_far
print("a point 3x farther along the same ray lands on the same pixel:", np.allclose(uvw2[:2] / uvw2[2], [u, v]))

# --- Volume rendering weights along one ray hitting a surface at depth 4 ----------------------------------
ts = np.linspace(2, 6, 64); delta = np.diff(ts, append=ts[-1] + (ts[1] - ts[0]))
sigma = np.where(np.abs(ts - 4.0) < 0.1, 50.0, 0.0)            # an opaque, thin surface
alpha = 1 - np.exp(-sigma * delta)
T = np.exp(-np.concatenate([[0], np.cumsum(sigma * delta)[:-1]]))
w = T * alpha
color = np.where(ts < 4.5, 1.0, 0.0)                           # red before the surface, black after
print(f"sum of weights {w.sum():.3f}, expected depth {np.sum(w * ts) / w.sum():.2f}, rendered value {np.sum(w * color):.3f}")
```

```python
# --- Spectral bias: fitting a high-frequency 1-D signal with and without Fourier features -------------------------
x = torch.linspace(0, 1, 512)[:, None]
y = torch.sin(2 * np.pi * 3 * x) + 0.5 * torch.sin(2 * np.pi * 25 * x)

def encode(x, L):
    if L == 0: return x
    freqs = 2.0 ** torch.arange(L) * np.pi
    return torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], dim=1)

for L in [0, 8]:
    net = nn.Sequential(nn.Linear(max(1, 2 * L), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(1500):
        loss = ((net(encode(x, L)) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print(f"positional encoding L={L}: final MSE {loss.item():.4f}" + ("  <- misses the fine detail" if L == 0 else ""))
```

---

## Pitfalls & misconceptions

- **Camera convention mix-ups:** OpenCV vs OpenGL axes, world-to-camera vs camera-to-world matrices. Most "nothing works" NeRF bugs are here.
- **Trusting SfM poses blindly.** Check the reprojection error, and look at the cameras visually.
- **Too few views, or views from a narrow arc**, then expecting good novel views far outside them.
- **Varying exposure or white balance across photos.** Use per-image appearance embeddings or fixed camera settings.
- **Comparing NeRF and splatting only on quality.** Rendering speed, memory, and editability differ a lot.

## Cheat sheet

| Item | Formula |
|---|---|
| Projection | $\lambda[u, v, 1]^\top = K[R\mid t][X_w; 1]$ |
| Pinhole | $u = f_xX_c/Z_c + c_x$ |
| Ray direction | $R^\top K^{-1}[u, v, 1]^\top$ |
| Opacity / transmittance | $\alpha_i = 1-e^{-\sigma_i\delta_i}$, $T_i = e^{-\sum_{j \lt i}\sigma_j\delta_j}$ |
| Rendered colour | $\sum_i T_i\alpha_i c_i$ (NeRF and splats alike) |
| Positional encoding | $(\sin 2^k\pi p, \cos 2^k\pi p)_{k \lt L}$ |
| Bundle adjustment | minimize the total reprojection error over poses + points |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Projection with K and [R | t].</summary>

$\lambda[u, v, 1]^\top = K(RX_w + t)$, so $u = f_x X_c/Z_c + c_x$ and $v = f_y Y_c/Z_c + c_y$, with $X_c = RX_w + t$. The worked example gives pixel (370, 215).
</details>

<details>
<summary>2. Why does NeRF need positional encoding?</summary>

MLPs on raw coordinates have spectral bias: they fit smooth functions and struggle with high frequencies. Sinusoidal multi-frequency features make fine detail easy to represent. Without them, renders are blurry and lose texture (the demo's MSE gap).
</details>

<details>
<summary>3. What does the volume-rendering integral accumulate, and why is it differentiable?</summary>

The expected colour along the ray: each sample's colour weighted by the probability that the ray terminates there, $T_i\alpha_i$. All its operations are smooth functions of the MLP's density and colour outputs, so the photometric loss backpropagates to the network weights.
</details>

<details>
<summary>4. Where do training camera poses come from?</summary>

Structure-from-motion (for example COLMAP): feature matching, triangulation, and bundle adjustment to minimize the reprojection error. Sometimes they come from device sensors (phone AR tracking) or calibrated rigs.
</details>

<details>
<summary>5. Why are Gaussian splats faster to render than NeRF?</summary>

They're rasterized: project the explicit Gaussians, sort them per tile, and alpha-blend a few splats per pixel on GPU hardware. NeRF queries an MLP at many samples along every ray.
</details>

<details>
<summary>6. Blurry, "foggy" NeRF renders.</summary>

Inaccurate or inconsistent camera poses (the model "hedges" with semi-transparent density), and insufficient frequency capacity or sampling (no or too little positional encoding, too few samples, bad near/far bounds). Also: sparse views, exposure changes, moving objects.
</details>

## Where this leads

**Path 5 is complete.** The electives start at [EL-01 notes](../electives/01-time-series-forecasting.md). Each one is self-contained, so pick by interest.
