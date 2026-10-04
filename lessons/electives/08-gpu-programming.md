# EL-08: GPU Programming for ML: CUDA Concepts, Triton & Kernels

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~11 h | L3 | DL-07 |

## Why this matters
FlashAttention, fused optimizers, and quantized matmuls are all custom kernels, and they often give bigger speed-ups than any model change.
Understanding the GPU execution model (threads, blocks, shared memory, memory coalescing) and writing a few kernels yourself lets you read
systems papers, use `torch.compile`/Triton output intelligently, and optimize the hot spots in your own code.

## Learning goals
By the end you can:
- Explain the GPU execution model (grids, blocks, threads, warps) and the memory hierarchy (registers, shared memory/SRAM, HBM).
- Write simple kernels: map, zip, reduction, softmax, and tiled matmul.
- Explain coalescing, bank conflicts, occupancy, and tiling, and use a roofline to reason about performance.
- Write and benchmark a Triton kernel, and compare it with PyTorch eager and `torch.compile`.
- Explain why FlashAttention's tiling makes attention IO-efficient.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: GPU Programming for ML](../../notes/electives/08-gpu-programming.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Horace He: Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | Reread it with kernels in mind (memory bandwidth, fusion) | 30 min |
| 2 | **Build** | [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | All puzzles (map → zip → guards → shared memory → pooling → dot product → matmul) | 3 h |
| 3 | **Watch + Build** | [GPU MODE lectures](https://github.com/gpu-mode/lectures) | The early lectures on profiling, CUDA basics, and Triton, with their code | 3 h |
| 4 | **Read** | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | The GPU chapter (streaming multiprocessors, the memory hierarchy, networking) | 1 h |
| 5 | **Read** | [FlashAttention](https://arxiv.org/abs/2205.14135) | §1–3 (standard attention's IO problem, the tiled algorithm) | 1 h |
| 6 | **Build** | [LeetGPU](https://leetgpu.com/) or [Tensara problems](https://github.com/tensara/problems) | Solve 5 problems (vector add, reduction, softmax, matmul, one of your choice) | 1.5 h |

## Check your understanding
1. Why are warps important, and what is warp divergence?
2. What makes a global-memory access pattern coalesced?
3. How does tiling a matmul through shared memory reduce HBM traffic?
4. Softmax needs a max and a sum over a row. How do you write it as a fused, numerically stable kernel?
5. Why does FlashAttention never materialize the full N×N attention matrix, and what does it recompute instead?
6. *(debug)* Your hand-written matmul kernel is 10× slower than cuBLAS. List the first three things to check.

## Mini-project
**Task:** Write a fused softmax and a tiled matmul in Triton. Benchmark them against PyTorch eager and `torch.compile` across matrix sizes,
and plot achieved GB/s or TFLOP/s against the hardware roofline.
**Deliverable:** A benchmark notebook and a roofline plot.

## Go deeper
- [Stanford CS336 assignment 2](https://github.com/stanford-cs336/assignment2-systems): FlashAttention in Triton, plus profiling.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691).
- [llm.c](https://github.com/karpathy/llm.c): GPT-2 training in raw C/CUDA.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md) (performance, GPUs & scale).
- **Papers:** [Efficiency & systems](../../papers/06-efficiency-systems.md) (attention kernels section).
- **Implement it yourself:** the [capstone rung](../../exercises/from-scratch-ladder.md): the GPT-2 forward pass in C/CUDA following llm.c.
- **Drills:** [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) · [LeetGPU](https://leetgpu.com/) · [Tensara](https://github.com/tensara/problems).
