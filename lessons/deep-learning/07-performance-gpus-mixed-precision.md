# DL-07: Making Training Fast: GPUs, Mixed Precision, Compilation & Profiling

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~6 h | L2→L3 | DL-02, DL-03 |

## Why this matters
The same model can train 2–5× faster with the right precision, batch size, data loading, and compilation. That's the difference
between 10 experiments a day and 2. More importantly, knowing *why* a step is slow (compute-bound, memory-bound, or overhead-bound)
stops you from guessing at optimizations.

## Learning goals
By the end you can:
- Classify a workload as compute-, memory-bandwidth-, or overhead-bound, and pick the right fix.
- Use mixed precision (`torch.autocast`, BF16/FP16), and explain loss scaling.
- Use `torch.compile`, and explain operator fusion.
- Profile a training step and find the bottleneck (the data loader, a Python-side overhead, or a slow kernel).
- Reason about memory: parameters, gradients, optimizer state, and activations, plus when to use gradient checkpointing.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Horace He: Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | The whole post (compute, memory bandwidth, overhead, operator fusion) | 45 min |
| 2 | **Read** | [PyTorch: Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | The whole guide: data loading, AMP, cuDNN settings, and more | 45 min |
| 3 | **Build** | [PyTorch: torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | Compile a model and measure the speed-up (with a warm-up) | 1 h |
| 4 | **Build** | [PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | Profile one training step of your DL-04 ResNet. Find the top 5 ops by time and memory. | 1 h |
| 5 | **Read** | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | The rooflines chapter and the GPU chapter (arithmetic intensity, how GPUs are structured) | 1.5 h |
| 6 | *Read (optional)* | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | The "memory usage in transformers" section (where memory goes during training) | 45 min |

**Notes for the learner:** Always measure throughput after a warm-up, and with `torch.cuda.synchronize()`. GPU calls are
asynchronous, so naive timing lies.

## Check your understanding
1. What is arithmetic intensity, and why is a big matmul compute-bound while an elementwise op is memory-bound?
2. Why does fusing `x.cos().cos()` into one kernel help, even though the FLOP count is identical?
3. BF16 vs FP16: why does BF16 usually not need loss scaling?
4. Estimate the training memory for a 1B-parameter model with AdamW in mixed precision (parameters + gradients + optimizer state).
5. What does gradient checkpointing trade off?
6. *(debug)* GPU utilization sits at 35% during training. What are the three most likely culprits, and how do you tell them apart?

## Mini-project
**Task:** Take your DL-04 CIFAR-10 ResNet and make it as fast as possible without losing accuracy. Try, in order: DataLoader workers
and pinned memory → channels_last → AMP (BF16) → `torch.compile` → a bigger batch with LR scaling. Log images/sec and final accuracy at each step.
**Deliverable:** A speed-up table and a profiler trace screenshot before and after.

## Go deeper
- [Tim Dettmers: Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): how GPU specs map to DL performance.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) and [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).
- [EL-08 GPU Programming](../electives/08-gpu-programming.md): write the kernels yourself.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 04: Deep learning fundamentals](../../toolbox/04-deep-learning-fundamentals.md) (performance & precision) · [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md).
- **Papers:** [Efficiency & systems](../../papers/06-efficiency-systems.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rung 24 (ResNet with AMP and a cosine schedule).
- **Drills:** [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) · more in [exercises/](../../exercises/README.md).
