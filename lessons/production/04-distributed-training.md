# PROD-04: Distributed Training at Scale

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Production | ~9 h | L3 | DL-07, GEN-01 |

## Why this matters
As soon as a model or batch doesn't fit on one GPU, or training takes too long, you need parallelism. Data parallelism, sharding
(ZeRO/FSDP), and tensor, pipeline, sequence, and expert parallelism are how every large model is trained. Even with a single node,
knowing the memory and communication maths lets you pick the right setup instead of guessing.

## Learning goals
By the end you can:
- Break down training memory (parameters, gradients, optimizer states, activations), and estimate it for a given model.
- Explain DDP (all-reduce of gradients) and when communication becomes the bottleneck.
- Explain ZeRO stages 1–3 / FSDP, and what each stage shards.
- Explain tensor, pipeline, sequence/context, and expert parallelism, and how they are combined ("5D parallelism").
- Launch DDP and FSDP training with `torchrun`, and measure scaling efficiency.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face) | Memory usage → data parallelism → ZeRO → tensor parallelism → pipeline parallelism → context & expert parallelism. The primary text of this lesson. | 3 h |
| 2 | **Read** | [Lilian Weng: How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) | The whole post, as a second view | 1 h |
| 3 | **Build** | [PyTorch: DDP series intro](https://pytorch.org/tutorials/beginner/ddp_series_intro.html) → [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | Convert your GPT training loop to DDP with `torchrun` (2 GPUs, or 2 CPU processes with the gloo backend) | 2 h |
| 4 | **Build** | [PyTorch: FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | Wrap the same model with FSDP. Compare peak memory per rank with DDP. | 1.5 h |
| 5 | **Build** | [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) | Work the memory/communication puzzles | 1.5 h |

**Notes for the learner:** You don't need a GPU cluster. The concepts and the code work with 2 processes on one machine. Do the maths
by hand for a 7B model at least once.

## Check your understanding
1. Estimate the training memory for a 7B model with AdamW in mixed precision, without any sharding. How much do ZeRO-1/2/3 save?
2. What does all-reduce do in DDP, and why can it overlap with the backward pass?
3. Tensor vs pipeline parallelism: what does each split, and what does each cost in communication or bubbles?
4. Why is tensor parallelism usually kept inside a node?
5. What does gradient accumulation give you, and what doesn't it give you, compared with a truly larger batch?
6. *(debug)* Going from 1 to 8 GPUs with DDP gives only a 3× speed-up. List the likely bottlenecks and how you'd measure each.

## Mini-project
**Task:** Train your nanoGPT (DL-06/DL-08) with a single process, DDP, and FSDP on the same hardware. Report tokens/sec, peak memory per rank,
and scaling efficiency. Write a one-page plan for training a 7B model on 64 GPUs (which parallelism dimensions, and why).
**Deliverable:** A benchmark table and the plan document.

## Go deeper
- Papers: [ZeRO](https://arxiv.org/abs/1910.02054) · [Megatron-LM](https://arxiv.org/abs/1909.08053) · [GPipe](https://arxiv.org/abs/1811.06965) · [PyTorch FSDP](https://arxiv.org/abs/2304.11277)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the parallelism chapters from the TPU/JAX side.
- [Stas Bekman: ML Engineering Open Book](https://github.com/stas00/ml-engineering): hardware, networking, debugging, and fault tolerance at scale.
- [Stanford CS336 assignment 2](https://github.com/stanford-cs336/assignment2-systems): distributed-training and Triton exercises.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md) (performance, GPUs & scale).
- **Papers:** [Efficiency & systems](../../papers/06-efficiency-systems.md) (distributed training section).
- **Implement it yourself:** a manual all-reduce data-parallel step with `torch.distributed` (no DDP wrapper). Check its gradients against a single-process run.
- **Drills:** [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) · more in [exercises/](../../exercises/README.md).
