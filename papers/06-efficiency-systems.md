# Papers: Efficiency, Fine-tuning & ML Systems

[← Papers library](README.md) · Background: [GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md), [GEN-08](../lessons/llms-genai/08-efficient-llm-inference.md), [PROD-04](../lessons/production/04-distributed-training.md) · [Toolbox: MLOps & systems](../toolbox/11-mlops-and-systems.md)

## Parameter-efficient fine-tuning (PEFT)
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Parameter-Efficient Transfer Learning for NLP (Adapters)](https://arxiv.org/abs/1902.00751) | 2019 | Small trainable modules inserted into a frozen model. | L2 | GEN-02 |
| [Prefix-Tuning](https://arxiv.org/abs/2101.00190) | 2021 | Learning continuous "virtual tokens". | L2 | GEN-02 |
| [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | 2021 | Soft prompts become competitive as models get bigger. | L2 | GEN-02 |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | ⭐ Low-rank weight updates, today's default PEFT method. | L2 | GEN-02 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | ⭐ LoRA on a 4-bit base model: fine-tune a 65B model on one GPU. | L2 | GEN-02 |
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | 2024 | Splits magnitude and direction for better LoRA. | L3 | GEN-02 |

## Quantization, pruning & distillation
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) | 2021 | ⭐ The best introduction to PTQ vs QAT. Start here. | L2 | GEN-08 |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | 2015 | Pruning + quantization + Huffman coding. | L2 | GEN-08 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 2022 | Outlier features and 8-bit inference for LLMs. | L3 | GEN-08 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 2022 | Accurate 3–4-bit post-training quantization. | L3 | GEN-08 |
| [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) | 2023 | Protect the most important weights. Widely deployed. | L3 | GEN-08 |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | 2015 | ⭐ Knowledge distillation. | L2 | GEN-08 |

## Attention kernels & inference serving
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | ⭐ Tiling to cut memory reads/writes. It teaches you to think about GPU memory hierarchies. | L3 | GEN-08 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2023 | Better parallelism and work partitioning. | L3 | GEN-08 |
| [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) | 2023 | ⭐ Paging for the KV cache, and continuous batching. | L2 | GEN-08 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 2022 | ⭐ Draft with a small model, verify with the big one. Same output distribution, faster. | L2 | GEN-08 |
| [Medusa](https://arxiv.org/abs/2401.10774) | 2024 | Multiple decoding heads instead of a separate draft model. | L3 | GEN-08 |
| [SGLang](https://arxiv.org/abs/2312.07104) | 2023 | Structured generation programs and a RadixAttention prefix cache. | L3 | GEN-08 |

## Distributed training
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Megatron-LM: Training Multi-Billion Parameter LMs Using Model Parallelism](https://arxiv.org/abs/1909.08053) | 2019 | ⭐ Tensor parallelism. | L3 | PROD-04 |
| [GPipe](https://arxiv.org/abs/1811.06965) | 2018 | Pipeline parallelism with micro-batches. | L3 | PROD-04 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 2019 | ⭐ Sharding optimizer state, gradients, and parameters (the basis of DeepSpeed and FSDP). | L3 | PROD-04 |
| [PyTorch FSDP](https://arxiv.org/abs/2304.11277) | 2023 | How ZeRO-3-style sharding is implemented in PyTorch. | L3 | PROD-04 |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | 2017 | FP16 training with loss scaling. | L2 | PROD-04 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 2016 | Gradient checkpointing. | L2 | PROD-04 |

> **Read these with:** [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face), which explains every technique above with measurements from 4,000+ experiments.
