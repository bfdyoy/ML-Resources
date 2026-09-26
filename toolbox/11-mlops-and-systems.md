# Toolbox 11: MLOps, Systems & Scale

[← Toolbox](README.md) · Guided version: [PROD-01](../lessons/production/01-ml-system-design.md), [PROD-02](../lessons/production/02-mlops-in-practice.md) · Papers: [06 Efficiency & systems](../papers/06-efficiency-systems.md), [11 Production](../papers/11-ml-in-production.md)

**Whole-field resources:** [Made With ML](https://madewithml.com/) · [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) · [Stanford CS329S: ML Systems Design](https://stanford-cs329s.github.io/) · [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/) · [Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) `[paid]` ([summaries](https://github.com/chiphuyen/dmls-book)) · [awesome-mlops](https://github.com/visenger/awesome-mlops)

## Designing ML systems
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Rules of thumb for ML engineering | [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) | [Hidden Technical Debt in ML Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) | Score an old project against the rules |
| System design (batch vs online, features, feedback loops) | [CS329S syllabus & notes](https://stanford-cs329s.github.io/) | [DMLS summaries](https://github.com/chiphuyen/dmls-book) | Write a design doc (PROD-01 mini-project) |
| Learning from real systems | [Evidently: 800 ML & LLM system-design case studies](https://www.evidentlyai.com/ml-system-design) | [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml) | Summarise one case study per week in 5 bullets |
| Problems people hit in practice | — | [Challenges in Deploying ML](https://arxiv.org/abs/2011.09926) · [Operationalizing ML](https://arxiv.org/abs/2209.09125) | — |

## The MLOps lifecycle
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| MLOps maturity levels & CI/CD/CT | [Google Cloud: MLOps pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | [Martin Fowler: CD4ML](https://www.martinfowler.com/articles/cd4ml.html) | A GitHub Actions workflow that trains, tests, and publishes a model |
| Experiment tracking & model registry | [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) (experiment tracking module) | [MLflow](https://github.com/mlflow/mlflow) docs | Track 10 runs, register the best one |
| Data & pipeline versioning, orchestration | [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) (orchestration module) | [DVC](https://github.com/iterative/dvc), [ZenML](https://github.com/zenml-io/zenml) docs | Version a dataset; rebuild a model from a commit |
| Feature stores | — | [Feast](https://github.com/feast-dev/feast) docs · [DMLS summaries](https://github.com/chiphuyen/dmls-book) (feature engineering) | Serve the same feature online and offline |
| Testing ML systems | [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) | [Made With ML](https://madewithml.com/) (testing lesson) | Data tests + model behaviour tests in pytest |
| Model serving & packaging | [Made With ML](https://madewithml.com/) (serving lesson) | [BentoML](https://github.com/bentoml/BentoML) docs · [learnpytorch.io 09](https://www.learnpytorch.io/09_pytorch_model_deployment/) | FastAPI + Docker service with a health check and batching |
| Monitoring, drift & observability | [Evidently: ML Observability course](https://www.evidentlyai.com/ml-observability-course) | [DMLS summaries](https://github.com/chiphuyen/dmls-book) (data distribution shifts) | [Evidently](https://github.com/evidentlyai/evidently) drift report on a shifted dataset |
| Data-centric quality in production | [MIT DCAI](https://dcai.csail.mit.edu/) | — | [cleanlab](https://github.com/cleanlab/cleanlab) on production-like noisy labels |
| LLMOps: tracing, eval-driven development, cost | [applied-llms.org](https://applied-llms.org/) (operational section) | [Chip Huyen: GenAI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) · [Chip Huyen: Common pitfalls when building GenAI apps](https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html) · [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) | Add tracing and an eval gate to your GEN-03 capstone |

## Performance, GPUs & scale
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| GPU performance mental model | [Horace He: Making DL Go Brrrr](https://horace.io/brrr_intro.html) | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (GPU chapter, rooflines) | Profile and fix one bottleneck ([PyTorch profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)) |
| Writing GPU kernels (CUDA, Triton) | [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | [GPU MODE lectures](https://github.com/gpu-mode/lectures) | [LeetGPU](https://leetgpu.com/) · [Tensara problems](https://github.com/tensara/problems) |
| Data parallelism (DDP) | [PyTorch DDP series intro](https://pytorch.org/tutorials/beginner/ddp_series_intro.html) | [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | Train on 2 GPUs (or 2 processes on CPU) with DDP |
| Sharded training (ZeRO / FSDP) | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) · [ZeRO](https://arxiv.org/abs/1910.02054) · [FSDP paper](https://arxiv.org/abs/2304.11277) | Fit a model that doesn't fit on one GPU |
| Tensor, pipeline, sequence & expert parallelism | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | [Lilian Weng: How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/) · [Megatron-LM](https://arxiv.org/abs/1909.08053) · [GPipe](https://arxiv.org/abs/1811.06965) | [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) |
| Training infrastructure in the real world | — | [Stas Bekman: ML Engineering Open Book](https://github.com/stas00/ml-engineering) | — |
| LLM inference serving | [HF: Optimizing LLMs](https://huggingface.co/blog/optimize-llm) | [vLLM](https://github.com/vllm-project/vllm) · [Lilian Weng: Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Benchmark throughput/latency vs batch size |
| Efficient ML (pruning, quantization, NAS, on-device) | — | [MIT 6.5940 EfficientML](https://hanlab.mit.edu/courses/2026-fall-65940) | Do one lab from the course |
