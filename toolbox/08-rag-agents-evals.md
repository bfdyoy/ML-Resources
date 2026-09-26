# Toolbox 08: RAG, Agents & LLM Evaluation

[← Toolbox](README.md) · **Taught in:** [GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md) · [GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md) · [GEN-05](../lessons/llms-genai/05-retrieval-engineering.md) · [GEN-06](../lessons/llms-genai/06-agents-tool-use.md) · [GEN-09](../lessons/llms-genai/09-llm-security-safety.md) · Papers: [05 Retrieval & RAG](../papers/05-retrieval-rag.md), [04 Agents & evals](../papers/04-alignment-reasoning-agents.md)

**Practitioner guides for the whole topic:** [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/) (Yan, Bischof, Frye, Husain, Liu, Shankar) · [Eugene Yan: Patterns for LLM Systems](https://eugeneyan.com/writing/llm-patterns/) · [Chip Huyen: Building a GenAI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) · [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) `[paid]`

## Retrieval foundations (most RAG failures start here)
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Inverted indexes, tf-idf, BM25 | — | [IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) Ch. 1, 6, 11 | Implement BM25 from scratch; compare with `rank_bm25` | — |
| Retrieval evaluation (recall@k, MRR, NDCG) | — | [IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) Ch. 8 | Build a 50-query labeled retrieval set for your own docs | — |
| Dense embeddings & bi-encoders | [Lilian Weng: How to build an open-domain QA system](https://lilianweng.github.io/posts/2020-10-29-odqa/) | [Sentence Transformers docs](https://sbert.net/) · [HF: MTEB](https://huggingface.co/blog/mteb) | Fine-tune an embedding model on your domain pairs | [SBERT](https://arxiv.org/abs/1908.10084), [DPR](https://arxiv.org/abs/2004.04906), [E5](https://arxiv.org/abs/2212.03533), [MTEB](https://arxiv.org/abs/2210.07316) |
| Vector search (ANN, HNSW, IVF, PQ) | — | [Faiss](https://github.com/facebookresearch/faiss) wiki · [HF: Embedding quantization](https://huggingface.co/blog/embedding-quantization) | Benchmark exact vs HNSW recall and latency with Faiss | [HNSW](https://arxiv.org/abs/1603.09320) |
| Hybrid search & reranking (cross-encoders, late interaction) | — | [Sentence Transformers: cross-encoders](https://sbert.net/) | BM25 + dense + cross-encoder rerank; measure each stage | [ColBERT](https://arxiv.org/abs/2004.12832), [SPLADE](https://arxiv.org/abs/2107.05720) |

## Building RAG
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Naive RAG end to end | [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) (notebooks) | [RAG survey](https://arxiv.org/abs/2312.10997) | Q&A over your own docs (GEN-02 mini-project) | [RAG](https://arxiv.org/abs/2005.11401) |
| Chunking, metadata, query rewriting, HyDE | [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) (one notebook per technique) | [applied-llms.org](https://applied-llms.org/) (tactical section on RAG) | Ablate chunk size and overlap on retrieval recall | [HyDE](https://arxiv.org/abs/2212.10496) |
| Context ordering & long context | — | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Needle-in-a-haystack test on your model | [Lost in the Middle](https://arxiv.org/abs/2307.03172) |
| Advanced RAG (self-reflective, corrective, hierarchical, graph) | [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | [RAG survey](https://arxiv.org/abs/2312.10997) | Implement CRAG-style retrieval grading | [Self-RAG](https://arxiv.org/abs/2310.11511), [CRAG](https://arxiv.org/abs/2401.15884), [RAPTOR](https://arxiv.org/abs/2401.18059), [GraphRAG](https://arxiv.org/abs/2404.16130) |
| Structured outputs & function calling | [OpenAI Cookbook](https://github.com/openai/openai-cookbook) (structured outputs, function calling examples) | [HF: Evaluating structured outputs](https://huggingface.co/blog/evaluation-structured-outputs) | Extract JSON reliably; validate it with Pydantic | — |

## Agents
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| What agents are, and when *not* to build one | [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Build the same task as a workflow and as an agent, then compare | [ReAct](https://arxiv.org/abs/2210.03629) |
| Tool use, planning, memory | [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | [GenAI_Agents notebooks](https://github.com/NirDiamant/GenAI_Agents) · [Anthropic courses](https://github.com/anthropics/courses) (tool use) | A research agent with web search + calculator tools | [Toolformer](https://arxiv.org/abs/2302.04761), [Reflexion](https://arxiv.org/abs/2303.11366) |
| Model Context Protocol (MCP) | [MCP docs](https://modelcontextprotocol.io/) | [MCP specification repo](https://github.com/modelcontextprotocol/modelcontextprotocol) | Write a small MCP server exposing one of your tools | — |
| Coding agents & agent–computer interfaces | — | [SWE-bench](https://arxiv.org/abs/2310.06770) · [SWE-agent](https://arxiv.org/abs/2405.15793) | Evaluate an agent on 10 SWE-bench Lite tasks | — |
| Harness design for agents | — | [Lilian Weng: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) | — | — |

## Evaluation
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Error analysis → evals → iterate | [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | [Hamel: Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) · [Evidently: LLM evaluation course](https://www.evidentlyai.com/llm-evaluations-course) | GEN-03 mini-project | — |
| LLM-as-judge (and its biases) | — | [HF Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) | Measure judge–human agreement (Cohen's κ) | [MT-Bench / Arena](https://arxiv.org/abs/2306.05685) |
| RAG-specific metrics | — | [RAGAS paper](https://arxiv.org/abs/2309.15217) | Faithfulness + context recall on your RAG | [RAGAS](https://arxiv.org/abs/2309.15217) |
| Benchmarks & harnesses | — | [HF Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) | Run [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on two small models | [MMLU](https://arxiv.org/abs/2009.03300), [HELM](https://arxiv.org/abs/2211.09110), [Chatbot Arena](https://arxiv.org/abs/2403.04132) |

## Security & safety for LLM apps
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Prompt injection | [Simon Willison: prompt injection series](https://simonwillison.net/series/prompt-injection/) | [Lilian Weng: Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | Red-team your own RAG app with 20 injection attempts |
| The OWASP LLM risk list | [OWASP Top 10 for LLM Apps (2025)](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) | — | Go through all 10 risks for your GEN-03 capstone |
| Reward hacking & specification gaming | — | [Lilian Weng: Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | — |
