# Papers: Retrieval & RAG

[← Papers library](README.md) · Background: [GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md) · [Toolbox: RAG, agents & evals](../toolbox/08-rag-agents-evals.md)

> **Before these papers:** read the classic IR fundamentals in [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/),
> Ch. 6 (scoring, tf-idf, the vector space model), Ch. 8 (evaluation), and Ch. 11 (probabilistic IR / BM25). Most "RAG problems" are retrieval problems.

## Embeddings & dense retrieval
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) | 2019 | ⭐ Bi-encoders for sentence embeddings. | L2 | GEN-02 |
| [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) | 2020 | ⭐ Dense retrieval beats BM25 for open-domain QA. | L2 | GEN-02 |
| [ColBERT: Late Interaction](https://arxiv.org/abs/2004.12832) | 2020 | Token-level matching, somewhere between bi- and cross-encoders. | L2 | GEN-02 |
| [ColBERTv2](https://arxiv.org/abs/2112.01488) | 2021 | A compressed, practical version of late interaction. | L3 | GEN-02 |
| [SPLADE](https://arxiv.org/abs/2107.05720) | 2021 | Learned *sparse* retrieval, which works with inverted indexes. | L3 | GEN-02 |
| [Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever)](https://arxiv.org/abs/2112.09118) | 2021 | Retrievers trained without labels. | L3 | GEN-02 |
| [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533) | 2022 | How modern general-purpose embedding models are trained. | L2 | GEN-02 |
| [C-Pack (BGE embeddings)](https://arxiv.org/abs/2309.07597) | 2023 | An open embedding-model recipe and data. | L3 | GEN-02 |
| [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) | 2022 | ⭐ How to compare embedding models, and why no single model wins everywhere. | L1 | GEN-02 |
| [HNSW: Efficient and Robust Approximate Nearest Neighbor Search](https://arxiv.org/abs/1603.09320) | 2016 | ⭐ The graph index inside most vector databases. | L2 | GEN-02 |

## Retrieval-augmented generation
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [REALM: Retrieval-Augmented LM Pre-Training](https://arxiv.org/abs/2002.08909) | 2020 | Retrieval built into pretraining. | L3 | GEN-02 |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) | 2020 | ⭐ The paper that named RAG. | L2 | GEN-02 |
| [RETRO: Improving LMs by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426) | 2021 | Retrieval as a substitute for parameters. | L3 | GEN-02 |
| [Precise Zero-Shot Dense Retrieval (HyDE)](https://arxiv.org/abs/2212.10496) | 2022 | Embed a *hypothetical answer* instead of the query. | L2 | GEN-02 |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 2023 | ⭐ Models miss information buried in the middle of a long context. It affects how you order retrieved chunks. | L1 | GEN-02 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 2023 | The model decides when to retrieve, and critiques its own output. | L3 | GEN-02 |
| [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884) | 2024 | Grade what was retrieved, and fall back to web search. | L2 | GEN-02 |
| [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | 2024 | Hierarchical summaries to retrieve at multiple levels of detail. | L2 | GEN-02 |
| [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | 2024 | Knowledge-graph-based RAG for "global" questions about a corpus. | L2 | GEN-02 |
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | 2023 | Faithfulness, answer relevance, and context relevance metrics. | L1 | GEN-03 |
| [RAG for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | 2023 | ⭐ A map of naive → advanced → modular RAG. | L1 | GEN-02 |
