# GEN-05: Retrieval Engineering for RAG: Search, Embeddings, Vector Indexes & Reranking

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~9 h | L2→L3 | GEN-02 |

## Why this matters
Most RAG failures are **retrieval failures**: the right passage never reaches the model. Classic information retrieval (BM25,
evaluation with recall@k and NDCG), modern embeddings, approximate nearest-neighbour search, hybrid search, and reranking are what
separate a demo from a system people trust.

## Learning goals
By the end you can:
- Build and evaluate a retriever on its own, using a labeled query set with recall@k, MRR, and NDCG.
- Explain tf-idf and BM25, and when lexical search beats embeddings.
- Choose and fine-tune an embedding model (bi-encoder), and read the MTEB leaderboard sensibly.
- Explain ANN indexes (HNSW, IVF, PQ), and tune the recall/latency/memory trade-off.
- Add hybrid search, a cross-encoder reranker, query rewriting (HyDE), and good chunking, and measure the effect of each.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) | Ch. 6 (tf-idf, vector space model), Ch. 8 (evaluation), and Ch. 11 (probabilistic IR, with BM25 in §11.4) | 2.5 h |
| 2 | **Read** | [Lilian Weng: How to Build an Open-Domain QA System](https://lilianweng.github.io/posts/2020-10-29-odqa/) | The retriever sections (classic IR, dense retrieval, retriever–reader) | 1 h |
| 3 | **Read + Build** | [Sentence Transformers docs](https://sbert.net/) | Quickstart, then the training overview. Embed your GEN-02 corpus, and try a cross-encoder reranker. | 1.5 h |
| 4 | **Read** | [HF: MTEB](https://huggingface.co/blog/mteb) + [HF: Embedding Quantization](https://huggingface.co/blog/embedding-quantization) | Both posts (choosing models; binary/int8 embeddings for cheap search) | 45 min |
| 5 | **Build** | [Faiss](https://github.com/facebookresearch/faiss) | Exact (Flat) vs HNSW vs IVF-PQ on 100k+ vectors. Plot recall@10 against latency. | 1.5 h |
| 6 | **Build** | [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | Pick 4 notebooks: chunking strategies, HyDE, hybrid (fusion) retrieval, and reranking. Apply each to your corpus and measure. | 2 h |

**Notes for the learner:** Build the **retrieval eval set first** (50+ queries with known relevant chunks). Without it, every change
in steps 5–6 is a guess.

## Check your understanding
1. What does BM25 add over tf-idf (saturation, length normalization)? When does it beat dense retrieval?
2. Bi-encoder vs cross-encoder: why use one for retrieval and the other for reranking?
3. What HNSW parameters trade recall against speed and memory?
4. How does chunk size affect retrieval recall and answer quality in opposite directions?
5. Why does HyDE help, and when could it hurt?
6. What is reciprocal rank fusion, and why is it a robust way to combine lexical and dense results?
7. *(debug)* Recall@10 is 0.9 on your eval set, but users still get wrong answers. Where do you look next?

## Mini-project
**Task:** For your GEN-02 corpus, build a retrieval leaderboard: BM25 → dense → hybrid (RRF) → hybrid + cross-encoder rerank → + HyDE.
Report recall@5/10, MRR, NDCG@10, and p95 latency for each.
**Deliverable:** A table, and a short recommendation of which stack to ship and why.

## Go deeper
- Papers: [DPR](https://arxiv.org/abs/2004.04906) · [ColBERT](https://arxiv.org/abs/2004.12832) · [SPLADE](https://arxiv.org/abs/2107.05720) · [HNSW](https://arxiv.org/abs/1603.09320) · [E5](https://arxiv.org/abs/2212.03533)
- Advanced RAG: [RAPTOR](https://arxiv.org/abs/2401.18059) · [GraphRAG](https://arxiv.org/abs/2404.16130) · [CRAG](https://arxiv.org/abs/2401.15884) · [Self-RAG](https://arxiv.org/abs/2310.11511)
- [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch): notebooks on query translation, routing, and indexing.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 08: RAG, agents & evals](../../toolbox/08-rag-agents-evals.md) (retrieval foundations, building RAG).
- **Papers:** [Retrieval & RAG](../../papers/05-retrieval-rag.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 37–39.
- **Drills:** more in [exercises/](../../exercises/README.md).
