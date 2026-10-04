# DL-09: Graph Neural Networks

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~8 h | L2→L3 | DL-03, DL-06 · math: [MATH-01](../math/01-linear-algebra.md) blocks A–B |

## Why this matters
Molecules, social networks, road maps, knowledge graphs, and user–item interactions are all graphs. GNNs learn from that structure
directly, with message passing. They power drug discovery, fraud detection, traffic forecasting, and many recommenders. And
attention is itself message passing on a fully connected graph.

## Learning goals
By the end you can:
- Represent graph data (adjacency, node/edge/graph features), and name the three task levels: node, edge, and graph.
- Explain message passing, and derive GCN, GraphSAGE, and GAT as special cases.
- Explain over-smoothing, and the expressivity limits of GNNs (the WL test).
- Train node classification and link prediction models with PyTorch Geometric.
- Describe how transformers relate to GNNs.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Graph Neural Networks](../../notes/deep-learning/09-graph-neural-networks.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) | The whole interactive article | 1 h |
| 2 | **Read** | [Distill: Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) | The whole article (from spectral to modern GNNs) | 1 h |
| 3 | **Read** | [Hamilton: Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) | Ch. 5 (the GNN model: message passing, GCN, GraphSAGE, attention) and Ch. 7 (theoretical motivations, the WL test) | 2 h |
| 4 | **Build** | [UvA DL Tutorial: GNNs](https://uvadlc-notebooks.readthedocs.io/) | The GNN tutorial notebook (GCN/GAT layers from scratch, then PyG) | 1.5 h |
| 5 | **Build** | [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | Node classification on Cora with GCN vs GAT; then link prediction | 1 h |

## Check your understanding
1. What does one message-passing layer compute for a node? What do K layers give it access to?
2. How do GCN, GraphSAGE, and GAT differ in how they aggregate neighbours?
3. What is over-smoothing, and why does it limit GNN depth?
4. Why can't a standard message-passing GNN distinguish some non-isomorphic graphs? What's the connection to the WL test?
5. How is a transformer a GNN on a complete graph? What replaces the adjacency matrix?
6. *(debug)* Your GCN's validation accuracy peaks at 2 layers and collapses at 8. What's happening, and what can you try?

## Mini-project
**Task:** Node classification on Cora or CiteSeer: compare an MLP (no graph), a GCN, and a GAT. Then do link prediction on the same graph,
evaluated with ROC-AUC on held-out edges.
**Deliverable:** A results table and a t-SNE/UMAP plot of the learned node embeddings, coloured by class.

## Go deeper
- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) + [course notes](https://snap-stanford.github.io/cs224w-notes/): the full course.
- [HF: Introduction to Graph Machine Learning](https://huggingface.co/blog/intro-graphml).
- [Geometric Deep Learning](https://arxiv.org/abs/2104.13478): the unifying theory of CNNs, GNNs, and transformers through symmetry.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 05: Architectures](../../toolbox/05-architectures.md) (graph neural networks).
- **Papers:** [GCN](https://arxiv.org/abs/1609.02907) · [GraphSAGE](https://arxiv.org/abs/1706.02216) · [GAT](https://arxiv.org/abs/1710.10903) · [GIN](https://arxiv.org/abs/1810.00826) · [MPNN](https://arxiv.org/abs/1704.01212).
- **Implement it yourself:** a GCN layer in pure PyTorch with a sparse adjacency matrix. Check it against PyG's `GCNConv`.
- **Drills:** more in [exercises/](../../exercises/README.md).
