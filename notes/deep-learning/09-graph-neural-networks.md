# DL-09 notes: Graph Neural Networks

[← Lesson DL-09](../../lessons/deep-learning/09-graph-neural-networks.md) · [All notes](../README.md) · [← DL-08 notes](08-modern-architectures-moe-ssm.md) · Next: [GEN-01 notes →](../llms-genai/01-how-llms-are-built.md) (or [CV-01 notes](../vision/01-object-detection-segmentation.md))

> **Reading time** ≈ 55 min. **You need:** [DL-06 notes](06-transformers.md) §1 (attention, permutation equivariance) and [MATH-01 notes](../math/01-linear-algebra.md) §C (eigenvectors).

---

## Where we are

Images are grids (CNNs) and text is a sequence (transformers). Molecules, social networks, road maps, citation networks, and knowledge graphs are **graphs**: nodes with an arbitrary number of neighbours, and no natural order.
GNNs extend the same principle of *share weights, respect the symmetry* to this setting. They also explain transformers from a new angle.

---

## 1. Graph data and tasks

A graph with $n$ nodes has an **adjacency matrix** $A$ ($A_{uv} = 1$ if there's an edge), a **degree matrix** $D = \operatorname{diag}(\sum_v A_{uv})$, and node features $X \in \mathbb{R}^{n\times d}$. It may also have edge features.

| Level | Example | Output |
|---|---|---|
| Node | Classify papers in a citation graph | One label per node |
| Edge | Predict friendships, drug–target interactions, recommendations | A score per node pair |
| Graph | Predict a molecule's toxicity | One label per graph (pool the node embeddings) |

**The symmetry to respect:** renumbering the nodes doesn't change the graph. So a node-level GNN must be **permutation-equivariant**: relabel the nodes, and the outputs are relabelled the same way. A graph-level GNN must be **invariant**.

---

## 2. Message passing: the one framework

Each layer updates every node from its neighbours:

```math
h_v^{(k)} = \operatorname{UPDATE}\Big(h_v^{(k-1)},\ \operatorname{AGG}\big(\lbrace \operatorname{MSG}(h_u^{(k-1)}, h_v^{(k-1)}, e_{uv}) : u \in \mathcal{N}(v)\rbrace\big)\Big).
```

- **AGG must be permutation-invariant** (sum, mean, max, or attention-weighted sum), because neighbours have no order. That's exactly what makes the whole layer equivariant.
- **The weights are shared across all nodes**, just as conv kernels are shared across pixels.
- **$K$ layers give each node a $K$-hop receptive field**: after 2 layers, a node "knows" about its neighbours' neighbours.

### 2.1 GCN (Kipf & Welling)

Add self-loops, $\tilde A = A + I$ (so a node keeps its own features), and normalize symmetrically:

```math
H^{(k)} = \sigma\big(\tilde D^{-1/2}\tilde A\,\tilde D^{-1/2} H^{(k-1)} W^{(k)}\big).
```

Per node, this is $h_v' = \sigma\big(\sum_{u \in \mathcal N(v)\cup\lbrace v\rbrace} \frac{1}{\sqrt{\tilde d_u \tilde d_v}} W h_u\big)$: a **degree-normalized average** of the transformed neighbours.
Without normalization, high-degree nodes would get huge sums. The symmetric $1/\sqrt{d_ud_v}$ form also keeps the operator's eigenvalues in $[-1, 1]$, which makes it stable to stack.

### 2.2 GraphSAGE

```math
h_v' = \sigma\Big(W\big[\,h_v \,\Vert\, \operatorname{mean}_{u\in\mathcal N(v)} h_u\,\big]\Big).
```

It keeps the node's own representation **separate** from the neighbourhood summary (concatenation instead of mixing). It **samples** a fixed number of neighbours per node, so it scales to huge graphs and new nodes (it's **inductive**).

### 2.3 GAT (graph attention)

Learn *how much* each neighbour matters:

```math
\alpha_{vu} = \operatorname{softmax}_{u\in\mathcal N(v)}\Big(\operatorname{LeakyReLU}\big(a^\top[\,Wh_v \Vert Wh_u\,]\big)\Big),\qquad h_v' = \sigma\Big(\sum_{u} \alpha_{vu} W h_u\Big).
```

This is attention restricted to the edges, usually with several heads. Compare it with the transformer's $\operatorname{softmax}(q^\top k)$: GAT is attention with the **graph as the mask**.

---

## 3. Over-smoothing: why GNNs stay shallow

A GCN layer, stripped of weights and non-linearity, is a multiplication by the normalized operator $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$. Stacking $k$ layers applies $\hat A^k$.
As $k$ grows, $\hat A^k$ converges to a projection onto its **top eigenvector**, which, after rescaling each node by $\sqrt{\tilde d_v}$, is constant on each connected component (MATH-01 §C; it's like power iteration).
So **all node representations converge to the same vector**, up to degree scaling. Classes become indistinguishable, and accuracy collapses (the lesson's debug question: best at 2 layers, collapsed at 8).

**Remedies:**

- fewer layers, with a wider MLP before or after;
- residual or skip connections, or "jumping knowledge" (concatenating all layers' outputs);
- normalization schemes (PairNorm);
- DropEdge;
- getting long-range information some other way: virtual nodes, graph transformers.

---

## 4. Expressivity: what message passing cannot see

**The Weisfeiler–Lehman (1-WL) test** repeatedly recolours each node by hashing (its colour, the multiset of its neighbours' colours). If two graphs end up with different colour histograms, they're definitely not isomorphic.
If the histograms match, the test can't tell them apart.

**Message-passing GNNs are at most as powerful as 1-WL** (Xu et al., GIN). A message-passing layer *is* a learned, soft version of the WL update. Two classic failures:

- **Two disjoint triangles vs one 6-cycle.** Every node has degree 2, and every neighbour also has degree 2, so 1-WL and every MPNN give identical outputs. Yet one graph contains triangles and the other doesn't. (That matters for chemistry: rings.)
- **The aggregator matters.** *Mean* can't tell the neighbour multiset {a, b} from {a, a, b, b}. *Max* can't tell {a, b} from {a, a, b}. **Sum** keeps the multiset information.
  So **GIN** uses sum aggregation followed by an MLP, $h_v' = \operatorname{MLP}\big((1+\epsilon)h_v + \sum_{u} h_u\big)$, and reaches the 1-WL ceiling.

More expressive variants add structural features (cycle counts), positional encodings (Laplacian eigenvectors, random-walk features), or higher-order message passing.

---

## 5. Transformers are GNNs on a complete graph

Self-attention computes $h_i' = \sum_j \alpha_{ij} W_V h_j$, with $\alpha_{ij} = \operatorname{softmax}_j(q_i^\top k_j)$: message passing where **every token is connected to every other**, and the **adjacency is replaced by learned, input-dependent attention weights**.
The causal mask is a directed graph (each token connects only to earlier tokens). Positional encodings play the role of graph structure, which is why **graph transformers** use Laplacian eigenvectors as "positions". The two families are one idea, with different choices of graph.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# --- A two-community graph (stochastic block model) with weak node features -----------------
n, half = 200, 100
labels = torch.cat([torch.zeros(half), torch.ones(half)]).long()
same = labels[:, None] == labels[None, :]
A = (torch.rand(n, n) < torch.where(same, torch.tensor(0.08), torch.tensor(0.01))).float()
A = torch.triu(A, 1); A = A + A.T                                 # undirected, no self-loops
X = torch.randn(n, 16) + 0.3 * labels[:, None].float()            # features: only weakly informative

def gcn_norm(A):
    At = A + torch.eye(len(A)); d = At.sum(1)
    return At / torch.sqrt(d[:, None] * d[None, :])
A_hat = gcn_norm(A)

class GCN(nn.Module):
    def __init__(self, d_in, d_h, n_cls, layers=2):
        super().__init__()
        dims = [d_in] + [d_h] * (layers - 1) + [n_cls]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
    def forward(self, X, A_hat):
        h = X
        for i, lin in enumerate(self.lins):
            h = A_hat @ lin(h)                      # transform, then aggregate over neighbours
            if i < len(self.lins) - 1: h = F.relu(h)
        return h

train = torch.zeros(n, dtype=torch.bool); train[torch.randperm(n)[:20]] = True    # only 20 labeled nodes
def fit(model, A_used):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        opt.zero_grad(); F.cross_entropy(model(X, A_used)[train], labels[train]).backward(); opt.step()
    return (model(X, A_used).argmax(1)[~train] == labels[~train]).float().mean().item()
print(f"MLP on features only (no graph): {fit(GCN(16, 32, 2), torch.eye(n)):.2f}")
for L in [2, 8, 16]:
    print(f"GCN with {L:2d} layers: test accuracy {fit(GCN(16, 32, 2, layers=L), A_hat):.2f}")
```

```python
# --- Over-smoothing: repeated averaging makes node features identical (after degree scaling) -------------
H = X.clone()
dsqrt = torch.sqrt((A + torch.eye(n)).sum(1, keepdim=True))
for k in [0, 2, 8, 32, 128]:
    Hk = torch.linalg.matrix_power(A_hat, k) @ H if k else H
    Z = Hk / dsqrt; Z = Z / Z.norm(dim=1, keepdim=True)
    print(f"k={k:3d}: mean pairwise cosine distance between nodes = {abs((1 - Z @ Z.T).mean()):.4f}")

# --- 1-WL / sum-aggregation cannot tell two triangles from a 6-cycle ----------------------------------------
def cycle(m, offset=0): return [(offset + i, offset + (i + 1) % m) for i in range(m)]
def adj(edges, n):
    M = torch.zeros(n, n)
    for u, v in edges: M[u, v] = M[v, u] = 1
    return M
two_triangles = adj(cycle(3) + cycle(3, 3), 6); hexagon = adj(cycle(6), 6)
W1, W2 = torch.randn(1, 8), torch.randn(8, 8)
def gin_readout(M):
    h = torch.ones(6, 1)                                # identical initial features
    h = torch.relu((h + M @ h) @ W1); h = torch.relu((h + M @ h) @ W2)
    return h.sum(0)
print("GIN-style readouts equal for 2 triangles vs hexagon:", torch.allclose(gin_readout(two_triangles), gin_readout(hexagon)))
print("...but they differ structurally: triangles (trace(A^3)/6) =", int(torch.trace(two_triangles @ two_triangles @ two_triangles) / 6),
      "vs", int(torch.trace(hexagon @ hexagon @ hexagon) / 6))

# Mean vs sum aggregation on neighbour multisets {a, b} and {a, a, b, b}
a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
m1, m2 = torch.stack([a, b]), torch.stack([a, a, b, b])
print("mean equal:", torch.equal(m1.mean(0), m2.mean(0)), "| sum equal:", torch.equal(m1.sum(0), m2.sum(0)))
```

---

## Pitfalls & misconceptions

- **Stacking 10+ GCN layers "because deeper is better".** Over-smoothing.
- **Leaking test labels through the graph.** In transductive node classification, the test nodes are present in the graph, which is fine for features, but their labels must never enter training. Also split *edges* carefully for link prediction: the edges you evaluate on must be removed from the message-passing graph.
- **Mean aggregation when counts matter.** Use sum (GIN) for structure-sensitive tasks.
- **Treating attention or edge weights as explanations** without validation.

## Cheat sheet

| Item | Formula |
|---|---|
| Message passing | $h_v' = \text{UPD}(h_v, \text{AGG}_{u\in\mathcal N(v)}\text{MSG}(h_u))$ |
| GCN | $\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$ |
| GraphSAGE | $\sigma(W[h_v \Vert \operatorname{mean}_u h_u])$ + neighbour sampling |
| GAT | $\alpha_{vu} = \operatorname{softmax}_u(\text{LeakyReLU}(a^\top[Wh_v\Vert Wh_u]))$ |
| GIN | $\text{MLP}((1+\epsilon)h_v + \sum_u h_u)$, reaching the 1-WL ceiling |
| Receptive field | $K$ layers = $K$ hops |
| Over-smoothing | $\hat A^k$ → projection onto the top eigenvector |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does one message-passing layer compute; what do K layers give?</summary>

Each node aggregates (permutation-invariantly) messages from its neighbours and combines them with its own state through shared weights. $K$ layers give each node information from its $K$-hop neighbourhood.
</details>

<details>
<summary>2. GCN vs GraphSAGE vs GAT aggregation.</summary>

GCN: a fixed, symmetric degree-normalized average including self-loops. GraphSAGE: concatenates self with a (sampled) neighbour mean, max, or LSTM aggregate, and is inductive. GAT: learned, input-dependent attention weights over neighbours.
</details>

<details>
<summary>3. What is over-smoothing, and why does it limit depth?</summary>

Repeated normalized averaging drives node representations toward the same vector (the top eigenvector of $\hat A$), so they lose discriminative information as layers are added. The demo shows the cosine distance collapsing and accuracy falling with depth.
</details>

<details>
<summary>4. Why can't message-passing GNNs distinguish some graphs? The WL link.</summary>

An MPNN's update is a (soft) version of 1-WL colour refinement, so it can't separate graphs that 1-WL can't. Example: two triangles vs a 6-cycle (all nodes look identical locally). Mean and max aggregators are even weaker than sum.
</details>

<details>
<summary>5. How is a transformer a GNN on a complete graph?</summary>

Every token aggregates from every other token, with learned attention weights $\operatorname{softmax}(q^\top k)$ replacing the fixed adjacency. Masks define the graph (a causal mask is a DAG), and positional encodings supply the structure.
</details>

<details>
<summary>6. Validation peaks at 2 layers and collapses at 8.</summary>

Over-smoothing (and harder optimization). Use 2–3 layers with residual or jumping-knowledge connections, PairNorm, or DropEdge. For long-range needs, add virtual nodes or use a graph transformer instead of stacking more layers.
</details>

## Where this leads

**Path 2 is complete.** You can now choose:

- [GEN-01 notes](../llms-genai/01-how-llms-are-built.md): scale the DL-06 transformer up to an LLM, and learn how LLMs are actually built.
- [CV-01 notes](../vision/01-object-detection-segmentation.md): take the DL-04 CNNs from classification to detection and segmentation.
