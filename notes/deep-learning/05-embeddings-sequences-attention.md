# DL-05 notes: Embeddings, Language Modeling & Sequences

[← Lesson DL-05](../../lessons/deep-learning/05-embeddings-sequences-attention.md) · [All notes](../README.md) · [← DL-04 notes](04-cnns-computer-vision.md) · Next: [DL-06 notes →](06-transformers.md)

> **Reading time** ≈ 60 min. **You need:** [DL-01 notes](01-neural-networks-from-scratch.md) (softmax + cross-entropy), [DL-03 notes](03-training-deep-networks.md) §1 (vanishing gradients), [MATH-03 notes](../math/03-probability-statistics.md) §C (cross-entropy, perplexity).

---

## Where we are

Images come as numbers on a grid. Text comes as **discrete symbols in a sequence**. This lesson builds the four ideas that lead to transformers:

1. **Embeddings:** turn symbols into vectors where similarity means something.
2. **Language modeling:** predict the next token. It's the training objective of every LLM.
3. **RNNs:** process a sequence step by step. We'll see exactly why they forget.
4. **Attention:** let the model *look back* at any position directly.

---

## 1. Embeddings

### 1.1 An embedding is a lookup that is also a linear layer

Represent token $i$ (out of a vocabulary of $V$) as a one-hot vector $e_i$. Multiplying by a matrix $C \in \mathbb{R}^{V\times d}$ picks out row $i$: $e_i^\top C = C_i$. So an **embedding table is a linear layer applied to one-hot inputs**, implemented as an indexing operation for speed.
It's trained by backprop like any other weight. Each row receives gradient only when its token appears.

### 1.2 Word2vec: learning embeddings from context

The **distributional hypothesis**: words that appear in similar contexts have similar meanings. **Skip-gram with negative sampling** gives each word a "centre" vector $v$ and a "context" vector $u$. It trains them to tell real (centre, context) pairs from random ones:

```math
\max\ \log\sigma(u_o^\top v_c) + \sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_c),
```

where $o$ is a word that really appeared near the centre word $c$, and $n_1..n_K$ are random "negative" words. It's **logistic regression on dot products**: real pairs are pushed to align, and random pairs to point apart.
Words used in similar contexts end up with similar vectors. (The same contrastive pattern returns in CLIP, [CV-03](../vision/03-clip-vision-language-models.md), and in retrieval embeddings, [GEN-05](../llms-genai/05-retrieval-engineering.md).)

### 1.3 Analogies, and why not to over-read them

$v_\text{king} - v_\text{man} + v_\text{woman} \approx v_\text{queen}$ works because consistent relationships (gender, tense, capital-of) show up as **roughly constant offset directions**: if "man→woman" and "king→queen" share the same contexts-difference, their vectors differ by about the same amount.
**Caveats:**

- the query words themselves are excluded from the nearest-neighbour search (otherwise "king" is often the answer);
- many analogies fail;
- the embeddings faithfully absorb societal biases present in the text.

---

## 2. Language modeling

### 2.1 The objective

By the chain rule of probability, *any* distribution over sequences factorizes left to right:

```math
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{\lt t}).
```

So a **language model** only needs to predict the **next token** given the previous ones. The loss is the average negative log-likelihood of each true next token, which is cross-entropy over the vocabulary (DL-01 §2).

### 2.2 Perplexity

```math
\text{PPL} = \exp\Big(-\frac{1}{T}\sum_t \log p(x_t\mid x_{\lt t})\Big) = e^{\text{cross-entropy}} .
```

**Intuition:** "as confused as choosing uniformly among PPL options at each step". A uniform guess over $V$ tokens has perplexity exactly $V$. A perfect model has 1. A character model with PPL 4 is, on average, as uncertain as a 4-way coin.
Perplexity is only comparable **between models that use the same tokenizer**, because it's measured per token.

### 2.3 From counts to neural models

- **Bigram model:** $p(x_t\mid x_{t-1}) = \frac{\text{count}(x_{t-1}x_t) + \alpha}{\text{count}(x_{t-1}) + \alpha V}$. The $\alpha$ (add-α smoothing) avoids zero probabilities, which would give infinite loss. It's exactly MLE (counting) with a prior.
- **Neural $n$-gram (the makemore MLP, Bengio 2003):** look up the embeddings of the previous $n$ tokens, concatenate them, pass them through an MLP, and apply softmax over the vocabulary.
  Because **the same table $C$ is shared across positions**, what's learned about "a" in position 1 transfers to "a" in position 3. Similar tokens get similar vectors, so the model **generalizes to contexts it never saw**. Count tables can't do that.

### 2.4 Sampling, and its failure modes

To generate, sample $x_t \sim p(\cdot\mid x_{\lt t})$, append it, and repeat. **Temperature** $T$ rescales the logits: $p \propto e^{z/T}$. $T \to 0$ is greedy decoding (the argmax), and $T > 1$ flattens the distribution.

*Debug: "the same letter over and over".* Possible causes:

- greedy decoding or a very low temperature, which falls into a high-probability loop;
- an undertrained model whose distribution is dominated by the most frequent character;
- a bug where the generated token isn't fed back in (the context never changes);
- a mismatched context window between training and sampling.

---

## 3. Recurrent neural networks

### 3.1 The recurrence

```math
h_t = \tanh(W_h h_{t-1} + W_x x_t + b),\qquad \hat y_t = \operatorname{softmax}(W_y h_t).
```

The same weights are applied at every step, so it handles any length. $h_t$ is a running summary of the whole prefix.

### 3.2 Why vanilla RNNs forget: the product of Jacobians

**Backpropagation through time** unrolls the recurrence. The gradient of a loss at step $t$ with respect to an earlier state $h_k$ contains

```math
\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \operatorname{diag}\big(\tanh'(a_j)\big)\, W_h .
```

That's a product of $t - k$ matrices, the **same multiplicative trap as in DL-03 §1**, now along time. If the largest singular value of $W_h$ (times $\lvert\tanh'\rvert \le 1$) is below 1, the signal from 100 steps back shrinks like $\rho^{100}$, so it **vanishes**. Above 1, it **explodes**.
Exploding gradients are tamed by **gradient clipping** (rescale the gradient if its norm exceeds a threshold). Vanishing is the real problem: the model can't learn long-range dependencies.

### 3.3 LSTM and GRU: an additive memory path

The LSTM keeps a **cell state** $c_t$ that is updated **additively**, under learned gates (sigmoids between 0 and 1):

```math
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t,\qquad h_t = o_t \odot \tanh(c_t).
```

When the forget gate $f_t \approx 1$, then $\partial c_t/\partial c_{t-1} \approx I$: an identity path, **the same trick as residual connections**, so information and gradients persist over many steps. The GRU is a lighter variant with two gates.

---

## 4. Seq2seq and attention

### 4.1 The bottleneck

An encoder–decoder RNN for translation squeezes the **whole source sentence into one fixed-size vector** $h_T$, then decodes from it. Long sentences can't fit: details from early words are overwritten, and quality drops as length grows.

### 4.2 Attention: look back instead of remembering

Keep **all** the encoder states $h_1..h_S$. At each decoder step $t$, compute a relevance score between the decoder state $s_{t-1}$ and every encoder state. Normalize the scores with softmax, and take the weighted average:

```math
\alpha_{t,i} = \operatorname{softmax}_i\big(\text{score}(s_{t-1}, h_i)\big),\qquad c_t = \sum_i \alpha_{t,i}\, h_i .
```

The context $c_t$ is **recomputed for every output word**: when translating "chat", the model attends to "cat". There's no bottleneck, and the gradient reaches every source position in **one step** instead of through a long chain of Jacobians.
The attention weights are also interpretable as a soft alignment. **Transformers (DL-06) keep only this mechanism and drop the recurrence entirely.**

```python
import numpy as np, math
from collections import Counter
rng = np.random.default_rng(0)

# --- A character bigram LM with add-one smoothing; NLL and perplexity ---------------
text = ("the cat sat on the mat. the dog sat on the log. a cat and a dog met on a mat. " * 20)
chars = sorted(set(text)); V = len(chars); idx = {c: i for i, c in enumerate(chars)}
split = int(0.9 * len(text)); train, test = text[:split], text[split:]
counts = np.ones((V, V))                                   # add-one smoothing
for a, b in zip(train, train[1:]): counts[idx[a], idx[b]] += 1
P = counts / counts.sum(1, keepdims=True)
nll = -np.mean([math.log(P[idx[a], idx[b]]) for a, b in zip(test, test[1:])])
print(f"vocab {V}; test NLL {nll:.3f} nats/char; perplexity {math.exp(nll):.2f} (uniform would be {V})")

# --- Temperature sampling --------------------------------------------------------------
def sample(start="t", n=40, T=1.0, seed=0):
    r = np.random.default_rng(seed); out = start
    for _ in range(n):
        logits = np.log(P[idx[out[-1]]]) / T
        p = np.exp(logits - logits.max()); p /= p.sum()
        out += chars[r.choice(V, p=p)]
    return out
for T in [0.05, 1.0, 3.0]:
    print(f"T={T:<4}: {sample(T=T)!r}")
```

```python
# --- Why vanilla RNN gradients vanish or explode: ||d h_t / d h_0|| vs the spectral radius ---------
d, steps = 32, 60
for rho in [0.5, 0.9, 1.0, 1.5]:
    W = rng.normal(size=(d, d)); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h, J = np.zeros(d), np.eye(d)
    for t in range(steps):
        a = W @ h + rng.normal(scale=0.1, size=d)              # small inputs keep tanh near-linear
        h = np.tanh(a)
        J = np.diag(1 - h ** 2) @ W @ J
    print(f"spectral radius {rho}: ||dh_60/dh_0|| = {np.linalg.norm(J, 2):.2e}")

# --- One step of attention: scores -> softmax -> weighted average ------------------------------
H = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])   # 3 encoder states ("le", "chat", "noir")
s = np.array([0.1, 1.0])                              # decoder state that is "looking for" the 2nd one
scores = H @ s; alpha = np.exp(scores) / np.exp(scores).sum()
print("attention weights:", alpha.round(3), " context:", (alpha @ H).round(3))
```

---

## Pitfalls & misconceptions

- **Comparing perplexities across different tokenizers.**
- **Zero probabilities in count models.** They give infinite loss, so smooth.
- **"LSTMs solve long-range dependencies."** They help a lot, but they still struggle beyond a few hundred steps, and they're sequential, so they're slow to train. Attention fixes both.
- **Reading analogies as proof of semantic understanding.**
- **Forgetting gradient clipping on RNNs.** One exploding batch can wreck training.

## Cheat sheet

| Item | Formula |
|---|---|
| Embedding | one-hot × $C$ = row lookup |
| Skip-gram NS | $\log\sigma(u_o^\top v_c) + \sum_k\log\sigma(-u_{n_k}^\top v_c)$ |
| LM factorization | $\prod_t p(x_t\mid x_{\lt t})$ |
| Perplexity | $\exp(\text{mean NLL})$; uniform = $V$ |
| RNN | $h_t = \tanh(W_hh_{t-1} + W_xx_t + b)$ |
| BPTT factor | $\prod \operatorname{diag}(\tanh')W_h$, which vanishes or explodes |
| LSTM cell | $c_t = f\odot c_{t-1} + i\odot\tilde c$ |
| Attention | $c_t = \sum_i \operatorname{softmax}_i(\text{score})\,h_i$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does king − man + woman ≈ queen work, and why not over-read it?</summary>

Consistent relations are encoded as roughly constant offset directions, because the words' contexts differ in consistent ways. But the inputs are excluded from the search, many analogies fail, the results are cherry-picked, and biases are encoded too.
</details>

<details>
<summary>2. What is perplexity, and how does it relate to cross-entropy?</summary>

$\text{PPL} = e^{\text{cross-entropy}}$: the effective number of equally likely choices per token. Lower is better. Uniform over $V$ gives $V$. It's only comparable for the same tokenizer.
</details>

<details>
<summary>3. What does makemore's embedding table C learn, and why share it?</summary>

A vector per character, placed so that characters which behave similarly as context are close together. Sharing it across positions means every occurrence trains the same vector, and knowledge transfers between positions and to unseen contexts.
</details>

<details>
<summary>4. Why do vanilla RNNs struggle 100 steps back?</summary>

The gradient passes through a product of 100 Jacobians $\operatorname{diag}(\tanh')W_h$. With a spectral radius below 1 it shrinks exponentially, so the long-range signal vanishes (the demo shows a norm of about $10^{-18}$ after 60 steps at $\rho = 0.5$).
</details>

<details>
<summary>5. What does the seq2seq bottleneck lose?</summary>

Everything that doesn't fit in one fixed-size vector: fine details, especially from early or long inputs, get overwritten. Attention removes the bottleneck by letting each decoder step read all the encoder states.
</details>

<details>
<summary>6. A char-LM generates the same letter repeatedly.</summary>

Greedy or very low-temperature decoding, an undertrained model collapsing onto frequent characters, a bug in feeding the sampled token back in, or a context-length mismatch between training and sampling (§2.4).
</details>

## Where this leads

Next: [DL-06 notes](06-transformers.md). Attention solved the bottleneck, but RNNs still process tokens one at a time. The transformer's bet: **attention is all you need**.
Let every token attend to every other token in parallel, add position information, and stack the blocks deep.
