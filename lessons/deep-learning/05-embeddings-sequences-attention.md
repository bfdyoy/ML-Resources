# DL-05: Embeddings, Language Modeling & Sequences

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~7 h | L2 | DL-02, DL-03 |

## Why this matters
Before transformers make sense, you need three ideas: **tokens become vectors** (embeddings), **a language model
predicts the next token**, and **fixed-size context is a bottleneck** (which motivates attention). This lesson builds
those ideas with small, understandable models.

## Learning goals
By the end you can:
- Explain word embeddings (word2vec/skip-gram with negative sampling) and what vector arithmetic on them shows.
- Build a character-level language model (bigram → MLP), and evaluate it with negative log-likelihood / perplexity.
- Explain how RNNs process sequences, and why they struggle with long-range dependencies.
- Explain the encoder-decoder bottleneck, and how attention fixes it.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) (Alammar) | The whole post | 40 min |
| 2 | **Read** | [Jurafsky & Martin, *Speech and Language Processing* (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/) | The chapters on **N-gram Language Models** (for perplexity) and **Vector Semantics and Embeddings**. Read for concepts, and skim the proofs. | 2 h |
| 3 | **Build** | [Karpathy Lecture 2: makemore (bigram)](https://www.youtube.com/watch?v=PaCmpygFfXo) → [Lecture 3: makemore MLP](https://youtu.be/TCH_1BHY58I) | Code along with both | 3.5 h |
| 4 | **Read** | [D2L](https://d2l.ai/) | Chapter "Recurrent Neural Networks" (sequence models, language models, RNN basics) and the first section of "Attention Mechanisms and Transformers" (queries, keys, values) | 1 h |

**Notes for the learner:** Don't spend long on LSTM/GRU details. They matter less now. What matters is the *story*:
fixed-size memory → bottleneck → attention.

## Check your understanding
1. Why does `king − man + woman ≈ queen` work, and why shouldn't you over-read it?
2. What is perplexity, intuitively, and how does it relate to cross-entropy loss?
3. In makemore's MLP, what does the embedding table `C` learn? Why does sharing it across positions matter?
4. Why do vanilla RNNs have trouble learning dependencies 100 steps back?
5. In a seq2seq model, what information is lost by compressing the whole input into one vector?
6. *(debug)* Your character-level LM generates the same letter over and over. What sampling or training issues might cause this?

## Mini-project
**Task:** Extend makemore to a new domain (e.g. Romanian first names, city names, or Pokémon names). Tune embedding size and
context length, report dev-set NLL, and sample 20 outputs.
**Deliverable:** A notebook with a hyperparameter table and samples.

## Go deeper
- [Géron, *Hands-On ML*, Ch. 13–14](https://github.com/ageron/handson-mlp): RNNs and CNNs for sequences, and NLP with RNNs and attention.
- [Karpathy Lecture 6: makemore WaveNet](https://github.com/karpathy/nn-zero-to-hero): hierarchical/convolutional sequence models.
