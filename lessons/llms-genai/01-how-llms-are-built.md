# GEN-01: How LLMs Are Built: Tokenization, Pretraining & Post-training

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~9 h | L2→L3 | DL-06 |

## Why this matters
You've built a toy GPT. Production LLMs follow the same recipe at vastly larger scale, and then go through
**post-training** (SFT, preference tuning, RL for reasoning) that turns a text predictor into a helpful assistant.
Knowing the whole pipeline explains LLM quirks: hallucination, tokenization bugs, sycophancy, and reasoning traces.

## Learning goals
By the end you can:
- Explain BPE tokenization, and predict the failure modes it causes (spelling, arithmetic, non-English text).
- Describe pretraining: data pipelines, next-token loss, scaling laws, and compute budgets (at a conceptual level).
- Describe post-training: supervised fine-tuning, RLHF/DPO-style preference optimization, and RL with verifiable rewards for reasoning models.
- Explain inference-time behaviour: sampling (temperature, top-p), KV caching, and context-length limits.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | The whole talk. It's the mental map for this lesson. | 1 h |
| 2 | **Read + Build** | [Raschka, *Build a LLM From Scratch*](https://github.com/rasbt/LLMs-from-scratch) | Ch. 2 (text data and tokenization) and Ch. 5 (pretraining on unlabeled data). Run `ch02.ipynb` and `ch05.ipynb`. The notebooks are free and heavily commented, even without the book. | 3 h |
| 3 | **Build** | [Karpathy Lecture 8: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | Code along for the first ~hour (BPE training and encode/decode). Skim the rest. | 1.5 h |
| 4 | **Watch** | [Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | Focus on the post-training and RL sections. Watch at 1.25–1.5× speed. | 2 h |
| 5 | **Read** | [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) (Alammar) | The whole post: how reasoning models are trained | 45 min |

**Notes for the learner:** This lesson leans on video more than others, because the best up-to-date end-to-end
explanations of LLM training *are* Karpathy's talks. The Raschka notebooks give you the written, hands-on side.
For a textbook view, see Jurafsky & Martin under "Go deeper".

## Check your understanding
1. Why do LLMs struggle to count the letters in "strawberry"? Answer in terms of tokens.
2. What's the difference between a base model and an instruction-tuned model, in terms of their training data and objective?
3. What does a preference-optimization step (RLHF / DPO) optimize that SFT doesn't?
4. What is a "verifiable reward", and why did it enable reasoning models?
5. How does temperature change the next-token distribution? When would you set it to 0?
6. *(debug)* A model trained mostly on English performs badly on Romanian and uses 3× more tokens for it. Explain the connection.

## Mini-project
**Task:** Train a BPE tokenizer (your own code, or `tokenizers`) on two corpora, one English and one in another language.
Compare token counts per word, and inspect odd merges. Then load a small open model (e.g. GPT-2 via
`transformers`), and compare greedy, temperature, and top-p samples for the same prompt.
**Deliverable:** A notebook with tokenization statistics and sampling comparisons.

## Go deeper
- [Jurafsky & Martin, SLP3](https://web.stanford.edu/~jurafsky/slp3/): the chapters on large language models, transformers, and post-training/alignment.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) ([lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)): the advanced, full-stack version of this lesson.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
