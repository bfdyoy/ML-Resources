# Papers: NLP, Transformers & LLMs

[← Papers library](README.md) · Background: [DL-05](../lessons/deep-learning/05-embeddings-sequences-attention.md), [DL-06](../lessons/deep-learning/06-transformers.md), [GEN-01](../lessons/llms-genai/01-how-llms-are-built.md) · [Toolbox: NLP & LLMs](../toolbox/07-nlp-and-llms.md)

## Word representations & tokenization
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Efficient Estimation of Word Representations (word2vec)](https://arxiv.org/abs/1301.3781) | 2013 | ⭐ Skip-gram and CBOW. | L1 | DL-05 |
| [Enriching Word Vectors with Subword Information (fastText)](https://arxiv.org/abs/1607.04606) | 2016 | Subword n-grams handle rare words. | L2 | DL-05 |
| [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) | 2015 | ⭐ Byte-pair encoding, the basis of GPT tokenizers. | L1 | GEN-01 |
| [SentencePiece](https://arxiv.org/abs/1808.06226) | 2018 | Language-independent tokenization, used by Llama, T5, and others. | L2 | GEN-01 |

## Sequence models → attention → transformers
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | 2014 | Encoder-decoder LSTMs. | L1 | DL-05 |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 2014 | ⭐ The birth of attention. | L2 | DL-05 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | ⭐ The Transformer. Read it alongside [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). | L2 | DL-06 |
| [Deep contextualized word representations (ELMo)](https://arxiv.org/abs/1802.05365) | 2018 | Contextual embeddings. | L2 | DL-05 |
| [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146) | 2018 | The transfer-learning recipe for NLP. Very readable. | L1 | DL-05 |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | ⭐ Masked-LM pretraining for encoders. | L2 | DL-06 |
| [Exploring the Limits of Transfer Learning (T5)](https://arxiv.org/abs/1910.10683) | 2019 | Text-to-text framing and a huge ablation study. Skim §3. | L2 | DL-06 |
| [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) | 2020 | A map of the "X-former" zoo. | L2 | DL-06 |

## Scaling & large language models
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | ⭐ In-context learning. Read §1–3. | L2 | GEN-01 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | ⭐ Power laws in data, parameters, and compute. | L2 | GEN-01 |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | ⭐ About 20 tokens per parameter. It changed how everyone trains. | L2 | GEN-01 |
| [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | 2022 | Abilities that seem to appear suddenly with scale… | L1 | GEN-01 |
| [Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004) | 2023 | …or do they? Metric choice matters. Read the two back to back. | L2 | GEN-01 |
| [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) | 2023 | A long, regularly updated survey. Use it as a reference. | L2 | GEN-01 |
| [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023 | Predictable scaling and evaluation practice. | L2 | GEN-01 |
| [Llama 2](https://arxiv.org/abs/2307.09288) | 2023 | A detailed open recipe: pretraining, SFT, RLHF, safety. | L2 | GEN-01 |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 2024 | ⭐ The most complete public account of training a frontier-scale model. | L2 | GEN-01 |
| [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) | 2024 | Fully open: data, code, and checkpoints. Good for learning. | L2 | GEN-01 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | MoE, multi-head latent attention, and FP8 training at scale. | L3 | GEN-01 |

## Modern architecture components
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) | 2021 | ⭐ Rotary position embeddings, now standard. | L2 | DL-06 |
| [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) | 2023 | Extending RoPE to longer contexts. | L3 | DL-06 |
| [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245) | 2023 | Smaller KV cache and faster inference. | L2 | DL-06 |
| [Efficient Streaming LMs with Attention Sinks](https://arxiv.org/abs/2309.17453) | 2023 | Why the first tokens soak up attention. | L3 | DL-06 |
| [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) | 2017 | The original mixture-of-experts layer. | L2 | DL-06 |
| [GShard](https://arxiv.org/abs/2006.16668) | 2020 | Scaling MoE with automatic sharding. | L3 | DL-06 |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | 2021 | ⭐ Simplified top-1 MoE routing. | L2 | DL-06 |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | An open sparse MoE LLM. A short, readable report. | L2 | DL-06 |
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 2024 | Fine-grained plus shared experts. | L3 | DL-06 |
| [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) | 2021 | State-space models. Math-heavy. | L3 | DL-05 |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | ⭐ Selective SSMs, the main alternative to attention. | L3 | DL-06 |
| [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) | 2024 | Connects SSMs and attention. | L3 | DL-06 |

## Pretraining data
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [The Pile](https://arxiv.org/abs/2101.00027) | 2020 | How a large open pretraining corpus is built. | L1 | GEN-01 |
| [FineWeb](https://arxiv.org/abs/2406.17557) | 2024 | ⭐ Web-data filtering ablations. Shows that data quality beats data quantity. | L2 | GEN-01 |
| [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644) | 2023 | Small models trained on high-quality synthetic data. | L2 | GEN-01 |
