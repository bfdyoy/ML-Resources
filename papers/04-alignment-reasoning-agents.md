# Papers: Post-training, Reasoning, Agents & LLM Evaluation

[← Papers library](README.md) · Background: [GEN-01](../lessons/llms-genai/01-how-llms-are-built.md), [GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md), [GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md) · [Toolbox: NLP & LLMs](../toolbox/07-nlp-and-llms.md), [Toolbox: RAG, agents & evals](../toolbox/08-rag-agents-evals.md)

## Instruction tuning & RLHF
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | The origin of learning a reward model from human comparisons. | L2 | EL-03 |
| [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) | 2020 | The full RLHF pipeline applied to summarization. Very clear. | L2 | GEN-01 |
| [Training LMs to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | ⭐ The SFT → reward model → PPO recipe. | L2 | GEN-01 |
| [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) | 2022 | Anthropic's HH-RLHF: the helpfulness/harmlessness trade-off, and online iteration. | L2 | GEN-01 |
| [Scaling Instruction-Finetuned Language Models (Flan)](https://arxiv.org/abs/2210.11416) | 2022 | Instruction tuning on over 1,800 tasks, with chain-of-thought data. | L2 | GEN-02 |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | ⭐ AI feedback guided by written principles. | L2 | GEN-01 |
| [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) | 2023 | Can AI feedback replace human feedback? | L2 | GEN-01 |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | ⭐ Preference learning without a reward model or RL loop. | L2 | GEN-02 |
| [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | 2024 | Alignment from thumbs-up/down signals instead of pairs. | L3 | GEN-02 |
| [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) | 2024 | SFT and preference optimization in one step. | L3 | GEN-02 |
| [Tülu 3: Pushing Frontiers in Open LM Post-Training](https://arxiv.org/abs/2411.15124) | 2024 | ⭐ A fully open, modern post-training recipe (SFT + DPO + RLVR). | L2 | GEN-02 |

## Reasoning
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | ⭐ | L1 | GEN-01 |
| [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | 2022 | "Let's think step by step." | L1 | GEN-01 |
| [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) | 2022 | Sample many reasoning paths, then take a majority vote. | L1 | GEN-01 |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | 2023 | Search over reasoning steps. | L2 | GEN-01 |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) | 2022 | A model learning from its own correct rationales. | L2 | GEN-01 |
| [Training Verifiers to Solve Math Word Problems (GSM8K)](https://arxiv.org/abs/2110.14168) | 2021 | Verifiers, and the GSM8K benchmark. | L2 | GEN-01 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | 2023 | Process vs outcome reward models. | L2 | GEN-01 |
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | 2024 | When "thinking longer" beats a bigger model. | L3 | GEN-01 |
| [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) | 2024 | ⭐ Group Relative Policy Optimization, the RL algorithm behind many reasoning models. | L3 | EL-03 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | ⭐ Reasoning that emerges from RL with verifiable rewards. Read it with [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1). | L2 | GEN-01 |

## Prompting & agents
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608) | 2024 | A taxonomy of 50+ prompting techniques. Use it as a reference. | L1 | GEN-02 |
| [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) | 2022 | ⭐ The thought → action → observation loop. | L1 | GEN-02 |
| [Toolformer](https://arxiv.org/abs/2302.04761) | 2023 | LMs teaching themselves to call tools. | L2 | GEN-02 |
| [Reflexion](https://arxiv.org/abs/2303.11366) | 2023 | Agents that learn from verbal self-feedback. | L2 | GEN-02 |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | 2023 | Memory, reflection, and planning in simulated characters. | L1 | GEN-02 |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 2023 | ⭐ Real GitHub issues as the benchmark for coding agents. | L2 | GEN-03 |
| [SWE-agent: Agent-Computer Interfaces](https://arxiv.org/abs/2405.15793) | 2024 | Tool and interface design matters as much as the model. | L2 | GEN-03 |

## Evaluating LLMs
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) | 2020 | The classic knowledge benchmark. | L1 | GEN-03 |
| [Evaluating LLMs Trained on Code (Codex / HumanEval)](https://arxiv.org/abs/2107.03374) | 2021 | pass@k and code evaluation. | L2 | GEN-03 |
| [Beyond the Imitation Game (BIG-bench)](https://arxiv.org/abs/2206.04615) | 2022 | 200+ diverse tasks. | L1 | GEN-03 |
| [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) | 2022 | Evaluation across many metrics and scenarios. | L2 | GEN-03 |
| [Judging LLM-as-a-Judge (MT-Bench, Chatbot Arena)](https://arxiv.org/abs/2306.05685) | 2023 | ⭐ The biases of LLM judges (position, verbosity, self-preference). | L1 | GEN-03 |
| [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) | 2024 | Elo/Bradley-Terry ranking from crowdsourced comparisons. | L2 | GEN-03 |
