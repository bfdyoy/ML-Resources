# GEN-07: Post-Training: SFT, RLHF, DPO & Reasoning Models

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~11 h | L3 | GEN-01, GEN-02 · helpful: EL-03 (policy gradients) |

## Why this matters
Post-training is what turns a base model into a helpful assistant, and more recently into a *reasoning* model. It's also where much
of today's frontier progress happens. Understanding SFT, reward models, PPO/GRPO, and direct preference methods lets you choose the
right tool for your own fine-tuning, and read model reports critically.

## Learning goals
By the end you can:
- Describe the canonical pipeline: SFT → reward model → RL (PPO), and what each stage fixes.
- Derive the DPO objective's intuition, and compare DPO/KTO/ORPO with RL-based methods.
- Explain RL with verifiable rewards (RLVR) and GRPO, and how they produce long chain-of-thought reasoning.
- Explain reward hacking, KL regularization, and why post-training can reduce diversity.
- Run SFT → DPO (and optionally GRPO) on a small model with TRL, and evaluate the result.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Post-Training: SFT, RLHF, DPO & Reasoning](../../notes/llms-genai/07-post-training-alignment-reasoning.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [HF: Illustrating RLHF](https://huggingface.co/blog/rlhf) | The whole post (the 3-stage picture) | 45 min |
| 2 | **Read** | [RLHF Book](https://rlhfbook.com/) (Lambert) | The chapters on instruction tuning, reward modeling, policy-gradient RL, and direct alignment algorithms | 3 h |
| 3 | **Read** | [HF: Fine-tune with DPO (TRL)](https://huggingface.co/blog/dpo-trl) + [HF: RLOO](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo) | Both posts (a practical DPO recipe; why simpler RL works) | 1 h |
| 4 | **Read** | [Raschka: Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) + [Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) | Both posts | 1.5 h |
| 5 | **Build** | [TRL](https://github.com/huggingface/trl) + [HF smol-course](https://github.com/huggingface/smol-course) | SFT, then DPO, on a ≤1B model. Compare win-rates with an LLM judge (GEN-03). | 2.5 h |
| 6 | *Build (stretch)* | [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch) or [HF: Open-R1](https://huggingface.co/blog/open-r1) | GRPO with a verifiable reward (exact-match math answers) on a small model | 2 h |

## Check your understanding
1. What does SFT teach that pretraining doesn't? What can't SFT fix on its own?
2. How is a reward model trained from pairwise preferences (Bradley–Terry)?
3. Why does PPO-based RLHF include a KL penalty to the reference model?
4. Explain intuitively why DPO needs no explicit reward model or sampling loop.
5. What makes a reward "verifiable", and why did RLVR unlock long reasoning chains?
6. How does GRPO estimate advantages without a value network?
7. *(debug)* After DPO, your model's answers got much longer and the judge prefers them, but humans don't. What's happening?

## Mini-project
**Task:** Take a small instruct model, build 300–1,000 preference pairs for a narrow behaviour (e.g. concise, well-cited answers), and run DPO.
Evaluate the base, SFT, and DPO versions on a held-out set with both an LLM judge and 30 human (your own) ratings.
**Deliverable:** A training log and a win-rate table, plus notes on any reward/length hacking you observed.

## Go deeper
- Papers: [InstructGPT](https://arxiv.org/abs/2203.02155) · [DPO](https://arxiv.org/abs/2305.18290) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [Tülu 3](https://arxiv.org/abs/2411.15124) · [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [Lilian Weng: Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) (test-time compute) · [Lilian Weng: Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [HF LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) Ch. 10–12 (fine-tuning, dataset curation, building reasoning models).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 07: NLP & LLMs](../../toolbox/07-nlp-and-llms.md) (using & adapting LLMs) · [Toolbox 10: Reinforcement learning](../../toolbox/10-reinforcement-learning.md) (RL ↔ LLMs).
- **Papers:** [Post-training, reasoning & agents](../../papers/04-alignment-reasoning-agents.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 34–35 (loss masking, DPO loss).
- **Drills:** [ARENA](https://github.com/callummcdougall/ARENA_3.0) RLHF exercises · more in [exercises/](../../exercises/README.md).
