# GEN-09: LLM Security & Safety: Prompt Injection, Red-Teaming & Guardrails

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~6 h | L2 | GEN-03, GEN-06 |

## Why this matters
As soon as an LLM reads untrusted text (web pages, emails, retrieved documents) and can take actions, **prompt injection** becomes a
real attack surface. LLM apps also leak data, get jailbroken, and optimize for the wrong thing. Shipping responsibly means knowing the
threat model, testing against it, and designing systems that stay safe even when the model is fooled.

## Learning goals
By the end you can:
- Explain direct vs indirect prompt injection, and why it can't be fully solved by prompting alone.
- Walk through the OWASP Top 10 for LLM applications, and map each risk to your own system.
- Red-team an app systematically (attack taxonomy, automated attacks, a regression suite).
- Design defences in depth: least-privilege tools, human confirmation, output handling, data-flow isolation, and monitoring.
- Explain reward hacking and specification gaming as the training-time version of the same problem.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: LLM Security & Safety](../../notes/llms-genai/09-llm-security-safety.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Simon Willison: Prompt injection series](https://simonwillison.net/series/prompt-injection/) | Start with the earliest posts that define the attack, then the posts on indirect injection and on design patterns for mitigation | 1.5 h |
| 2 | **Read** | [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) | All 10 risks, with their example scenarios and mitigations | 1.5 h |
| 3 | **Read** | [Lilian Weng: Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | Threat model, attack types (token manipulation, jailbreak prompting, automated red-teaming) | 1 h |
| 4 | **Read** | [Lilian Weng: Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | The sections on reward hacking in RLHF and LLM tasks | 45 min |
| 5 | **Build** | Your GEN-06 agent | Write 20 attacks (direct, indirect via a poisoned document, data exfiltration via tool calls). Add mitigations and re-run. | 1.5 h |

## Check your understanding
1. Why is indirect prompt injection more dangerous than direct injection?
2. Why is "just add 'ignore malicious instructions' to the system prompt" not a real defence?
3. Which OWASP risks is a RAG system with public document upload most exposed to?
4. What makes an agent with email access plus web browsing especially risky? What's the minimal-privilege redesign?
5. How is reward hacking during RLHF related to Goodhart's law?
6. *(debug)* Your red-team suite passes 100%, but a user still extracts the system prompt. What was missing from the suite?

## Mini-project
**Task:** Write a threat model for your GEN-06 capstone (assets, entry points, attacker goals), then build an automated red-team regression suite
(≥30 cases) that runs in CI next to your GEN-03 evals.
**Deliverable:** A threat-model doc, the suite, and a before/after pass-rate table.

## Go deeper
- [Constitutional AI](https://arxiv.org/abs/2212.08073) and [Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862): safety via training.
- [ARENA](https://github.com/callummcdougall/ARENA_3.0): the LLM evals chapter (dangerous-capability and alignment evaluations).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 08: RAG, agents & evals](../../toolbox/08-rag-agents-evals.md) (security & safety) · [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (AI safety & security).
- **Papers:** [Post-training, reasoning & agents](../../papers/04-alignment-reasoning-agents.md) (instruction tuning & RLHF section).
- **Implement it yourself:** a tool-call "policy layer" that blocks any tool action touching data the current request didn't originate. Unit-test it with your attacks.
- **Drills:** more in [exercises/](../../exercises/README.md).
