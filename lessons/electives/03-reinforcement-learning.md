# EL-03: Reinforcement Learning

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~13 h | L2→L3 | DL-03 · math: [MATH-03](../math/03-probability-statistics.md) (expectations) |

## Why this matters
RL is how agents learn from interaction, and it's now central to LLM post-training (RLHF, and RL for reasoning).
The core ideas (value functions, policies, exploration, credit assignment) are distinct enough from supervised
learning to deserve their own careful study.

## Learning goals
By the end you can:
- Formalize a problem as an MDP (states, actions, rewards, discount), and explain the Bellman equations.
- Explain and implement tabular methods: Monte Carlo, TD(0), SARSA, Q-learning.
- Explain the exploration/exploitation trade-off (ε-greedy, and bandits).
- Explain deep RL basics: DQN, and policy gradients (REINFORCE → actor-critic → PPO) at a conceptual level.
- Connect policy gradients to how RLHF-style fine-tuning works.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Reinforcement Learning](../../notes/electives/03-reinforcement-learning.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) | Ch. 1 (introduction), Ch. 2 (multi-armed bandits), Ch. 3 (finite MDPs), Ch. 6 (temporal-difference learning). Skim Ch. 4–5. | 5 h |
| 2 | **Build** | [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | Units 1–3 (intro, Q-learning, deep Q-learning), with the hands-on notebooks | 4 h |
| 3 | **Read** | [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html) | "Introduction to RL" parts 1–3 (key concepts, kinds of RL algorithms, intro to policy optimization) | 2 h |
| 4 | **Read** | [UDL](https://udlbook.github.io/udlbook/) | Ch. 19 "Reinforcement Learning": a compact modern summary that connects everything | 1 h |

## Check your understanding
1. What's the difference between the state-value and action-value functions?
2. Why is Q-learning "off-policy" and SARSA "on-policy"? When does it matter?
3. Why does DQN need a replay buffer and a target network?
4. What does the policy-gradient theorem let you do that value-based methods can't?
5. *(debug)* Your agent's reward curve collapses suddenly after steady improvement. What are the likely causes?

## Mini-project
**Task:** Solve `FrozenLake` with tabular Q-learning, then `CartPole` with DQN (Gymnasium). Plot learning curves over 5 random seeds.
**Deliverable:** A notebook with mean ± std reward curves.

## Go deeper
- [Géron, *Hands-On ML*, Ch. 19 "Reinforcement Learning"](https://github.com/ageron/handson-mlp)
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1): RL applied to LLM reasoning.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 10: Reinforcement learning](../../toolbox/10-reinforcement-learning.md), for every concept in this lesson, with alternatives.
- **Papers:** [Reinforcement learning](../../papers/08-reinforcement-learning.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 43–45.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
