# Papers: Reinforcement Learning

[← Papers library](README.md) · Background: [EL-03](../lessons/electives/03-reinforcement-learning.md) · [Toolbox: RL](../toolbox/10-reinforcement-learning.md)

> Read [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html) Ch. 1–6 and [Spinning Up Part 1–3](https://spinningup.openai.com/en/latest/user/introduction.html) before these.

## Value-based deep RL
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Playing Atari with Deep RL (DQN)](https://arxiv.org/abs/1312.5602) | 2013 | ⭐ The start of deep RL: replay buffer plus target network. | L2 | EL-03 |
| [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) | 2017 | Six DQN improvements with an ablation study. A good map of the field. | L2 | EL-03 |

## Policy gradients & actor-critic
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) | 2015 | Monotonic improvement with KL constraints. Math-heavy. | L3 | EL-03 |
| [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) | 2015 | ⭐ Generalized Advantage Estimation, used inside PPO. | L3 | EL-03 |
| [Asynchronous Methods for Deep RL (A3C)](https://arxiv.org/abs/1602.01783) | 2016 | Parallel actor-critic. | L2 | EL-03 |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | 2017 | ⭐ Clipped objective. The default algorithm, including for RLHF. | L2 | EL-03 |
| [Continuous Control with Deep RL (DDPG)](https://arxiv.org/abs/1509.02971) | 2015 | Deterministic policy gradients for continuous actions. | L3 | EL-03 |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) | 2018 | Maximum-entropy RL, and strong off-policy continuous control. | L3 | EL-03 |

## Model-based RL, planning & games
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) | 2018 | An agent trained inside its own "dream". Beautifully presented. | L2 | EL-03 |
| [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815) | 2017 | ⭐ MCTS plus self-play with no human knowledge. | L2 | EL-03 |
| [Mastering Atari, Go, Chess and Shogi with a Learned Model (MuZero)](https://arxiv.org/abs/1911.08265) | 2019 | Planning with a learned model, no game rules given. | L3 | EL-03 |
| [Decision Transformer](https://arxiv.org/abs/2106.01345) | 2021 | RL as sequence modeling. | L2 | DL-06 |

## Perspective
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) *(blog)* | 2018 | ⭐ An honest account of why deep RL is hard: sample efficiency, reward design, instability. | L1 | EL-03 |
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | The bridge from RL to RLHF. | L2 | EL-03 |
