# EL-03 notes: Reinforcement Learning

[← Lesson EL-03](../../lessons/electives/03-reinforcement-learning.md) · [All notes](../README.md) · [← EL-02 notes](02-bayesian-probabilistic-ml.md) · Next: [EL-04 notes →](04-recommender-systems.md)

> **Reading time** ≈ 65 min. **You need:** [MATH-03 notes](../math/03-probability-statistics.md) §A3 (expectations), [MATH-02 notes](../math/02-calculus-optimization.md) (gradients), and [DL-01 notes](../deep-learning/01-neural-networks-from-scratch.md) (networks, for the deep RL part).

---

## Where we are

Supervised learning gets the right answer for every input. In reinforcement learning, an **agent** only gets a **reward** after acting, possibly much later, and its own actions decide what data it sees next. Two problems arise that Paths 1–2 never had:

- **credit assignment:** which earlier action earned this reward?
- **exploration vs exploitation:** should it try something new, or use what it knows?

The math here also underlies RLHF and reasoning models (GEN-07).

---

## 1. Markov decision processes

An MDP is made of:

- states $s$, actions $a$;
- transition probabilities $P(s'\mid s, a)$;
- rewards $r(s, a)$;
- a discount factor $\gamma\in[0,1)$.

A **policy** $\pi(a\mid s)$ chooses actions. The **return** from time $t$ is the discounted sum of future rewards, $G_t = \sum_{k\ge0}\gamma^k r_{t+k}$. The discount makes infinite sums finite, and prefers sooner rewards.

**Value functions:**

- **state value** $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$: how good it is to *be* in $s$ and follow $\pi$;
- **action value** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$: how good it is to *take* $a$ in $s$, and then follow $\pi$.

They're related by $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s,a)$. $Q$ is what you need to act **without a model**: just pick $\arg\max_a Q(s,a)$.

### 1.1 The Bellman equations

The value of a state is the immediate reward plus the discounted value of where you land:

```math
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s,a) + \gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\Big],\qquad
Q^*(s,a) = r(s,a) + \gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a').
```

The second one, the **Bellman optimality equation**, defines the optimal $Q^*$. With a known model, **value iteration** applies it repeatedly as an update until convergence. It's a contraction with factor $\gamma$, so it always converges.

---

## 2. Learning from experience (no model)

### 2.1 Monte Carlo vs temporal difference

- **Monte Carlo:** play the episode to the end, then move $V(s)$ toward the actual return: $V(s)\leftarrow V(s) + \alpha\,(G_t - V(s))$. It's unbiased, high-variance, and must wait until the end of the episode.
- **TD(0):** update after **one step**, using the current estimate of the next state as a stand-in for the rest of the return (**bootstrapping**):

```math
V(s_t)\leftarrow V(s_t) + \alpha\big[\underbrace{r_t + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\big].
```

TD has lower variance and learns online, but it's biased by its own estimates.

### 2.2 SARSA (on-policy) vs Q-learning (off-policy)

```math
\textbf{SARSA: } Q(s,a)\leftarrow Q(s,a) + \alpha\big[r + \gamma Q(s', a') - Q(s,a)\big]\quad(a' = \text{the action actually taken next})
```

```math
\textbf{Q-learning: } Q(s,a)\leftarrow Q(s,a) + \alpha\big[r + \gamma\max_{a''}Q(s', a'') - Q(s,a)\big]
```

- **SARSA** learns the value of **the policy it's actually following**, including its exploration (say ε-greedy). It's *on-policy*.
- **Q-learning** learns the value of the **greedy (optimal) policy**, whatever exploratory action it actually took. It's *off-policy*, so it can learn from old data or from another agent's behaviour.
- **When it matters (the classic "cliff walking" example):** the shortest path runs along a cliff edge. Q-learning learns that optimal edge path, and while exploring it occasionally steps off the cliff, collecting big penalties. SARSA accounts for its own random steps and learns a **safer path** further from the edge.
  So SARSA earns more reward *during training*, and Q-learning finds the better greedy policy. The code reproduces this.

### 2.3 Exploration

- **ε-greedy:** act randomly with probability ε, and greedily otherwise. Decay ε over time.
- **Optimism / UCB (multi-armed bandits):** pick the arm with the highest upper confidence bound, $\hat\mu_a + c\sqrt{\ln t / n_a}$. Arms tried rarely get a large bonus. That's "optimism in the face of uncertainty", and it gives logarithmic regret.
- **Thompson sampling:** sample a plausible value per arm from its posterior (EL-02), and pick the best sample.

---

## 3. Deep RL

### 3.1 DQN

Replace the Q-table with a network $Q_\theta(s, a)$ trained to minimize $\big(r + \gamma\max_{a'}Q_{\theta^-}(s', a') - Q_\theta(s,a)\big)^2$. Two stabilizers:

- **Replay buffer:** store transitions and train on *random* mini-batches from it. That breaks the strong correlation between consecutive steps (SGD assumes roughly i.i.d. data), and reuses each transition many times.
- **Target network** $\theta^-$: a lagged copy of the weights, used to compute the targets. Without it, every update moves its own target (the "moving target" problem), which causes oscillation or divergence.

Function approximation + bootstrapping + off-policy learning is the **"deadly triad"**, which can diverge. These tricks tame it.

### 3.2 Policy gradients

Parameterize the policy directly, $\pi_\theta(a\mid s)$, and maximize $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$. The **log-derivative trick**, $\nabla_\theta p = p\,\nabla_\theta\log p$, gives:

```math
\nabla_\theta J = \mathbb{E}_{\pi_\theta}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\;(G_t - b(s_t))\Big].
```

- **What it buys you:** you never differentiate through the environment or the reward. You only need to *sample* trajectories and differentiate the log-probability of your own actions. So it works with **non-differentiable rewards**, **stochastic policies**, and **continuous or huge action spaces** (like the vocabulary of an LLM), where $\arg\max_a Q$ is impractical.
- **REINFORCE** uses the sampled return $G_t$. It's unbiased and very noisy. Subtracting a **baseline** $b(s)$ (such as $V(s)$) leaves the gradient unbiased, because $\mathbb{E}[\nabla\log\pi\cdot b] = b\,\nabla\sum_a\pi = 0$, and it **reduces the variance**.
- **Actor-critic:** learn $V_w(s)$ as the baseline (the critic), and use the **advantage** $A = G - V$ (or the TD error) in the actor's update.
- **PPO:** limits each policy update with the clipped ratio objective (GEN-07 §3), for stability.

### 3.3 The connection to RLHF

For an LLM:

- the **state** is the prompt plus the tokens generated so far;
- an **action** is the next token;
- the **policy** is the model;
- the **reward** is a reward-model score (or a verifier's verdict) at the end of the response.

RLHF with PPO, and GRPO (GEN-07 §5, which uses the group mean as the baseline instead of a critic), are policy-gradient methods on exactly this MDP.

**Debug: the reward curve suddenly collapses after steady improvement.** Likely causes:

- a **too-large policy update** (the learning rate, no clipping or KL constraint), so one bad step moves the policy somewhere it can't recover from;
- **Q-value divergence or overestimation** (the deadly triad; target-network update too frequent);
- **exploration collapse:** entropy goes to 0, or ε decays to 0 too early;
- **catastrophic forgetting** as the replay buffer fills with only recent, narrow experience;
- **non-stationarity or bugs:** a reward-scale change, environment resets, observation-normalization drift.

Check the entropy, the KL between successive policies, the Q-value magnitudes, and the gradient norms around the collapse.

```python
import numpy as np
rng = np.random.default_rng(0)

# --- Value iteration on a 4x4 gridworld (goal at the corner, -1 per step) ---------------------------------
S, gamma = 16, 1.0
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def step(s, a):
    if s == 15: return s, 0.0
    r, c = divmod(s, 4); dr, dc = moves[a]
    r2, c2 = min(max(r + dr, 0), 3), min(max(c + dc, 0), 3)
    return r2 * 4 + c2, -1.0
V = np.zeros(S)
for it in range(100):
    V_new = np.array([max(rw + gamma * V[s2] for s2, rw in (step(s, a) for a in range(4))) for s in range(S)])
    if np.allclose(V_new, V): break
    V = V_new
print(f"value iteration converged in {it} sweeps; V (minus the steps to the goal):\n{V.reshape(4, 4)}")
```

```python
# --- Cliff walking: SARSA (on-policy) vs Q-learning (off-policy) --------------------------------------------------
H, W = 4, 12; start, goal = (3, 0), (3, 11)
def env_step(pos, a):
    r, c = pos; dr, dc = moves[a]
    r, c = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    if r == 3 and 0 < c < 11: return start, -100.0, False     # fell off the cliff
    return (r, c), -1.0, (r, c) == goal

def run(method, episodes=2000, eps=0.1, alpha=0.5, seed=0):
    r_ = np.random.default_rng(seed); Q = np.zeros((H, W, 4)); totals = []
    pick = lambda s: r_.integers(4) if r_.random() < eps else int(np.argmax(Q[s]))
    for _ in range(episodes):
        s, total, done = start, 0.0, False; a = pick(s)
        while not done:
            s2, rew, done = env_step(s, a); a2 = pick(s2); total += rew
            target = rew + (0 if done else (Q[s2][a2] if method == "sarsa" else Q[s2].max()))
            Q[s][a] += alpha * (target - Q[s][a]); s, a = s2, a2
        totals.append(total)
    return np.mean(totals[-100:]), Q
def greedy_path_row_min(Q):
    s, rows = start, []
    for _ in range(60):
        s, _, done = env_step(s, int(np.argmax(Q[s]))); rows.append(s[0])
        if done: break
    return min(rows) if rows else None
for m in ["sarsa", "qlearning"]:
    avg, Q = run(m)
    print(f"{m:9s}: average reward per training episode (last 100) {avg:7.1f}; greedy path climbs to row {greedy_path_row_min(Q)} (row 2 = hugging the cliff, row 0 = far from it)")
```

```python
# --- REINFORCE on a 3-armed bandit: the baseline cuts gradient variance without bias -----------------------------------
true_means = np.array([1.0, 1.5, 2.0]); theta = np.zeros(3)
def grads(theta, baseline, n=5000):
    p = np.exp(theta) / np.exp(theta).sum()
    a = rng.choice(3, size=n, p=p); R = true_means[a] + rng.normal(0, 1, n)
    glog = -np.tile(p, (n, 1)); glog[np.arange(n), a] += 1      # gradient of log softmax(theta)[a]
    return glog * (R - baseline)[:, None]
g0, g1 = grads(theta, 0.0), grads(theta, true_means.mean())
print("mean gradient, no baseline  :", g0.mean(0).round(3), " variance", g0.var(0).round(2))
print("mean gradient, with baseline:", g1.mean(0).round(3), " variance", g1.var(0).round(2))
```

---

## Pitfalls & misconceptions

- **Judging the agent by the training reward under exploration.** Evaluate the greedy policy separately.
- **No target network or replay buffer in DQN.** Instability.
- **REINFORCE without a baseline**, or with a tiny batch. The gradient is mostly noise.
- **Reward bugs.** Agents exploit any loophole (reward hacking, GEN-09 §6).
- **Single-seed conclusions.** RL results vary hugely across seeds. Report several.

## Cheat sheet

| Item | Formula |
|---|---|
| Return | $G_t = \sum_k\gamma^kr_{t+k}$ |
| Bellman optimality | $Q^*(s,a) = r + \gamma\,\mathbb{E}_{s'}\max_{a'}Q^*(s',a')$ |
| TD(0) | $V\mathrel{+}=\alpha(r + \gamma V(s') - V(s))$ |
| SARSA / Q-learning target | $r + \gamma Q(s',a')$ / $r + \gamma\max Q(s',\cdot)$ |
| UCB | $\hat\mu_a + c\sqrt{\ln t/n_a}$ |
| Policy gradient | $\mathbb{E}[\nabla\log\pi(a\mid s)(G - b(s))]$ |
| DQN stabilizers | replay buffer + target network |

## Answer sketches for the lesson's self-check

<details>
<summary>1. State-value vs action-value function.</summary>

$V^\pi(s)$: the expected return from state $s$ when following $\pi$. $Q^\pi(s,a)$: the expected return when taking action $a$ in $s$ first, then following $\pi$. $Q$ lets you choose actions without a model of the environment.
</details>

<details>
<summary>2. Why Q-learning is off-policy and SARSA on-policy; when it matters.</summary>

Q-learning's target uses $\max_{a'}Q(s',a')$, the greedy policy, regardless of the action actually taken. SARSA's target uses the action the behaviour policy actually takes. It matters when exploration is risky: on cliff walking, SARSA learns a safer path and earns more reward during training, while Q-learning learns the optimal edge path but falls more while exploring (see the demo).
</details>

<details>
<summary>3. Why DQN needs a replay buffer and a target network.</summary>

The replay buffer decorrelates consecutive samples and reuses data. The target network keeps the bootstrap target stable for a while, so updates don't chase their own changing predictions. Together they prevent oscillation and divergence.
</details>

<details>
<summary>4. What does the policy-gradient theorem enable?</summary>

Optimizing a parameterized stochastic policy directly, by sampling, without differentiating through the environment or the reward. It handles continuous or huge action spaces and non-differentiable rewards: exactly the RLHF setting.
</details>

<details>
<summary>5. The reward curve collapses after steady improvement.</summary>

A too-large policy step (learning rate, missing clipping or KL), value divergence (the deadly triad, target updates too frequent), exploration or entropy collapse, replay-buffer forgetting, or reward and environment bugs and normalization drift. Inspect the entropy, the policy KL, the Q magnitudes, and the gradient norms.
</details>

## Where this leads

Next: [EL-04 notes](04-recommender-systems.md). Recommenders are where exploration, feedback loops, and ranking all meet real users.
