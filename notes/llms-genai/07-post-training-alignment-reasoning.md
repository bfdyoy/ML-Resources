# GEN-07 notes: Post-Training: SFT, RLHF, DPO & Reasoning

[← Lesson GEN-07](../../lessons/llms-genai/07-post-training-alignment-reasoning.md) · [All notes](../README.md) · [← GEN-09 notes](09-llm-security-safety.md) · Next: [GEN-08 notes →](08-efficient-llm-inference.md)

> **Reading time** ≈ 75 min. This is the most mathematical note in Path 3. **You need:** [GEN-01 notes](01-how-llms-are-built.md) §3, [MATH-03 notes](../math/03-probability-statistics.md) §B and §C (likelihood, KL, especially forward vs reverse KL), [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §1 (the sigmoid, the logistic loss). [EL-03 notes](../electives/03-reinforcement-learning.md) (policy gradients) help but aren't required.

---

## Where we are

GEN-01 sketched the pipeline: **pretrain → SFT → preference optimization → (RL with verifiable rewards)**. This note derives each stage's objective, and shows how DPO falls out of the RLHF objective with one clever substitution.

---

## 1. SFT: imitation

Minimize the NLL of the demonstration responses $y$ given the prompts $x$ (the loss is computed on the response tokens only):

```math
\mathcal{L}_\text{SFT} = -\mathbb{E}_{(x,y)}\sum_t \log\pi_\theta(y_t\mid x, y_{\lt t}).
```

- **What SFT teaches:** the chat format, following instructions, the target style and persona, and when to refuse. It turns "text continuer" into "assistant".
- **What it can't fix:** it only ever sees *good* examples, so it never learns *which of two plausible answers is better*. It can't exceed the quality of its demonstrations. And it trains on teacher-written text that the model didn't generate itself (*off-policy*), so its own mistakes are never corrected.

---

## 2. Reward models from pairwise preferences (Bradley–Terry)

Humans find it easier to *compare* two responses than to score one. Assume each response has a latent quality $r(x, y)$, and that the probability of preferring $y_w$ (winner) over $y_l$ (loser) is a sigmoid of the difference:

```math
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big).
```

Train a reward model $r_\phi$ (an LLM with a scalar head) by maximum likelihood on the preference pairs:

```math
\mathcal{L}_\text{RM} = -\mathbb{E}\big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big].
```

That's logistic regression on reward *differences* (CORE-03). Only differences matter, so the reward is defined **up to a constant per prompt**.

---

## 3. RLHF: maximize reward, stay close to the reference

```math
\max_{\pi}\ \mathbb{E}_{x,\ y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\ \Vert\ \pi_\text{ref}(\cdot\mid x)\big).
```

**Why the KL penalty?** Three reasons:

1. **Reward hacking:** the reward model is only accurate near the responses it was trained on. Without a leash, the policy drifts to strange outputs that the RM *overrates* (Goodhart, GEN-09 §6).
2. **Preserving capabilities and fluency:** it keeps the policy near the SFT model's knowledge and language.
3. **Diversity:** without it, the policy collapses onto a single highest-reward answer. With it, the optimum is a *distribution*.

$\beta$ trades reward against staying close to the reference.

**PPO** optimizes this with policy gradients. It samples responses from the current policy, scores them with the RM (minus the KL term), estimates advantages with a learned value network, and takes **clipped** steps:

```math
\mathcal{L}_\text{PPO} = -\mathbb{E}\Big[\min\big(\rho_t A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big],\qquad \rho_t = \frac{\pi_\theta(y_t\mid\cdot)}{\pi_\text{old}(y_t\mid\cdot)}.
```

The clipping stops a single batch from moving the policy too far. PPO works, but it needs **four models** in memory (policy, reference, reward, value) and a slow online sampling loop.

### 3.1 The optimal policy has a closed form

For a fixed $x$, the objective is $\sum_y \pi(y)r(y) - \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_\text{ref}(y)}$. Define $\pi^*(y) = \frac{1}{Z}\pi_\text{ref}(y)\,e^{r(y)/\beta}$, where $Z = \sum_y \pi_\text{ref}(y)e^{r(y)/\beta}$. A few lines of algebra rewrite the objective as

```math
-\beta\,\mathrm{KL}(\pi\,\Vert\,\pi^*) + \beta\log Z ,
```

which is maximized exactly at $\pi = \pi^*$, because KL is zero only there. So:

```math
\boxed{\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_\text{ref}(y\mid x)\,\exp\big(r(x,y)/\beta\big)}
```

**Reading it:** reweight the reference policy by exponentiated reward. A small $\beta$ means aggressive reweighting toward the best responses. A large $\beta$ stays near $\pi_\text{ref}$. We can't *sample* from this directly ($Z$ sums over all possible texts). But it unlocks DPO.

---

## 4. DPO: the reward is hidden inside the policy

Take logs of the boxed equation and solve for the reward:

```math
r(x, y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta\log Z(x).
```

Substitute this into the Bradley–Terry likelihood. **$\beta\log Z(x)$ appears for both $y_w$ and $y_l$, so it cancels in the difference.** Then replace $\pi^*$ with our trainable $\pi_\theta$:

```math
\mathcal{L}_\text{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\Big)\Big].
```

That's the whole trick: **the policy *is* the reward model** (its *implicit reward* is $\hat r = \beta\log\pi_\theta/\pi_\text{ref}$). DPO is just a classification loss on preference pairs:

- no separate reward model;
- no sampling loop (it trains on a fixed dataset of pairs);
- no value network;
- only two models in memory (the policy, and the frozen reference).

**Gradient intuition:** each pair's update is weighted by $\sigma(\hat r_l - \hat r_w)$, which is large when the model currently *ranks the pair wrongly*. Like logistic regression, it focuses on its mistakes. It raises the likelihood of $y_w$ and lowers that of $y_l$, both relative to the reference.

**Trade-offs vs PPO:** DPO is *offline* (it learns from fixed pairs, not from its own fresh samples), so it can over-optimize on quirks of the dataset. A known failure is that it lowers the likelihood of *both* responses.
Online and iterative DPO, and variants like **KTO** (learns from single thumbs-up/down labels, using a prospect-theory loss) and **ORPO** (folds the preference term into SFT with no reference model), each relax one assumption.

---

## 5. RL with verifiable rewards (RLVR) and GRPO

For math, code, and logic, the reward is a **program**: 1 if the final answer matches or the tests pass, 0 otherwise. It's cheap and objective, and much harder to hack than a learned RM. So RL can run for many steps without the policy drifting into nonsense.

**GRPO (group relative policy optimization)** drops PPO's value network. For each prompt:

1. sample a **group** of $G$ responses;
2. score each one, $r_1..r_G$;
3. use the **group-normalized reward** as the advantage:

```math
A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G})} .
```

Responses better than their siblings get pushed up, and worse ones get pushed down: a **per-prompt baseline** with no learned critic. Then apply PPO-style clipped updates plus a KL term.
If every response in the group gets the same reward (all right, or all wrong), the advantages are 0 and the prompt teaches nothing. That's why **curriculum and difficulty filtering** matter: you want prompts the model solves *sometimes*.

**Why long reasoning emerges:** nothing rewards length directly. But on hard problems, responses that check their work, try alternatives, and backtrack are *correct more often*. The policy gradient amplifies whatever raises correctness, so chains of thought grow longer and more careful over training.
(It also amplifies quirks, such as language mixing, which later SFT and RL stages clean up.)

---

## 6. Costs of alignment: hacking and lost diversity

- **Reward hacking:** the policy exploits the RM's blind spots. Signs: reward rising while human ratings stall, responses getting longer, sycophancy. Watch the KL to the reference and the length.
- **Diversity collapse:** the KL-regularized optimum is $\pi_\text{ref}\,e^{r/\beta}$, so as $\beta$ shrinks, probability concentrates on the top responses. And the reverse-KL nature of the objective is **mode-seeking** (MATH-03 §C3).
  Aligned models are therefore less diverse and less calibrated than their base models: fine for an assistant, bad for creative sampling or for using the probabilities as estimates.

**Debug: DPO answers got much longer; the judge prefers them, humans don't.** It's **length exploitation**:

- the preference data, or the AI judge that labeled it, has a **verbosity bias** (longer answers win more often), so DPO learned "longer = better";
- the evaluation judge shares that bias, so it rewards the same thing.

Fixes:

- check the win rate against the length difference in your data, and balance or length-control the pairs;
- use length-normalized objectives (SimPO-style average log-probabilities), or add an explicit length penalty;
- use a stronger $\beta$;
- **evaluate with length-controlled win rates**, and validate the judge against humans (GEN-03 §4).

```python
import numpy as np
rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))

# --- Closed-form optimal policy: pi* ∝ pi_ref * exp(r / beta) over 5 candidate responses ---------------
r = np.array([1.0, 2.0, 0.5, 3.0, 2.8])            # true rewards
pi_ref = np.array([0.30, 0.25, 0.25, 0.10, 0.10])
for beta in [10.0, 1.0, 0.1]:
    w = pi_ref * np.exp(r / beta); pi_star = w / w.sum()
    kl = np.sum(pi_star * np.log(pi_star / pi_ref))
    print(f"beta={beta:5}: pi*={np.round(pi_star, 3)}  E[r]={pi_star @ r:.2f}  KL to ref={kl:.2f}")

# --- Bradley-Terry reward model from pairwise preferences -------------------------------------------
d, n_pairs = 5, 4000
w_true = rng.normal(size=d)
F = rng.normal(size=(200, d))                        # features of 200 candidate responses
true_r = F @ w_true
i, j = rng.integers(0, 200, n_pairs), rng.integers(0, 200, n_pairs)
pref_i = rng.random(n_pairs) < sig(true_r[i] - true_r[j])     # humans prefer i with BT probability
win = np.where(pref_i, i, j); lose = np.where(pref_i, j, i)
w = np.zeros(d)
for _ in range(500):
    p = sig(F[win] @ w - F[lose] @ w)
    w += 0.5 * ((1 - p)[:, None] * (F[win] - F[lose])).mean(0)     # gradient ascent on log sigma(r_w - r_l)
print("learned RM vs true reward, correlation:", np.corrcoef(F @ w, true_r)[0, 1].round(3))
```

```python
# --- DPO on a tabular "policy": the implicit reward recovers the true reward (up to a constant) ------------
K, beta = 6, 0.5
r_true = np.array([0.0, 1.0, 2.0, -1.0, 0.5, 1.5])
logits_ref = rng.normal(size=K); pi_ref = np.exp(logits_ref) / np.exp(logits_ref).sum()
a, b = rng.integers(0, K, 20000), rng.integers(0, K, 20000)
keep = a != b; a, b = a[keep], b[keep]
a_wins = rng.random(len(a)) < sig(r_true[a] - r_true[b])
yw, yl = np.where(a_wins, a, b), np.where(a_wins, b, a)

theta = logits_ref.copy()
for step in range(3000):
    logp = theta - np.log(np.exp(theta).sum())
    h = beta * ((logp[yw] - np.log(pi_ref[yw])) - (logp[yl] - np.log(pi_ref[yl])))
    g_h = -(1 - sig(h)) * beta                         # d(-log sigma(h))/dh, times beta
    grad = np.zeros(K)
    np.add.at(grad, yw, g_h); np.add.at(grad, yl, -g_h)
    # (the log-partition terms cancel between y_w and y_l, so this is the full gradient)
    theta -= 2.0 * grad / len(yw)
pi = np.exp(theta) / np.exp(theta).sum()
implicit = beta * np.log(pi / pi_ref)
print("implicit reward (centred):", np.round(implicit - implicit.mean(), 2))
print("true reward (centred)    :", np.round(r_true - r_true.mean(), 2))

# --- GRPO advantages for one prompt ------------------------------------------------------------------------------
group_rewards = np.array([1, 0, 0, 1, 1, 0, 0, 0], float)       # 8 sampled answers, 3 correct
A = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
print("GRPO advantages:", A.round(2), "| all-correct group gives:", np.round((np.ones(8) - 1) / (0 + 1e-8), 2))
```

---

## Pitfalls & misconceptions

- **"DPO has no reward model."** It has an *implicit* one, $\beta\log\pi_\theta/\pi_\text{ref}$, and it inherits every bias in the preference data.
- **Too small a $\beta$ (or too many epochs).** It over-optimizes: verbosity, sycophancy, collapse.
- **Evaluating aligned models only with an AI judge** that shares the preference data's biases.
- **RLVR on prompts that are too easy or too hard.** Every response in the group gets the same reward, so the gradient is zero.
- **Forgetting the reference model** when computing DPO log-ratios, or using a different tokenizer or chat template for it.

## Cheat sheet

| Item | Formula |
|---|---|
| SFT | $-\sum_t\log\pi(y_t\mid x, y_{\lt t})$ on the responses |
| Bradley–Terry | $P(y_w\succ y_l) = \sigma(r_w - r_l)$ |
| RLHF objective | $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\Vert\pi_\text{ref})$ |
| Optimal policy | $\pi^* \propto \pi_\text{ref}\,e^{r/\beta}$ |
| Implicit reward | $\beta\log\frac{\pi}{\pi_\text{ref}}$ (+ a per-prompt constant) |
| DPO | $-\log\sigma\big(\beta\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)}\big)$ |
| PPO clip | $\min(\rho A, \operatorname{clip}(\rho, 1\pm\epsilon)A)$ |
| GRPO advantage | $(r_i - \bar r)/\operatorname{std}(r)$ within the group |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What SFT teaches that pretraining doesn't, and what it can't fix.</summary>

The assistant format, following instructions, style, and refusals. It can't express relative preferences between plausible answers, can't exceed its demonstrations, and never corrects the model's own (on-policy) mistakes.
</details>

<details>
<summary>2. How is a reward model trained from pairwise preferences?</summary>

Assume $P(y_w\succ y_l) = \sigma(r_w - r_l)$ (Bradley–Terry), and fit $r_\phi$ by minimizing $-\log\sigma(r_\phi(y_w) - r_\phi(y_l))$: logistic regression on reward differences. The demo recovers the true reward ordering.
</details>

<details>
<summary>3. Why the KL penalty in PPO-based RLHF?</summary>

To limit reward hacking (the RM is only reliable near its training distribution), to preserve the reference model's capabilities and fluency, and to keep a diverse distribution instead of collapsing. The optimum is $\pi_\text{ref}e^{r/\beta}/Z$.
</details>

<details>
<summary>4. Why does DPO need no explicit reward model or sampling loop?</summary>

The KL-regularized optimum links reward and policy: $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z$. In the Bradley–Terry likelihood, the intractable $\log Z$ cancels, leaving a supervised loss on fixed preference pairs in terms of the policy's own log-probabilities.
</details>

<details>
<summary>5. What makes a reward verifiable, and why did RLVR unlock long reasoning?</summary>

It's computed by a reliable checker (exact answer, unit tests): cheap, objective, hard to game, so long RL runs stay stable. Careful, longer reasoning raises the correctness rate, and the policy gradient amplifies it.
</details>

<details>
<summary>6. How does GRPO estimate advantages without a value network?</summary>

It samples a group of responses per prompt and normalizes each reward by the group's mean and standard deviation. The group mean acts as the baseline (the demo computes one).
</details>

<details>
<summary>7. After DPO: longer answers, the judge likes them, humans don't.</summary>

Length or verbosity exploitation, learned from biased preference data and rewarded by an equally biased judge. Length-balance the data, use length-normalized objectives or penalties and a stronger $\beta$, and evaluate with length-controlled win rates and human checks.
</details>

## Where this leads

Next: [GEN-08 notes](08-efficient-llm-inference.md). Post-trained models are served billions of times. GEN-08 is the systems side: why decoding is memory-bound, how the KV cache, quantization, speculative decoding, and batching make serving affordable.
