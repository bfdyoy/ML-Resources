# GEN-03 notes: Evaluating & Shipping LLM Applications

[← Lesson GEN-03](../../lessons/llms-genai/03-evaluating-llm-apps.md) · [All notes](../README.md) · [← GEN-02 notes](02-adapting-llms-finetuning-rag.md) · Next: [GEN-05 notes →](05-retrieval-engineering.md)

> **Reading time** ≈ 50 min. **You need:** [MATH-03 notes](../math/03-probability-statistics.md) §D (standard errors, paired comparisons), [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §3 (TPR/TNR), and [CORE-10 notes](../core-ml/10-data-centric-ml.md) §3 (Cohen's κ).

---

## Where we are

GEN-02 gave you several levers. This lesson decides **which lever, and whether it worked**. LLM outputs are free-form text, so "accuracy" isn't free. You have to *build* the measuring instrument, and then check that the instrument itself is accurate.
The statistics from Path 1 (error bars, paired tests, agreement) are exactly what makes LLM evals trustworthy.

---

## 1. Error analysis first

Before writing a single metric, **read your outputs**:

1. Collect 50–100 realistic traces: real or realistic inputs, plus the full outputs (and the retrieved context, and the tool calls).
2. **Open coding:** write a short free-text note on each failure ("ignored the date constraint", "cited the wrong doc", "too verbose").
3. **Axial coding:** group the notes into categories, and **count** them.
4. Prioritize by frequency × severity.

**Why first?** Because you can't know which metrics matter until you've seen how the system actually fails. Generic metrics ("helpfulness 4.2/5") measure things that may not be your problem, and they miss the things that are. Every eval you then write targets a real failure category.

---

## 2. Three levels of evals

| Level | What it is | Cost | Use for |
|---|---|---|---|
| **1. Assertions / unit tests** | Code checks: parses as JSON, matches the schema, contains the required field, no banned phrases, cites an existing doc ID, a regex | ~Free, deterministic | Anything that can be checked mechanically. Run on every change, in CI. |
| **2. LLM-as-judge** | A model grades the output against a specific, *binary* criterion ("does the answer contradict the context? yes/no") | Moderate | Fuzzy qualities: faithfulness, relevance, tone |
| **3. Human review** | Experts label samples | Expensive | Ground truth; validating the judges; high-stakes checks |

**When is an assertion better than a judge?** Whenever the property *can* be checked by code. It's exact, free, never drifts, and needs no validation.
Use judges only for what code can't check. And prefer **binary, specific** judge questions over 1–10 scales, which are noisy and poorly calibrated.

---

## 3. Your eval score has an error bar

A pass rate on $n$ test cases is a binomial proportion (MATH-03 §D1):

```math
\mathrm{SE} = \sqrt{\frac{\hat p(1-\hat p)}{n}} .
```

At $\hat p = 0.8$, **50 cases give ±11 percentage points** (95% CI) and **500 cases give ±3.5**. A prompt change that moves 80% to 84% on 50 cases is noise.
Compare two versions **on the same cases with a paired test** (a paired bootstrap, or McNemar on the cases where they disagree; MATH-03 §D3). That's far more sensitive than comparing two independent pass rates.

---

## 4. Validating an LLM judge

A judge is a **classifier**, so measure it like one, against human labels on a held-out set:

- **TPR** (it says "pass" when humans say pass) and **TNR** (it says "fail" when humans say fail). Report both: a judge that passes everything has TPR = 1 and is useless.
- **Cohen's κ** for agreement beyond chance (CORE-10 §3).
- Iterate on the judge prompt (criteria, examples of hard cases) on a *development* split, and report the final agreement on a separate *test* split. Otherwise you overfit the judge, just as you would a model.

### 4.1 Correcting the measured pass rate

If the judge has known TPR and TNR, the observed pass rate $p_\text{obs}$ is biased. The true rate $\theta$ satisfies $p_\text{obs} = \theta\,\text{TPR} + (1 - \theta)(1 - \text{TNR})$. Solving for $\theta$ gives the Rogan–Gladen correction:

```math
\hat\theta = \frac{p_\text{obs} + \text{TNR} - 1}{\text{TPR} + \text{TNR} - 1}.
```

*Example:* a judge with TPR = 0.95 and TNR = 0.60 reports 90% passing. The corrected estimate is $(0.90 + 0.60 - 1)/(0.95 + 0.60 - 1) = 0.50/0.55 \approx 0.91$. And if it reports 85%, it's $0.45/0.55 \approx 0.82$.
A lenient judge (low TNR) inflates scores when the true pass rate is low. The code shows this.

### 4.2 Judge biases, and how to detect them

| Bias | What happens | Detection / fix |
|---|---|---|
| **Position bias** | In pairwise comparisons, prefers whichever answer is shown first (or second) | Run both orders. Count only consistent verdicts. Measure the flip rate. |
| **Verbosity bias** | Prefers longer answers | Correlate verdicts with length. Compare length-matched pairs. Instruct the judge about conciseness. |
| **Self-preference** | Prefers outputs from its own model family | Use a judge from a different family. Check agreement with humans per source. |
| **Leniency / criteria drift** | Passes nearly everything, or interprets the criteria loosely | A low TNR against human labels. Make the criteria binary and specific, with failing examples. |

---

## 5. Offline vs online

- **Offline eval set:** fixed, versioned test cases. Reproducible, so you can compare versions before shipping. Its weakness: it **goes stale** and may not represent real traffic.
- **Online metrics:** user feedback (thumbs, edits, retries), task completion, escalation rate, latency, cost, and production traces scored by your judges. They reflect reality, but they're noisy and delayed, and they can only measure what you've already shipped.

You need both. **Feed the online failures back into the offline set**: that's the loop that keeps it representative.

### 5.1 Production patterns

- **Guardrails:** validate outputs (schema, safety filters, PII checks) before showing them, with fallbacks when validation fails.
- **Caching:** exact or semantic caching for repeated queries.
- **Defensive UX:** show sources, allow edits, set expectations.
- **Collect feedback** tied to traces, so every complaint becomes a debuggable example.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
rng = np.random.default_rng(0)

# --- Error bars on pass rates ------------------------------------------------------------------
for n in [50, 200, 500]:
    se = np.sqrt(0.8 * 0.2 / n)
    print(f"n={n:3d}: pass rate 0.80 ± {1.96*se:.3f} (95% CI)")

# --- Paired comparison of two prompt versions on the same 200 cases ------------------------------------
n = 200
a = rng.random(n) < 0.80                                  # version A passes ~80%
b = a.copy(); flip = rng.random(n) < 0.12                 # B differs on ~12% of cases...
b[flip] = rng.random(flip.sum()) < 0.70                   # ...where it passes 70% of the time
diffs = [b[i].mean() - a[i].mean() for i in (rng.integers(0, n, n) for _ in range(2000))]
print(f"A={a.mean():.3f} B={b.mean():.3f}; paired-bootstrap 95% CI of B-A: {np.percentile(diffs, 2.5):+.3f} .. {np.percentile(diffs, 97.5):+.3f}")

# --- Validating a judge against human labels -------------------------------------------------------------
human = rng.random(400) < 0.6                             # true pass rate 60%
judge = np.where(human, rng.random(400) < 0.95, rng.random(400) < 0.40)   # TPR .95, TNR .60 (lenient)
TPR = judge[human].mean(); TNR = (~judge[~human]).mean()
print(f"judge TPR {TPR:.2f}  TNR {TNR:.2f}  kappa {cohen_kappa_score(human, judge):.2f}")
p_obs = judge.mean()
theta = (p_obs + TNR - 1) / (TPR + TNR - 1)
print(f"judge says {p_obs:.2f} pass; corrected estimate {theta:.2f}; human truth {human.mean():.2f}")

# --- Position bias: how often does the verdict flip when the order is swapped? -------------------------------
quality_gap = rng.normal(0, 1, 300)                       # >0 means answer X is truly better
def judge_pair(gap, first_bonus=0.8):                     # this judge favours whatever is shown first
    return gap + first_bonus + rng.normal(0, 0.5, gap.shape) > 0
xy = judge_pair(quality_gap)                              # X shown first
yx = ~judge_pair(-quality_gap)                            # Y shown first; convert to "X wins"
print(f"verdict flips when order is swapped: {np.mean(xy != yx):.0%}  -> count only consistent verdicts")
```

---

## Pitfalls & misconceptions

- **Writing generic metrics before reading outputs.**
- **1–10 rating scales from judges.** They're noisy and uncalibrated. Use binary, specific criteria.
- **Trusting a judge without measuring TPR/TNR** against human labels.
- **Declaring a winner from a 2-point difference on 50 cases.**
- **A frozen eval set** that never absorbs production failures.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Pass-rate SE | $\sqrt{p(1-p)/n}$; 50 cases ≈ ±11 points at 80% |
| Compare versions | a paired test on the same cases |
| Judge quality | TPR, TNR, κ against human labels (dev/test split) |
| Corrected rate | $(p_\text{obs} + \text{TNR} - 1)/(\text{TPR} + \text{TNR} - 1)$ |
| Position bias | evaluate both orders |
| Order of work | error analysis → assertions → validated judges → humans |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why error analysis before writing evals?</summary>

You don't know which failure modes matter until you've read real outputs. Evals written first measure generic qualities, miss your actual failures, and waste effort. Error analysis produces the categories that each eval should target.
</details>

<details>
<summary>2. Three LLM-judge biases and how to detect them.</summary>

Position bias (swap the order and measure the flip rate), verbosity bias (correlate verdicts with length; use length-matched pairs), and self-preference (compare judges from different families against human labels). Also leniency, which shows as a low TNR against humans.
</details>

<details>
<summary>3. Offline eval set vs online metrics.</summary>

Offline: fixed, reproducible, used before shipping, but it can drift from real use. Online: real traffic and user behaviour, but noisy, delayed, and only for shipped versions. Use offline to gate releases and online to discover new failures, and feed those back into the offline set.
</details>

<details>
<summary>4. When is an assertion better than a model-graded eval?</summary>

Whenever the property is mechanically checkable (valid JSON, schema, required fields, citation IDs exist, length limits). Assertions are exact, free, deterministic, and need no validation.
</details>

<details>
<summary>5. The judge rates 95% good, users complain constantly.</summary>

Check the judge against human labels: it's probably lenient (low TNR), or judging the wrong criteria. Check that the eval set represents real traffic (sample production traces). Do error analysis on the complaints themselves, and add those failures as new test cases and judge criteria.
</details>

## Where this leads

Next: [GEN-05 notes](05-retrieval-engineering.md). Error analysis on RAG systems usually points at **retrieval**. GEN-05 opens the retriever up: BM25, embeddings, vector indexes, hybrid search and reranking, each measured with the retrieval metrics you're now ready to trust.
