# Playbook 6: Debugging, from symptom to cause

[← Playbook](README.md) · Previous: [Outside the box](05-outside-the-box.md)

> ML bugs rarely crash. They show up as a number that's a little too good, a little too bad, or one that won't move. Find your **symptom** below,
> then check the **likely causes** in order: they're sorted by how often they turn out to be the culprit. The method is Karpathy's
> [recipe](http://karpathy.github.io/2019/04/25/recipe/): become one with the data, build up from something that works, change one thing at a time, and visualize everything.

---

## A. Tabular and classical ML

| Symptom | Likely causes (most common first) | Check |
|---|---|---|
| CV score far better than the test or production score | Leakage (target-derived features fitted outside the folds; future data); a random split on grouped or time data; tuning on the test set | Shuffled-label run must score chance ([outside the box §9](05-outside-the-box.md#9-shuffle-the-labels-a-lie-detector-for-your-pipeline)); group or time splits; adversarial validation ([tabular §1](02-tabular-tricks.md#1-adversarial-validation-can-a-model-tell-train-from-test)) |
| A score that's "too good to be true" (AUC 0.99 on a hard problem) | A feature that encodes the label (an ID, a timestamp, a post-outcome field); duplicate rows across the split | Top feature importances: would each be known *at prediction time*? Search for duplicates across the train/test split |
| CV scores vary a lot between folds | Small data; rare classes landing unevenly; group structure | Repeated stratified/group CV; report mean ± std; are differences between models within the spread? |
| The model is no better than the baseline | Weak features; the wrong metric; a bug in feature joins (misaligned rows) | Inspect 20 rows end to end, raw → features; plot the target against the top features; check the join row counts ([Lab 02](../labs/02-pandas-wrangling/README.md)) |
| Good AUC, bad decisions | The threshold isn't chosen from costs; probabilities are uncalibrated; the label is a proxy for what you actually care about | Cost-based threshold ([Lab 04](../labs/04-logistic-regression-metrics/README.md)); reliability diagram ([Lab 12](../labs/12-calibration-conformal-drift/README.md)) |
| Performance decays after deployment | Data drift; training/serving skew (features computed differently online); label delay hiding the problem | PSI and a classifier two-sample test on recent data ([PROD-03](../lessons/production/03-testing-monitoring-drift.md)); log the served features and diff them against the training pipeline |

## B. Deep learning training

**First, the 5-minute sanity checks** (run them before every new training setup):

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
C = 10
model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, C))
xb, yb = torch.randn(32, 3, 8, 8), torch.randint(0, C, (32,))

# 1. The initial loss should be about ln(C): the model starts out not knowing anything.
print(f"initial loss {F.cross_entropy(model(xb), yb).item():.3f} vs ln(C) = {math.log(C):.3f}")

# 2. Overfit ONE batch: the loss must go to ~0. If it doesn't, the bug is in the model/loss/optimizer, not the data.
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"loss after 300 steps on one batch: {loss.item():.4f}")

# 3. Every parameter receives a gradient (none are None or all zeros).
dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
print("parameters with no gradient:", dead)
```

The initial loss is **2.385** against ln 10 = 2.303, which is close enough. One batch is memorized to a loss of **0.0012** in 300 steps, and every parameter gets a gradient. If any of these three checks fails, *stop and fix it first*.

| Symptom | Likely causes | Check |
|---|---|---|
| Loss is NaN or inf | LR too high; log(0) or division by 0 in a custom loss; fp16 overflow; bad input values | Lower the LR 10×; `torch.autograd.set_detect_anomaly(True)`; assert the inputs are finite; use BF16 or a grad scaler ([DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md)) |
| Loss doesn't decrease at all | LR far too low or too high; forgot `zero_grad()`/`step()`; labels misaligned with inputs; gradients not flowing (detached tensor, frozen layers) | The checks above; an LR range test ([DL tricks §1](03-deep-learning-tricks.md#1-find-the-learning-rate-with-a-range-test)); plot a batch with its labels |
| Initial loss ≫ ln C | Bad init scale; the last layer outputs large logits; wrong loss (e.g. softmax applied twice) | Zero or scale down the last layer; pass logits (not probabilities) to `cross_entropy` |
| Train loss falls, val loss rises early | Overfitting; data leakage between augment and val; val preprocessing differs from train | Augmentation, weight decay, smaller model, more data; check that val uses the eval transforms |
| Val metric is great in training, bad at inference | `model.eval()` not called (dropout/BatchNorm in train mode); different preprocessing at inference (normalization, resize, channel order) | Run the same batch through the training-time and inference-time code paths and diff the outputs |
| Loss plateaus, then drops suddenly | Bad init, or too little warmup; the learning rate is too low for the start | Warmup + a higher peak LR; check the init ([DL tricks §2–§4](03-deep-learning-tricks.md#2-warm-up-then-decay-one-cycle-or-cosine)) |
| Training is slow | A data-loading bottleneck (GPU idle); no AMP; small batches; CPU↔GPU syncs (`.item()` every step) | The profiler ([DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md)): if GPU utilization is < 80%, the bottleneck is usually the input pipeline |
| Results differ every run | Unseeded randomness; non-deterministic kernels; a real high variance between seeds | Seed everything; run ≥ 3 seeds before believing a difference |

## C. LLM applications and RAG

| Symptom | Likely causes | Check |
|---|---|---|
| Confident wrong answers | Retrieval missed the fact (most common); the right chunk was retrieved but buried; a prompt that invites guessing | Log the retrieved chunks for every failure: was the answer *in* them? Measure recall@k separately from answer quality ([LLM tricks §4](04-llm-and-retrieval-tricks.md#4-retrieval-hybrid-first-rerank-second-and-contextualize-chunks)) |
| Good on your 5 test questions, bad in use | The eval set is too small or too easy; real users phrase things differently | Collect real queries; grow the eval to ≥ 50 including hard and unanswerable cases ([GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md)) |
| Output format breaks occasionally | Free-text parsing; long outputs truncated at max tokens | Structured outputs / JSON mode; validate with a schema and retry; raise max tokens |
| A prompt change fixed one case, broke others | No regression suite | Re-run the full eval on every prompt change, in CI ([PROD-05](../lessons/production/05-llmops-genai-platforms.md)) |
| The LLM judge disagrees with you | An unvalidated rubric; position or length bias | Measure κ against your labels; randomize the order ([LLM tricks §2](04-llm-and-retrieval-tricks.md#2-use-an-llm-as-a-labeler-then-measure-it-like-one)) |
| Latency or cost too high | Long prompts; no prefix caching; a big model for an easy task | Stable prefix first ([LLM tricks §5](04-llm-and-retrieval-tricks.md#5-put-the-stable-part-of-the-prompt-first)); cascade or distill ([outside the box §11](05-outside-the-box.md#11-route-by-confidence-cascades)) |
| The model follows instructions found *inside* documents | Prompt injection | Treat retrieved and tool content as data; privilege separation; a red-team suite ([GEN-09](../lessons/llms-genai/09-llm-security-safety.md)) |

## D. Production

| Symptom | Likely causes | Check |
|---|---|---|
| Offline metric up, online metric flat or down | The offline metric isn't the business metric; training/serving skew; feedback loops | An A/B test; log the online features and compare; are the offline labels a proxy? |
| Sudden drop in quality | An upstream schema or unit change; a broken feature pipeline; a new category value | Data-quality checks on every batch (nulls, ranges, categories); alert on input drift, not just on output metrics |
| Slow decline | Concept drift; the population changed | A scheduled drift report; retraining cadence; a backtest of "retrain monthly" vs "retrain quarterly" |
| Can't reproduce last month's model | Unpinned data or code versions; unseeded training | Version the data and code together; log the full config and environment ([PROD-02](../lessons/production/02-mlops-in-practice.md)) |

---

## The universal moves

1. **Look at the data**: 20 random rows, 20 errors, and 20 of the most confident predictions.
2. **Shrink the problem** until it's fast: a subset, a smaller model, one batch.
3. **Change one thing at a time**, and keep a log of what you changed and what happened.
4. **Compare against a known-good reference**: a library implementation, a simpler model, yesterday's run.
5. **Write the bug down as a test** once it's fixed (data checks, shape asserts, a regression eval), so it can't come back.
