# The playbook: tips, tricks, and A-vs-B decisions

[← Back to the README](../README.md) · [Courses](../courses/README.md) · [Labs](../labs/README.md) · [Toolbox](../toolbox/README.md)

The lessons teach *how each method works*. The playbook is about **judgment**: which method to pick, the small moves that make a model noticeably better,
the unexpected ways to use a tool, and how to find the bug when the numbers look wrong. It's the material that experienced practitioners carry in their heads
and that textbooks rarely write down.

Every trick follows the same format: **when** to use it → **why** it works → **how** → **the trap**. Wherever possible, there's a small, seeded, runnable demo that shows the mechanism.
All demos are checked by `scripts/check_notes.py`, so every number quoted in the text matches the code's output.

| Page | What's in it | Read after |
|---|---|---|
| [1. Choosing algorithms, A vs B](01-choosing-algorithms.md) | 14 decision guides: linear vs GBM vs kNN, RF vs GBM, clustering, prompting vs RAG vs fine-tuning, metrics, optimizers, vision, search, forecasting, anomalies, HPO, imbalance, explanations | CORE-05 (then dip in as needed) |
| [2. Tabular tricks](02-tabular-tricks.md) | Adversarial validation, out-of-fold features, the noise-feature null, target transforms, monotone constraints, kNN features, residual boosting, hill-climbing blends, pseudo-labels, quantiles, group aggregates | CORE-07 |
| [3. Deep learning tricks](03-deep-learning-tricks.md) | LR range test, warmup + one-cycle, log-prior bias init, zero-init residuals, label smoothing, mixup/CutMix, EMA/SWA/soups, TTA, gradient accumulation, discriminative LRs, SAM, distillation | DL-03 |
| [4. LLM & retrieval tricks](04-llm-and-retrieval-tricks.md) | Self-consistency, LLM-as-labeler with κ, reasoning-then-answer, hybrid + rerank + contextual chunks, prefix caching, lost in the middle, distillation, few-shot from failures | GEN-03 |
| [5. Outside the box](05-outside-the-box.md) | No-model baselines, classifier two-sample tests, compression as similarity, reframing, error models for slice discovery, pinball loss for asymmetric costs, random projections, frozen embeddings, shuffled-label tests, planted signals, cascades | CORE-07 |
| [6. Debugging playbook](06-debugging-playbook.md) | Symptom → cause → check tables for tabular ML, DL training, LLM apps, and production, plus the 5-minute DL sanity checks | Any time something looks wrong |

**How to use it:** read a page once after its "read after" lesson, then come back when you face the decision. The syllabi in [courses/](../courses/README.md) schedule specific sections into specific weeks.
When a trick helps on a real project, write down *how much* it helped. Your own measured experience is the real playbook.
