# Course 2: Deep Learning, an 11-week syllabus

[← Courses](README.md) · Path: [Path 2: Deep Learning](../paths/02-deep-learning.md) · Labs: [index](../labs/README.md) · Cards: [deep-learning deck](flashcards/README.md)

> **Pace** ≈ 8 h/week for 11 weeks: 9 teaching weeks, 1 review/midterm week, and 1 capstone week. **Before you start:** CORE-01…04, or equivalent.
> **Running project:** one small model that grows. Karpathy's *Zero to Hero* builds one codebase from micrograd to GPT, and this course does the same.

The [weekly loop](README.md#32-the-weekly-loop-about-8-hours) applies every week.

## How this course is shaped

- **Use it, then open it up** (fast.ai). Week 1 starts with a two-hour fine-tune of a pretrained image model, deployed as a demo, *then* opens the box with backprop by hand.
- **Two implementations of everything** (D2L). From scratch in the labs, then with `torch.nn` in the lessons. At the end, compare the two and check that they match to 1e-6.
- **Gradient checks are non-negotiable** (CS231n). Every hand-written backward pass is verified against finite differences.
- **Overfit one batch first** ([Karpathy's recipe](http://karpathy.github.io/2019/04/25/recipe/)). Every training run in this course starts by driving the loss on 1 batch to ≈0. It's in the [debugging playbook](../playbook/06-debugging-playbook.md#b-deep-learning-training).
- **Projects from week 1** (Neuromatch). You choose the capstone option in week 1, not week 10.

---

## Week 1: Whole game, then backprop by hand
- **Whole game (2 h):** fine-tune a pretrained ResNet on 2–3 classes of your own images (follow DL-04's fine-tuning step), and put it in a Gradio demo. Don't try to understand everything yet.
- **Do:** [DL-01](../lessons/deep-learning/01-neural-networks-from-scratch.md) with its [notes](../notes/deep-learning/01-neural-networks-from-scratch.md), first half.
- **Lab:** [Lab 07: micrograd](../labs/07-micrograd/README.md).
- **Project:** choose your capstone option, (a), (b) or (c) from the [path capstone](../paths/02-deep-learning.md#capstone), and get its data.

## Week 2: Backprop, vectorized
- **Warm-up:** DL-01 cards + CORE-02 cards. *Interleave:* "Logistic regression is a one-layer net. What is its gradient with respect to the logits?"
- **Do:** DL-01, second half.
- **Lab:** [Lab 08: MLP backprop with a gradient check](../labs/08-mlp-backprop/README.md).
- **Project:** a tiny version of your capstone model that **overfits one batch**.

## Week 3: PyTorch fluency
- **Warm-up:** DL-01 cards. *Interleave:* "Your gradient check gives a relative error of 1e-2. Name three suspects."
- **Do:** [DL-02](../lessons/deep-learning/02-pytorch-fluency.md).
- **Exercise:** rewrite Lab 08's network with `nn.Module` and confirm the gradients match your NumPy version (the D2L "concise implementation" move).
- **Project:** a modular `train.py` with config, logging, and seeding.

## Week 4: Training deep networks well
- **Warm-up:** DL-02 + DL-01 cards. *Interleave:* "`model.eval()` vs `torch.no_grad()`: what does each one switch off?"
- **Do:** [DL-03](../lessons/deep-learning/03-training-deep-networks.md).
- **Playbook:** [Deep learning tricks](../playbook/03-deep-learning-tricks.md) §1–§4 (LR range test, warmup + one-cycle/cosine, init the last-layer bias to the log prior, zero-init residuals).
- **Project:** an LR range test and a schedule. Log the before/after.

## Week 5: CNNs
- **Warm-up:** DL-03 + DL-01 cards. *Interleave:* "BatchNorm with batch size 2: what breaks, and what would you use instead?"
- **Do:** [DL-04](../lessons/deep-learning/04-cnns-computer-vision.md).
- **Playbook:** deep learning tricks §5–§8 (label smoothing, mixup/CutMix, EMA/SWA/soups, TTA).
- **Project:** a strong baseline for your capstone, with 3 tricks you plan to ablate.

## Week 6: Review week + **midterm**
- **Interleaved quiz:** 20 cards from DL-01…04 and CORE-03/04.
- **Redo from a blank file:** Lab 08's `softmax_cross_entropy` forward and backward.
- **Midterm (rubric below):** a short report on your capstone baseline. Include the curves, the "overfit one batch" proof, and an error analysis of 20 mistakes.

## Week 7: Embeddings, language modeling & sequences
- **Warm-up:** DL-04 + DL-02 cards. *Interleave:* "A 3×3 conv layer and an MLP over 3×3 patches: what is shared?"
- **Do:** [DL-05](../lessons/deep-learning/05-embeddings-sequences-attention.md).
- **Project:** if you chose option (c), your character-level model now trains end to end.

## Week 8: Transformers
- **Warm-up:** DL-05 + DL-03 cards. *Interleave:* "Why does a char-level MLP language model need a fixed context length, and how does attention remove that limit?"
- **Do:** [DL-06](../lessons/deep-learning/06-transformers.md).
- **Lab:** [Lab 09: attention](../labs/09-attention/README.md) (masked scaled dot-product attention and multi-head attention, checked against `F.scaled_dot_product_attention`).
- **Project:** options (b) and (c) now have their transformer.

## Week 9: Making training fast
- **Warm-up:** DL-06 + DL-04 cards. *Interleave:* "Which costs more memory at long context: the attention scores or the weights? Why?"
- **Do:** [DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md).
- **Playbook:** [gradient accumulation and progressive resizing](../playbook/03-deep-learning-tricks.md#9-progressive-resizing-and-gradient-accumulation).
- **Project:** a speed-up table for your capstone's training loop.

## Week 10: Modern architectures (+ GNNs, optional)
- **Warm-up:** DL-07 + DL-05 cards. *Interleave:* "Mixed precision: why keep a float32 master copy of the weights?"
- **Do:** [DL-08](../lessons/deep-learning/08-modern-architectures-moe-ssm.md). [DL-09](../lessons/deep-learning/09-graph-neural-networks.md) is optional this week, or take it later as a module.
- **Project:** the capstone ablation runs (3 changes, ≥2 seeds each).

## Week 11: **Capstone**
- **Interleaved quiz:** 30 cards from the whole deck.
- **Capstone:** a short paper-style write-up (rubric below).

---

## Midterm rubric (week 6)

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Sanity checks | None | Loss decreases | Initial loss ≈ log(C) checked, one batch overfit, gradients nonzero in every layer |
| Baseline | None | Trains | Trains, with a simple non-DL baseline for comparison |
| Curves | None | Train loss only | Train/val loss and metric vs steps, with the LR shown |
| Error analysis | None | Accuracy only | 20 mistakes, grouped, each with a hypothesis |
| Plan | None | A list of tricks | 3 ablations, each with a predicted effect |

## Capstone rubric (week 11)

| Criterion | What "2" looks like |
|---|---|
| Reproduction | Matches a reference number (paper or known baseline) within a stated tolerance, or explains the gap |
| Training code | Your own loop: seeded, logged, resumable from a checkpoint |
| Ablations | ≥3 changes, ≥2 seeds each, reported as mean ± spread |
| Analysis | Curves, failure cases, and a paragraph on *why* each ablation did what it did |
| Efficiency | Throughput measured; at least one speed-up applied (AMP, compile, data loading) |
| Write-up | Paper-style: abstract, method, results table, limitations |

Pass: ≥ 9/12.

## Assessment summary

| Component | Weight |
|---|---|
| Labs 07–09 passing | 20% |
| Weekly self-checks from memory | 15% |
| Midterm | 25% |
| Capstone | 40% |
