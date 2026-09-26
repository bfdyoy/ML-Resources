# DL-01: Neural Networks from Scratch & Backpropagation

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~8 h | L2 | CORE-02, CORE-03 · math: [MATH-02](../math/02-calculus-optimization.md) (chain rule) |

## Why this matters
Backprop is the single algorithm under all of deep learning. If you've built it once yourself, frameworks
stop being magic. Then vanishing gradients, exploding losses, and "why won't it train" become debuggable.

## Learning goals
By the end you can:
- Describe an MLP as a composition of linear maps and non-linearities, and say why depth adds expressive power.
- Derive backprop for a small MLP using the chain rule on a computational graph.
- Implement a tiny autograd engine (micrograd-style), and use it to train a classifier.
- Explain the role of the loss function (MSE vs cross-entropy) from a maximum-likelihood view.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [3Blue1Brown: But what is a neural network?](https://www.3blue1brown.com/lessons/neural-networks/) → [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/) → [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus/) | The three lesson pages (each has the video plus a written version) | 1 h |
| 2 | **Read** | [Nielsen, *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) | [Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) (skim, since you know most of it) and **[Ch. 2 "How the backpropagation algorithm works"](http://neuralnetworksanddeeplearning.com/chap2.html)** (read carefully, with pen and paper) | 2 h |
| 3 | **Read** | [UDL](https://udlbook.github.io/udlbook/) (Prince) | Ch. 3 "Shallow Neural Networks", Ch. 4 "Deep Neural Networks", Ch. 5 "Loss Functions". Do the Chapter 3–5 notebooks from the [repo](https://github.com/udlbook/udlbook). | 2.5 h |
| 4 | **Build** | [Karpathy, Zero to Hero, Lecture 1: micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) ([repo](https://github.com/karpathy/nn-zero-to-hero)) | Code along. Don't just watch. Build `Value`, `backward()`, then an MLP. | 2.5 h |

**Notes for the learner:** This is the one lesson where the "Build" step is a video, and that's on purpose. Karpathy
codes live, and typing along with him *is* the exercise. Read Nielsen Ch. 2 *before* the video, so the video is
confirming something you already worked out.

## Check your understanding
1. Why does an MLP with no activation functions collapse to a linear model?
2. Walk through the backward pass for `L = (w·x + b − y)²`, one node at a time.
3. Why do we accumulate (`+=`) gradients in micrograd instead of assigning them?
4. Show that minimizing cross-entropy is the same as maximizing likelihood under a categorical model.
5. What's the computational cost of backprop compared with a forward pass, and why is it so cheap?
6. *(debug)* Your hand-written backprop gives gradients that differ from a finite-difference check by 30%. How do you find the bug?

## Mini-project
**Task:** Train your micrograd MLP on `sklearn.datasets.make_moons`, and plot the decision boundary. Then reimplement the
same network in pure NumPy with vectorised backprop, and check that its gradients match micrograd's.
**Deliverable:** A notebook with a gradient-check cell (finite differences vs your backprop).

## Go deeper
- [Nielsen Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html): a visual proof that neural nets can compute any function.
- [D2L](https://d2l.ai/): the chapter "Multilayer Perceptrons", especially the section on forward/backward propagation and computational graphs.
- [Karpathy Lecture 5 "Becoming a Backprop Ninja"](https://github.com/karpathy/nn-zero-to-hero): manual backprop through a full MLP with BatchNorm.
