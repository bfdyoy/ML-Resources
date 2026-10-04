# Lab 05: CV splitters, tree splits & leak-free target encoding

[← Labs](../README.md) · Lessons: [CORE-04](../../lessons/core-ml/04-generalization-validation-regularization.md) (part A) · [CORE-05](../../lessons/core-ml/05-trees-and-ensembles.md) (part B) · [CORE-07](../../lessons/core-ml/07-feature-engineering-pipelines-leakage.md) (part C)

**Time** ≈ 2.5 h in three sittings · **You'll practise:** writing the splitters you use every day, the exact computation a tree does at each node, and the single most common tabular leak and its fix.

| Part | Function | Checked against |
|---|---|---|
| A | `kfold_indices` | partition properties; seeded reproducibility |
| A | `group_kfold_indices` | no group in both train and val; balanced folds |
| B | `gini`, `best_split` | hand values; the root split of `sklearn`'s `DecisionTreeClassifier(max_depth=1)` |
| C | `oof_target_encode` | **flipping a row's own label never changes its encoding** |

```bash
pytest labs/05-trees-and-cross-validation -k "kfold"        # part A only
pytest labs/05-trees-and-cross-validation -k "gini or split" # part B
pytest labs/05-trees-and-cross-validation -k "oof"           # part C
```

**Bonus:**
1. Make `best_split` O(n log n): sort once, then sweep with cumulative class counts instead of re-counting at each threshold.
2. Replace `oof_target_encode` with an in-sample encoder (fit and transform on all rows). Use it as the only feature for a GBM on
   **pure-noise** labels with 100 categories, and compare its training AUC with the out-of-fold version. ([Playbook: the noise-feature null](../../playbook/02-tabular-tricks.md#3-the-noise-feature-null).)
