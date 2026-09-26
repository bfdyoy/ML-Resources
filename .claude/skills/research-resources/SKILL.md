---
name: research-resources
description: Research and shortlist the best learning resources (book chapters, visual essays, notebooks, videos) for a machine learning topic, verify their URLs, and add them to resources/catalog.md. Use when the user asks to find, compare, or refresh resources for an ML topic, or when a lesson needs better material.
---

# Research resources for an ML topic

## Inputs
- `topic`: e.g. "gradient boosting", "LoRA fine-tuning", "calibration"
- optional `lesson`: lesson ID it's for (e.g. `CORE-05`)

## Steps

1. **Check what we already have.** Grep `resources/catalog.md` and `lessons/` for the topic.
   Re-use existing resources where they cover the topic. Don't add near-duplicates.

2. **Search in layers** (use WebSearch; WebFetch where the host is reachable):
   - *Books with free online editions*: ISLP, UDL (Prince), D2L, MML, Nielsen, Bishop DLFC,
     Murphy PML, Molnar IML, Sutton & Barto, FPP3, SLP3 (Jurafsky & Martin). Find the exact chapter/section.
   - *Visual/interactive*: MLU-Explain, Distill, Polo Club explainers, Seeing Theory, setosa.io, 3Blue1Brown lessons.
   - *Long-form explainers*: Jay Alammar, Lilian Weng, Sebastian Raschka, Chip Huyen, Eugene Yan, Karpathy's blog.
   - *Hands-on*: official notebooks for the books above, Kaggle Learn, learnpytorch.io, Hugging Face courses, scikit-learn MOOC.
   - *Video complements*: 3Blue1Brown, StatQuest, Karpathy, Stanford/MIT lecture playlists.
   Useful query shapes: `"<topic>" chapter site:<book-site>`, `<topic> visual explanation interactive`,
   `<topic> from scratch notebook`.

3. **Score each candidate** against `.claude/rules/resource-quality.md`
   (explains-not-shows, credibility, correctness, access, level). Drop anything that fails a rule.

4. **Verify URLs** per `.claude/rules/link-policy.md`. Record the method (`fetch` or `search`) and today's date.

5. **Pick one primary per step type** (Intuition / Read / Build / Watch). List the rest as "Go deeper".

6. **Update `resources/catalog.md`**: add rows in the right section, keeping alphabetical order within the section.

## Output
A short report with the primary pick per step type (and why), the alternatives,
anything you rejected (and why), and the catalog rows you added.
