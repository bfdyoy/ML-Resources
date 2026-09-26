# CLAUDE.md — ML-Resources

This repo is a **curated, opinionated curriculum for learning machine learning**.
It does not host content. It finds the best free (or clearly marked paid)
resources on the internet and puts them in order as **lessons** and **paths**, so
learning ML is as easy as possible *and still makes sense*.

## Who it's for

- **Primary learner:** already at an **intermediate** level. They know some
  Python, have trained a model or two, and have seen gradient descent. They
  want depth and intuition, not a from-zero tutorial.
- **Learning style:** *not video-only*. Prefer **book chapters and written
  explanations with worked examples**, interactive visual essays, and runnable
  notebooks. Use videos as a complement, never as the only way into a topic.

## Repository map

```
README.md                 Start here: overview, how to use, path picker
PROGRESS.md               Learner's personal checklist
paths/                    Learning paths (ordered sequences of lessons)
lessons/<track>/NN-*.md   One lesson = one concept cluster, ~5–12 hours of work
  core-ml/                Classical ML done properly
  deep-learning/          Neural nets → CNNs → transformers → performance, modern architectures, GNNs
  vision/                 Detection/segmentation, ViT & self-supervised, CLIP/VLMs, 3D
  llms-genai/             LLMs, fine-tuning, RAG/retrieval, agents, evals, post-training, inference, security, diffusion
  production/             ML systems design, MLOps, testing & drift, distributed training, LLMOps
  electives/              Time series, Bayesian, RL, recsys, causal, anomaly, mech interp, GPU, audio, tabular DL
  math/                   Just-in-time math refreshers linked from lessons
toolbox/NN-*.md           Reference shelf: every concept → intuition, deeper reading, practice, paper
papers/NN-*.md            ~250 verified papers by topic, with a 30-paper must-read list
exercises/*.md            Drills, 45-rung from-scratch ladder, assignments, projects, interview prep
resources/catalog.md      Resources used in lessons, typed and tagged
resources/verified-urls.tsv  Verification record for EVERY URL in the repo (method + evidence)
templates/lesson-template.md
scripts/check_links.py    Live link checker (stdlib only; run where network allows)
scripts/audit_urls.py     Fails if any URL lacks a verification record (runs anywhere)
.claude/rules/            Standing rules (quality bar, lesson format, link policy)
.claude/skills/           Repeatable workflows (research, build lesson, check links, review path)
.claude/agents/           Subagents (resource scout, curriculum architect, link auditor, pedagogy reviewer)
```

## The four layers
1. **Paths → lessons**: the guided route (what to do next, in what order).
2. **Toolbox**: the reference shelf for every concept, including ones no lesson covers.
3. **Papers**: primary sources, each paired with an explainer and a "read after" lesson.
4. **Exercises**: practice at every scale, from 20-minute drills to capstones.
Lessons link down into the other three (see the "Toolbox, papers & practice" section of each lesson).

## Core principles (read `.claude/rules/` for detail)

1. **Intuition → Read → Build → Check.** Every lesson starts with something
   visual or intuitive, then a written chapter with examples, then hands-on
   code, then self-check questions. This order is the core of the repo.
2. **One primary resource per step.** Don't hand the learner a list of 10 links.
   Choose the best one, and put the alternatives under "Go deeper".
3. **Chapter-level precision.** Link to the specific chapter or section, and
   estimate the reading time. "Read ISLP" is useless. "Read ISLP §8.2 (≈45 min)" is useful.
4. **Free first.** Mark paid resources `[paid]`. A paid resource can be *primary*
   only when no free resource comes close, and a free fallback must be listed.
5. **Verified links only.** Every URL must be confirmed to exist, via a fetch or
   a search result showing the exact URL. Never invent a URL or a chapter
   number. If unsure, link to the resource's landing page and say what to look for.
6. **Coherent sequencing.** A lesson may only rely on concepts from earlier
   lessons in the same path, or on explicitly listed prerequisites.

## Working conventions

- Lesson files: `lessons/<track>/NN-kebab-title.md`, using `templates/lesson-template.md`.
- New resources go into `resources/catalog.md` **first**, and lessons reference them from there.
- Keep lesson IDs stable (`CORE-03`, `DL-02`, `CV-01`, …). Paths refer to lessons by ID + link.
- **Coverage rule:** every toolbox section should be taught by at least one lesson. When you add a toolbox topic, add or extend a lesson.
- When you change a lesson, update every path that includes it.
- **Every new URL gets a row in `resources/verified-urls.tsv`** (method codes in `.claude/rules/link-policy.md`).
  `python3 scripts/audit_urls.py` must pass before committing.
- Run `python3 scripts/check_links.py` before committing if network allows.
  In restricted sandboxes, many hosts are blocked. Treat those results as
  "unknown", not "broken". When arxiv.org is blocked, verify IDs against GitHub citation
  corpora (link-policy §8).
- Commit messages: imperative, scoped (e.g. `lessons: add DL-04 CNNs`).

## Workflows

| Task | Use |
|---|---|
| Find resources for a topic | skill `research-resources`, or agent `resource-scout` |
| Write or rewrite a lesson | skill `build-lesson` |
| Validate URLs | skill `check-links`, or agent `link-auditor` |
| Check a path's ordering and difficulty | skill `review-path`, or agent `pedagogy-reviewer` |
| Restructure paths or add a new track | agent `curriculum-architect` |
| Add concepts, papers, or exercises to the toolbox | skill `expand-toolbox` |
