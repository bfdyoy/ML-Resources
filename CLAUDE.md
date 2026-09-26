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
paths/                    Learning paths (ordered sequences of lessons)
lessons/<track>/NN-*.md   One lesson = one concept cluster, 3–8 hours of work
  core-ml/                Classical ML done properly
  deep-learning/          Neural nets → CNNs → transformers
  llms-genai/             LLMs, fine-tuning, RAG, evals, diffusion
  production/             ML systems design + MLOps
  electives/              Time series, Bayesian ML, RL
  math/                   Just-in-time math refreshers linked from lessons
resources/catalog.md      Every resource used, typed and tagged (source of truth)
templates/lesson-template.md
scripts/check_links.py    Link checker (stdlib only)
.claude/rules/            Standing rules (quality bar, lesson format, link policy)
.claude/skills/           Repeatable workflows (research, build lesson, check links, review path)
.claude/agents/           Subagents (resource scout, curriculum architect, link auditor, pedagogy reviewer)
```

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
- Keep lesson IDs stable (`CORE-03`, `DL-02`, …). Paths refer to lessons by ID + link.
- When you change a lesson, update every path that includes it.
- Run `python3 scripts/check_links.py` before committing if network allows.
  In restricted sandboxes, many hosts are blocked. Treat those results as
  "unknown", not "broken".
- Commit messages: imperative, scoped (e.g. `lessons: add DL-04 CNNs`).

## Workflows

| Task | Use |
|---|---|
| Find resources for a topic | skill `research-resources`, or agent `resource-scout` |
| Write or rewrite a lesson | skill `build-lesson` |
| Validate URLs | skill `check-links`, or agent `link-auditor` |
| Check a path's ordering and difficulty | skill `review-path`, or agent `pedagogy-reviewer` |
| Restructure paths or add a new track | agent `curriculum-architect` |
