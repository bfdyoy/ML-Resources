---
name: expand-toolbox
description: Add new concepts, papers, or exercises to the toolbox/, papers/, and exercises/ reference layers, with verified links and cross-links to lessons. Use when the user wants broader coverage of a topic, asks to add a paper or resource to the library, or wants more practice material.
---

# Expand the toolbox

## Decide where it goes
| Kind of item | File |
|---|---|
| A concept with explainer/reading/practice | `toolbox/NN-<domain>.md` (add a table row in the right section) |
| A paper | `papers/NN-<topic>.md` (table row: title+link, year, why, level, "after" lesson). Mark ⭐ only if it's essential. |
| A drill / puzzle / problem set | `exercises/drills-and-puzzles.md` or `exercises/course-assignments.md` |
| An implement-it-yourself task | `exercises/from-scratch-ladder.md` (it must have a testable **check**) |
| A project idea | `exercises/projects-and-competitions.md` |
| A free book | the bookshelf in `toolbox/README.md` |

## Steps
1. Search the repo first (`grep -rn`) to avoid duplicates. Extend an existing row rather than adding a near-duplicate.
2. Research and verify per `.claude/rules/link-policy.md`. For arXiv, confirm the ID ↔ title pair.
3. Fill every column you can: **Start here** (intuition) → **Go deeper** (text) → **Practice** → **Paper**. Use `—` for an empty cell, and never pad with weak resources.
4. Add each new URL to `resources/verified-urls.tsv`, then run `python3 scripts/audit_urls.py`.
5. If the item belongs to a lesson's topic, make sure that lesson's "Toolbox, papers & practice" section points to the right page.
6. If you added papers, update the count in `papers/README.md`. If an item is a must-read, consider the top-30 list.
