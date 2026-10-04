---
name: build-lab
description: Create or update a practice-layer artifact — a test-driven lab (labs/NN-*/ with solution, generated exercise stubs and pytest tests), a playbook trick or decision guide with a runnable demo (playbook/), or a week in a course syllabus (courses/). Use when the user asks for exercises to implement, "build it yourself" practice, tips and tricks, A-vs-B comparisons, or a study schedule.
---

# Build a lab, a playbook entry, or a syllabus week

Read `.claude/rules/practice-layers.md` first.

## A lab
1. Pick the lesson, and the 4–7 functions that capture its core mechanics (the things a learner should be able to write from memory).
2. Write `labs/NN-name/solution.py`: a clear module docstring, then each function with a docstring stating the contract, its subgoals, and a hint.
3. Write `test_labNN.py`: one test per function against an independent reference, plus one test for the topic's classic mistake. Seed everything.
4. `LAB_IMPL=solution python3 -m pytest labs/NN-name` must pass. Then run `python3 scripts/make_lab_stubs.py labs/NN-name`, and check that
   `python3 -m pytest labs/NN-name` fails only with `NotImplementedError`.
5. Write the lab's `README.md` (template: any existing lab), and add it to `labs/README.md`, the lesson's practice section, the course week, and `PROGRESS.md`.

## A playbook entry
1. Decide the page (choosing algorithms, tabular, DL, LLM & retrieval, outside the box, debugging), and write when → why → how → the trap.
2. Write the smallest seeded demo that shows the mechanism, run it (`python3 scripts/check_notes.py playbook/<page>.md`), and then write the numbers into the prose.
3. Verify any new URL (link policy), add it to `resources/verified-urls.tsv`, and add the source to `resources/catalog.md` §11 (papers to `papers/`).
4. Add a cheat-sheet row, then run `python3 scripts/make_flashcards.py`.

## A syllabus week
Follow the weekly loop in `courses/README.md` §3.2: warm-up tags + an interleaving question, notes → lesson, lab, playbook reading, and a milestone that grows the running project.

## Before committing
`python3 scripts/check_notes.py notes/ playbook/ courses/ labs/`, `LAB_IMPL=solution python3 -m pytest labs/`, `python3 scripts/make_lab_stubs.py --check`,
`python3 scripts/make_flashcards.py --check`, and `python3 scripts/audit_urls.py`.
