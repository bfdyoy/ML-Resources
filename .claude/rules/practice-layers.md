---
paths:
  - "labs/**"
  - "playbook/**"
  - "courses/**"
---
# Rule: Labs, playbook, and courses

## Labs (`labs/NN-name/`)
1. Four files: `README.md` (lesson link, time, what you practise, a table of functions → what checks them, the run command, a Bonus),
   `solution.py` (the source of truth), `exercise.py` (**generated**: never edit by hand), and `test_labNN.py` (a unique file name per lab).
2. `solution.py` docstrings state the **contract** (shapes, conventions, tie-breaking) and, early in the sequence, **subgoals** and hints.
   Fade the scaffolding in later labs. Mark a function `(given)` in its docstring to keep it whole in the exercise (worked examples, boilerplate).
3. Tests compare against an **independent reference** wherever one exists: scikit-learn, PyTorch autograd or `F.*`, finite differences, or values
   computed by hand in a comment. Include at least one test that catches the classic mistake for the topic (leakage, `=` vs `+=`, wrong axis, a future-token leak).
4. Tests are seeded, CPU-only, and fast (the whole suite runs in seconds). They use the `impl` fixture from `labs/conftest.py`.
5. After editing: `python3 scripts/make_lab_stubs.py --force labs/<lab>`, then `LAB_IMPL=solution python3 -m pytest labs/<lab>` must pass,
   and `python3 -m pytest labs/<lab>` must fail with `NotImplementedError` (not with import errors).
6. Add the lab to `labs/README.md`, to the lesson's "Toolbox, papers & practice" section, and to the course week that uses it.

## Playbook (`playbook/NN-*.md`)
1. Every trick follows **when → why → how (demo) → the trap**. Decision guides use "pick A when / pick B when / settle it with".
2. A demo is a seeded, CPU-fast ```python block that shows the *mechanism*. Say when data is synthetic. Every number in the prose must
   match the demo's output (`python3 scripts/check_notes.py playbook/` runs the demos).
3. Cite sources with verified URLs. Prefer pairing a paper with a practical explainer. Never present a trick as universally helpful: say how to measure it.
4. End each page with a `## Cheat sheet` table. Its rows become flashcards (`python3 scripts/make_flashcards.py`).

## Courses (`courses/NN-*.md`)
1. One syllabus per path, week by week. Each week names: the lesson and notes, a warm-up (two earlier lessons' flashcard tags plus one
   interleaving question that mixes topics), the lab (if any), a playbook reading (if any), and a project milestone.
2. Week 1 is a **whole-game** pass. A review week every 4–6 weeks. A midterm checkpoint and a capstone, each with a published rubric.
3. When a lesson, lab, or playbook section is renamed or moved, update every syllabus that links to it (`check_notes.py` checks links and anchors).
