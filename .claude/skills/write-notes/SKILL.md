---
name: write-notes
description: Write or update the study notes for a lesson (notes/<track>/<file>.md) — an original, step-by-step explanation of the lesson's ideas and math with worked examples, runnable code whose output is verified, pitfalls, a cheat sheet, and answer sketches for the lesson's self-check questions. Use when the user asks for explanations, the math behind a topic, notes for a lesson, or after a lesson's content changes.
---

# Write study notes

## Inputs
- The lesson file `lessons/<track>/<file>.md` (its learning goals and *Check your understanding* questions define the scope).

## Steps

1. Read `.claude/rules/notes-format.md`, `templates/notes-template.md`, `notes/notation.md`, the lesson, and the
   **previous and next notes in path order** (so the "Where we are" / "Where this leads" bridges are true).
2. List the concepts the learning goals and the self-check questions need. Each must be explained in the body,
   and each answer sketch must point to the section that explains it.
3. Draft the body: plain words → math (define symbols, derive or motivate every formula) → a worked example with numbers →
   code that reproduces the example and demonstrates the main claims (seeded, CPU, fast).
4. **Run the code** (`python3 scripts/check_notes.py notes/<track>/<file>.md`). Then reconcile: every number the prose quotes
   must match the output. If a demo contradicts the text, fix the demo or the claim, never paper over it.
5. Write the pitfalls, the cheat sheet, the answer sketches (one `<details>` per question, same order), and the bridge.
6. Make sure the lesson's study plan has step **0 · Primer** linking to the notes, `notes/README.md` lists the note
   (reading time + one-line summary), and the README lesson index has the 📘 link.
7. Run `python3 scripts/check_notes.py notes/` (with `npm install katex` available for the KaTeX parse) and
   `python3 scripts/audit_urls.py`. Both must pass.

## Quality checklist
- [ ] Nothing used before it is explained or linked ("You need" points to exact sections)
- [ ] Every formula derived or motivated; every symbol defined
- [ ] At least one worked example with concrete numbers
- [ ] Code runs top to bottom; every quoted number matches its output
- [ ] Every self-check question has an answer sketch
- [ ] Math passes the lint (no `$$`, no `\{` inline, `\lt` instead of `<letter`, no `|` in table math)
