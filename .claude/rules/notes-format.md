---
paths:
  - "notes/**/*.md"
  - "templates/notes-template.md"
---
# Rule: Study notes format

Every lesson `lessons/<track>/NN-name.md` has study notes at `notes/<track>/NN-name.md` (same filename).
The notes are **original explanations** written for this repo. They are not summaries or copies of the linked resources.
Follow `templates/notes-template.md`.

## Required parts, in order
1. **Title** `# <ID> notes: <Lesson title>` and a nav line: lesson link · all notes · previous notes · next notes (path order).
2. **Header quote**: the reading time (≈ N min) and **"You need"**: links to the *exact sections* of earlier notes that this one builds on.
3. **Where we are**: 1–3 sentences connecting to the previous note. No topic should appear out of nowhere.
4. **Body**: plain words first, then the math, step by step:
   - Define every symbol (use `notes/notation.md` conventions). Derive or motivate every formula; don't just state it.
   - At least one **worked example with concrete numbers**, small enough to check by hand.
   - **Runnable code** (```python blocks: NumPy / scikit-learn / pandas / PyTorch-CPU) that reproduces the worked numbers and demonstrates the key claims.
5. **Pitfalls & misconceptions**: 4–6 bullets.
6. **Cheat sheet**: a table of the formulas and rules.
7. **Answer sketches for the lesson's self-check**: one `<details>` per question in the lesson's *Check your understanding*, in the same order.
8. **Where this leads**: a bridge to the next note in the path.

## Truthfulness
- **Every number in the prose that comes from code must match the code's output.** Run the code, then write or adjust the text.
  If a demo doesn't show what you expected, change the demo or the claim. Never keep a claim the code contradicts.
- Synthetic demos must say they are synthetic when real-world magnitudes would differ.
- Don't invent citations, chapter numbers, or benchmark results. Link only URLs already in `resources/verified-urls.tsv` (or verify new ones).

## Math syntax (GitHub renders with MathJax; we validate with KaTeX)
- Display math: fenced ```math blocks. **Never `$$`.**
- Inline math: `$...$`, with no space just inside the dollars, and not directly followed by a letter or digit.
- Inline math must not use `\{ \}` (use `\lbrace \rbrace`), `<` before a letter (use `\lt`), `\\`, or `|` inside table rows (use `\lvert \rvert`, `\mid`, `\Vert`).
- Write currency as "USD"/"EUR" in prose. A bare `$` starts math.

## Code
- Cells within a note share one namespace and run top to bottom. They must finish in well under a minute on a laptop CPU.
- Seed every random generator, so the printed numbers are reproducible.
- Illustrative code that can't run on CPU (CUDA, Triton) goes in a ```py block, which the checker skips.

## Before committing
`python3 scripts/check_notes.py notes/` must pass (structure, links, math lint, KaTeX parse if `npm install katex`, and code execution).
