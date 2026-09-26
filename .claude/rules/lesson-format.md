---
paths:
  - "lessons/**/*.md"
  - "templates/**/*.md"
---
# Rule: Lesson format

Every lesson file follows `templates/lesson-template.md`. Non-negotiables:

1. **Header block**: ID, track, estimated time, level, prerequisites (lesson IDs).
2. **Why this matters**: 2–4 sentences that tie the concept to real practice.
3. **Learning goals**: 3–6 bullets, each starting with a verb ("Explain…", "Implement…", "Diagnose…").
4. **Study plan**: ordered steps, each labelled with one of:
   - **Intuition** — a visual, interactive, or short explainer (≤ 30 min)
   - **Read** — the primary written chapter/section, with a time estimate
   - **Watch** — optional video complement
   - **Build** — notebook, exercise, or coding task
   Each step names ONE primary resource, gives an exact scope (chapter/section/pages), and a time estimate.
5. **Check your understanding**: 4–8 questions the learner should be able to answer
   without notes. Mix conceptual ("why") and practical ("what would you do if…") questions.
6. **Mini-project**: a small concrete task (1–3 h) with a suggested dataset.
7. **Go deeper**: optional alternatives and advanced material, max 5 items.
8. **Math refresher** (if needed): link to a `lessons/math/` file with the specific section.

Style:
- Use plain, friendly language. Explain *why* the resource was chosen in a short phrase.
- Time estimates are for an intermediate learner reading carefully, not skimming.
- Never paste large chunks of copyrighted text. Summarise and link.
- Resource names must match their entry in `resources/catalog.md`.
