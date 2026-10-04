---
name: build-lesson
description: Write or rewrite a lesson file in lessons/ from the template, sequencing intuition, then reading, then hands-on build, then self-check, using verified resources from the catalog. Use when the user asks to create a lesson, expand a topic into a lesson, or improve an existing lesson.
---

# Build a lesson

## Inputs
- `id` (e.g. `DL-07`), `title`, `track` (core-ml | deep-learning | vision | llms-genai | production | electives | math)
- Prerequisite lesson IDs

## Steps

1. Read `templates/lesson-template.md`, `.claude/rules/lesson-format.md`, and the lessons
   immediately before and after this one in its path(s). Match their depth and tone.
2. Write 3–6 **learning goals** before choosing resources. Resources serve the goals, not the reverse.
3. Pull resources from `resources/catalog.md`. If a goal isn't covered, run the
   `research-resources` skill first.
4. Build the **study plan** in this order: Intuition (short, visual) → Read (primary chapter,
   exact sections) → Build (notebook/exercise) → optional Watch. Put a time estimate on every step.
   Target total: 6–10 hours including the ~1 h primer (hands-on-heavy lessons may reach 13).
5. Write **Check your understanding** questions that test the goals. At least one question
   should be "debug this situation" style.
6. Design a **mini-project** with a named public dataset (sklearn built-ins, Kaggle, UCI, Hugging Face Datasets).
7. Add **Go deeper** (≤ 5 items) and a **Math refresher** link if needed.
8. Add step **0 · Primer** linking to `notes/<track>/<same-filename>.md`, and write those study notes with the
   `write-notes` skill (every lesson has notes; the self-check answers live there).
9. Update every `paths/*.md` that should include the lesson, the lesson index in `README.md` (with its 📘 notes link),
   and `notes/README.md`.
10. Run `python3 scripts/check_links.py lessons/<track>/<file>.md` if network allows.

## Quality checklist
- [ ] Every step has exactly one primary resource with exact scope and time
- [ ] Nothing relies on a concept that hasn't been taught earlier in the path
- [ ] At least one non-video resource per step
- [ ] Paid items marked `[paid]` with a free fallback
- [ ] All URLs are in the catalog and verified
