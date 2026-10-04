# Flashcards: spaced repetition for the whole curriculum

[← Courses](../README.md) · [Study notes](../../notes/README.md) · [Playbook](../../playbook/README.md)

Reading a note once leads to understanding. Remembering it six months later takes **retrieval, spaced out over time**.
Michael Nielsen's [*Augmenting Long-term Memory*](https://augmentingcognition.com/ltm.html) estimates a few minutes of total review per card to keep it for years.
These decks turn every study note's **cheat sheet** and every lesson's **self-check question** (with the note's answer sketch) into cards, so the whole curriculum can live in that system.

| Deck | Cards | Source |
|---|---|---|
| [python.tsv](python.tsv) | 77 | PY-01…05 notes |
| [math.tsv](math.tsv) | 41 | MATH-01…03 notes |
| [core-ml.tsv](core-ml.tsv) | 146 | CORE-01…12 notes |
| [deep-learning.tsv](deep-learning.tsv) | 115 | DL-01…09 notes |
| [llms-genai.tsv](llms-genai.tsv) | 129 | GEN-01…10 notes |
| [production.tsv](production.tsv) | 56 | PROD-01…05 notes |
| [vision.tsv](vision.tsv) | 50 | CV-01…04 notes |
| [electives.tsv](electives.tsv) | 123 | EL-01…10 notes |
| [playbook.tsv](playbook.tsv) | 40 | Playbook cheat sheets |

## Import into Anki

1. In Anki: **File → Import**, and pick a `.tsv` file. The file headers set the tab separator, HTML, and the tags column automatically.
2. Note type **Basic**: field 1 = Front, field 2 = Back. Choose (or create) a deck such as `ML::core-ml`.
3. Math renders through Anki's built-in MathJax (`\( … \)` and `\[ … \]`).

Every card is tagged with its **lesson ID** (e.g. `CORE-02`), its **track**, and its **kind** (`cheat-sheet`, `self-check`, or `playbook`).

## How the courses use them

- **Daily (10 min):** review whatever Anki schedules. Add each lesson's cards to your active deck **after** you've studied the lesson. Cards for material you haven't learned yet are noise.
- **Weekly warm-up:** each [syllabus](../README.md#the-syllabi) week names two earlier lessons. Make a filtered deck with the search `tag:CORE-02 OR tag:CORE-04`, draw 5 cards, and answer them from memory *before* you start the new lesson.
  That's spacing and interleaving in 15 minutes.
- **Review weeks:** a filtered deck over the whole course (`tag:core-ml`), 20–30 cards, shuffled.

## Make the cards yours

Matuschak's notes on [writing good prompts](https://notes.andymatuschak.org/Writing_good_spaced_repetition_memory_prompts_is_hard) apply here:
prompts should be focused (one idea), precise (one right answer), and **effortful** (you must retrieve, not recognize). The generated cards are a starting point.

- **Edit** any card whose answer is too long to recall. Split it into two or three cards.
- **Add** your own cards for anything that surprised you, for each lab bug you fixed, and for each playbook trick that worked on your data.
- **Delete** cards for things you'll never need to recall without looking them up.

## Regenerating

The decks are generated from the notes and the playbook. After editing those, run:

```bash
python3 scripts/make_flashcards.py           # rewrite the decks
python3 scripts/make_flashcards.py --check   # what CI runs: fails if a deck is out of date
```

When you re-import an updated deck, choose the option to **update** existing notes: Anki matches cards by their first field (the front), updates their backs, and adds the new cards.
