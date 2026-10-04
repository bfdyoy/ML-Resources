#!/usr/bin/env python3
"""Generate spaced-repetition flashcard decks (Anki-importable TSV) from the study notes and the playbook.

Cards come from three places:
  1. Every row of a note's "## Cheat sheet" table      -> front: the item, back: the rule/formula
  2. Every "Answer sketches" <details> block of a note -> front: the full question from the lesson's
     "Check your understanding" list (same order), back: the answer sketch
  3. Every row of a playbook page's "## Cheat sheet"   -> front: the situation, back: the trick

Output: courses/flashcards/<deck>.tsv with Anki file headers (#separator:tab, #html:true, #tags column:3).
Math is converted for Anki's MathJax: $..$ -> \\(..\\), ```math blocks -> \\[..\\].
Tags: the lesson ID (e.g. CORE-02), the track, and the card kind (cheat-sheet / self-check / playbook).

Usage:  python3 scripts/make_flashcards.py           # write the decks
        python3 scripts/make_flashcards.py --check   # CI: fail if the decks are out of date
"""
from __future__ import annotations

import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "courses" / "flashcards"
HEADER = "#separator:tab\n#html:true\n#tags column:3\n"


def md_to_html(text: str) -> str:
    """A small Markdown subset -> HTML for Anki: math, code, bold/italics, links, line breaks."""
    text = re.sub(r"```math\n(.*?)```", lambda m: "\\[" + m.group(1).strip() + "\\]", text, flags=re.S)
    text = re.sub(r"```\w*\n(.*?)```", lambda m: "<pre>" + html.escape(m.group(1).rstrip()) + "</pre>", text, flags=re.S)
    parts = re.split(r"(`[^`]+`|\$[^$\n]+\$|\\\[.*?\\\]|<pre>.*?</pre>)", text, flags=re.S)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append("<code>" + html.escape(part[1:-1]) + "</code>")
        elif part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append("\\(" + html.escape(part[1:-1], quote=False) + "\\)")
        elif part.startswith(("\\[", "<pre>")):
            out.append(part if part.startswith("<pre>") else html.escape(part, quote=False))
        else:
            p = html.escape(part, quote=False)
            p = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", p)                 # links -> their text
            p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
            p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
            out.append(p)
    s = "".join(out).strip()
    s = re.sub(r"\n\s*\n", "<br><br>", s)
    return s.replace("\n", " ").replace("\t", " ")


def section(text: str, heading: str) -> str:
    m = re.search(rf"^{re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    return m.group(1) if m else ""


def table_rows(block: str) -> list[list[str]]:
    rows = [l for l in block.splitlines() if l.strip().startswith("|")]
    cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
    return [c for c in cells[2:] if any(c)]                                 # skip the header and the --- row


def lesson_questions(lesson: pathlib.Path) -> list[str]:
    if not lesson.exists():
        return []
    block = section(lesson.read_text(encoding="utf-8"), "## Check your understanding")
    if not block:
        block = section(lesson.read_text(encoding="utf-8"), "## Self-check")
    qs, cur = [], None
    for line in block.splitlines():
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            cur = [m.group(1)]
            qs.append(cur)
        elif cur is not None and line.strip() and not line.startswith("#"):
            cur.append(line.strip())
    return [" ".join(q) for q in qs]


def note_cards(note: pathlib.Path) -> tuple[str, list[tuple[str, str, str]]]:
    text = note.read_text(encoding="utf-8")
    m = re.match(r"#\s+(\S+)\s+notes:", text)
    lid = m.group(1) if m else note.stem
    track = note.parent.name
    cards = []
    for row in table_rows(section(text, "## Cheat sheet")):
        if len(row) >= 2:
            cards.append((f"<small>{lid}</small><br>{md_to_html(row[0])}", md_to_html(" · ".join(row[1:])), f"{lid} {track} cheat-sheet"))
    questions = lesson_questions(ROOT / "lessons" / track / note.name)
    answers = re.findall(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", section(text, "## Answer sketches for the lesson's self-check"), flags=re.S)
    for i, (summary, body) in enumerate(answers):
        q = questions[i] if i < len(questions) else re.sub(r"^\d+\.\s*", "", html.unescape(re.sub(r"<[^>]+>", "`", summary)))
        cards.append((f"<small>{lid} · self-check</small><br>{md_to_html(q)}", md_to_html(body.strip()), f"{lid} {track} self-check"))
    return track, cards


def playbook_cards(page: pathlib.Path) -> list[tuple[str, str, str]]:
    text = page.read_text(encoding="utf-8")
    title = re.match(r"#\s+(.*)", text).group(1)
    tag = "playbook-" + page.stem.split("-", 1)[0]
    return [(f"<small>{html.escape(title)}</small><br>{md_to_html(r[0])}", md_to_html(" · ".join(r[1:])), f"{tag} playbook")
            for r in table_rows(section(text, "## Cheat sheet")) if len(r) >= 2]


def build() -> dict[str, str]:
    decks: dict[str, list] = {}
    for note in sorted((ROOT / "notes").glob("*/*.md")):
        track, cards = note_cards(note)
        decks.setdefault(track, []).extend(cards)
    for page in sorted((ROOT / "playbook").glob("0*.md")):
        decks.setdefault("playbook", []).extend(playbook_cards(page))
    return {name: HEADER + "".join(f"{f}\t{b}\t{t}\n" for f, b, t in cards) for name, cards in decks.items()}


def main() -> int:
    decks = build()
    OUT.mkdir(parents=True, exist_ok=True)
    stale = 0
    for name, content in sorted(decks.items()):
        path = OUT / f"{name}.tsv"
        n = content.count("\n") - 3
        if "--check" in sys.argv:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                print(f"OUT OF DATE {path.relative_to(ROOT)} (run python3 scripts/make_flashcards.py)")
                stale += 1
            continue
        path.write_text(content, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {n} cards")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
