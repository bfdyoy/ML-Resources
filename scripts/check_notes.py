#!/usr/bin/env python3
"""Validate the study notes in notes/.

Checks, for every notes/**/*.md file (or the files/directories given):
  1. Structure: required sections exist, every <details> block is closed.
  2. Relative links resolve to existing files.
  3. Math is safe for GitHub's renderer: no $$ (use ```math blocks), paired inline $,
     no \\{ or \\} inline (use \\lbrace/\\rbrace), no "<letter" inline (use \\lt), no "|" inside
     inline math in table rows, no \\\\ inline. If node and the npm package `katex` are available
     (npm install katex), every formula is also parsed with KaTeX via scripts/katex_check.mjs.
  4. Code: every ```python block runs, top to bottom, in one namespace per note
     (```py blocks are illustrative and skipped). Use --no-run to skip this step.

Usage:  python3 scripts/check_notes.py [paths...] [--no-run]
Needs (for step 4): numpy scipy scikit-learn pandas torch
Exit code 1 if anything fails.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIRED = ["## Where we are", "## Pitfalls & misconceptions", "## Cheat sheet", "## Where this leads"]
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
INLINE_MATH_RE = re.compile(r"(^|[^\\\w$])\$(?!\s)([^$]+?)(?<!\s)\$(?![\w$])")


def note_files(args: list[str]) -> list[pathlib.Path]:
    targets = [pathlib.Path(a) for a in args] or [ROOT / "notes"]
    files: list[pathlib.Path] = []
    for t in targets:
        files += sorted(t.rglob("*.md")) if t.is_dir() else [t]
    return [f.resolve() for f in files]


def check_structure(f: pathlib.Path, text: str) -> list[str]:
    errs = []
    if f.parent.name != "notes":                        # per-lesson notes, not README/notation
        errs += [f"missing section '{h}'" for h in REQUIRED if h not in text]
    if text.count("<details>") != text.count("</details>"):
        errs.append("unbalanced <details> blocks")
    return errs


def check_links(f: pathlib.Path, text: str) -> list[str]:
    errs = []
    prose = re.sub(r"^(```|~~~).*?^(```|~~~)\s*$", "", text, flags=re.S | re.M)   # drop fenced code
    prose = re.sub(r"`[^`\n]*`", "", prose)                                           # drop inline code
    for target in LINK_RE.findall(prose):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0]
        if path and not (f.parent / path).exists():
            errs.append(f"broken link -> {target}")
    return errs


def math_spans(text: str) -> tuple[list[tuple[int, str, bool]], list[str]]:
    """Return (spans as (line, tex, display), lint errors)."""
    spans, errs, fence, buf, start = [], [], None, [], 0
    for n, line in enumerate(text.split("\n"), 1):
        m = re.match(r"^\s*(```+|~~~+)\s*(\w*)", line)
        if fence:
            if m and m.group(1).startswith(fence[0]) and not m.group(2):
                if fence[1] == "math":
                    spans.append((start, "\n".join(buf), True))
                fence, buf = None, []
            else:
                buf.append(line)
            continue
        if m:
            fence, start = (m.group(1), m.group(2)), n
            continue
        if "$$" in line:
            errs.append(f"line {n}: use a ```math block instead of $$")
            continue
        plain = re.sub(r"`[^`]*`", "", line)
        found = [mm.group(2) for mm in INLINE_MATH_RE.finditer(plain)]
        if plain.replace("\\$", "").count("$") != 2 * len(found):
            errs.append(f"line {n}: unpaired or unparsable inline $")
        for tex in found:
            spans.append((n, tex, False))
            if re.search(r"\\[{}]", tex):
                errs.append(f"line {n}: inline \\{{ or \\}} (use \\lbrace / \\rbrace)")
            if re.search(r"<[A-Za-z!/]", tex):
                errs.append(f"line {n}: '<' before a letter in inline math (use \\lt)")
            if line.lstrip().startswith("|") and "|" in tex:
                errs.append(f"line {n}: '|' inside inline math in a table row (use \\lvert / \\mid)")
            if "\\\\" in tex:
                errs.append(f"line {n}: \\\\ in inline math (use a ```math block)")
    if fence:
        errs.append(f"unclosed code fence starting at line {start}")
    return spans, errs


def katex_parse(files: list[pathlib.Path]) -> list[str]:
    script = ROOT / "scripts" / "katex_check.mjs"
    if not shutil.which("node"):
        return []
    r = subprocess.run(["node", str(script), *map(str, files)], capture_output=True, text=True)
    if "Cannot find package 'katex'" in r.stderr or "ERR_MODULE_NOT_FOUND" in r.stderr:
        print("  (KaTeX parse skipped: run `npm install katex` to enable it)")
        return []
    return [l for l in r.stdout.splitlines() if l.strip()] if r.returncode else []


def run_code(f: pathlib.Path, text: str) -> list[str]:
    blocks = re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)
    ns: dict = {"__name__": "__main__"}
    errs = []
    for i, block in enumerate(blocks, 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(block, f"{f.name}#block{i}", "exec"), ns)
        except Exception:
            errs.append(f"code block {i} failed:\n" + traceback.format_exc(limit=2))
            break
    return errs


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = "--no-run" not in sys.argv
    files = note_files(args)
    failures = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        t0 = time.time()
        _, lint = math_spans(text)
        errs = check_structure(f, text) + check_links(f, text) + lint
        if run and not errs:
            errs += run_code(f, text)
        status = "FAIL" if errs else "ok"
        print(f"{status:4} {f.relative_to(ROOT)} ({time.time() - t0:.1f}s)")
        for e in errs:
            print("     " + e.replace("\n", "\n     "))
        failures += bool(errs)
    for e in katex_parse(files):
        print("KaTeX " + e)
        failures += 1
    print(f"\n{len(files)} files checked, {failures} with problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
