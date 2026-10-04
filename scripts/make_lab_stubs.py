#!/usr/bin/env python3
"""Generate each lab's exercise.py from its solution.py.

Every function and method body is replaced by `raise NotImplementedError`, keeping the signature and
docstring (which carries the subgoals and hints). Module-level code, imports, and constants are kept.
A function stays implemented in the exercise when its docstring contains the marker "(given)".

Usage:
  python3 scripts/make_lab_stubs.py labs/07-micrograd   # write exercise.py (refuses to overwrite edited work)
  python3 scripts/make_lab_stubs.py --force labs/...    # overwrite anyway
  python3 scripts/make_lab_stubs.py --check             # CI: every exercise.py matches its solution.py
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ("# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.\n"
          "# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).\n")


def stub(source: str) -> str:
    tree = ast.parse(source)
    lines = source.split("\n")
    edits: list[tuple[int, int, str]] = []          # (first body line, last line, indent), 1-based inclusive

    def visit(node: ast.AST, inside_stub: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not inside_stub:
                doc = ast.get_docstring(child) or ""
                if "(given)" in doc:
                    continue
                body = child.body
                first = body[1] if doc and len(body) > 1 else (None if doc else body[0])
                if first is None:
                    continue
                start = getattr(first, "decorator_list", None)
                start = min(d.lineno for d in start) if start else first.lineno
                indent = " " * first.col_offset
                edits.append((start, child.end_lineno, indent))
                visit(child, True)
            else:
                visit(child, inside_stub)

    visit(tree, False)
    for start, end, indent in sorted(edits, reverse=True):
        lines[start - 1:end] = [f'{indent}raise NotImplementedError("your code here")']
    return HEADER + "\n".join(lines).rstrip("\n") + "\n"


def lab_dirs(args: list[str]) -> list[pathlib.Path]:
    dirs = [pathlib.Path(a) for a in args] or sorted(p.parent for p in (ROOT / "labs").glob("*/solution.py"))
    return [d.resolve() for d in dirs]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check, force = "--check" in sys.argv, "--force" in sys.argv
    bad = 0
    for d in lab_dirs(args):
        want = stub((d / "solution.py").read_text(encoding="utf-8"))
        ex = d / "exercise.py"
        have = ex.read_text(encoding="utf-8") if ex.exists() else None
        if check:
            if have != want:
                print(f"OUT OF DATE {ex.relative_to(ROOT)} (run scripts/make_lab_stubs.py --force {d.relative_to(ROOT)})")
                bad += 1
            continue
        if have is not None and have != want and not force:
            print(f"skip {ex.relative_to(ROOT)}: it differs from a fresh stub (your work?). Use --force to overwrite.")
            continue
        ex.write_text(want, encoding="utf-8")
        print(f"wrote {ex.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
