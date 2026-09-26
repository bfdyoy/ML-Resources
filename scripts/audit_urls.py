#!/usr/bin/env python3
"""Audit that every URL in the repo's markdown has a verification record.

Usage:
    python3 scripts/audit_urls.py            # report unverified URLs, exit 1 if any
    python3 scripts/audit_urls.py --stats    # also print counts per verification method

The log lives in resources/verified-urls.tsv with columns:
    url <TAB> method <TAB> date <TAB> evidence

Methods (strongest first):
    F  page fetched successfully
    S  exact URL seen in web search results with a matching title
    G  GitHub repository confirmed to exist (git ls-remote)
    R  source file confirmed in the site's own GitHub repo (blog posts, docs pages)
    I  arXiv ID + title confirmed in curated GitHub citation lists
    P  same URL scheme as verified sibling pages; pending direct check (weakest)

Stdlib only. templates/ is ignored (it contains placeholder URLs).
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "resources" / "verified-urls.tsv"
URL_RE = re.compile(r"https?://[^\s<>()\"'`\]]+")
SKIP = {".git", "templates", "node_modules"}


def repo_urls() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for p in ROOT.rglob("*.md"):
        if SKIP & set(p.relative_to(ROOT).parts):
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for u in URL_RE.findall(line):
                found.setdefault(u.rstrip(".,;:*"), []).append(f"{p.relative_to(ROOT)}:{n}")
    return found


def load_log() -> dict[str, tuple[str, str, str]]:
    log: dict[str, tuple[str, str, str]] = {}
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                log[parts[0]] = (parts[1], parts[2], parts[3] if len(parts) > 3 else "")
    return log


def main() -> int:
    urls, log = repo_urls(), load_log()
    valid = {"F", "S", "G", "R", "I", "P"}
    missing = sorted(u for u in urls if u not in log or log[u][0] not in valid)
    if "--stats" in sys.argv:
        counts = collections.Counter(log[u][0] for u in urls if u in log)
        print("Verified URLs by method:", dict(sorted(counts.items())))
        weak = sorted(u for u in urls if u in log and log[u][0] == "P")
        if weak:
            print(f"\n{len(weak)} URL(s) only pattern-verified (P), re-check when possible:")
            for u in weak:
                print("  ", u)
    if missing:
        print(f"\n{len(missing)} URL(s) have no verification record:")
        for u in missing:
            print(f"  {u}  (at {urls[u][0]})")
        print("\nVerify each (see .claude/rules/link-policy.md), then add it to resources/verified-urls.tsv.")
        return 1
    print(f"All {len(urls)} URLs have a verification record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
