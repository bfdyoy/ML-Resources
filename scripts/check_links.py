#!/usr/bin/env python3
"""Check every http(s) link in the repo's markdown files.

Usage:
    python3 scripts/check_links.py                 # all *.md files
    python3 scripts/check_links.py lessons/core-ml # a folder
    python3 scripts/check_links.py README.md paths/01-core-ml.md

Statuses:
    OK           2xx
    REDIRECT     3xx that ends somewhere else (consider updating the link)
    BROKEN       404 / 410 (the page is gone)
    UNREACHABLE  network/proxy errors, timeouts, 401/403/429/5xx (often bot-blocking
                 or a sandbox policy, so re-check by hand before calling it broken)

Exit code is 1 if anything is BROKEN, else 0. Stdlib only.
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s<>()\"'`\]]+")
SKIP_DIRS = {".git", ".claude", "templates", "node_modules", ".venv"}


def is_reserved(url: str) -> bool:
    """RFC 2606/6761 example hosts (e.g. evil.example, example.com) are illustrations, not links."""
    m = re.match(r"https?://([^/?#:]+)", url.replace("\\", ""))
    host = m.group(1).lower() if m else ""
    return host in {"example.com", "example.org", "example.net", "localhost"} or host.endswith(
        (".example", ".test", ".invalid", ".localhost", ".example.com", ".example.org"))
UA = "Mozilla/5.0 (ML-Resources link checker; +https://github.com/bfdyoy/ML-Resources)"


def md_files(args: list[str]) -> list[pathlib.Path]:
    targets = [ROOT / a for a in args] if args else [ROOT]
    files: list[pathlib.Path] = []
    for t in targets:
        if t.is_file():
            files.append(t)
        else:
            files += [p for p in t.rglob("*.md") if not SKIP_DIRS & set(p.parts)]
    return sorted(set(files))


def extract(files: list[pathlib.Path]) -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {}
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for u in URL_RE.findall(line):
                u = u.rstrip(".,;:*")
                if is_reserved(u):
                    continue
                urls.setdefault(u, []).append(f"{f.relative_to(ROOT)}:{n}")
    return urls


def check(url: str) -> tuple[str, str]:
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                final = r.geturl()
                if final.rstrip("/") != url.rstrip("/"):
                    return "REDIRECT", final
                return "OK", str(r.status)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                if method == "HEAD":
                    continue  # some servers 404 on HEAD only
                return "BROKEN", str(e.code)
            if e.code in (405, 400) and method == "HEAD":
                continue
            return "UNREACHABLE", f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - report any network failure
            if method == "HEAD":
                continue
            return "UNREACHABLE", type(e).__name__
    return "UNREACHABLE", "unknown"


def main() -> int:
    urls = extract(md_files(sys.argv[1:]))
    print(f"Checking {len(urls)} unique URLs…\n")
    counts: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        results = dict(zip(urls, ex.map(check, urls)))
    for url, (status, detail) in sorted(results.items(), key=lambda kv: kv[1][0]):
        counts[status] = counts.get(status, 0) + 1
        if status != "OK":
            print(f"[{status}] {url} -> {detail}")
            for loc in urls[url][:3]:
                print(f"    at {loc}")
    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("BROKEN") else 0


if __name__ == "__main__":
    sys.exit(main())
