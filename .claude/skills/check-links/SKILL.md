---
name: check-links
description: Validate every URL in the repo's markdown files, classify failures (broken vs blocked-by-sandbox vs moved), and propose official replacements. Use when the user asks to check, audit, or fix links, or before a release.
---

# Check links

1. Run `python3 scripts/check_links.py` (optionally pass specific files).
   The script prints `OK`, `REDIRECT`, `BROKEN`, and `UNREACHABLE` per URL, and exits non-zero if anything is `BROKEN`.
2. Treat results carefully:
   - `UNREACHABLE` (proxy 403, DNS failure, timeout): probably the sandbox. Re-check with
     WebSearch for the exact URL before calling it broken.
   - `REDIRECT` to a new permanent location: update the link to the final URL.
   - `BROKEN` (404/410): follow `.claude/rules/link-policy.md` §6 to find the official new location.
3. Update both `resources/catalog.md` and every lesson that uses the URL (`grep -rn "<old-url>"`).
4. Update the `Verified` date in the catalog for everything you re-checked.
5. Report what you fixed, what's still unknown, and what you replaced (with reasons).
