# Rule: Link policy

1. **Never fabricate URLs, chapter numbers, or section titles.** If you can't verify a
   deep link, link the landing page and write *what to look for* (e.g. "chapter titled 'Tree-Based Methods'").
2. **Verification counts only if** there is first-hand evidence. Record every URL in
   `resources/verified-urls.tsv` (`url  method  date  evidence`) with one of these methods:
   - `F`: page fetched successfully
   - `S`: exact URL seen in web-search results with a matching title
   - `G`: GitHub repo confirmed with `git ls-remote` (or a successful clone)
   - `R`: source file confirmed in the site's own GitHub repo (e.g. `huggingface/blog/<slug>.md`,
     `lilianweng/lilianweng.github.io/posts/<slug>`, `scikit-learn/doc/modules/<x>.rst`)
   - `I`: arXiv ID + title confirmed together in curated GitHub citation lists (see below)
   - `P`: same URL scheme as verified sibling pages. This is the weakest; replace it with a stronger method when possible.
   `python3 scripts/audit_urls.py` must pass before every commit.
3. **Prefer stable URLs**: author homepages, official book sites, GitHub repos, arXiv abs pages.
   Avoid tracking parameters, `?authuser=`, translated-proxy URLs, and URL shorteners.
4. **Official sources only**: link the author's, publisher's, or university's copy.
   Never link pirated mirrors.
5. **YouTube**: link a specific video or an official playlist. Add the duration.
6. **When a link dies**: find the official new location first (author sites move),
   then try the Internet Archive, and only then replace the resource. Log the change in the commit message.
7. Sandboxed sessions may block many hosts. A blocked fetch means "unverified", **not** "broken".
8. **arXiv when arxiv.org is blocked:** raw GitHub is usually reachable. Clone paper lists and docs that cite
   papers (e.g. `huggingface/transformers` model docs, `huggingface/trl`, `dair-ai/ML-Papers-Explained`,
   `Hannibal046/Awesome-LLM`) with `--depth 1 --filter=blob:limit=300k`, index `arxiv.org/abs/<id>` and
   `huggingface.co/papers/<id>` occurrences, and accept an ID only if the **title keywords appear next to it**.
   Web-search anything that isn't found. Link papers as `https://arxiv.org/abs/<id>` (never PDF mirrors).
