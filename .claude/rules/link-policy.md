# Rule: Link policy

1. **Never fabricate URLs, chapter numbers, or section titles.** If you can't verify a
   deep link, link the landing page and write *what to look for* (e.g. "chapter titled 'Tree-Based Methods'").
2. **Verification counts only if** you (a) fetched the page successfully, or (b) saw
   the exact URL in a search result together with a matching title. Record how in the
   catalog's `Verified` column (`fetch` / `search` + date).
3. **Prefer stable URLs**: author homepages, official book sites, GitHub repos, arXiv abs pages.
   Avoid tracking parameters, `?authuser=`, translated-proxy URLs, and URL shorteners.
4. **Official sources only**: link the author's, publisher's, or university's copy.
   Never link pirated mirrors.
5. **YouTube**: link a specific video or an official playlist. Add the duration.
6. **When a link dies**: find the official new location first (author sites move),
   then try the Internet Archive, and only then replace the resource. Log the change in the commit message.
7. Sandboxed sessions may block many hosts. A blocked fetch means "unverified", **not** "broken".
