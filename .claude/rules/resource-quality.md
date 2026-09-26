# Rule: Resource quality bar

Apply this whenever you add, replace, or recommend a learning resource.

## A resource qualifies if it meets ALL of:
1. **Explains, not just shows.** Builds intuition with worked examples, figures,
   or interactive elements. Reference docs and API manuals are fine for a "Build" step only.
2. **Author credibility.** Written by a recognised practitioner, researcher, or
   institution (university course, well-known book, core library maintainers).
3. **Still correct.** No deprecated APIs as the *main* code path (e.g. TF1,
   `sklearn.cross_validation`), and no claims that the field has since overturned.
   Older classics are fine if the concepts are timeless (e.g. Nielsen's book, Distill articles).
4. **Accessible.** Free, or clearly marked `[paid]` with a free fallback.
   No sign-up walls that need a credit card.
5. **Right level.** Assumes an intermediate learner. Skip "What is Python?" material.
   Graduate-level material only goes under "Go deeper".

## Preference order for the "Read" step
1. A chapter from a well-written book with examples (ISLP, UDL, D2L, Géron, Nielsen, MML)
2. A long-form explainer blog or visual essay (Distill, Alammar, MLU-Explain, Lilian Weng)
3. Course notes (CS231n, Inria scikit-learn MOOC)
4. Video (3Blue1Brown, StatQuest, Karpathy): **only as a complement**, never the sole path in

## Papers & PDFs
- Papers are welcome as the **"Paper" layer** (toolbox and papers library), and as the Read step only when no better explainer exists.
- Pair every paper with an explainer (blog, visual essay, annotated implementation) where one exists.
- Official PDFs from the author, a university, or a publisher are fine (e.g. MML, ESL, Boyd, Sutton & Barto).

## Red flags → reject
- SEO listicles, "Top 10 ML algorithms you must know", Medium posts without code or derivations
- Paywalled Medium or Towards Data Science articles as primary material
- Course platforms where you can't audit the content for free
- Pirated PDFs (dokumen.pub, scribd uploads, random GitHub PDF mirrors). **Never link these.**
  Link the author's or publisher's official page only.

## Tagging (used in resources/catalog.md)
- **Type:** `book` `book-chapter` `visual-essay` `interactive` `course` `course-notes` `video` `notebook` `blog` `paper` `docs`
- **Cost:** `free` `free-online/paid-print` `paid`
- **Level:** `L1` intro · `L2` intermediate · `L3` advanced
