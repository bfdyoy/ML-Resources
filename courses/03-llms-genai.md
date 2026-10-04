# Course 3: LLMs & Generative AI, a 12-week syllabus

[← Courses](README.md) · Path: [Path 3: LLMs & Generative AI](../paths/03-llms-genai.md) · Labs: [index](../labs/README.md) · Cards: [llms-genai deck](flashcards/README.md)

> **Pace** ≈ 8–9 h/week for 12 weeks. **Before you start:** DL-05 and DL-06 (or equivalent: you can explain attention and train a small GPT).
> **Running project:** one domain assistant over documents you care about. It starts in week 1 as a 30-line prototype, and every week adds a piece and an eval.

The [weekly loop](README.md#32-the-weekly-loop-about-8-hours) applies every week.

## How this course is shaped

- **Whole game in week 1** (fast.ai, the HF course). A naive "stuff the top chunks into the prompt" assistant and 10 hand-graded questions, before any theory.
- **Evals before improvements** (Hamel Husain's and Eugene Yan's practice, linked from GEN-03). From week 3 on, **no change ships without a number**. Each week's milestone is a row in your eval table.
- **Build the core once** (CS336). Lab 10 trains a BPE tokenizer from scratch, and Lab 11 builds BM25 and the ranking metrics. Then use the libraries.
- **Three-layer structure** (Labonne's *LLM course*): fundamentals (weeks 1–2) → engineer (weeks 3–7) → scientist (weeks 8–9) → generative media (weeks 10–11).

---

## Week 1: Whole game + how LLMs are built
- **Whole game (2 h):** pick ~50 pages of documents. Split them into chunks, embed, retrieve the top 3, and prompt any model you have access to. Write 10 questions and grade the answers by hand (right / partly / wrong).
- **Do:** [GEN-01](../lessons/llms-genai/01-how-llms-are-built.md) with its [notes](../notes/llms-genai/01-how-llms-are-built.md).
- **Lab:** [Lab 10: BPE tokenizer](../labs/10-bpe-tokenizer/README.md).
- **Explain it back:** why tokenization makes LLMs bad at counting letters.

## Week 2: Adapting LLMs
- **Warm-up:** GEN-01 + DL-06 cards. *Interleave:* "Temperature 0 vs top-p 0.9: which one can change the *set* of possible tokens, and which only reshapes it?"
- **Do:** [GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md).
- **Playbook:** [Choosing: prompting vs RAG vs fine-tuning](../playbook/01-choosing-algorithms.md#4-prompting-vs-rag-vs-fine-tuning).
- **Project:** grow the test set to 20 questions with reference answers. Compare prompt-only with RAG.

## Week 3: Evaluating LLM apps
- **Warm-up:** GEN-02 + GEN-01 cards. *Interleave:* "LoRA rank 8 on a 4096×4096 matrix: how many trainable parameters?"
- **Do:** [GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md).
- **Playbook:** [LLM-as-labeler, validated](../playbook/04-llm-and-retrieval-tricks.md#2-use-an-llm-as-a-labeler-then-measure-it-like-one).
- **Project:** a rerunnable eval harness: assertions + an LLM judge, with judge agreement measured on 30 of your own labels.

## Week 4: Retrieval engineering
- **Warm-up:** GEN-03 + GEN-02 cards. *Interleave:* "Your judge agrees with you 95% of the time, but 90% of answers are correct. Is 95% impressive?"
- **Do:** [GEN-05](../lessons/llms-genai/05-retrieval-engineering.md).
- **Lab:** [Lab 11: BM25, ranking metrics & RRF](../labs/11-retrieval-metrics/README.md).
- **Playbook:** [hybrid search + rerank, contextual chunks](../playbook/04-llm-and-retrieval-tricks.md#4-retrieval-hybrid-first-rerank-second-and-contextualize-chunks).
- **Project:** a retrieval leaderboard (BM25 → dense → hybrid → rerank) on your own eval set.

## Week 5: Review week + **midterm**
- **Interleaved quiz:** 20 cards from GEN-01, 02, 03 and 05, plus CORE-03 (metrics).
- **Redo from a blank file:** Lab 11's `ndcg_at_k` and `reciprocal_rank_fusion`.
- **Midterm (rubric below):** your assistant's eval report, version 1.

## Week 6: Agents & tool use
- **Warm-up:** GEN-05 + GEN-03 cards. *Interleave:* "Recall@10 rose, but answer quality didn't. Two hypotheses?"
- **Do:** [GEN-06](../lessons/llms-genai/06-agents-tool-use.md).
- **Project:** a workflow version and an agent version of one task. Compare success rate, steps and cost.

## Week 7: Security & safety
- **Warm-up:** GEN-06 + GEN-02 cards. *Interleave:* "Why are structured outputs a security feature, not just a convenience?"
- **Do:** [GEN-09](../lessons/llms-genai/09-llm-security-safety.md).
- **Project:** a threat model and a ≥30-case red-team suite that runs with your evals.

## Week 8: Post-training
- **Warm-up:** GEN-09 + GEN-01 cards. *Interleave:* "Indirect prompt injection reaches the model through which component of your RAG system?"
- **Do:** [GEN-07](../lessons/llms-genai/07-post-training-alignment-reasoning.md).
- **Playbook:** [self-consistency (majority voting)](../playbook/04-llm-and-retrieval-tricks.md#1-sample-several-answers-and-vote-self-consistency), and the length-bias trap in judges.
- **Project (optional track):** DPO on a narrow behaviour of your assistant, evaluated with your harness.

## Week 9: Efficient inference
- **Warm-up:** GEN-07 + GEN-03 cards. *Interleave:* "DPO's win rate went up and answers got 40% longer. What do you check?"
- **Do:** [GEN-08](../lessons/llms-genai/08-efficient-llm-inference.md).
- **Playbook:** [prompt-prefix caching order](../playbook/04-llm-and-retrieval-tricks.md#5-put-the-stable-part-of-the-prompt-first).
- **Project:** latency and cost per query, before and after one optimization.

## Week 10: Generative models & diffusion
- **Warm-up:** GEN-08 + DL-07 cards. *Interleave:* "The KV cache grows with what three quantities?"
- **Do:** [GEN-04](../lessons/llms-genai/04-generative-models-diffusion.md).

## Week 11: Advanced diffusion & flow matching
- **Warm-up:** GEN-04 + GEN-07 cards. *Interleave:* "Classifier-free guidance and DPO both compare two models' outputs. What is each one comparing?"
- **Do:** [GEN-10](../lessons/llms-genai/10-advanced-diffusion-flow-matching.md).
- **Project:** freeze features; write the capstone report draft.

## Week 12: **Capstone**
- **Interleaved quiz:** 30 cards from the whole deck.
- **Capstone:** the [path capstone](../paths/03-llms-genai.md#capstone) (rubric below).

---

## Midterm rubric (week 5)

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Test set | < 10 questions | 20 questions | ≥ 30 questions, covering easy, hard and unanswerable |
| Retrieval eval | None | One metric | Recall@k + MRR/NDCG on a labeled set, with a leaderboard |
| Generation eval | Vibes | LLM judge, unvalidated | Assertions + a judge with measured agreement |
| Error analysis | None | A list of failures | Failures grouped (retrieval miss / reasoning / formatting), with counts |
| Next steps | None | Ideas | Prioritized by the error counts |

## Capstone rubric (week 12)

| Criterion | What "2" looks like |
|---|---|
| Comparison | ≥2 approaches compared on the same eval, with uncertainty (bootstrap CI or a paired test) |
| Retrieval | A leaderboard with latency; the shipped choice justified |
| Evals | A rerunnable harness; a judge validated against human labels; a regression caught |
| Safety | A threat model, a red-team suite in CI, and the pass rate reported |
| Cost & latency | Measured per query, with one optimization applied |
| Write-up | Error analysis, limitations, and what you'd do next |

Pass: ≥ 9/12.

## Assessment summary

| Component | Weight |
|---|---|
| Labs 10–11 passing | 15% |
| Weekly self-checks from memory | 15% |
| Midterm | 25% |
| Capstone | 45% |
