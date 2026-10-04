# PROD-05 notes: LLMOps: Running GenAI Applications in Production

[← Lesson PROD-05](../../lessons/production/05-llmops-genai-platforms.md) · [All notes](../README.md) · [← PROD-04 notes](04-distributed-training.md) · Next: [CV-01 notes →](../vision/01-object-detection-segmentation.md)

> **Reading time** ≈ 45 min. **You need:** [GEN-03 notes](../llms-genai/03-evaluating-llm-apps.md) (evals), [GEN-05 notes](../llms-genai/05-retrieval-engineering.md) and [GEN-06 notes](../llms-genai/06-agents-tool-use.md) (RAG, agents), [GEN-08 notes](../llms-genai/08-efficient-llm-inference.md) (serving cost), and [PROD-02 notes](02-mlops-in-practice.md) (CI/CD, promotion gates).

---

## Where we are

LLM apps break the classic MLOps assumptions:

- the "model" is often a third-party API that changes under you;
- the "code" includes prompts;
- outputs are free text;
- cost scales with *tokens*, not requests.

This note adapts PROD-02's discipline to that world.

---

## 1. Platform architecture, layer by layer

A mature GenAI platform grows these layers, roughly in this order:

1. **Context construction:** retrieval (GEN-05), tools (GEN-06), conversation memory, and the prompt templates that assemble them.
2. **Guardrails:** input checks (PII, injection tripwires, abuse) and output checks (schema, safety, leakage; GEN-09).
3. **Model gateway / router:** one API in front of many models. It handles keys, rate limits, retries, fallbacks, logging, and **routing** (§4).
4. **Caching:** exact and semantic (§3).
5. **Orchestration:** chains, agents, and workflows (GEN-06).
6. **Observability:** traces, evals, cost, and feedback (§2).

---

## 2. Eval-driven development and tracing

### 2.1 Prompts are code

A one-word prompt change can shift behaviour as much as a model swap. So **version prompts in git**, review them like code, and **gate them in CI on the offline eval suite** (GEN-03): assertions plus validated judges, compared against the current version with a *paired* test.
The same gate applies when a provider updates a model: pin model versions and re-run the suite before switching.

### 2.2 What a trace must record

For a RAG + tools request, record enough to replay and debug it a week later:

- **request:** a user or session ID (pseudonymous), a timestamp, the app version, the prompt template version, the model and its parameters (temperature, max tokens);
- **context:** the retrieval query, the retrieved chunk IDs with their scores and ranks, and what was actually packed into the prompt (and what was truncated);
- **tool calls:** names, arguments, raw results, errors, and latencies, as nested **spans**;
- **outputs:** the raw model output, the parsed or validated output, and the guardrail decisions;
- **cost and latency:** input, output, cached, and reasoning tokens; time to first token; total time; the cost per span;
- **feedback:** user ratings, edits, retries, escalations, linked to the trace ID.

---

## 3. Caching, and its risks

- **Exact cache** (keyed on a hash of the full prompt and parameters). Safe in principle. **Risks:** serving **stale** answers after the documents or prices change (key the cache on the index version, or set a TTL), and **leaking personalized answers** across users if the key omits the user context.
- **Semantic cache** (return a cached answer if a new query's embedding is within a similarity threshold of a cached one). **Risks:** **false hits**. "How do I cancel my order?" and "How do I cancel my subscription?" can be very similar in embedding space but need different answers.
  The threshold trades hit rate against wrong answers. Calibrate it on labeled pairs, scope the cache per tenant, and avoid it for high-stakes answers.
- **Provider prompt caching** (reusing the KV cache for a shared prefix, GEN-08 §2): no correctness risk, and it rewards putting **stable content first** in the prompt.

---

## 4. Routing between a small and a large model

Send each request to the cheapest model that will handle it well. Suppose a fraction $f$ of the traffic is routed to the small model, which costs $c_s$ per request (against $c_\ell$ for the large one), and the router itself costs $c_r$. Then:

```math
\text{cost per request} = c_r + f\,c_s + (1-f)\,c_\ell, \qquad \text{saving} = f\,(c_\ell - c_s) - c_r .
```

**Routing pays off** when a large share of the traffic is "easy" (high $f$ at acceptable quality), the price gap is big, and the router is cheap and accurate. Ways to decide the route:

- a classifier trained on labeled examples of whether the small model's answer was acceptable;
- the small model's own confidence or a self-check, escalating when it's uncertain (a **cascade**);
- rules (by task type or input length).

Evaluate the router like any classifier: the quality you lose on misrouted hard queries vs the money saved.

---

## 5. Online quality signals

- **Explicit:** thumbs up/down, ratings, reported issues.
- **Implicit:** the user **regenerates or retries**, **edits** the output heavily, copies or accepts it (a good sign), abandons the session, escalates to a human, rephrases the same question, or has to do follow-up correction turns.
- **Automated:** online judges on sampled traces (validated, GEN-03 §4), guardrail trigger rates, and refusal rates.

Watch the *rates*, segmented by feature, prompt version, and model, and alert on changes.

## 6. Cost and latency levers

- **Fewer input tokens:** trim conversation history (summarize), retrieve fewer and better chunks (rerank), compress prompts, move stable instructions to a cached prefix.
- **Fewer output tokens:** ask for concise formats, cap `max_tokens`, and watch the **reasoning-token** budgets of thinking models.
- **Cheaper tokens:** routing and cascades, batch APIs for offline jobs, self-hosting at high steady volume (GEN-08).
- **Fewer calls:** caching, and fixing agent loops and retries.

**Debug: the API bill doubled overnight with no traffic increase.** The four most likely causes:

1. **Longer inputs:** a prompt or template change, more retrieved chunks, conversation history no longer trimmed, a bigger tool output being stuffed in.
2. **More calls per request:** agent loops, retry storms on errors or timeouts, a broken cache (hit rate collapsed).
3. **A pricier model or more output:** a routing or config change sending traffic to the large model, a provider model update, reasoning effort raised, `max_tokens` removed.
4. **Abuse or bugs:** a single key or user, a scraper, a runaway batch job.

Check the per-trace token breakdowns and the cache hit rate, and diff the deployment.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
rng = np.random.default_rng(0)

# --- Request cost from tokens -------------------------------------------------------------------
def cost(in_tok, out_tok, price_in_per_m, price_out_per_m, cached_tok=0, cached_discount=0.9):
    billable_in = in_tok - cached_tok + cached_tok * (1 - cached_discount)
    return (billable_in * price_in_per_m + out_tok * price_out_per_m) / 1e6
base = cost(3000, 400, 3.0, 15.0)
print(f"base request ${base:.4f}")
print(f"history no longer trimmed (9k input): ${cost(9000, 400, 3.0, 15.0):.4f}")
print(f"stable 2.5k prefix cached: ${cost(3000, 400, 3.0, 15.0, cached_tok=2500):.4f}")

# --- Routing break-even -----------------------------------------------------------------------------
c_small, c_large, c_router = 0.001, 0.02, 0.0002
for f in [0.2, 0.5, 0.8]:
    print(f"route {f:.0%} to the small model: cost/request ${c_router + f*c_small + (1-f)*c_large:.4f} (large only: ${c_large:.4f})")
```

```python
# --- Semantic cache false hits: similar wording, different intent ---------------------------------------
cached = ["how do i cancel my order", "what is your refund policy", "how do i reset my password"]
queries = [("how can i cancel my order", 0), ("how do i cancel my subscription", None),
           ("what's the refund policy", 1), ("how do i reset my router", None)]
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(cached + [q for q, _ in queries])
C = vec.transform(cached)
for q, true_idx in queries:
    sims = (vec.transform([q]) @ C.T).toarray()[0]
    verdict = "correct reuse" if sims.argmax() == true_idx else "WRONG answer if reused"
    print(f"{q!r:36} best match {cached[sims.argmax()]!r:30} sim {sims.max():.2f} -> {verdict}")
# The wrong-intent queries score as high as (or higher than) the right ones, so NO threshold separates them
# with this representation. Use a better embedding, calibrate the threshold on labeled pairs, and keep
# semantic caching away from answers where a near-miss is harmful.
```

---

## Pitfalls & misconceptions

- **Editing prompts in production without an eval gate.**
- **Unpinned model versions.** Silent provider updates change behaviour.
- **Semantic caching with an uncalibrated threshold**, or one shared across tenants.
- **Logging only final outputs.** Without the context and tool spans, you can't debug.
- **Watching total cost but not tokens per request and cache hit rate.**

## Cheat sheet

| Item | Rule / formula |
|---|---|
| Platform layers | context → guardrails → gateway/router → cache → orchestration → observability |
| Prompt changes | versioned, reviewed, gated by the eval suite (paired comparison) |
| Trace | request + context (chunk IDs, ranks) + tool spans + outputs + tokens/latency/cost + feedback |
| Routing saving | $f(c_\ell - c_s) - c_r$ |
| Cache risks | exact: staleness, personalization leaks; semantic: false hits |
| Bill doubled | input length, call count, model/output mix, abuse |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why should a prompt change go through review and an eval gate?</summary>

Prompts determine behaviour as much as code or weights do. Small edits cause regressions you can't see without measurement. Versioning, review, and a paired eval comparison catch them before users do.
</details>

<details>
<summary>2. What a RAG + tools trace should record.</summary>

Versions (app, prompt, model, parameters), the retrieval query and retrieved chunk IDs with scores and ranks, the packed context (and what was truncated), every tool call's arguments, results, errors and latency, the raw and validated outputs, guardrail decisions, token counts, cost, latency, and the linked user feedback (§2.2).
</details>

<details>
<summary>3. Exact vs semantic caching risks.</summary>

Exact: stale answers after the underlying data changes, and cross-user leakage if the key omits the user context. Semantic: false hits, where near-duplicate wording has a different intent (in the demo, the wrong-intent queries score higher than the correct ones). Calibrate the threshold, scope per tenant, and invalidate on data changes.
</details>

<details>
<summary>4. When does small/large routing pay off, and how do you route?</summary>

When much of the traffic is easy, the price gap is large, and the router is cheap and accurate: saving = $f(c_\ell - c_s) - c_r$. Route with a trained difficulty classifier, a confidence or self-check cascade, or rules. Validate the quality loss on misrouted queries.
</details>

<details>
<summary>5. Online signals that quality is dropping.</summary>

Explicit: thumbs-down rate, reports. Implicit: regenerations or retries, heavy edits, rephrased repeat questions, abandonment, escalations. Automated: judge pass rates on sampled traces, guardrail and refusal rates, all segmented by version.
</details>

<details>
<summary>6. The bill doubled overnight with no traffic increase.</summary>

Longer inputs (template, history, retrieval k), more calls (agent loops, retries, a broken cache), a pricier model or longer outputs (routing or config change, reasoning budget), or abuse and bugs (one key, a scraper, a runaway job) (§6).
</details>

## Where this leads

**Path 4 is complete.** Next, Path 5: [CV-01 notes](../vision/01-object-detection-segmentation.md) takes vision beyond classification. Or pick an elective from [EL-01 notes](../electives/01-time-series-forecasting.md) onwards.
