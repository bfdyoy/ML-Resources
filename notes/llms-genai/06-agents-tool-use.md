# GEN-06 notes: Agents: Tool Use, Structured Outputs & MCP

[← Lesson GEN-06](../../lessons/llms-genai/06-agents-tool-use.md) · [All notes](../README.md) · [← GEN-05 notes](05-retrieval-engineering.md) · Next: [GEN-09 notes →](09-llm-security-safety.md)

> **Reading time** ≈ 50 min. **You need:** [GEN-01 notes](01-how-llms-are-built.md) §4 (sampling from logits), [GEN-03 notes](03-evaluating-llm-apps.md) (evals), and [GEN-05 notes](05-retrieval-engineering.md) (retrieval as a tool).

---

## Where we are

So far, the LLM answered in one shot. An **agent** runs in a loop: it decides on an action (call a tool), observes the result, and decides again, until it's done. That unlocks multi-step tasks. It also makes errors **compound**, makes cost unpredictable, and makes evaluation harder.
This note is about getting the benefits while keeping control.

---

## 1. Workflows vs agents

- **Workflow:** *you* write the control flow, and the LLM fills in steps. The common patterns:
  - **chaining:** step A's output feeds step B;
  - **routing:** classify, then send to a specialized prompt;
  - **parallelization:** independent subtasks, or voting;
  - **orchestrator–workers:** one LLM splits the task, and others do the parts;
  - **evaluator–optimizer:** generate, critique, revise.
- **Agent:** the *LLM* chooses the next step, in an open-ended loop with tools.

**Rule:** use the simplest thing that works. A fixed workflow is cheaper, faster, more predictable, and easier to test. Choose it when the steps are known in advance ("extract fields → validate → write to DB").
An agent is necessary when the path **depends on what's discovered along the way** and can't be enumerated: debugging an unfamiliar codebase, open-ended research, multi-site web tasks.

---

## 2. Why errors compound

If each step succeeds independently with probability $p$, a task needing $n$ correct steps succeeds with probability

```math
P(\text{success}) = p^{\,n}.
```

At $p = 0.95$: 10 steps give 0.60, and 20 steps give **0.36**. Small per-step error rates destroy long trajectories. Worse, real errors aren't independent: one wrong observation (a misread file) poisons every later step.

**What helps:**

- **Verification at checkpoints:** run the tests, validate the schema, re-read the file. A check that catches a fraction $c$ of step errors effectively raises the per-step success rate to $p + (1-p)c$ (if retries succeed).
  At $p = 0.95$ and $c = 0.8$, the effective rate is 0.99, and 20 steps then succeed with probability 0.82.
- **Shorter trajectories:** better tools that do more per call, and decomposition into verified sub-tasks.
- **Recovery:** tool errors that explain how to fix the problem, and the ability to backtrack.
- **Humans** at the irreversible steps.

---

## 3. Tools and the ReAct loop

**ReAct:** alternate *Thought* (reasoning about what to do), *Action* (a tool call with arguments), and *Observation* (the tool's result appended to the context), until the agent emits a final answer **or hits a stopping condition**: max steps, max cost, a repeated-action detector.

**A good tool definition** is a prompt for the model:

- a clear **name** and a **description** that says *when* to use the tool (and when not to);
- **typed, documented parameters** with constraints;
- **informative error messages** that tell the model how to fix the call ("date must be YYYY-MM-DD; you sent 03/04");
- outputs that are concise, relevant, and paginated, not 50 KB dumps;
- tools that are idempotent and safe where possible, with irreversible actions clearly flagged.

**Memory:** the context window is short-term memory. For longer tasks, summarize old steps, keep a scratchpad or todo file, and store and retrieve long-term facts.

---

## 4. Structured outputs: constrained decoding

"Please answer in JSON" is a request, and a model can ignore it: trailing prose, a missing brace, an invented field. **Constrained decoding** enforces the format *inside the sampler*. At each step, it computes which tokens are valid under the grammar or JSON schema, and **masks out everything else**:

```math
p'(t) = \frac{p(t)\,\mathbb{1}[t \in \text{Valid}]}{\sum_{u} p(u)\,\mathbb{1}[u \in \text{Valid}]} .
```

The output is **guaranteed to parse**. Function calling is the same idea, trained in and enforced. Two caveats:

- the format being valid doesn't make the *content* correct;
- an overly rigid schema can push the model into awkward tokens.

So keep validating the semantics (types, ranges, business rules) after parsing. And the grammar has to match the real spec exactly: the first version of the demo below allowed `05`, which JSON rejects, and only 90% of its outputs parsed.

---

## 5. MCP: one protocol instead of N×M integrations

Without a standard, every app (Claude Desktop, IDEs, your own agent) needs custom glue code for every tool (GitHub, a database, a file system): **N × M integrations**.
The **Model Context Protocol** defines one client–server protocol. Tool providers write an **MCP server** once, exposing tools, resources and prompts with schemas, and any **MCP client** can use it. That's **N + M** integrations, with discovery and schemas built in.
Security still matters: a server's tool descriptions are text the model reads, so a malicious server can inject instructions ([GEN-09](09-llm-security-safety.md)).

---

## 6. Evaluating agents

The final answer isn't enough. Evaluate the **trajectory**:

- **Outcome:** did the task succeed? Prefer a programmatic check (tests pass, the right DB state).
- **Process:** were the right tools called, with valid arguments, in a sensible order? Were there redundant steps or loops?
- **Cost and latency:** steps, tokens, time.
- **Reliability:** agents are stochastic, so run each task $k$ times.
  - **pass@k** = P(at least one of $k$ runs succeeds) = $1 - (1-p)^k$. It's optimistic: it measures what's *possible*.
  - **pass^k** = P(*all* $k$ runs succeed) = $p^k$. That's what users experience as **reliability**.
  - At $p = 0.8$ and $k = 5$: pass@5 = 0.9997, but pass^5 = **0.33**.

**Log everything** so failures can be replayed: the full prompts (system + tools), every model output (including reasoning), the tool calls with their arguments, the raw tool results and errors, timing, token counts, model and prompt versions, and the random seeds or sampling parameters.

**A looping agent** (the lesson's debug question):

- a max-steps cap, plus **repeated-call detection** (the same tool with near-identical arguments);
- better tool feedback ("no results; try a broader query, or answer with what you have");
- explicit instructions on when to stop, with a budget visible to the model;
- recording which queries have already been tried, in the scratchpad;
- a better search tool, or deduplicated results.

```python
import numpy as np, json, re
rng = np.random.default_rng(0)

# --- Compounding errors, and the effect of verification ------------------------------------------
p, c = 0.95, 0.8
for n in [5, 10, 20, 50]:
    print(f"{n:2d} steps: no checks {p**n:.2f} | with checks catching 80% of errors {(p + (1 - p) * c) ** n:.2f}")

# --- pass@k vs pass^k ------------------------------------------------------------------------------
p_task, k = 0.8, 5
print(f"pass@{k} = {1 - (1 - p_task) ** k:.4f}   pass^{k} = {p_task ** k:.3f}")

# --- Constrained decoding toy: a 'model' that sometimes rambles; a mask that only allows valid tokens ----
vocab = ['{', '}', '"age"', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' Sure!', ' Here', '\n']
def model_probs(prefix):
    p = np.ones(len(vocab)); p[vocab.index(' Sure!')] = 3; p[vocab.index(' Here')] = 2   # chatty model
    return p / p.sum()
def valid_next(prefix):                       # a tiny grammar: { "age" : DIGITS }
    s = "".join(prefix)
    if s == "": return {'{'}
    if s == "{": return {'"age"'}
    if s == '{"age"': return {':'}
    if s.endswith(':'): return set("123456789")                              # JSON forbids leading zeros
    if re.fullmatch(r'\{"age":\d', s): return set("0123456789") | {'}'}     # one digit so far
    if re.fullmatch(r'\{"age":\d\d', s): return {'}'}                      # two digits: must close
    return set()
def generate(constrained, max_len=8):
    out = []
    for _ in range(max_len):
        p = model_probs(out)
        if constrained:
            mask = np.array([t in valid_next(out) for t in vocab], float)
            if mask.sum() == 0: break
            p = p * mask; p /= p.sum()
        out.append(vocab[rng.choice(len(vocab), p=p)])
        if out[-1] == '}': break
    return "".join(out)
def parses(s):
    try: json.loads(s); return True
    except Exception: return False
for constrained in [False, True]:
    ok = np.mean([parses(generate(constrained)) for _ in range(500)])
    print(f"constrained={constrained}: valid JSON {ok:.0%}, e.g. {generate(constrained)!r}")
```

```python
# --- A minimal ReAct-style loop with a stopping rule and loop detection -------------------------------
def search(q):
    kb = {"capital of france": "Paris", "population of paris": "about 2.1 million (city proper)"}
    return kb.get(q.lower(), "NO RESULTS. Try a shorter, more general query.")
TOOLS = {"search": search}

def fake_llm(history):
    """Stand-in for a model: decides the next action from the transcript."""
    obs = [h for h in history if h.startswith("Observation")]
    if not obs: return 'Action: search("population of the capital city of France")'
    if "NO RESULTS" in obs[-1] and len(obs) == 1: return 'Action: search("capital of France")'
    if "Paris" in obs[-1]: return 'Action: search("population of Paris")'
    if "million" in obs[-1]: return "Final: Paris has about 2.1 million inhabitants (city proper)."
    return 'Action: search("capital of France")'

def run_agent(max_steps=6):
    history, seen = [], set()
    for step in range(max_steps):
        out = fake_llm(history); history.append(out)
        if out.startswith("Final:"): return out, history
        tool, arg = re.match(r'Action: (\w+)\("(.*)"\)', out).groups()
        if (tool, arg.lower()) in seen:
            history.append("Observation: You already ran this exact call. Use what you have or change approach.")
            continue
        seen.add((tool, arg.lower()))
        history.append(f"Observation: {TOOLS[tool](arg)}")
    return "Stopped: step budget exhausted", history
answer, trace = run_agent()
print("\n".join(trace)); print("=>", answer)
```

---

## Pitfalls & misconceptions

- **Reaching for an agent when a workflow would do.**
- **No step or cost limits.** Loops burn money.
- **Vague tool descriptions and unhelpful errors.** The model can only be as good as its tool interface.
- **Trusting "JSON mode" for semantics.** It only guarantees syntax.
- **Measuring only pass@k.** Users experience pass^k.
- **Not logging the raw tool results.** Many agent failures are tool or observation problems.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Compounding | $P = p^n$; with checks, $(p + (1-p)c)^n$ |
| Constrained decoding | renormalize $p(t)$ over the grammar-valid tokens |
| MCP | N×M integrations → N+M |
| pass@k / pass^k | $1-(1-p)^k$ / $p^k$ |
| Stopping | max steps, cost cap, repeated-call detection, explicit "done" criteria |

## Answer sketches for the lesson's self-check

<details>
<summary>1. A task for a fixed workflow vs one that needs an agent.</summary>

Workflow: invoice processing (extract → validate → store), where the steps are known. Agent: "fix the failing CI build in this unfamiliar repo", where the steps depend on what each investigation reveals.
</details>

<details>
<summary>2. Why do agent errors compound, and how do checkpoints help?</summary>

Success needs every step right, roughly $p^n$, and early mistakes corrupt later context. Verification at checkpoints catches errors early and allows a retry, raising the effective per-step success rate (§2: 0.36 → 0.82 for 20 steps in the example).
</details>

<details>
<summary>3. What makes a good tool definition?</summary>

A descriptive name, a description of when (and when not) to use it, typed and documented parameters with constraints, concise relevant outputs, and error messages that tell the model how to correct the call. Flag irreversible effects.
</details>

<details>
<summary>4. Constrained decoding vs asking nicely for JSON.</summary>

Asking nicely leaves the model free to emit invalid tokens, so it sometimes breaks the format. Constrained decoding masks the logits to grammar-valid tokens at every step, so valid syntax is guaranteed (the demo: about 0% vs 100% valid). Semantics still need validation.
</details>

<details>
<summary>5. What does MCP solve?</summary>

The N×M integration problem. A standard protocol (tools, resources, prompts with schemas and discovery) means each tool is wrapped once as a server and works in any compliant client.
</details>

<details>
<summary>6. What to log for post-hoc debugging.</summary>

The full prompts and tool schemas, every model output, the tool calls and arguments, raw results and errors, timings, token counts, model and prompt versions, and the sampling parameters. Enough to replay the trajectory.
</details>

<details>
<summary>7. Three fixes for an agent looping on search.</summary>

Repeated-call detection plus a step cap. Tool feedback that guides the next move ("no results; broaden the query, or answer"). Explicit stopping criteria and a visible budget, plus a record of the queries already tried. Also consider a better search tool.
</details>

## Where this leads

Next: [GEN-09 notes](09-llm-security-safety.md). Agents read untrusted text (web pages, emails, tool descriptions) and take real actions. That combination is the core security problem of LLM applications: **prompt injection**.
