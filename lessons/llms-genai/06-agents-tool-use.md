# GEN-06: Agents: Tool Use, Structured Outputs, Planning & MCP

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~9 h | L2→L3 | GEN-02, GEN-03 (GEN-05 helps) |

## Why this matters
Agents are LLMs that call tools in a loop to reach a goal: search, code execution, APIs, other models. They are powerful but
fragile. Knowing when a simple **workflow** beats an **agent**, how to design tools and structured outputs, and how to evaluate
multi-step behaviour is now a core AI-engineering skill.

## Learning goals
By the end you can:
- Tell workflows (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) apart from autonomous agents, and choose between them.
- Get reliable structured output (JSON schemas, function calling), and validate it.
- Implement the ReAct loop (thought → action → observation) with tools, memory, and a stopping condition.
- Expose tools through the Model Context Protocol (MCP).
- Evaluate agents on trajectories, not just final answers, and name the common failure modes (loops, tool misuse, error compounding).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | The whole post | 45 min |
| 2 | **Read** | [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Planning, memory, and tool use sections | 1 h |
| 3 | **Read** | [ReAct](https://arxiv.org/abs/2210.03629) | §1–3 and Figure 1 | 45 min |
| 4 | **Build** | [OpenAI Cookbook](https://github.com/openai/openai-cookbook) (structured outputs and function-calling examples) + [HF: Evaluating structured outputs](https://huggingface.co/blog/evaluation-structured-outputs) | Extract JSON reliably from messy text; validate with Pydantic; measure the failure rate | 1.5 h |
| 5 | **Build** | [HF AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | Unit 1 (what agents are: thought-action-observation, tools) and one framework unit of your choice | 2.5 h |
| 6 | **Build** | [MCP docs](https://modelcontextprotocol.io/) | Write a small MCP server that exposes one of your GEN-05 retrieval functions as a tool | 1.5 h |
| 7 | *Build (optional)* | [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) / [Anthropic courses](https://github.com/anthropics/courses) (tool use) | One more agent pattern: a multi-agent setup or an evaluator-optimizer loop | 1 h |

## Check your understanding
1. Give a task where a fixed workflow is better than an agent, and one where an agent is necessary.
2. Why do agent errors compound over long trajectories? How do checkpoints and verification help?
3. What makes a good tool definition (name, description, parameters, error messages)?
4. How does constrained decoding for structured output differ from "asking nicely" for JSON?
5. What problem does MCP solve compared with hand-wiring tools into each app?
6. What should you log so you can debug an agent failure after the fact?
7. *(debug)* Your agent keeps calling the same search tool in a loop with slightly different queries. List three fixes.

## Mini-project
**Task:** Build a "research assistant" agent with three tools: your GEN-05 retriever (via MCP), a web search or calculator tool, and a
note-writer. Create 20 tasks with expected outcomes, and evaluate success rate, average steps, and cost. Compare it with a fixed workflow version.
**Deliverable:** A repo with traces for 5 successes and 5 failures, each failure annotated with a hypothesis.

## Go deeper
- [SWE-bench](https://arxiv.org/abs/2310.06770) and [SWE-agent](https://arxiv.org/abs/2405.15793): agents on real software tasks, and why interface design matters.
- [Toolformer](https://arxiv.org/abs/2302.04761) · [Reflexion](https://arxiv.org/abs/2303.11366) · [Generative Agents](https://arxiv.org/abs/2304.03442)
- [Lilian Weng: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/): designing the environment around an agent.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 08: RAG, agents & evals](../../toolbox/08-rag-agents-evals.md) (agents).
- **Papers:** [Post-training, reasoning & agents](../../papers/04-alignment-reasoning-agents.md) (prompting & agents section).
- **Implement it yourself:** a 100-line ReAct loop with no framework: a tool registry, a JSON action parser, and a max-steps guard.
- **Drills:** more in [exercises/](../../exercises/README.md).
