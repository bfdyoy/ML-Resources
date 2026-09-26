# GEN-02: Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~9 h | L2 | GEN-01 |

## Why this matters
Almost nobody trains an LLM from scratch. The real skill is **adapting** existing models: prompt well, fine-tune
cheaply (LoRA), or ground the model in your own data (RAG). Knowing which lever to pull, and in what order,
saves weeks of work.

## Learning goals
By the end you can:
- Use the Hugging Face stack (`transformers`, `datasets`, tokenizers, the Hub) to load, run, and fine-tune models.
- Fine-tune a small model for classification and for instruction following, and explain the difference.
- Explain LoRA (low-rank updates to frozen weights), and tune its key hyperparameters (rank, alpha, target modules).
- Build a basic RAG pipeline (chunk → embed → retrieve → generate), and name its common failure modes.
- Choose between prompting, RAG, and fine-tuning for a given problem.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read + Build** | [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Ch. 1–3 (transformer models, using 🤗 Transformers, fine-tuning a pretrained model). Do the code sections in Colab. | 3 h |
| 2 | **Read + Build** | [Raschka, *Build a LLM From Scratch*](https://github.com/rasbt/LLMs-from-scratch) | Ch. 6 (fine-tuning for classification), Ch. 7 (instruction fine-tuning), Appendix E (LoRA). Run the notebooks. | 3 h |
| 3 | **Read** | [Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) | The whole post | 45 min |
| 4 | **Read** | [Eugene Yan: Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) | The sections on RAG and fine-tuning (you'll come back to evals, guardrails, and caching in GEN-03) | 1 h |
| 5 | **Build** | Your own RAG pipeline | Embed a document set with a sentence-transformer model, retrieve top-k, generate an answer with an open model or an API. Log what gets retrieved. | 1.5 h |

## Check your understanding
1. Why does LoRA use so much less GPU memory than full fine-tuning? Where does the saving come from?
2. What goes wrong if you fine-tune an instruction model on a tiny dataset with a high learning rate?
3. Your RAG system gives confident wrong answers. List the pipeline stages where the error could come from, and how you'd test each one.
4. When is fine-tuning the right call over RAG, and when is it the other way round?
5. What does chunk size trade off in RAG?
6. *(debug)* After fine-tuning, the model follows your format perfectly but has "forgotten" general knowledge. What happened, and what can you do?

## Mini-project
**Task:** Build a Q&A assistant over a document set you care about (course notes, docs of a library you use). Version 1
is prompt-only. Version 2 adds RAG. Write 20 test questions with reference answers, and compare the two versions.
**Deliverable:** A notebook plus a results table (you'll reuse this test set in GEN-03).

## Go deeper
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (Chip Huyen) `[paid]`: the chapters on RAG, agents, and fine-tuning. The most complete practitioner treatment.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) Ch. 10–12: advanced fine-tuning, dataset curation, and reasoning models.
