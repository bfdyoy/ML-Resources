# Lab 10: A byte-level BPE tokenizer

[← Labs](../README.md) · Lesson: [GEN-01 How LLMs are built](../../lessons/llms-genai/01-how-llms-are-built.md) · Notes: [GEN-01 notes](../../notes/llms-genai/01-how-llms-are-built.md)

**Time** ≈ 1.5 h · **You'll practise:** the BPE training loop, applying merges in *training order* at encode time, and why byte-level vocabularies never produce out-of-vocabulary tokens.

| Function | Checked against |
|---|---|
| `get_pair_counts`, `merge` | hand examples, including the overlapping `[1, 1, 1]` case |
| `train_bpe` | the classic `aaabdaaabac` example: three merges, hand-computed |
| `encode`, `decode` | exact round trips, including Romanian diacritics and an emoji |
| — | trained on a small corpus, the encoding is < 40% of the byte count |

```bash
pytest labs/10-bpe-tokenizer
```

**Bonus:**
1. Train on English text, then encode a Romanian sentence. How many tokens does each language use per word? This is the "token tax" from the GEN-01 notes.
2. GPT-2 first splits text with a regex, so merges never cross word or punctuation boundaries. Add that pre-split. Which odd merges disappear?
