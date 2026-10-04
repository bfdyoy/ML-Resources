# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 10: a byte-level BPE tokenizer (the GPT-2 family's core idea, without the regex pre-split).

Token ids 0..255 are raw UTF-8 bytes. Training adds merges: each merge maps a pair of ids to a new id 256, 257, ...
`merges` is a dict {(a, b): new_id} whose insertion order is the training order.
"""


def get_pair_counts(ids):
    """Counts of adjacent pairs, as a dict {(a, b): count} in order of first appearance.
    Example: [1, 2, 1, 2, 3] -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}."""
    raise NotImplementedError("your code here")


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair`, scanning left to right, with new_id.
    Example: merge([1, 1, 1], (1, 1), 9) -> [9, 1]."""
    raise NotImplementedError("your code here")


def train_bpe(text, vocab_size):
    """Learn vocab_size - 256 merges from the UTF-8 bytes of `text`. At each step merge the most frequent pair;
    ties go to the pair that appears FIRST in the current sequence. Stop early if no pair occurs at least twice.
    Return the merges dict."""
    raise NotImplementedError("your code here")


def encode(text, merges):
    """Bytes of `text`, then repeatedly apply the merge with the LOWEST new_id (the earliest learned)
    among the pairs present, until no learned pair remains. Return the id list."""
    raise NotImplementedError("your code here")


def decode(ids, merges):
    """Map ids back to text: build vocab {id: bytes} (0..255 are single bytes; each merge concatenates its pair's
    bytes, in training order), join the bytes, and decode UTF-8 with errors="replace"."""
    raise NotImplementedError("your code here")
