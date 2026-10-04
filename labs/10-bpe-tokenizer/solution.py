"""Lab 10: a byte-level BPE tokenizer (the GPT-2 family's core idea, without the regex pre-split).

Token ids 0..255 are raw UTF-8 bytes. Training adds merges: each merge maps a pair of ids to a new id 256, 257, ...
`merges` is a dict {(a, b): new_id} whose insertion order is the training order.
"""


def get_pair_counts(ids):
    """Counts of adjacent pairs, as a dict {(a, b): count} in order of first appearance.
    Example: [1, 2, 1, 2, 3] -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair`, scanning left to right, with new_id.
    Example: merge([1, 1, 1], (1, 1), 9) -> [9, 1]."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, vocab_size):
    """Learn vocab_size - 256 merges from the UTF-8 bytes of `text`. At each step merge the most frequent pair;
    ties go to the pair that appears FIRST in the current sequence. Stop early if no pair occurs at least twice.
    Return the merges dict."""
    ids = list(text.encode("utf-8"))
    merges = {}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)          # max returns the first of equal maxima
        if counts[pair] < 2:
            break
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges


def encode(text, merges):
    """Bytes of `text`, then repeatedly apply the merge with the LOWEST new_id (the earliest learned)
    among the pairs present, until no learned pair remains. Return the id list."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = get_pair_counts(ids)
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, merges):
    """Map ids back to text: build vocab {id: bytes} (0..255 are single bytes; each merge concatenates its pair's
    bytes, in training order), join the bytes, and decode UTF-8 with errors="replace"."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
