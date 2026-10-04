import pytest

CORPUS = ("the cat sat on the mat. the cat ate the rat. "
          "o pisică stă pe preș; pisica a mâncat șoarecele. ") * 20


def test_pair_counts_and_merge(impl):
    assert impl.get_pair_counts([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}
    assert list(impl.get_pair_counts([1, 2, 1, 2, 3])) == [(1, 2), (2, 1), (2, 3)]
    assert impl.merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert impl.merge([5, 1, 2, 1, 2], (1, 2), 7) == [5, 7, 7]


def test_train_on_the_classic_example(impl):
    merges = impl.train_bpe("aaabdaaabac", 259)
    a, b = ord("a"), ord("b")
    assert merges == {(a, a): 256, (256, a): 257, (257, b): 258}
    assert impl.encode("aaabdaaabac", merges) == [258, ord("d"), 258, a, ord("c")]


def test_roundtrip_including_unicode(impl):
    merges = impl.train_bpe(CORPUS, 300)
    for text in [CORPUS[:200], "unseen words: zebra, ăîșț, 🙂", ""]:
        assert impl.decode(impl.encode(text, merges), merges) == text


def test_compression(impl):
    merges = impl.train_bpe(CORPUS, 320)
    assert len(merges) == 64
    n_bytes = len(CORPUS.encode("utf-8"))
    assert len(impl.encode(CORPUS, merges)) < 0.4 * n_bytes


def test_stops_when_nothing_repeats(impl):
    assert impl.train_bpe("abcdef", 300) == {}
