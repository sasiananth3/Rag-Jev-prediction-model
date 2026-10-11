import pytest

from metrics import precision_at_k, recall_at_k
from rag_pipeline import split_text


def test_precision_at_k():
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["a", "x"], ["a", "b"], 2) == pytest.approx(0.5)


def test_empty_retrieval():
    assert precision_at_k([], ["a"], 5) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0


def test_split_text_preserves_ends():
    chunks = split_text("alpha beta gamma delta", chunk_size=12, overlap=3)
    assert chunks
    assert "alpha" in chunks[0]
    assert "delta" in chunks[-1]


def test_split_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=10)
