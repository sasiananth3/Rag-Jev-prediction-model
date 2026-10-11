from __future__ import annotations

from collections.abc import Iterable


def precision_at_k(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = list(retrieved_ids)[:k]
    if not retrieved:
        return 0.0
    relevant = set(relevant_ids)
    return sum(item in relevant for item in retrieved) / len(retrieved)


def recall_at_k(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = set(list(retrieved_ids)[:k])
    relevant = set(relevant_ids)
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0
