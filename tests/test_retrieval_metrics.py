
import pytest
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = ["B"]

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_zero_reciprocal_rank_when_no_match():
    retrieved = ["A", "B", "C"]
    relevant = ["D"]

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], ["A"], 0)


def test_empty_relevant_ids():
    with pytest.raises(ValueError):
        recall_at_k(["A"], [], 3)
