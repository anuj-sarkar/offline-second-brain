"""
tests/test_metrics.py

Tests for app.evaluation.metrics, using small hand-built result lists
where the correct metric value can be computed by hand and verified.

Run with: python -m pytest tests/test_metrics.py -v
"""

from app.evaluation.metrics import recall_at_k, precision_at_k, mean_reciprocal_rank


def _fake_results(pages: list[tuple[str, int]]) -> list[dict]:
    """Build fake ranked results from a list of (filename, page) tuples, in rank order."""
    return [{"filename": f, "page_number": p} for f, p in pages]


def test_recall_at_k_finds_all_relevant():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3)])
    relevant = [("a.pdf", 2)]
    assert recall_at_k(results, relevant, k=3) == 1.0


def test_recall_at_k_misses_relevant_outside_k():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3)])
    relevant = [("a.pdf", 2)]
    assert recall_at_k(results, relevant, k=1) == 0.0  # relevant page is rank 2, outside top-1


def test_recall_at_k_partial_when_only_some_relevant_pages_found():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 2)])
    relevant = [("a.pdf", 2), ("a.pdf", 99)]  # page 99 never retrieved
    assert recall_at_k(results, relevant, k=2) == 0.5


def test_precision_at_k_all_relevant():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 2)])
    relevant = [("a.pdf", 1), ("a.pdf", 2)]
    assert precision_at_k(results, relevant, k=2) == 1.0


def test_precision_at_k_half_relevant():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 99)])  # only page 1 is relevant
    relevant = [("a.pdf", 1)]
    assert precision_at_k(results, relevant, k=2) == 0.5


def test_mrr_first_result_relevant():
    results = _fake_results([("a.pdf", 1), ("a.pdf", 2)])
    relevant = [("a.pdf", 1)]
    assert mean_reciprocal_rank(results, relevant) == 1.0


def test_mrr_second_result_relevant():
    results = _fake_results([("a.pdf", 99), ("a.pdf", 1)])
    relevant = [("a.pdf", 1)]
    assert mean_reciprocal_rank(results, relevant) == 0.5


def test_mrr_no_relevant_result_found():
    results = _fake_results([("a.pdf", 99), ("a.pdf", 98)])
    relevant = [("a.pdf", 1)]
    assert mean_reciprocal_rank(results, relevant) == 0.0