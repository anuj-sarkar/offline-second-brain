"""
app/evaluation/metrics.py

Phase 11 deliverable: standard information-retrieval metrics for
scoring retrieval quality against known-relevant pages.

These operate on (filename, page_number) pairs, not raw similarity
scores - relevance here is binary (either a retrieved chunk's page
is in the ground-truth relevant set, or it isn't), which is the
standard, simplest formulation of these metrics.
"""


def _retrieved_pages(results: list[dict]) -> list[tuple[str, int]]:
    """Extract (filename, page_number) from ranked retrieval results, in rank order."""
    return [(r["filename"], r["page_number"]) for r in results]


def recall_at_k(results: list[dict], relevant_pages: list[tuple[str, int]], k: int) -> float:
    """
    Recall@K: of all the relevant pages that exist, what fraction did
    we find within the top K retrieved results?

    Input:  ranked retrieval results, the ground-truth relevant pages,
            and K (how many top results to consider)
    Output: a float in [0, 1] - 1.0 means every relevant page was
            found somewhere in the top K, 0.0 means none were

    This answers: "did we find everything we needed?" - it does NOT
    penalize for retrieving irrelevant chunks alongside the relevant
    ones (that's precision's job).
    """
    if not relevant_pages:
        return 0.0

    top_k_pages = set(_retrieved_pages(results[:k]))
    relevant_set = set(relevant_pages)

    found = relevant_set & top_k_pages
    return len(found) / len(relevant_set)


def precision_at_k(results: list[dict], relevant_pages: list[tuple[str, int]], k: int) -> float:
    """
    Precision@K: of the top K results we retrieved, what fraction
    were actually relevant?

    Output: a float in [0, 1] - 1.0 means every one of the top K
            results was relevant, 0.0 means none were

    This answers: "how much noise did we retrieve alongside the
    signal?" - a system can have perfect recall but terrible
    precision if it just returns a huge K stuffed with irrelevant
    chunks alongside the relevant ones.
    """
    if k == 0 or not results:
        return 0.0

    top_k_pages = _retrieved_pages(results[:k])
    relevant_set = set(relevant_pages)

    relevant_count = sum(1 for page in top_k_pages if page in relevant_set)
    return relevant_count / min(k, len(top_k_pages))


def mean_reciprocal_rank(results: list[dict], relevant_pages: list[tuple[str, int]]) -> float:
    """
    Reciprocal Rank (for ONE query - averaging across queries in the
    caller gives you MRR): 1 / (rank of the FIRST relevant result).

    Output: a float in [0, 1] - 1.0 means the very first result was
            relevant, 0.5 means the first relevant result was at
            rank 2, 0.0 means no relevant result was found at all

    This specifically measures "how quickly does the user see
    something useful?" - it only cares about the first hit, ignoring
    everything after, which makes it a good complement to recall
    (which cares about finding ALL relevant pages, not just the first).
    """
    relevant_set = set(relevant_pages)
    for rank, page in enumerate(_retrieved_pages(results), start=1):
        if page in relevant_set:
            return 1.0 / rank
    return 0.0