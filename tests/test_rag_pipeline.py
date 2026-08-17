"""
tests/test_rag_pipeline.py

Tests for app.generation.rag_pipeline. These are integration tests -
they exercise the real vectorstore (populated from earlier phases)
and real Ollama generation. Run this AFTER app.retrieval.vectorstore
has been run at least once (so the collection has data).

Run with: python -m pytest tests/test_rag_pipeline.py -v
"""

from app.generation.rag_pipeline import construct_context, answer_question


def test_construct_context_formats_chunks_with_source_labels():
    chunks = [
        {"filename": "paper.pdf", "page_number": 3, "text": "Some relevant text.", "similarity": 0.8},
        {"filename": "paper.pdf", "page_number": 5, "text": "More relevant text.", "similarity": 0.7},
    ]
    context = construct_context(chunks)

    assert "[Source: paper.pdf, Page 3]" in context
    assert "[Source: paper.pdf, Page 5]" in context
    assert "Some relevant text." in context
    assert "More relevant text." in context


def test_construct_context_handles_empty_chunks():
    context = construct_context([])
    assert "No relevant context" in context


def test_answer_question_returns_grounded_answer_with_sources():
    result = answer_question("What is the Transformer architecture based on?", top_k=3)

    assert result.answer
    assert len(result.retrieved_chunks) > 0
    # The answer should reference attention, since that's the actual
    # subject matter of the retrieved paper - a loose but meaningful
    # sanity check that generation used the context, not generic filler
    assert "attention" in result.answer.lower()


def test_answer_question_declines_when_answer_not_in_documents():
    # A question with essentially zero chance of being covered by
    # research papers on transformers/genetic algorithms
    result = answer_question("What is the capital of Australia?", top_k=3)

    # We can't guarantee exact wording, but the model should NOT
    # confidently state "Canberra" as if it came from the documents -
    # it should indicate the documents don't cover this
    answer_lower = result.answer.lower()
    declined_or_flagged = (
        "don't contain" in answer_lower
        or "not" in answer_lower and ("found" in answer_lower or "information" in answer_lower)
        or "general knowledge" in answer_lower
    )
    assert declined_or_flagged, f"Expected a decline/flag, got: {result.answer}"