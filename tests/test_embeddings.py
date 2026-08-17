"""
tests/test_embeddings.py

Tests for app.embeddings.embedder. These make real calls to your local
Ollama server (no mocking) - that's intentional for this phase, since
the whole point is proving the local embedding pipeline actually works
end to end. Requires `ollama pull nomic-embed-text` to have been run.

Run with: python -m pytest tests/test_embeddings.py -v
"""

import pytest
from app.embeddings.embedder import embed_text, cosine_similarity


def test_embed_text_returns_correct_dimensionality():
    vector = embed_text("This is a test sentence.")
    assert isinstance(vector, list)
    assert len(vector) == 768  # nomic-embed-text's fixed dimensionality
    assert all(isinstance(v, float) for v in vector)


def test_identical_text_has_similarity_near_one():
    text = "The Transformer architecture relies on self-attention."
    vec_a = embed_text(text)
    vec_b = embed_text(text)
    sim = cosine_similarity(vec_a, vec_b)
    assert sim > 0.99  # same text embedded twice should be ~identical


def test_similar_sentences_score_higher_than_unrelated_ones():
    query = "How does the attention mechanism work?"
    related = "The attention mechanism computes weighted relationships between tokens."
    unrelated = "I need to buy groceries this weekend."

    vec_query = embed_text(query)
    vec_related = embed_text(related)
    vec_unrelated = embed_text(unrelated)

    sim_related = cosine_similarity(vec_query, vec_related)
    sim_unrelated = cosine_similarity(vec_query, vec_unrelated)

    assert sim_related > sim_unrelated


def test_mismatched_vector_lengths_raise_error():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])