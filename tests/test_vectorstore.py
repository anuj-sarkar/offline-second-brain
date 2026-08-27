"""
tests/test_vectorstore.py

Tests for app.retrieval.vectorstore. Uses a temporary directory for
the ChromaDB persist path so tests don't pollute or depend on your
real vectorstore/ folder, and uses small hand-built fake embeddings
(3-dimensional) instead of calling Ollama - these tests check
ChromaDB plumbing, not embedding quality (that's already covered in
test_embeddings.py).

Run with: python -m pytest tests/test_vectorstore.py -v
"""

import tempfile
from dataclasses import dataclass

from app.retrieval.vectorstore import (
    get_client, create_collection, add_documents, search,
    delete_document, list_documents,
)


@dataclass
class FakeEmbeddedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    embedding: list[float]


def _make_test_collection():
    """Fresh client + collection in a temp dir, isolated per test."""
    tmp_dir = tempfile.mkdtemp()
    client = get_client(persist_dir=tmp_dir)
    collection = create_collection(client, name="test_collection")
    return collection


def test_add_and_search_returns_most_similar_first():
    collection = _make_test_collection()

    chunks = [
        FakeEmbeddedChunk("c1", "d1", "paper_a.pdf", 1, "about cats", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c2", "d1", "paper_a.pdf", 2, "about dogs", [0.9, 0.1, 0.0]),
        FakeEmbeddedChunk("c3", "d2", "paper_b.pdf", 1, "about finance", [0.0, 0.0, 1.0]),
    ]
    add_documents(collection, chunks)

    # Query vector closest to "cats"/"dogs" direction, far from "finance"
    results = search(collection, query_embedding=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"  # most similar first
    assert results[0]["similarity"] > results[1]["similarity"]


def test_search_with_filename_filter():
    collection = _make_test_collection()

    chunks = [
        FakeEmbeddedChunk("c1", "d1", "paper_a.pdf", 1, "text one", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c2", "d2", "paper_b.pdf", 1, "text two", [1.0, 0.0, 0.0]),
    ]
    add_documents(collection, chunks)

    results = search(collection, query_embedding=[1.0, 0.0, 0.0], top_k=5,
                      filename_filter="paper_a.pdf")

    assert len(results) == 1
    assert results[0]["filename"] == "paper_a.pdf"


def test_search_with_multi_filename_filter():
    collection = _make_test_collection()

    chunks = [
        FakeEmbeddedChunk("c1", "d1", "paper_a.pdf", 1, "text one", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c2", "d2", "paper_b.pdf", 1, "text two", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c3", "d3", "paper_c.pdf", 1, "text three", [1.0, 0.0, 0.0]),
    ]
    add_documents(collection, chunks)

    # Filter by 2 out of 3 files
    results = search(collection, query_embedding=[1.0, 0.0, 0.0], top_k=5,
                      filename_filter=["paper_a.pdf", "paper_c.pdf"])

    assert len(results) == 2
    filenames = {r["filename"] for r in results}
    assert filenames == {"paper_a.pdf", "paper_c.pdf"}
    assert "paper_b.pdf" not in filenames


def test_delete_document_removes_only_its_chunks():
    collection = _make_test_collection()

    chunks = [
        FakeEmbeddedChunk("c1", "doc_to_delete", "a.pdf", 1, "text", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c2", "doc_to_keep", "b.pdf", 1, "text", [0.0, 1.0, 0.0]),
    ]
    add_documents(collection, chunks)

    deleted_count = delete_document(collection, doc_id="doc_to_delete")
    assert deleted_count == 1

    remaining = list_documents(collection)
    assert "a.pdf" not in remaining
    assert "b.pdf" in remaining


def test_list_documents_counts_chunks_per_file():
    collection = _make_test_collection()

    chunks = [
        FakeEmbeddedChunk("c1", "d1", "a.pdf", 1, "t", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c2", "d1", "a.pdf", 2, "t", [1.0, 0.0, 0.0]),
        FakeEmbeddedChunk("c3", "d2", "b.pdf", 1, "t", [0.0, 1.0, 0.0]),
    ]
    add_documents(collection, chunks)

    counts = list_documents(collection)
    assert counts == {"a.pdf": 2, "b.pdf": 1}