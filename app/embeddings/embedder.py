"""
app/embeddings/embedder.py

Phase 3 deliverable: turn text into vectors using a local embedding
model (nomic-embed-text via Ollama), and provide the similarity
function used to compare them.

Kept deliberately thin - this module has exactly two jobs:
  1. call the embedding model
  2. compute cosine similarity
Nothing about chunking or vector storage lives here (that's Phase 2
and Phase 4's job respectively) - this is intentional separation of
concerns, same as the ingestion/chunking split.
"""

import logging
import math
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"


@dataclass
class EmbeddedChunk:
    """
    A Chunk (from Phase 2) plus its embedding vector attached.
    Kept as a separate type rather than bolting a `vector` field onto
    Chunk itself - this keeps chunking (Phase 2) fully independent of
    embeddings (Phase 3); Chunk objects don't need to know embeddings
    exist at all.
    """
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    embedding: list[float]


def embed_text(text: str) -> list[float]:
    """
    Get the embedding vector for a single piece of text.

    Input:  a string (a chunk, a query, a sentence - doesn't matter)
    Output: a list of floats (768 numbers for nomic-embed-text)

    This is a thin wrapper around Ollama's embedding endpoint. Note
    it's a single blocking network call to your local Ollama server -
    for many chunks, embed_chunks() below batches these calls but
    still does them one at a time (Ollama's Python client doesn't
    batch-embed in one request as of this version - worth knowing,
    since it's a real performance limitation you'd hit at scale).
    """
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors, from scratch -
    no numpy, so you see exactly what's happening arithmetically.

    Input:  two equal-length lists of floats
    Output: a float, typically in [0, 1] for text embeddings
            (1 = identical direction/meaning, 0 = unrelated)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}. "
            "This usually means the two texts were embedded with "
            "different models - they must match."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def embed_chunks(chunks: list) -> list[EmbeddedChunk]:
    """
    Embed a list of Chunk objects (from app.chunking.splitter).

    Input:  list[Chunk]
    Output: list[EmbeddedChunk] - same chunks, now with vectors attached

    Logs progress every 20 chunks since embedding 200+ chunks one at a
    time can take a noticeable while, and silent long-running loops
    are a common source of "is this stuck?" confusion.
    """
    embedded: list[EmbeddedChunk] = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk.text)
        embedded.append(EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            text=chunk.text,
            embedding=vector,
        ))
        if (i + 1) % 20 == 0:
            logger.info(f"Embedded {i + 1}/{len(chunks)} chunks")

    logger.info(f"Finished embedding {len(embedded)} chunks")
    return embedded


if __name__ == "__main__":
    # --- Experiment: semantically similar vs. dissimilar sentences ---
    # This proves cosine similarity actually tracks meaning before we
    # trust it to power retrieval over your real documents.

    pairs = [
        # Similar meaning, different words - should score HIGH
        ("The cat sat on the mat.", "A cat was sitting on a rug."),
        ("Transformers use self-attention mechanisms.",
         "The attention mechanism is central to transformer architectures."),
        # Unrelated meaning - should score LOW
        ("The cat sat on the mat.", "Stock markets fell sharply today."),
        ("Transformers use self-attention mechanisms.",
         "I need to buy groceries this weekend."),
        # Same topic, different specifics - should score MEDIUM
        ("NSGA-II is a genetic algorithm for multi-objective optimization.",
         "Genetic algorithms are used to solve optimization problems."),
    ]

    print(f"Using embedding model: {EMBEDDING_MODEL}\n")
    print("=" * 70)

    for text_a, text_b in pairs:
        vec_a = embed_text(text_a)
        vec_b = embed_text(text_b)
        sim = cosine_similarity(vec_a, vec_b)

        print(f"\nText A: {text_a}")
        print(f"Text B: {text_b}")
        print(f"Cosine similarity: {sim:.4f}")
        print(f"Vector dimensionality: {len(vec_a)}")
        print("-" * 70)