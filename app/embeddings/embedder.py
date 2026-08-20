"""
app/embeddings/embedder.py

Turns text into embedding vectors using local Ollama embedding models (nomic-embed-text).
Applies asymmetric task prefixing:
  - Document chunks: "search_document: <text>"
  - Query questions: "search_query: <query>"
This aligns with nomic-embed-text's training and maximizes retrieval quality.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import math
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"


@dataclass
class EmbeddedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    embedding: list[float]
    section_title: str = "General"


def embed_text(text: str, is_query: bool = False) -> list[float]:
    """
    Get the embedding vector for a piece of text.
    Applies asymmetric prefixing for nomic-embed-text.
    """
    prefix = "search_query: " if is_query else "search_document: "
    formatted_input = f"{prefix}{text.strip()}"
    response = ollama.embed(model=EMBEDDING_MODEL, input=formatted_input)
    return response["embeddings"][0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}. "
            "Embeddings must come from the same model."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def _embed_single_chunk(chunk) -> EmbeddedChunk:
    vector = embed_text(chunk.text, is_query=False)
    section_title = getattr(chunk, "section_title", "General")
    return EmbeddedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        text=chunk.text,
        embedding=vector,
        section_title=section_title,
    )


def embed_chunks(chunks: list, max_workers: int = 4) -> list[EmbeddedChunk]:
    """
    Embed a list of Chunk objects concurrently using a worker pool.
    """
    if not chunks:
        return []

    logger.info(f"Embedding {len(chunks)} chunks using {max_workers} workers...")

    # We preserve original chunk order by embedding via indexed mapping
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_embed_single_chunk, chunks))

    logger.info(f"Finished embedding {len(results)} chunks")
    return results