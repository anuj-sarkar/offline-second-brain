"""
app/embeddings/embedder.py

Embedding interface using LangChain's OllamaEmbeddings.
Applies asymmetric task prefixing automatically:
  - embed_documents() → "search_document: <text>"  (for indexing)
  - embed_query()     → "search_query: <text>"      (for retrieval)
This aligns with nomic-embed-text's training conventions.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_ollama import OllamaEmbeddings

from config.model_config import DEFAULT_EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = DEFAULT_EMBEDDING_CONFIG.model_name

# Singleton embedder — created once, reused across calls
_embedder: OllamaEmbeddings | None = None


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
        logger.info(f"Initialized OllamaEmbeddings with model='{EMBEDDING_MODEL}'")
    return _embedder


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
    Uses embed_query() for retrieval-time queries (adds 'search_query:' prefix)
    and embed_documents() for document chunks (adds 'search_document:' prefix).
    """
    embedder = _get_embedder()
    if is_query:
        return embedder.embed_query(text.strip())
    else:
        return embedder.embed_documents([text.strip()])[0]


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
    Returns EmbeddedChunk instances with the vector stored in .embedding.
    """
    if not chunks:
        return []

    logger.info(f"Embedding {len(chunks)} chunks using {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_embed_single_chunk, chunks))

    logger.info(f"Finished embedding {len(results)} chunks")
    return results