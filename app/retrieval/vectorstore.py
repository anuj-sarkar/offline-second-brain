"""
app/retrieval/vectorstore.py

Persistent local vector storage and semantic search using ChromaDB.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION_NAME = "research_library"


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    """Create a ChromaDB client that persists to disk."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def create_collection(
    client: chromadb.ClientAPI,
    name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Create or retrieve a collection with cosine distance metric."""
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{name}' ready ({collection.count()} items)")
    return collection


def add_documents(collection: Collection, embedded_chunks: list) -> None:
    """Add or update EmbeddedChunks in the ChromaDB collection."""
    if not embedded_chunks:
        logger.warning("add_documents called with empty list")
        return

    ids = [c.chunk_id for c in embedded_chunks]
    embeddings = [c.embedding for c in embedded_chunks]
    documents = [c.text for c in embedded_chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "filename": c.filename,
            "page_number": c.page_number,
            "section_title": getattr(c, "section_title", "General"),
        }
        for c in embedded_chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(embedded_chunks)} chunks in collection")


def search(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 5,
    filename_filter: str | None = None,
) -> list[dict]:
    """Semantic search returning ranked chunk dicts."""
    where_clause = {"filename": filename_filter} if filename_filter else None

    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause,
    )

    formatted = []
    if not results or not results["ids"] or not results["ids"][0]:
        return formatted

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        similarity = 1.0 - distances[i]
        meta = metadatas[i]
        formatted.append({
            "chunk_id": ids[i],
            "text": documents[i],
            "filename": meta.get("filename", "unknown"),
            "page_number": meta.get("page_number", 1),
            "doc_id": meta.get("doc_id", ""),
            "section_title": meta.get("section_title", "General"),
            "similarity": similarity,
        })

    return formatted


def delete_document(collection: Collection, doc_id: str) -> int:
    """Delete chunks matching doc_id."""
    existing = collection.get(where={"doc_id": doc_id})
    count = len(existing["ids"])
    if count == 0:
        return 0
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted {count} chunks for doc_id={doc_id}")
    return count


def delete_document_by_filename(collection: Collection, filename: str) -> int:
    """Delete all chunks belonging to a filename."""
    existing = collection.get(where={"filename": filename})
    count = len(existing["ids"])
    if count == 0:
        return 0
    collection.delete(where={"filename": filename})
    logger.info(f"Deleted {count} chunks for filename={filename}")
    return count


def list_documents(collection: Collection) -> dict[str, int]:
    """List document filenames and chunk counts."""
    all_items = collection.get()
    counts: dict[str, int] = {}
    for metadata in all_items.get("metadatas", []):
        filename = metadata.get("filename", "unknown")
        counts[filename] = counts.get(filename, 0) + 1
    return counts