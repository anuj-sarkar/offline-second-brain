"""
app/retrieval/vectorstore.py

Persistent local vector storage and semantic search using LangChain's Chroma wrapper.
Public API is preserved so that app.py and hybrid_search.py work without changes:
  - get_client()                       → returns a raw chromadb.ClientAPI (for BM25 + count checks)
  - create_collection()                → returns a raw chromadb Collection (for BM25 + count checks)
  - add_documents(collection, chunks)  → upserts EmbeddedChunks into ChromaDB
  - search(collection, vector, top_k)  → semantic search returning ranked chunk dicts
  - delete_document(collection, doc_id)
  - delete_document_by_filename(collection, filename)
  - list_documents(collection)
  - get_langchain_vectorstore()        → returns the LangChain Chroma instance (for LCEL chains)
"""

import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config.model_config import DEFAULT_EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION_NAME = "research_library"

# ── Singleton LangChain vectorstore ──────────────────────────────────────────
_lc_vectorstore: Chroma | None = None


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=DEFAULT_EMBEDDING_CONFIG.model_name)


def get_langchain_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Return (or lazily create) the singleton LangChain Chroma vectorstore."""
    global _lc_vectorstore
    if _lc_vectorstore is None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _lc_vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"LangChain Chroma vectorstore ready (collection='{collection_name}')")
    return _lc_vectorstore


# ── Raw chromadb helpers (kept for BM25 index + chunk-count checks) ───────────

def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    """Create a ChromaDB client that persists to disk (used by hybrid_search for BM25)."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def create_collection(
    client: chromadb.ClientAPI,
    name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Create or retrieve a raw ChromaDB collection (used by hybrid_search and app.py)."""
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{name}' ready ({collection.count()} items)")
    return collection


# ── Write operations ──────────────────────────────────────────────────────────

def add_documents(collection: Collection, embedded_chunks: list) -> None:
    """Add or update EmbeddedChunks in ChromaDB via the raw client (preserves chunk IDs)."""
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

    # Invalidate the LangChain singleton so it re-attaches to the updated collection
    global _lc_vectorstore
    _lc_vectorstore = None

    logger.info(f"Upserted {len(embedded_chunks)} chunks in collection")


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 5,
    filename_filter: str | list[str] | None = None,
) -> list[dict]:
    """Semantic search returning ranked chunk dicts (used by hybrid_search dense leg)."""
    where_clause = None
    if isinstance(filename_filter, str):
        if filename_filter.strip():
            where_clause = {"filename": filename_filter.strip()}
    elif isinstance(filename_filter, (list, tuple, set)):
        filtered_list = [f.strip() for f in filename_filter if isinstance(f, str) and f.strip()]
        if not filtered_list:
            return []
        elif len(filtered_list) == 1:
            where_clause = {"filename": filtered_list[0]}
        else:
            where_clause = {"filename": {"$in": filtered_list}}

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


# ── Delete operations ─────────────────────────────────────────────────────────

def delete_document(collection: Collection, doc_id: str) -> int:
    """Delete chunks matching doc_id."""
    existing = collection.get(where={"doc_id": doc_id})
    count = len(existing["ids"])
    if count == 0:
        return 0
    collection.delete(where={"doc_id": doc_id})
    global _lc_vectorstore
    _lc_vectorstore = None
    logger.info(f"Deleted {count} chunks for doc_id={doc_id}")
    return count


def delete_document_by_filename(collection: Collection, filename: str) -> int:
    """Delete all chunks belonging to a filename."""
    existing = collection.get(where={"filename": filename})
    count = len(existing["ids"])
    if count == 0:
        return 0
    collection.delete(where={"filename": filename})
    global _lc_vectorstore
    _lc_vectorstore = None
    logger.info(f"Deleted {count} chunks for filename={filename}")
    return count


# ── Listing ───────────────────────────────────────────────────────────────────

def list_documents(collection: Collection) -> dict[str, int]:
    """List document filenames and chunk counts."""
    all_items = collection.get()
    counts: dict[str, int] = {}
    for metadata in all_items.get("metadatas", []):
        filename = metadata.get("filename", "unknown")
        counts[filename] = counts.get(filename, 0) + 1
    return counts