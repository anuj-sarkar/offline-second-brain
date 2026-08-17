"""
app/retrieval/vectorstore.py

Phase 4 deliverable: persistent local vector storage and search using
ChromaDB.

This module wraps ChromaDB with the exact operations your project
needs: create_collection, add_documents, search, delete_document,
list_documents. It deliberately does NOT do embedding itself - it
takes already-embedded chunks (from app.embeddings.embedder) and
stores/searches them. Keeping embedding and storage separate means
you could swap ChromaDB for Qdrant later without touching Phase 3 at
all (see the migration notes at the bottom of this file).
"""

import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION_NAME = "research_library"


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    """
    Create a ChromaDB client that persists to disk at `persist_dir`.

    This is the key line that makes storage survive between program
    runs: PersistentClient writes its index and data to this folder
    automatically after every add/delete - no manual save() needed.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def create_collection(
    client: chromadb.ClientAPI,
    name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """
    Create (or get, if it already exists) a named collection.

    Input:  a ChromaDB client, a collection name
    Output: a Collection object - the handle you use for all further
            add/search/delete operations

    get_or_create_collection is used instead of create_collection so
    that re-running ingestion doesn't crash on "collection already
    exists" - this matters because you'll re-run ingestion scripts
    many times while developing.
    """
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # explicitly use cosine distance,
                                              # matching what we used by hand
                                              # in Phase 3 - ChromaDB defaults
                                              # to squared L2 otherwise, which
                                              # would silently rank results
                                              # differently than you'd expect
    )
    logger.info(f"Collection '{name}' ready ({collection.count()} items)")
    return collection


def add_documents(collection: Collection, embedded_chunks: list) -> None:
    """
    Add a list of EmbeddedChunk (from app.embeddings.embedder) to the
    collection.

    Input:  list[EmbeddedChunk]
    Output: None (writes to the collection, persisted to disk automatically)

    ChromaDB's `add` wants four parallel lists: ids, embeddings,
    documents (the raw text), and metadatas (a dict per item). We
    build all four from EmbeddedChunk fields here. `upsert` is used
    instead of `add` so re-running ingestion on the same chunk_id
    updates rather than errors on a duplicate ID.
    """
    if not embedded_chunks:
        logger.warning("add_documents called with an empty list - nothing to add")
        return

    ids = [c.chunk_id for c in embedded_chunks]
    embeddings = [c.embedding for c in embedded_chunks]
    documents = [c.text for c in embedded_chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "filename": c.filename,
            "page_number": c.page_number,
        }
        for c in embedded_chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(f"Added/updated {len(embedded_chunks)} chunks in collection")


def search(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 5,
    filename_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search: find the top_k most similar chunks to a query vector.

    Input:  the collection, a query embedding (from embed_text() on the
            user's question), how many results to return, and an
            optional metadata filter
    Output: list of dicts, each with text, filename, page_number,
            chunk_id, and similarity - ranked best-first

    Note the query embedding must come from the SAME embedding model
    used to embed the chunks (nomic-embed-text) - this is the "same
    coordinate system" requirement from Phase 3.
    """
    where_clause = {"filename": filename_filter} if filename_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )

    # ChromaDB returns parallel lists nested one level for batch queries -
    # we only sent one query, so we unwrap index [0] throughout.
    formatted = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        # With cosine space, ChromaDB returns cosine DISTANCE (0=identical,
        # 2=opposite), not similarity. Convert back to the similarity scale
        # we used in Phase 3 so results are directly comparable/readable.
        similarity = 1 - distances[i]

        formatted.append({
            "chunk_id": ids[i],
            "text": documents[i],
            "filename": metadatas[i]["filename"],
            "page_number": metadatas[i]["page_number"],
            "doc_id": metadatas[i]["doc_id"],
            "similarity": similarity,
        })

    return formatted


def delete_document(collection: Collection, doc_id: str) -> int:
    """
    Delete all chunks belonging to a specific document (by doc_id).

    Input:  the collection, a doc_id (from PageRecord/Chunk metadata)
    Output: number of chunks deleted

    Useful when a source PDF is updated or removed and you need to
    re-index it without leaving stale chunks behind.
    """
    existing = collection.get(where={"doc_id": doc_id})
    count = len(existing["ids"])

    if count == 0:
        logger.warning(f"No chunks found for doc_id={doc_id}")
        return 0

    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted {count} chunks for doc_id={doc_id}")
    return count


def list_documents(collection: Collection) -> dict[str, int]:
    """
    List every distinct document currently in the collection, with
    how many chunks each contributed.

    Input:  the collection
    Output: dict mapping filename -> chunk count

    Note: for a large collection, collection.get() without limits
    pulls everything into memory - fine at our current scale (a few
    hundred chunks), but worth flagging as something to paginate if
    your library grows into the tens of thousands of chunks.
    """
    all_items = collection.get()
    counts: dict[str, int] = {}

    for metadata in all_items["metadatas"]:
        filename = metadata["filename"]
        counts[filename] = counts.get(filename, 0) + 1

    return counts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from app.ingestion.loader import load_pdfs_from_directory
    from app.chunking.splitter import chunk_page_records
    from app.embeddings.embedder import embed_chunks, embed_text

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # --- Full pipeline: ingest -> chunk -> embed -> store ---
    print("Step 1: Loading and chunking documents...")
    records = load_pdfs_from_directory("data/raw")
    chunks = chunk_page_records(records, chunk_size=1000, chunk_overlap=200)

    print(f"Step 2: Embedding {len(chunks)} chunks (this will take a bit)...")
    embedded_chunks = embed_chunks(chunks)

    print("Step 3: Storing in ChromaDB...")
    client = get_client()
    collection = create_collection(client)
    add_documents(collection, embedded_chunks)

    print("\nStep 4: Listing documents in the collection:")
    for filename, count in list_documents(collection).items():
        print(f"  {filename}: {count} chunks")

    # --- Prove search works with a real query ---
    print("\nStep 5: Test search")
    query = "How does the attention mechanism work in transformers?"
    query_vector = embed_text(query)
    results = search(collection, query_vector, top_k=3)

    print(f"\nQuery: {query}\n")
    for rank, r in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Document: {r['filename']}")
        print(f"Page: {r['page_number']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"Text: {r['text'][:200]}...")
        print()