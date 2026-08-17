"""
experiments/semantic_search_cli.py

Phase 5 deliverable: a standalone semantic search application, with
zero LLM involvement. This exists to prove retrieval works correctly
on its own, before any generation is layered on top in Phase 6/7 -
if something goes wrong later in the full RAG pipeline, this script
is your tool for isolating whether the bug is in retrieval or generation.

Usage:
    python -m experiments.semantic_search_cli "your question here"
"""

import sys

from app.retrieval.vectorstore import get_client, create_collection, search
from app.embeddings.embedder import embed_text


def run_search(query: str, top_k: int = 5) -> None:
    client = get_client()
    collection = create_collection(client)

    query_vector = embed_text(query)
    results = search(collection, query_vector, top_k=top_k)

    if not results:
        print("No results found. Has the collection been populated yet? "
              "Run `python -m app.retrieval.vectorstore` first to ingest documents.")
        return

    print(f"\nQuery: {query}\n")
    for rank, r in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Document: {r['filename']}")
        print(f"Page: {r['page_number']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"\nRelevant text:\n{r['text']}\n")
        print("-" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m experiments.semantic_search_cli "your question"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    run_search(query)