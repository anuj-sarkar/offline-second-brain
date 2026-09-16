"""
backend/main.py

FastAPI backend for Offline Second Brain.
Features:
  - Document management (upload, sync/index, delete, list).
  - Dynamic local model discovery via Ollama.
  - ND-JSON streaming RAG generation with robust error handling and citation verification.
  - Document-level filtering support.

Run with: uvicorn backend.main:app --reload --port 8000
"""

import json
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion.loader import load_pdfs_from_directory
from app.chunking.splitter import chunk_page_records
from app.embeddings.embedder import embed_chunks, embed_text
from app.retrieval.vectorstore import (
    get_client, create_collection, add_documents, list_documents, delete_document,
)
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.vectorstore import search as dense_search
from app.generation.rag_pipeline import construct_context, RAG_PROMPT_TEMPLATE
from app.generation.llm import generate_stream
from app.generation.citations import verify_citations
from config.model_config import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "data/raw"

app = FastAPI(title="Offline Second Brain API")

# Local-only tool - frontend dev server runs on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_strategy: str = "hybrid_no_rerank"  # Best on academic corpora (MRR 0.415). Avoid
                                                   # "hybrid"/"dense_rerank" — FlashRank (ms-marco)
                                                   # degrades MRR to 0.325 on research PDFs.
    model_name: str = "llama3.2:3b"
    temperature: float = 0.1
    filename_filter: str | list[str] | None = None


# ---------- Health ----------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- Models (Dynamic Ollama discovery) ----------
@app.get("/api/models")
def get_models():
    """
    Returns list of installed local LLM models from Ollama,
    filtering out embedding-only models like nomic-embed-text.
    """
    try:
        import ollama
        res = ollama.list()
        models_list = []
        for m in getattr(res, "models", []):
            name = getattr(m, "model", "") or getattr(m, "name", "")
            # filter out embed models
            if "embed" in name.lower():
                continue
            if name:
                models_list.append(name)

        if not models_list:
            models_list = ["llama3.2:3b"]

        return {"models": models_list, "default": models_list[0]}
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {e}")
        return {"models": ["llama3.2:3b"], "default": "llama3.2:3b"}


# ---------- Documents ----------
@app.get("/api/documents")
def get_documents():
    client = get_client()
    collection = create_collection(client)
    return list_documents(collection)


@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.lower().endswith((".pdf", ".txt", ".md")):
            continue
        dest = Path(RAW_DATA_DIR) / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"saved": saved}


@app.post("/api/index")
def run_indexing(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Runs the full ingestion pipeline synchronously.
    """
    records = load_pdfs_from_directory(RAW_DATA_DIR)
    chunks = chunk_page_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedded_chunks = embed_chunks(chunks)

    client = get_client()
    collection = create_collection(client)
    add_documents(collection, embedded_chunks)

    return {
        "pages_processed": len(records),
        "chunks_indexed": len(chunks),
        "documents": list_documents(collection),
    }


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    client = get_client()
    collection = create_collection(client)
    deleted = delete_document(collection, doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="No chunks found for that doc_id")
    return {"deleted_chunks": deleted}


# ---------- Ask (streaming) ----------
@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """
    Streams the RAG response as newline-delimited JSON events:
      {"type": "sources", "chunks": [...]}          - sent first, retrieved chunks
      {"type": "token", "text": "..."}               - sent repeatedly, one per generated token
      {"type": "done", "citation_report": {...}}     - sent last, full citation verification
      {"type": "error", "error": "..."}              - sent on failure
    """
    def event_stream():
        try:
            client = get_client()
            collection = create_collection(client)

            if req.retrieval_strategy == "hybrid":
                retrieved_chunks = hybrid_search(
                    req.question,
                    top_k=req.top_k,
                    filename_filter=req.filename_filter,
                )
            else:
                query_vector = embed_text(req.question, is_query=True)
                retrieved_chunks = dense_search(
                    collection,
                    query_vector,
                    top_k=req.top_k,
                    filename_filter=req.filename_filter,
                )

            yield json.dumps({"type": "sources", "chunks": retrieved_chunks}) + "\n"

            if not retrieved_chunks:
                if isinstance(req.filename_filter, list):
                    if len(req.filename_filter) == 1:
                        filter_desc = f" within '{req.filename_filter[0]}'"
                    else:
                        filter_desc = f" within the {len(req.filename_filter)} selected documents"
                elif isinstance(req.filename_filter, str) and req.filename_filter.strip():
                    filter_desc = f" within '{req.filename_filter.strip()}'"
                else:
                    filter_desc = ""

                msg = (
                    f"No relevant chunks found in the library for this query"
                    + (f"{filter_desc}." if filter_desc else ".")
                    + " Please ensure documents are uploaded and indexed."
                )
                yield json.dumps({"type": "token", "text": msg}) + "\n"
                yield json.dumps({
                    "type": "done",
                    "citation_report": {
                        "verified_count": 0,
                        "fabricated_count": 0,
                        "all_verified": True,
                        "has_any_citations": False,
                    },
                }) + "\n"
                return

            context = construct_context(retrieved_chunks)
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=req.question)
            llm_config = LLMConfig(model_name=req.model_name, temperature=req.temperature)

            full_answer = ""
            for token in generate_stream(prompt, config=llm_config):
                if not token:
                    continue
                full_answer += token
                yield json.dumps({"type": "token", "text": token}) + "\n"

            report = verify_citations(full_answer, retrieved_chunks)
            yield json.dumps({
                "type": "done",
                "citation_report": {
                    "verified_count": report.verified_count,
                    "fabricated_count": report.fabricated_count,
                    "all_verified": report.all_verified,
                    "has_any_citations": report.has_any_citations,
                },
            }) + "\n"

        except Exception as e:
            logger.exception(f"Error in ask_stream: {e}")
            yield json.dumps({
                "type": "error",
                "error": f"Generation Error: {str(e)}. Please verify that the model '{req.model_name}' is installed in Ollama (`ollama list`).",
            }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")