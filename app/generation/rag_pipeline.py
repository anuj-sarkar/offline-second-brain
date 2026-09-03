"""
app/generation/rag_pipeline.py

Complete RAG pipeline orchestrating multi-turn query condensation,
hybrid retrieval with FlashRank cross-encoder reranking, and grounded answer generation.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
import logging
from typing import Optional

from app.embeddings.embedder import embed_text
from app.generation.llm import generate, generate_stream
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank_chunks
from app.retrieval.vectorstore import (
    create_collection,
    get_client,
    search as dense_search,
)
from config.model_config import DEFAULT_LLM_CONFIG, LLMConfig

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions using ONLY the provided context from the user's document library.

STRICT RULES:
1. Answer using ONLY information found in the context below. Do not use outside/pretrained knowledge to fill gaps.
2. If the context does not contain enough information to answer the question, say so explicitly: "The provided documents don't contain enough information to answer this."
3. If the context contains mathematical formulas, equations, algorithms, matrices, or scientific symbols, faithfully transcribe and format them using standard LaTeX ($...$ for inline math, $$...$$ for display equations). If an equation appears fragmented or broken across lines in the raw text, reconstruct it cleanly into standard LaTeX syntax.
4. Every claim you make must be traceable to a specific source. After EACH distinct claim or sentence, cite it like this: [Source: filename, Page X]. If your answer combines facts from multiple pages, cite each fact separately at the point it's made - do not put one single citation at the very end covering the whole answer.
5. If you genuinely need to add general knowledge NOT found in the context to make the answer understandable, clearly label it as: "(General knowledge, not from your documents: ...)" - never blend it in silently as if it came from the documents.
6. Do not fabricate page numbers, filenames, or quotes. Only cite what is actually shown in the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

CONDENSE_QUERY_TEMPLATE = """Given the previous chat history and the latest user question, rephrase the question into a clear, standalone search query that preserves all relevant context. Do NOT answer the question, only output the standalone query.

Chat History:
{chat_history}

Latest Question: {question}

Standalone Search Query:"""


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved_chunks: list[dict]
    standalone_query: str = ""


def format_chat_history(chat_history: list[dict], max_turns: int = 3) -> str:
    """Format recent chat turns for the query condenser prompt."""
    if not chat_history:
        return ""
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        q = turn.get("question", "")
        a = turn.get("answer", "")
        # Truncate answer to avoid prompt bloat
        a_snippet = (a[:250] + "...") if len(a) > 250 else a
        lines.append(f"User: {q}\nAssistant: {a_snippet}")
    return "\n\n".join(lines)


def condense_query(
    question: str,
    chat_history: list[dict] | None = None,
    llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> str:
    """
    Rewrite follow-up questions containing pronouns into standalone search queries.
    """
    if not chat_history:
        return question

    history_text = format_chat_history(chat_history)
    if not history_text:
        return question

    prompt = CONDENSE_QUERY_TEMPLATE.format(
        chat_history=history_text,
        question=question,
    )

    try:
        standalone = generate(prompt, config=llm_config).strip()
        # Clean any accidental quotes
        standalone = standalone.strip('"\'')
        logger.info(f"Condensed query: '{question}' -> '{standalone}'")
        return standalone if standalone else question
    except Exception as e:
        logger.warning(f"Query condensation failed: {e}. Using original question.")
        return question


def construct_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into labeled context blocks."""
    if not chunks:
        return "(No relevant context was found in the document library.)"

    blocks = []
    for chunk in chunks:
        sec = chunk.get("section_title", "General")
        section_label = f" | Section: {sec}" if sec and sec != "General" else ""
        block = (
            f"[Source: {chunk['filename']}, Page {chunk['page_number']}{section_label}]\n"
            f"{chunk['text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def retrieve_context(
    query: str,
    top_k: int = 5,
    retrieval_strategy: str = "hybrid",
) -> list[dict]:
    """Retrieve top-K chunks according to the chosen strategy."""
    client = get_client()
    collection = create_collection(client)

    if collection.count() == 0:
        return []

    if retrieval_strategy == "hybrid":
        # Hybrid (Dense + BM25) with FlashRank reranking
        return hybrid_search(query, top_k=top_k, use_reranker=True)
    elif retrieval_strategy == "hybrid_no_rerank":
        return hybrid_search(query, top_k=top_k, use_reranker=False)
    elif retrieval_strategy == "dense_rerank":
        query_vector = embed_text(query, is_query=True)
        candidates = dense_search(collection, query_vector, top_k=top_k * 3)
        return rerank_chunks(query, candidates, top_k=top_k)
    else:  # "dense"
        query_vector = embed_text(query, is_query=True)
        return dense_search(collection, query_vector, top_k=top_k)


def answer_question(
    question: str,
    top_k: int = 5,
    llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
    retrieval_strategy: str = "hybrid",
    chat_history: list[dict] | None = None,
) -> RAGResult:
    """Execute complete RAG pipeline synchronously."""
    standalone_query = condense_query(question, chat_history, llm_config=llm_config)
    retrieved_chunks = retrieve_context(standalone_query, top_k=top_k, retrieval_strategy=retrieval_strategy)

    context = construct_context(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate(prompt, config=llm_config)

    return RAGResult(
        question=question,
        standalone_query=standalone_query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
    )


def answer_question_stream(
    question: str,
    top_k: int = 5,
    llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
    retrieval_strategy: str = "hybrid",
    chat_history: list[dict] | None = None,
) -> tuple[Generator[str, None, None], list[dict], str]:
    """
    Prepare RAG pipeline for streaming generation.
    Returns (token_generator, retrieved_chunks, standalone_query).
    """
    standalone_query = condense_query(question, chat_history, llm_config=llm_config)
    retrieved_chunks = retrieve_context(standalone_query, top_k=top_k, retrieval_strategy=retrieval_strategy)

    context = construct_context(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    token_generator = generate_stream(prompt, config=llm_config)

    return token_generator, retrieved_chunks, standalone_query