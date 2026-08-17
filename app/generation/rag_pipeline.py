"""
app/generation/rag_pipeline.py

Phase 7 deliverable: the complete RAG pipeline. Combines retrieval
(app.retrieval.vectorstore) and generation (app.generation.llm) via
a prompt template specifically designed to keep the LLM grounded in
retrieved context rather than its own pretrained knowledge.

This module is intentionally the ONLY place retrieval and generation
are wired together - both underlying modules stay independently
testable (Phase 5 proved retrieval alone, Phase 6 proved generation
alone). If the full pipeline misbehaves, you can isolate whether the
bug is in retrieval, generation, or the glue logic here.
"""

import logging
from dataclasses import dataclass

from app.embeddings.embedder import embed_text
from app.retrieval.vectorstore import get_client, create_collection, search
from app.generation.llm import generate
from config.model_config import LLMConfig, DEFAULT_LLM_CONFIG

logger = logging.getLogger(__name__)


RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions using ONLY the provided context from the user's document library.

STRICT RULES:
1. Answer using ONLY information found in the context below. Do not use outside/pretrained knowledge to fill gaps.
2. If the context does not contain enough information to answer the question, say so explicitly: "The provided documents don't contain enough information to answer this."
3. Every claim you make must be traceable to a specific source. After EACH distinct claim or sentence, cite it like this: [Source: filename, Page X]. If your answer combines facts from multiple pages, cite each fact separately at the point it's made - do not put one single citation at the very end covering the whole answer.
4. If you genuinely need to add general knowledge NOT found in the context to make the answer understandable, clearly label it as: "(General knowledge, not from your documents: ...)" - never blend it in silently as if it came from the documents.
5. Do not fabricate page numbers, filenames, or quotes. Only cite what is actually shown in the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


@dataclass
class RAGResult:
    """
    Everything about one RAG query - the answer, plus the exact
    chunks used to produce it. Keeping retrieved_chunks attached
    (not just the final text) is what makes Phase 8's citation
    verification possible - you can check the model's claimed
    citations against what was actually retrieved.
    """
    question: str
    answer: str
    retrieved_chunks: list[dict]


def construct_context(chunks: list[dict]) -> str:
    """
    Turn retrieved chunks into a single formatted context block for
    the prompt.

    Input:  list of chunk dicts from vectorstore.search() (text,
            filename, page_number, similarity, ...)
    Output: a single string, each chunk clearly labeled with its
            source, so the LLM can produce accurate citations back
            to filename + page number.

    Formatting each chunk with an explicit "[Source: ...]" header
    (rather than just concatenating raw text) is what lets the LLM
    copy accurate citations into its answer - it can only cite
    correctly if the source info is visibly attached to each piece
    of text it's reading.
    """
    if not chunks:
        return "(No relevant context was found in the document library.)"

    blocks = []
    for chunk in chunks:
        block = (
            f"[Source: {chunk['filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    top_k: int = 5,
    llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> RAGResult:
    """
    Run the full RAG pipeline: embed the question, retrieve top-k
    chunks, construct a grounded prompt, generate an answer.

    Input:  a natural-language question, how many chunks to retrieve,
            optional LLM config
    Output: RAGResult containing the answer text AND the retrieved
            chunks it was based on (needed for citation verification
            in Phase 8, and for debugging retrieval vs. generation
            issues separately)
    """
    logger.info(f"RAG query: {question!r}")

    # Step 1-3: embed query, search, get top-k chunks
    client = get_client()
    collection = create_collection(client)
    query_vector = embed_text(question)
    retrieved_chunks = search(collection, query_vector, top_k=top_k)

    # Step 4: build context block
    context = construct_context(retrieved_chunks)

    # Step 5: fill the prompt template
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 6: generate, grounded in that context
    answer = generate(prompt, config=llm_config)

    return RAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_questions = [
        # A question the documents SHOULD answer well
        "What is the main innovation of the Transformer architecture?",
        # The exact question that exposed a retrieval weakness in Phase 5 -
        # testing whether grounded generation can still produce a correct
        # answer even when the "best" defining chunk didn't rank #1
        "What is NSGA-II?",
        # A question the documents almost certainly CANNOT answer -
        # tests whether the model correctly says "not found" instead
        # of hallucinating from pretrained knowledge
        "What is the capital of Australia?",
    ]

    for q in test_questions:
        result = answer_question(q, top_k=5)
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {result.question}")
        print(f"{'=' * 70}")
        print(f"\nANSWER:\n{result.answer}")
        print(f"\nBased on {len(result.retrieved_chunks)} retrieved chunks:")
        for c in result.retrieved_chunks:
            print(f"  - {c['filename']} p.{c['page_number']} (similarity: {c['similarity']:.4f})")