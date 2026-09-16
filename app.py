"""
app.py

Streamlit Application for Offline Second Brain.
Features:
  - Multi-turn conversational chat with automatic query condensation.
  - Real-time token streaming (st.write_stream).
  - Multi-format document ingestion (PDF via PyMuPDF, Markdown, Text).
  - Incremental SHA-256 caching and library management.
  - FlashRank cross-encoder reranking and citation grounding inspection.
"""

from pathlib import Path
import shutil
import streamlit as st

from app.chunking.splitter import chunk_page_records
from app.embeddings.embedder import embed_chunks
from app.generation.citations import verify_citations
from app.generation.rag_pipeline import answer_question_stream, retrieve_context
from app.ingestion.loader import load_documents_from_directory, load_single_markdown_or_text, load_single_pdf
from app.ingestion.manifest import IngestionManifest, compute_file_sha256
from app.retrieval.hybrid_search import load_or_build_bm25_index
from app.retrieval.vectorstore import (
    add_documents,
    create_collection,
    delete_document_by_filename,
    get_client,
    list_documents,
)
from config.model_config import LLMConfig

RAW_DATA_DIR = "data/raw"

st.set_page_config(
    page_title="Offline Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session state ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

manifest = IngestionManifest()

# ---------- Sidebar: Settings ----------
with st.sidebar:
    st.title("⚙️ Engine Settings")

    model_name = st.selectbox(
        "LLM Model (Ollama)",
        options=["llama3.2:3b", "llama3.1:8b", "phi4-mini", "mistral"],
        index=0,
        help="Local LLM model via Ollama.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Low = factual grounding; High = creative variation.",
    )

    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)

    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        options=[
            ("hybrid_no_rerank", "⚖️ Hybrid RRF — Dense + BM25 (Recommended)"),
            ("dense", "🔍 Dense-only (Semantic)"),
            ("hybrid", "⚡ Hybrid + FlashRank Rerank (⚠️ lower MRR on academic docs)"),
            ("dense_rerank", "🎯 Dense + FlashRank Rerank (⚠️ lower MRR on academic docs)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    st.divider()
    st.subheader("Chunking Parameters")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Overlap (chars)", min_value=0, max_value=500, value=200, step=50)

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("🔒 100% Offline & Private. No data leaves your machine.")


# ---------- Main Header ----------
st.title("🧠 Offline Second Brain")
st.caption("High-precision, local retrieval-augmented intelligence for your research and personal knowledge base.")

tab_chat, tab_library, tab_explorer = st.tabs([
    "💬 Second Brain Chat",
    "📚 Knowledge Library & Ingestion",
    "🔍 Retrieval Explorer",
])


# ==============================================================================
# TAB 1: Conversational Chat (Top Input & Newest Messages on Top)
# ==============================================================================
with tab_chat:
    # Check if any documents exist
    try:
        client = get_client()
        collection = create_collection(client)
        total_chunks = collection.count()
    except Exception:
        total_chunks = 0

    if total_chunks == 0:
        st.info("👋 Welcome! Your library is currently empty. Go to the **'Knowledge Library'** tab to upload and index documents.")

    # 1. Top Chat Bar
    user_query = st.chat_input("Ask a question across your research library...")

    # 2. Live response container right below top chat bar
    active_turn_container = st.container()

    if user_query:
        if total_chunks == 0:
            st.error("No documents indexed. Please index your documents in the Knowledge Library tab first.")
        else:
            with active_turn_container:
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    llm_config = LLMConfig(model_name=model_name, temperature=temperature)

                    with st.spinner("Analyzing and retrieving context..."):
                        token_gen, retrieved_chunks, standalone_query = answer_question_stream(
                            question=user_query,
                            top_k=top_k,
                            llm_config=llm_config,
                            retrieval_strategy=retrieval_strategy,
                            chat_history=st.session_state.chat_history,
                        )

                    if standalone_query != user_query:
                        st.caption(f"🔍 *Standalone Search Query: '{standalone_query}'*")

                    full_answer = st.write_stream(token_gen)
                    citation_report = verify_citations(full_answer, retrieved_chunks)

                    if citation_report.has_any_citations:
                        if citation_report.all_verified and citation_report.all_grounded:
                            st.caption("✅ All citations verified & grounded in source text")
                        elif citation_report.all_verified:
                            st.caption(f"⚠️ {citation_report.ungrounded_count} citation(s) weak grounding")
                        else:
                            st.caption(f"🚨 {citation_report.fabricated_count} unverified citation(s)")

                    with st.expander(f"📖 Context Sources ({len(retrieved_chunks)} chunks retrieved)"):
                        for c in retrieved_chunks:
                            score = c.get("rerank_score", c.get("similarity", c.get("fused_score", 0)))
                            sec = c.get("section_title", "General")
                            st.markdown(f"**{c.get('filename')}** | Page/Section {c.get('page_number')} ({sec}) — Score: `{score:.4f}`")
                            st.text(c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else ""))
                            st.divider()

            st.session_state.chat_history.append({
                "question": user_query,
                "standalone_query": standalone_query,
                "answer": full_answer,
                "chunks": retrieved_chunks,
                "citation_report": citation_report,
            })
            st.rerun()

    # 3. Render previous conversation (Newest First, Older Below)
    if st.session_state.chat_history:
        st.markdown("---")
        for turn in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(turn["question"])
                if turn.get("standalone_query") and turn["standalone_query"] != turn["question"]:
                    st.caption(f"🔍 *Searched as: '{turn['standalone_query']}'*")

            with st.chat_message("assistant"):
                st.write(turn["answer"])
                report = turn.get("citation_report")
                if report and report.has_any_citations:
                    if report.all_verified and report.all_grounded:
                        st.caption("✅ All citations verified & grounded in source text")
                    elif report.all_verified:
                        st.caption(f"⚠️ {report.ungrounded_count} citation(s) weak grounding")
                    else:
                        st.caption(f"🚨 {report.fabricated_count} unverified citation(s)")

                chunks = turn.get("chunks", [])
                with st.expander(f"📖 Context Sources ({len(chunks)} chunks retrieved)"):
                    for c in chunks:
                        score = c.get("rerank_score", c.get("similarity", c.get("fused_score", 0)))
                        sec = c.get("section_title", "General")
                        st.markdown(f"**{c.get('filename')}** | Page/Section {c.get('page_number')} ({sec}) — Score: `{score:.4f}`")
                        st.text(c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else ""))
                        st.divider()


# ==============================================================================
# TAB 2: Knowledge Library & Ingestion
# ==============================================================================
with tab_library:
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Add PDFs, Markdown notes (.md), or text files (.txt)",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for uploaded_file in uploaded_files:
            dest_path = Path(RAW_DATA_DIR) / uploaded_file.name
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)
            saved_count += 1
        st.success(f"Saved {saved_count} file(s) to `{RAW_DATA_DIR}/`")

    st.divider()
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.subheader("📂 Document Inventory")
    with col_btn:
        reindex_btn = st.button("🔄 Sync & Re-Index", type="primary", use_container_width=True)

    # Perform Ingestion
    if reindex_btn:
        with st.status("Executing layout-aware ingestion & indexing...", expanded=True) as status:
            raw_dir = Path(RAW_DATA_DIR)
            raw_dir.mkdir(parents=True, exist_ok=True)

            st.write("1. Reading documents with PyMuPDF & Markdown parsers...")
            records = load_documents_from_directory(str(raw_dir))

            if records:
                st.write(f"2. Chunking {len(records)} page/section records...")
                chunks = chunk_page_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                st.write(f"3. Embedding {len(chunks)} chunks with task prefixes ('search_document:')...")
                embedded_chunks = embed_chunks(chunks)

                st.write("4. Updating ChromaDB and persisting BM25 index...")
                client = get_client()
                collection = create_collection(client)
                add_documents(collection, embedded_chunks)
                load_or_build_bm25_index(collection, force_rebuild=True)

                # Update manifest
                by_doc = {}
                for c in chunks:
                    by_doc[c.filename] = by_doc.get(c.filename, 0) + 1

                for file_path in raw_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".md", ".txt"}:
                        sha = compute_file_sha256(file_path)
                        chunk_cnt = by_doc.get(file_path.name, 0)
                        manifest.update_entry(
                            filename=file_path.name,
                            sha256=sha,
                            file_type=file_path.suffix.lower(),
                            char_count=file_path.stat().st_size,
                            page_count=len([r for r in records if r.filename == file_path.name]),
                            chunk_count=chunk_cnt,
                        )

                status.update(label=f"Completed: {len(chunks)} chunks indexed across {len(manifest.entries)} documents.", state="complete")
            else:
                status.update(label="No document files found in data/raw/", state="error")
        st.rerun()

    # List Indexed Documents
    client = get_client()
    collection = create_collection(client)
    doc_counts = list_documents(collection)

    if doc_counts:
        for filename, count in sorted(doc_counts.items()):
            col_doc, col_meta, col_del = st.columns([3, 2, 1])
            with col_doc:
                icon = "📄" if filename.endswith(".pdf") else "📝"
                st.markdown(f"**{icon} {filename}**")
            with col_meta:
                st.caption(f"{count} chunks indexed in vector store")
            with col_del:
                if st.button("🗑️ Delete", key=f"del_{filename}"):
                    delete_document_by_filename(collection, filename)
                    file_path = Path(RAW_DATA_DIR) / filename
                    if file_path.exists():
                        file_path.unlink()
                    manifest.remove_entry(filename)
                    load_or_build_bm25_index(collection, force_rebuild=True)
                    st.success(f"Removed '{filename}'")
                    st.rerun()
            st.divider()
    else:
        st.info("No documents currently indexed. Upload files and click 'Sync & Re-Index'.")


# ==============================================================================
# TAB 3: Semantic Explorer
# ==============================================================================
with tab_explorer:
    st.subheader("🔬 Retrieval & Reranker Diagnostic")
    st.caption("Inspect raw candidate retrieval and test cross-encoder scores directly without LLM generation.")

    exp_query = st.text_input("Test Query", value="What is NSGA-II?")
    exp_strat = st.selectbox(
        "Diagnostic Strategy",
        options=["hybrid_no_rerank", "dense", "hybrid", "dense_rerank"],
        index=0,
    )
    exp_k = st.slider("Diagnostic Top-K", 1, 10, 5)

    if st.button("Run Diagnostic Search"):
        with st.spinner("Retrieving..."):
            diag_results = retrieve_context(exp_query, top_k=exp_k, retrieval_strategy=exp_strat)

        if diag_results:
            st.success(f"Retrieved {len(diag_results)} results:")
            for rank, r in enumerate(diag_results, 1):
                rerank_val = r.get("rerank_score")
                sim_val = r.get("similarity")
                fused_val = r.get("fused_score")

                score_str = ""
                if rerank_val is not None:
                    score_str += f" | **FlashRank Score:** `{rerank_val:.4f}`"
                if sim_val is not None:
                    score_str += f" | **Dense Sim:** `{sim_val:.4f}`"
                if fused_val is not None:
                    score_str += f" | **RRF Score:** `{fused_val:.4f}`"

                sec = r.get("section_title", "General")
                st.markdown(f"### Rank {rank}: {r.get('filename')} (Page {r.get('page_number')} — {sec})")
                st.markdown(f"*{score_str}*")
                st.code(r.get("text", ""))
                st.divider()
        else:
            st.warning("No results returned. Ensure documents are indexed.")