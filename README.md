# 🧠 Offline Second Brain

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) system for semantic search and Q&A over your own research library — PDFs, papers, and notes. Nothing leaves your machine.

Built from first principles across 13 phases: document ingestion → chunking → embeddings → vector search → local LLM generation → citation verification → evaluation → failure analysis → a full-stack web UI.

---

## Why this exists

Most "chat with your PDFs" tools send your documents to a cloud API. This one doesn't — every component (embedding model, LLM, vector database) runs locally via [Ollama](https://ollama.com) and [ChromaDB](https://www.trychroma.com/). It's suited for:

- Literature review across a personal paper library, with page-cited answers you can verify
- Working with unpublished, confidential, or pre-submission research
- Offline / air-gapped environments
- Anyone who wants a real, evidence-backed understanding of how RAG systems actually behave — including where they fail

---

## Architecture

```
PDF / TXT / MD documents
        │
        ▼
  Document Loader ──► Text Extraction ──► Cleaning (boilerplate/footnote removal)
        │
        ▼
     Chunking (recursive character splitting, configurable size/overlap)
        │
        ▼
  Embedding Model (nomic-embed-text, local via Ollama)
        │
        ▼
   Vector Database (ChromaDB, persistent)
        ▲
        │                    User Query
        │                        │
        │                        ▼
        │                 Query Embedding
        │                        │
        └───────── Semantic Search (dense) or Hybrid (dense + BM25) 
                                  │
                                  ▼
                          Top-K Relevant Chunks
                                  │
                                  ▼
                          Context Construction
                                  │
                                  ▼
                    Grounded Prompt Template
                                  │
                                  ▼
                     Local LLM (Ollama, streamed)
                                  │
                                  ▼
                  Answer + Verified Citations
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM inference | Ollama (`llama3.2:3b`) | Fully local, fits comfortably on a 6GB GPU |
| Embeddings | `nomic-embed-text` | Fast, strong general-purpose local embedding model |
| Vector DB | ChromaDB | Simple, persistent, good default for a first RAG build |
| Retrieval | Dense + BM25 hybrid (RRF fusion) | Compared empirically, not assumed |
| Backend | FastAPI | Thin API layer over a modular `app/` package — zero business logic in the API itself |
| Frontend | React + Vite | Streamed, token-by-token responses; custom design system |
| Testing | pytest | 50+ tests across every module |

---

## Project structure

```
offline-second-brain/
├── app/
│   ├── ingestion/       # PDF/TXT/MD loading, cleaning, low-content detection
│   ├── chunking/        # Recursive character splitting
│   ├── embeddings/      # Local embedding + cosine similarity
│   ├── retrieval/       # ChromaDB, hybrid search, reranking
│   ├── generation/      # LLM interface, RAG pipeline, citation verification
│   └── evaluation/      # Recall@K / Precision@K / MRR framework
├── backend/             # FastAPI API layer
├── frontend/            # React + Vite UI
├── experiments/         # Saved evaluation runs, standalone scripts
├── tests/                # pytest suite
├── data/raw/             # Your PDFs (gitignored)
├── vectorstore/          # ChromaDB persistence (gitignored)
├── FAILURE_ANALYSIS.md   # Documented bugs found, fixed, and open limitations
├── start.bat             # One-click launcher (Windows)
└── app.py                # Streamlit UI (earlier prototype, still functional)
```

---

## Setup

### Prerequisites
- Python 3.11+, Node.js 18+
- [Ollama](https://ollama.com) installed

### 1. Clone and install
```bash
git clone https://github.com/anuj-sarkar/offline-second-brain.git
cd offline-second-brain

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 2. Pull local models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Add documents
Drop PDFs into `data/raw/`.

### 4. Run
**Easiest:** double-click `start.bat` (Windows) — launches backend, frontend, and opens the browser automatically.

**Manual:**
```bash
uvicorn backend.main:app --reload --port 8000    # terminal 1
cd frontend && npm run dev                        # terminal 2
```
Then open `http://localhost:5173`, upload PDFs, click **Index**, and ask questions.

---

## Evaluation results

Built a real evaluation framework (Recall@K, Precision@K, MRR) rather than eyeballing individual queries. Comparing dense-only vs. hybrid (BM25 + dense) retrieval across 5 real questions against the indexed papers:

| Metric | Dense | Hybrid |
|---|---|---|
| Recall@5 | **0.900** | 0.700 |
| Precision@5 | 0.280 | **0.320** |
| MRR | **0.717** | 0.600 |

**Finding:** dense-only retrieval outperformed hybrid on this set — contrary to the common assumption that hybrid search is strictly better. Root cause: BM25's term-frequency scoring favors pages that repeat a term often (e.g. results/discussion sections) over the single, concise definitional mention (e.g. an abstract), which is exactly the opposite of what definitional queries need. Full writeup in [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md).

---

## Failure analysis

Rather than stopping at "it works," this project includes a dedicated failure-analysis pass — six real, reproduced cases (four fixed, two documented as open limitations), each with root cause and measured before/after impact. See [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) for the full writeup, including:
- A BM25 tokenizer bug that silently broke keyword matching on punctuation-adjacent terms
- A structural low-content detection gap letting figure-caption noise pollute the vector index
- Cross-publisher PDF boilerplate contamination (NeurIPS vs. IEEE footnote conventions)
- A Unicode character variant silently breaking a regex pattern

---

## Features

- 📄 PDF / TXT / Markdown ingestion with automatic boilerplate/footnote cleaning
- 🔍 Dense and hybrid (BM25 + dense, RRF-fused) semantic search
- 💬 Streamed, grounded LLM answers — explicitly instructed to decline when context is insufficient
- ✅ Automatic citation verification against actually-retrieved chunks (flags fabricated citations)
- 📊 Quantitative retrieval evaluation framework
- 🖥️ Full-stack web UI (FastAPI + React) with live token streaming
- 🔒 100% local — no cloud API calls, ever