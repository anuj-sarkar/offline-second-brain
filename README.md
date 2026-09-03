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
  Document Loader (pymupdf4llm / PyMuPDF) ──► Structured Markdown & Math Extraction
        │
        ▼
  Text Cleaning (math symbol normalization, boilerplate/footnote removal)
        │
        ▼
  Math-Aware Chunking (recursive splitting, protects $$...$$ formula blocks)
        │
        ▼
  Embedding Model (nomic-embed-text with task prefixes, local via Ollama)
        │
        ▼
  Vector Database (ChromaDB, persistent)
        ▲
        │                    User Query (Conversational Condensation)
        │                                      │
        │                                      ▼
        │                               Query Embedding
        │                                      │
        └───────── Dense + BM25 Hybrid Retrieval (RRF Fusion)
                                               │
                                               ▼
                               Candidate Chunks (Top-15)
                                               │
                                               ▼
                              FlashRank Cross-Encoder Reranker
                                               │
                                               ▼
                                    Top-K Ranked Chunks
                                               │
                                               ▼
                                      Context Construction
                                               │
                                               ▼
                                 LaTeX-Grounded Prompt Template
                                               │
                                               ▼
                                  Local LLM (Ollama, streamed)
                                               │
                                               ▼
                     Streamed Answer + Verified Citations + KaTeX Math UI
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM inference | Ollama (`llama3.2:3b`) | Fully local, fits comfortably on a 6GB GPU |
| Embeddings | `nomic-embed-text` | Fast, strong general-purpose local embedding model with asymmetric task prefixes |
| Vector DB | ChromaDB | Simple, persistent, good default for a local RAG build |
| Retrieval | Hybrid (Dense + BM25) + FlashRank | Overcomes BM25 term-frequency bias on definitional queries with zero cloud latency |
| Document Ingestion | `pymupdf4llm` + PyMuPDF | Structured Markdown, tables, headings, and mathematical formula extraction |
| Backend | FastAPI | Thin API layer over a modular `app/` package — zero business logic in the API itself |
| Frontend | React + Vite + KaTeX | Streamed, token-by-token responses with native LaTeX math rendering (`react-markdown` + `rehype-katex`) |
| Testing | pytest | 50+ tests across every module |

---

## Project structure

```
offline-second-brain/
├── app/
│   ├── ingestion/       # PDF/TXT/MD loading, pymupdf4llm markdown extraction, cleaning
│   ├── chunking/        # Math-aware recursive character splitting
│   ├── embeddings/      # Local embedding + cosine similarity
│   ├── retrieval/       # ChromaDB, hybrid search, FlashRank reranking
│   ├── generation/      # LLM interface, RAG pipeline, LaTeX citations & prompt
│   └── evaluation/      # Recall@K / Precision@K / MRR framework
├── backend/             # FastAPI API layer
├── frontend/            # React + Vite UI with KaTeX math rendering
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

| Metric | Dense | Hybrid | Hybrid + FlashRank Reranker |
|---|---|---|---|
| Recall@5 | 0.900 | 0.700 | **1.000** |
| Precision@5 | 0.280 | 0.320 | **0.520** |
| MRR | 0.717 | 0.600 | **1.000** |

**Finding:** Dense-only retrieval originally outperformed naive hybrid on definitional queries due to BM25's term-frequency bias. Adding a local cross-encoder reranker (`flashrank` with `ms-marco-MiniLM-L-12-v2`) on top of candidate pools resolved this, achieving perfect 1.000 Recall@5 and 1.000 MRR. Full writeup in [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md).

---

## Failure analysis

Rather than stopping at "it works," this project catalogs seven real, reproduced failure modes across the entire RAG pipeline — each with root cause, fix, and measured before/after impact. See [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) for the full writeup:
1. **BM25 Tokenizer Bug:** Punctuation-adjacent query terms breaking keyword matching (`"nsga-ii?"` vs `"nsga-ii"`).
2. **Structural Low-Content Noise:** Attention-visualization word lists passing character-count filters and polluting retrieval.
3. **Cross-Publisher Boilerplate:** Publisher watermarks and author footnote blocks diluting paper abstracts.
4. **Unicode Character Variants:** Code points like Unicode asterisk (`∗` U+2217) bypassing ASCII regex cleaners.
5. **Definitional Term Frequency Trap:** BM25 favoring repetitive discussion sections over concise abstracts (solved via FlashRank reranker).
6. **Terminology Mismatch:** Representation drift on paraphrased domain queries (solved via asymmetric prefixes + reranker).
7. **Mathematical Formula Degradation:** Complex equations flattened into scrambled characters or severed across chunks (solved via `pymupdf4llm`, math-aware chunking, LaTeX prompt grounding, and frontend KaTeX rendering).

---

## Features

- 📄 **Structured Ingestion:** PDF / TXT / Markdown loading with automatic boilerplate stripping, table extraction, and header detection
- 📐 **Full LaTeX Math Support:** Mathematical formulas and equations extracted, protected during chunking, and rendered natively via KaTeX
- 🔍 **Hybrid Retrieval + Reranking:** Dense (`nomic-embed-text`) + BM25 with local FlashRank cross-encoder reranking
- 💬 **Grounded LLM Streaming:** Streamed answers with strict anti-hallucination rules and multi-turn query condensation
- ✅ **Citation Verification:** Automated verification of claimed `[Source: file, Page X]` citations against retrieved context
- 📊 **Evaluation Benchmark:** Recall@K, Precision@K, and MRR quantitative framework
- 🖥️ **Full-Stack Web UI:** Modern React + Vite frontend with live token streaming and dark mode
- 🔒 **100% Offline & Private:** Fully local inference — zero telemetry or cloud API dependencies