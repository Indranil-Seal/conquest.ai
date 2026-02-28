# conquest.ai

**An AI-powered research assistant for Data Science, Machine Learning, and AI.**

conquest.ai combines Claude Sonnet 4.6 with Retrieval-Augmented Generation (RAG) over the DREAM library — a curated collection of DS/ML/AI textbooks and research papers — to give you accurate, well-sourced, and richly formatted answers with LaTeX equations and Mermaid diagrams.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    conquest.ai — System Overview                │
└─────────────────────────────────────────────────────────────────┘

  User Browser
      │
      │ HTTP
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Chainlit UI Layer                            │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Chat Interface  │  │  LaTeX Equations  │  │    Mermaid    │  │
│  │  (Streaming)     │  │  (MathJax 3)      │  │   Diagrams    │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ Python (async)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RAG Orchestration (LlamaIndex)                  │
│                                                                 │
│   User Query                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────┐      ┌──────────────────────────────┐  │
│  │  VectorIndexRetriever│      │    Response Synthesizer      │  │
│  │  (top-5 similarity) │─────▶│    (compact mode)            │  │
│  └──────────┬──────────┘      └──────────────┬───────────────┘  │
│             │                                │                   │
│             ▼                                ▼                   │
│  ┌──────────────────────┐      ┌──────────────────────────────┐  │
│  │  ChromaDB            │      │  Claude Sonnet 4.6           │  │
│  │  (Local Vector Store)│      │  (Anthropic API)             │  │
│  │                      │      │                              │  │
│  │  ~50 documents       │      │  Streaming response          │  │
│  │  chunked at 512 tok  │      │  Max 2048 tokens             │  │
│  └──────────────────────┘      └──────────────────────────────┘  │
│             ▲                                                     │
│  ┌──────────┴────────────────────────────────────────────────┐   │
│  │  Embeddings: sentence-transformers/all-MiniLM-L6-v2       │   │
│  │  (local — no API calls required for retrieval)            │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                             ▲
                             │ (built by ingest.py)
┌────────────────────────────┴────────────────────────────────────┐
│                      DREAM Library                              │
│                                                                 │
│  GitHub: https://github.com/Indranil-Seal/DREAM                 │
│                                                                 │
│  📘 Textbooks (~10)     │  Topics:                              │
│  📄 Research Papers(~35)│  • Machine Learning algorithms        │
│  📋 References  (~5)    │  • Statistics & probability           │
│                         │  • Deep learning & neural networks    │
│  Formats: PDF, EPUB,    │  • Programming (Python, R)            │
│  DOCX  (~400 MB total)  │  • Domain applications                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User asks a question
        │
        ▼
[1] Embed query with all-MiniLM-L6-v2 (local)
        │
        ▼
[2] ChromaDB similarity search → top-5 most relevant chunks
        │
        ▼
[3] Assemble prompt: System prompt + retrieved context + user query
        │
        ▼
[4] Claude Sonnet 4.6 generates streamed response (LaTeX, Mermaid, code)
        │
        ▼
[5] Chainlit displays response + collapsible source citations
```

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| **UI** | Chainlit | ≥ 1.3.0 |
| **LLM** | Claude Sonnet 4.6 (Anthropic) | `claude-sonnet-4-6` |
| **RAG Framework** | LlamaIndex | ≥ 0.11.0 |
| **Vector Database** | ChromaDB (local, persistent) | ≥ 0.5.0 |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | ≥ 3.0.0 |
| **PDF Parsing** | PyMuPDF (`fitz`) | ≥ 1.24.0 |
| **EPUB Parsing** | ebooklib + BeautifulSoup4 | — |
| **DOCX Parsing** | python-docx | ≥ 1.1.0 |

---

## Project Structure

```
conquest.ai/
├── README.md                    ← This file
├── about_conquest_ai.txt        ← Project specification
├── .env.example                 ← API key template (copy to .env)
├── .gitignore                   ← Excludes .env, DREAM/, data/
├── requirements.txt             ← Python dependencies
├── chainlit.md                  ← Chatbot welcome screen
├── app.py                       ← Chainlit app (main entry point)
├── ingest.py                    ← Document ingestion script (run once)
├── src/
│   ├── __init__.py
│   ├── prompts.py               ← System prompt & prompt templates
│   ├── indexer.py               ← Document loading, chunking, embedding
│   └── rag.py                   ← LlamaIndex RAG pipeline
├── data/
│   └── chroma_db/               ← ChromaDB vector store (auto-created)
└── DREAM/                       ← DREAM library (cloned by ingest.py)
```

---

## Setup Guide

### Prerequisites
- Python 3.11+
- Git
- An [Anthropic API key](https://console.anthropic.com)

### Installation

**1. Clone this repository**
```bash
git clone <your-repo-url>
cd conquest.ai
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API key**
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

**4. Build the knowledge base** *(first time only — takes 10–30 min)*
```bash
python ingest.py
```
This will:
- Clone the DREAM library from GitHub (~400 MB of PDFs)
- Parse and chunk all documents
- Build the ChromaDB vector index at `data/chroma_db/`

**5. Start conquest.ai**
```bash
chainlit run app.py
```
Open your browser at `http://localhost:8000`

---

## Usage Examples

```
Explain how gradient boosting works, with equations
```
```
What is the LASSO regularization objective function?
```
```
Show me a diagram of the neural network training pipeline
```
```
How does k-means++ differ from standard k-means?
```
```
Give me Python code for a Random Forest classifier
```

---

## Configuration

Override defaults via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CONQUEST_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |
| `DREAM_PATH` | `./DREAM` | DREAM library path |

---

## Re-indexing

If you update the DREAM library or want to add documents:
```bash
python ingest.py --force
```

## DREAM Library

The **DREAM** *(Data Science, Research, and Engineering Artifacts for Machine Learning)* library is a curated collection hosted at:
https://github.com/Indranil-Seal/DREAM

It contains ~50 files covering:
- Core ML textbooks (ESL, ISLP, Hands-On ML)
- Research papers on algorithms (SVM, AdaBoost, LASSO, k-means++, Isolation Forest)
- Domain applications (finance, healthcare, manufacturing)
- Programming references (Python, R, Pandas, dplyr)
