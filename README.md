# conquest.ai

> AI-powered research assistant for Data Science, Machine Learning, and AI

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-FF4B4B?logo=chainlink&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.6-D97706?logo=anthropic&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex-6366F1?logo=llama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00?logo=databricks&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-FFD21E?logo=huggingface&logoColor=black)

conquest.ai combines **Claude Sonnet 4.6** with **Retrieval-Augmented Generation** over the DREAM library — a curated collection of DS/ML/AI textbooks and research papers — to give rigorous, two-part answers with LaTeX equations, Mermaid diagrams, and conversation memory.

---

## Architecture

```mermaid
flowchart TD
    User(["👤 User\n(Browser)"])

    subgraph UI ["🖥️ Chainlit UI Layer"]
        Chat["💬 Chat Interface\n(Streaming)"]
        LaTeX["∑ LaTeX Equations\n(MathJax 3)"]
        Mermaid["📊 Mermaid Diagrams"]
    end

    subgraph RAG ["🧠 RAG Orchestration — LlamaIndex"]
        Memory["🗄️ ChatMemoryBuffer\n(Conversation History)"]
        Condense["🔀 Query Condenser\n(Reverse Prompt Injection)"]
        Retriever["🔍 VectorIndexRetriever\ntop-k = 5"]
        Synthesizer["⚙️ Response Synthesizer\n(compact mode)"]
    end

    subgraph LLM ["🤖 LLM — Anthropic"]
        Claude["✦ Claude Sonnet 4.6\nStreaming · max 4096 tokens"]
    end

    subgraph VDB ["📦 Vector Store — ChromaDB (local)"]
        Chunks["🗂️ ~50 Documents\nChunked · 512 tokens · overlap 50"]
        Embed["🔢 Embeddings\nall-MiniLM-L6-v2 (local)"]
    end

    subgraph DREAM ["📚 DREAM Library (Git)"]
        PDFs["📄 ~48 PDFs\nTextbooks & Papers"]
        EPUB["📖 1 EPUB"]
        DOCX["📝 1 DOCX"]
    end

    User -->|"message"| Chat
    Chat -->|"query + history"| Memory
    Memory -->|"history + query"| Condense
    Condense -->|"condensed query"| Retriever
    Retriever -->|"similarity search"| Chunks
    Chunks --> Synthesizer
    Memory -->|"chat history"| Synthesizer
    Synthesizer -->|"prompt"| Claude
    Claude -->|"token stream"| Chat
    Chat -->|"rendered response"| User
    LaTeX -.->|"renders"| Chat
    Mermaid -.->|"renders"| Chat
    Embed -->|"indexes"| Chunks
    DREAM -->|"ingest.py\n(one-time)"| Embed
```

### Data Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant C as 💬 Chainlit
    participant M as 🗄️ Memory
    participant Q as 🔀 Condenser
    participant V as 📦 ChromaDB
    participant L as ✦ Claude

    U->>C: "How does learning rate affect gradient descent?"
    C->>M: append user message
    M->>Q: history + new message
    Q->>L: condense into standalone query
    L-->>Q: "How does learning rate affect gradient descent convergence?"
    Q->>V: similarity search (top-5)
    V-->>C: relevant chunks from DREAM library
    C->>L: system prompt + chat history + context + query
    L-->>C: stream tokens (Part 1 + Part 2)
    C-->>U: rendered response + citations
    C->>M: append assistant response
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| ![Chainlit](https://img.shields.io/badge/-Chainlit-FF4B4B?logo=chainlink&logoColor=white&style=flat-square) | Chainlit ≥ 1.3 | Chat UI — LaTeX, Mermaid, streaming |
| ![Claude](https://img.shields.io/badge/-Claude%204.6-D97706?logo=anthropic&logoColor=white&style=flat-square) | Claude Sonnet 4.6 | LLM — reasoning, math, code generation |
| ![LlamaIndex](https://img.shields.io/badge/-LlamaIndex-6366F1?style=flat-square) | LlamaIndex ≥ 0.11 | RAG orchestration + chat memory |
| ![ChromaDB](https://img.shields.io/badge/-ChromaDB-FF6F00?logo=databricks&logoColor=white&style=flat-square) | ChromaDB ≥ 0.5 | Local persistent vector store |
| ![HuggingFace](https://img.shields.io/badge/-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black&style=flat-square) | sentence-transformers | Local embeddings — no API needed |
| ![PyMuPDF](https://img.shields.io/badge/-PyMuPDF-3776AB?logo=python&logoColor=white&style=flat-square) | PyMuPDF (fitz) | PDF parsing — best for math symbols |
| | ebooklib + BS4 | EPUB parsing |
| | python-docx | DOCX parsing |

---

## Key Features

- **Two-part responses** — every answer starts with a technical definition + intuition (Part 1), followed by a full mathematical/statistical deep-dive with LaTeX equations (Part 2)
- **Conversation memory** — the chat engine remembers the full session; ask follow-up questions naturally
- **Reverse prompt injection** — prior conversation turns are condensed back into each new retrieval query, keeping context coherent across multi-turn sessions
- **DREAM library RAG** — responses are grounded in curated textbooks and research papers
- **Local embeddings** — `all-MiniLM-L6-v2` runs fully offline; only the LLM call uses the API

---

## Project Structure

```
conquest.ai/
├── README.md                    ← This file
├── .env.example                 ← API key template (copy to .env)
├── .gitignore
├── requirements.txt
├── chainlit.md                  ← Chatbot welcome screen
├── app.py                       ← Chainlit app (main entry point)
├── ingest.py                    ← Document ingestion script (run once)
├── src/
│   ├── prompts.py               ← System prompt, two-part format, LaTeX rules
│   ├── indexer.py               ← PDF/EPUB/DOCX loading, chunking, ChromaDB
│   └── rag.py                   ← CondensePlusContextChatEngine + streaming
├── .chainlit/
│   └── config.toml              ← LaTeX enabled, assistant name set
├── data/
│   └── chroma_db/               ← ChromaDB vector store (auto-created)
└── DREAM/                       ← DREAM library (cloned by ingest.py, gitignored)
```

---

## Setup

**Prerequisites:** Python 3.11+, Git, [Anthropic API key](https://console.anthropic.com)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env → set ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the knowledge base (first time only — clones ~400MB DREAM library)
python ingest.py

# 4. Start the chatbot
chainlit run app.py
# Open http://localhost:8000
```

**Re-index after adding documents:**
```bash
python ingest.py --force
```

---

## Example Queries

```
Explain gradient boosting with equations
```
```
What is the LASSO objective function and why does it produce sparse solutions?
```
```
Show me a diagram of the neural network training pipeline
```
```
How does k-means++ initialization improve over random initialization?
```
```
What are the bias-variance tradeoff implications for Random Forests?
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CONQUEST_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |
| `DREAM_PATH` | `./DREAM` | DREAM library path |

---

## DREAM Library

**DREAM** *(Data Science, Research, and Engineering Artifacts for Machine Learning)*
— [github.com/Indranil-Seal/DREAM](https://github.com/Indranil-Seal/DREAM)

~50 files · ~400 MB · Topics: ML algorithms, statistics, deep learning, Python/R, domain applications
