# Week 13 – Agentic RAG App Setup Instructions

# Agentic RAG with LangGraph, Gemini, and FAISS

This project demonstrates a fully working **Agentic RAG (Retrieval-Augmented Generation)** system combining:

- LangGraph for multi-step agentic decision flows  
- Gemini 2.x API for LLM reasoning + embeddings  
- FAISS for local semantic search  
- SQLite for persistent chat memory  
- React frontend with follow-up buttons  
- Domain guardrails for in-domain/out-of-domain detection  

---

## Folder Structure

```
agentic-rag-langgraph/
├── agentic-rag-ui/                # React frontend
│   └── (Vite + React code)
│
├── langgraph-server/              # Python backend
│   ├── app.py
│   ├── routes/
│   │   └── rag_routes.py
│   ├── core/
│   │   └── langgraph_runner.py
│   ├── data/
│   │   └── info.txt
│   ├── faiss_index/
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── utils/
│   │   └── cosine_similarity.py
│   ├── build_vectorstore.py
│   ├── document_loader.py
│   ├── memory.db
│   ├── requirements.txt
│   └── .env
│
├── anim.html
├── diagram.html
└── README.md
```

---

## Backend Setup

### 0. Build the FAISS Vector Store

```bash
cd langgraph-server
python build_vectorstore.py
```

---

### 1. Create a Virtual Environment

```bash
python -m venv venv
venv/Scripts/activate      # Windows
# or
source venv/bin/activate   # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Gemini API Key

Create:

```
langgraph-server/.env
```

Inside it:

```
GEMINI_API_KEY=your_key_here
```

### 4. Run the Flask Server

```bash
python app.py
```

Runs at:

```
http://localhost:5000
```

---

## Frontend Setup

```bash
cd agentic-rag-ui
npm install
npm run dev
```

Runs at:

```
http://localhost:5173
```

---

## LangGraph Agent Workflow

```
User Query
      ↓
domain_check
      ↓
retrieve
      ↓
reason
      ├── sufficient → respond
      └── insufficient → retrieve_again → respond
      ↓
reflect
      ↓
Return JSON → React UI → stored in SQLite
```

---

## Features

- Semantic retrieval (FAISS)
- Multi-hop retrieval
- Domain guardrails
- Strict grounding (no hallucinations)
- Follow-up question generation
- Persistent conversation memory
- React UI with markdown
- Reset session + chat history API

---

