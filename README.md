# End-to-End RAG Pipeline

A fully working Retrieval-Augmented Generation pipeline in pure Python.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the demo
python rag_pipeline.py
```

---

## How It Works

```
Your Documents
     │
     ▼
┌─────────────┐
│  1. Loader  │  Reads text from strings, .txt files, etc.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  2. Chunker │  Splits docs into overlapping windows (e.g. 500 chars, 100 overlap)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  3. Embedder │  Converts each chunk to a dense vector (all-MiniLM-L6-v2)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  4. FAISS DB │  Stores vectors for fast cosine similarity search
└──────┬───────┘
       │ (index is built once, queries happen many times)
       ▼

User Question
     │
     ▼
┌──────────────┐
│  5. Retrieve │  Embeds question → finds top-k most similar chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  6. Generate │  Sends question + chunks to Claude → grounded answer
└──────┬───────┘
       │
       ▼
     Answer ✓
```

---

## Key Configuration

| Parameter | Default | What it does |
|---|---|---|
| `chunk_size` | 500 | Max characters per chunk |
| `chunk_overlap` | 100 | Shared characters between adjacent chunks |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `llm_model` | `claude-sonnet-4-20250514` | Claude model for generation |
| `top_k` | 4 | Number of chunks retrieved per query |

---

## Using Your Own Documents

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# From raw text strings
pipeline.ingest(["Your document text here...", "Another document..."])

# From .txt files
pipeline.ingest(["doc1.txt", "doc2.txt"])

# Save the index so you don't re-embed every time
pipeline.save("my_index/")

# Later: reload without re-embedding
pipeline.load("my_index/")

# Query
answer = pipeline.query("What does the document say about X?")
```

---

## Extending the Pipeline

| What to extend | Where |
|---|---|
| Support PDF/DOCX/URLs | `load_documents()` function |
| Smarter chunking (by paragraph/heading) | `TextChunker._split_text()` |
| Faster/bigger embedding model | `VectorStore.__init__()` model name |
| Scalable vector DB (Pinecone, Weaviate) | Replace `VectorStore` |
| Better prompts / conversation history | `AnswerGenerator.SYSTEM_PROMPT` |
