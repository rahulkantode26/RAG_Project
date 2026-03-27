"""
========================================================
  END-TO-END RAG PIPELINE (Retrieval-Augmented Generation)
========================================================

WHAT IS RAG?
------------
RAG combines a retrieval system with a language model.
Instead of relying solely on the LLM's training data,
you feed it *relevant documents* at query time so it
can answer questions grounded in YOUR data.

PIPELINE STAGES:
  1. Load documents
  2. Chunk documents into smaller pieces
  3. Embed chunks into vectors
  4. Store vectors in a vector database
  5. At query time: embed the question, retrieve top-k chunks
  6. Feed question + retrieved chunks to the LLM
  7. Return the grounded answer

DEPENDENCIES (install once):
  pip install anthropic sentence-transformers faiss-cpu numpy

HOW TO RUN:
  python rag_pipeline.py
"""

import os
import re
import json
import textwrap
import numpy as np
from typing import List, Dict, Tuple, Optional

# ── Third-party ──────────────────────────────────────────────────────────────
import anthropic                                      # Anthropic LLM client
from sentence_transformers import SentenceTransformer # Local embedding model
import faiss                                          # Facebook vector store


# =============================================================================
# STAGE 1 – DOCUMENT LOADER
# =============================================================================

class Document:
    """A single document with text content and metadata."""

    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Document(text='{preview}...', metadata={self.metadata})"


def load_documents(source: str | List[str]) -> List[Document]:
    """
    Load documents from:
      - A plain string  (treated as raw text)
      - A list of strings (each is raw text)
      - A .txt file path
      - A list of .txt file paths

    In production you'd add loaders for PDF, DOCX, URLs, databases, etc.
    """
    if isinstance(source, str):
        source = [source]

    docs = []
    for item in source:
        if os.path.isfile(item):
            with open(item, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(text=text, metadata={"source": item}))
        else:
            # Treat as raw text
            docs.append(Document(text=item, metadata={"source": "inline"}))

    print(f"[Loader]   Loaded {len(docs)} document(s)")
    return docs


# =============================================================================
# STAGE 2 – TEXT CHUNKER
# =============================================================================

class TextChunker:
    """
    Splits long documents into overlapping chunks.

    WHY CHUNK?
      LLMs have context-window limits, and similarity search
      works better on small focused passages than huge documents.

    WHY OVERLAP?
      Sentences at chunk boundaries don't get cut off – the
      overlap ensures every piece of information is fully
      represented in at least one chunk.

    Parameters
    ----------
    chunk_size  : maximum characters per chunk (default 500)
    chunk_overlap: characters shared between consecutive chunks (default 100)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        assert chunk_overlap < chunk_size, "overlap must be less than chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            doc_chunks = self._split_text(doc.text)
            for i, chunk_text in enumerate(doc_chunks):
                meta = {**doc.metadata, "chunk_index": i, "total_chunks": len(doc_chunks)}
                chunks.append(Document(text=chunk_text, metadata=meta))
        print(f"[Chunker]  {len(documents)} doc(s) → {len(chunks)} chunk(s)  "
              f"(size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Sliding-window split that respects sentence boundaries when possible."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Try to end at a sentence boundary (. ! ?) within the last 20 %
                boundary = text.rfind(".", start + int(self.chunk_size * 0.8), end)
                if boundary == -1:
                    boundary = text.rfind(" ", start + int(self.chunk_size * 0.8), end)
                if boundary != -1:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap  # slide back by overlap
        return chunks


# =============================================================================
# STAGE 3 & 4 – EMBEDDER + VECTOR STORE
# =============================================================================

class VectorStore:
    """
    Embeds text chunks and stores them in a FAISS index for fast
    nearest-neighbour retrieval.

    WHAT IS AN EMBEDDING?
      A high-dimensional float vector that captures the *meaning* of text.
      Sentences with similar meaning end up close together in vector space.

    WHAT IS FAISS?
      A library (by Meta) that lets you search millions of vectors in
      milliseconds using approximate nearest-neighbour algorithms.

    Parameters
    ----------
    model_name : any sentence-transformers model
                 'all-MiniLM-L6-v2' is small (80 MB) and fast.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading embedding model '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None  # inner-product (cosine after norm)
        self.chunks: List[Document] = []
        self.dim: Optional[int] = None

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Document]) -> None:
        """Embed all chunks and build the FAISS index."""
        texts = [c.text for c in chunks]
        print(f"[Embedder] Embedding {len(texts)} chunk(s) …")
        embeddings = self._embed(texts)          # shape: (N, dim)

        self.dim = embeddings.shape[1]
        # IndexFlatIP = exact brute-force with inner product
        # After L2-normalisation this equals cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)               # add all vectors
        self.chunks = chunks
        print(f"[VectorDB] Index built: {self.index.ntotal} vectors, dim={self.dim}")

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        Return the top_k most similar chunks to the query.

        Returns a list of (Document, score) tuples sorted by relevance.
        Score is cosine similarity in [0, 1].
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty – call build() first.")

        q_vec = self._embed([query])             # shape: (1, dim)
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results  # already sorted by score desc

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save the FAISS index and chunk metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump([{"text": c.text, "metadata": c.metadata} for c in self.chunks], f, indent=2)
        print(f"[VectorDB] Saved to '{directory}/'")

    def load(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json")) as f:
            raw = json.load(f)
        self.chunks = [Document(text=r["text"], metadata=r["metadata"]) for r in raw]
        self.dim = self.index.d
        print(f"[VectorDB] Loaded {self.index.ntotal} vectors from '{directory}/'")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalised float32 vectors."""
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        # L2 normalise so inner product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)          # avoid div-by-zero
        return vecs / norms


# =============================================================================
# STAGE 5 – ANSWER GENERATOR (LLM)
# =============================================================================

class AnswerGenerator:
    """
    Sends the user question + retrieved context to Claude and returns an answer.

    PROMPT DESIGN:
      - System prompt tells the model to be a grounded assistant
      - We inject the retrieved chunks as numbered [Source N] blocks
      - We instruct it to cite sources and say "I don't know" when unsure

    Parameters
    ----------
    model   : Claude model to use (claude-sonnet-4-20250514 recommended)
    max_tokens: maximum tokens in the generated answer
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a helpful assistant that answers questions ONLY based on the
        provided context passages. Follow these rules strictly:

        1. Base your answer solely on the context below – do NOT use outside knowledge.
        2. Cite the source number(s) you used, e.g. [Source 1].
        3. If the context doesn't contain enough information, say:
           "I don't have enough information in the provided documents to answer that."
        4. Be concise and factual.
    """).strip()

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024):
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, context_chunks: List[Tuple[Document, float]]) -> str:
        """
        Build the prompt with context and query, call the API, return the answer.
        """
        # Format retrieved chunks as numbered passages
        context_block = "\n\n".join(
            f"[Source {i+1}] (score={score:.3f}, from={chunk.metadata.get('source','?')})\n{chunk.text}"
            for i, (chunk, score) in enumerate(context_chunks)
        )

        user_message = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            "Answer:"
        )

        print(f"[LLM]      Calling {self.model} with {len(context_chunks)} context chunk(s) …")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()


# =============================================================================
# STAGE 6 – RAG PIPELINE (orchestrator)
# =============================================================================

class RAGPipeline:
    """
    Ties all stages together into a single easy-to-use pipeline.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.ingest(documents)    # build the index once
    answer = pipeline.query("…")  # ask questions any number of times
    pipeline.save("my_index/")    # optional persistence
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "claude-sonnet-4-20250514",
        top_k: int = 4,
    ):
        self.chunker   = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_db = VectorStore(model_name=embedding_model)
        self.generator = AnswerGenerator(model=llm_model)
        self.top_k     = top_k

    # ── Indexing ─────────────────────────────────────────────────────────────

    def ingest(self, source: str | List[str]) -> None:
        """
        Full ingestion pipeline:
          load → chunk → embed → index
        """
        print("\n" + "="*55)
        print("  INGESTION PIPELINE")
        print("="*55)
        docs   = load_documents(source)
        chunks = self.chunker.split(docs)
        self.vector_db.build(chunks)
        print("  ✓ Ingestion complete\n")

    def save(self, directory: str) -> None:
        self.vector_db.save(directory)

    def load(self, directory: str) -> None:
        self.vector_db.load(directory)

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(self, question: str, verbose: bool = True) -> str:
        """
        Full query pipeline:
          embed question → retrieve chunks → generate answer
        """
        print("\n" + "="*55)
        print(f"  QUERY: {question}")
        print("="*55)

        # Retrieval
        results = self.vector_db.retrieve(question, top_k=self.top_k)

        if verbose:
            print(f"[Retrieval] Top {len(results)} chunks:")
            for i, (chunk, score) in enumerate(results):
                preview = chunk.text[:80].replace("\n", " ")
                print(f"  {i+1}. score={score:.3f}  \"{preview}…\"")

        # Generation
        answer = self.generator.generate(question, results)

        print(f"\n[Answer]\n{answer}\n")
        return answer


# =============================================================================
# DEMO  –  run `python rag_pipeline.py` to try it
# =============================================================================

SAMPLE_DOCUMENTS = [
    """
    Photosynthesis is the process used by plants, algae, and some bacteria to convert
    light energy—usually from the sun—into chemical energy stored in glucose.
    The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
    Photosynthesis occurs mainly in the chloroplasts, organelles that contain the
    green pigment chlorophyll. There are two main stages: the light-dependent reactions
    (which produce ATP and NADPH) and the Calvin cycle (which uses that energy to
    fix CO2 into glucose). Temperature, light intensity, and CO2 concentration all
    affect the rate of photosynthesis.
    """,
    """
    The mitochondria are often called the 'powerhouse of the cell' because they
    generate most of the cell's supply of adenosine triphosphate (ATP), used as a
    source of chemical energy. Mitochondria have two membranes: an outer membrane
    and a highly folded inner membrane called cristae. The process by which mitochondria
    generate ATP is called cellular respiration, which involves the Krebs cycle and
    oxidative phosphorylation. Mitochondria also regulate the cell cycle and cell growth,
    and play a key role in apoptosis (programmed cell death).
    """,
    """
    DNA (deoxyribonucleic acid) is the molecule that carries genetic information in
    all living organisms. It is made of two polynucleotide strands coiled around
    each other in a double helix. Each strand is made of nucleotides containing
    a phosphate group, a sugar (deoxyribose), and one of four nitrogenous bases:
    adenine (A), thymine (T), guanine (G), and cytosine (C). Base pairing follows
    strict rules: A pairs with T, and G pairs with C. DNA replication is
    semi-conservative—each new double helix contains one original strand and one
    newly synthesised strand. The central dogma of molecular biology describes the
    flow of genetic information: DNA → RNA → Protein.
    """,
]


def main():
    # ── Ensure API key is set ────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY is not set.")
        print("   export ANTHROPIC_API_KEY='sk-ant-…'")
        print("   The retrieval demo will still run; only LLM generation will fail.\n")

    # ── Build pipeline ───────────────────────────────────────────────────────
    pipeline = RAGPipeline(
        chunk_size=300,
        chunk_overlap=60,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="claude-sonnet-4-20250514",
        top_k=3,
    )

    pipeline.ingest(SAMPLE_DOCUMENTS)

    # ── Optional: save & reload the index ────────────────────────────────────
    pipeline.save("rag_index")
    # To reload later instead of re-indexing:
    # pipeline.load("rag_index")

    # ── Ask questions ────────────────────────────────────────────────────────
    questions = [
        "What is photosynthesis and where does it take place?",
        "How do mitochondria produce energy?",
        "What are the base-pairing rules in DNA?",
        "What is the difference between ATP and DNA?",   # tests cross-document reasoning
    ]

    for q in questions:
        pipeline.query(q)


if __name__ == "__main__":
    main()
